from __future__ import annotations

from collections import defaultdict
import re
import pandas as pd

NO_RESOLUTION = "didn\'t needed any resolving"

def clean_value(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value): return ""
    except Exception: pass
    return str(value).strip()

def normalized_text(value: str) -> str:
    text=clean_value(value).upper().replace("&", " AND ")
    text=re.sub(r"[^A-Z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def normalized_pin(value: str) -> str:
    text=clean_value(value)
    if re.fullmatch(r"\d+\.0+", text): text=text.split(".")[0]
    digits=re.sub(r"\D", "", text)
    return digits if len(digits)==6 else ""

def is_decimal_only_pin_change(old: str, new: str) -> bool:
    old_s=clean_value(old)
    return bool(re.fullmatch(r"\d+\.0+", old_s)) and normalized_pin(old_s)==new

class AutoResolver:
    """Conservative State/City/PIN recovery layer."""

    def __init__(self, masters, indexes, resolver, resolver_module):
        self.masters = masters
        self.indexes = indexes
        self.resolver = resolver
        self.resolver_module = resolver_module

        self.pin_states = indexes.get("pin_states", {})
        self.pin_rows = indexes.get("pin_rows", {})
        self.pin_post_offices = indexes.get("pin_post_offices", {})

        self.state_aliases = getattr(resolver, "STATE_ALIASES", {})

        self.ulb_names = set()
        self.ulb_token_index = defaultdict(set)
        ulb = masters.get("ulb")
        if ulb is not None and not ulb.empty:
            for value in ulb.get("ulb_name_clean", pd.Series(dtype=str)).dropna():
                name = normalized_text(value)
                if len(name) < 3:
                    continue
                self.ulb_names.add(name)
                for token in set(name.split()):
                    if len(token) >= 3:
                        self.ulb_token_index[token].add(name)

        # India Post reverse indexes.
        self.office_to_pins = defaultdict(set)
        self.state_names = set()
        india = masters.get("india_post")
        if india is not None and not india.empty:
            for _, row in india[["pincode_clean", "statename_clean", "officename_clean", "district_clean"]].fillna("").iterrows():
                pin = clean_value(row["pincode_clean"])
                state = normalized_text(row["statename_clean"])
                office = normalized_text(row["officename_clean"])
                if state:
                    self.state_names.add(state)
                if pin and office:
                    self.office_to_pins[office].add(pin)

        self.state_candidates = set(self.state_names)
        for alias, canonical in getattr(resolver, "STATE_ALIASES", {}).items():
            alias_n = normalized_text(alias)
            canonical_n = normalized_text(canonical)
            if alias_n:
                self.state_candidates.add(alias_n)
            if canonical_n:
                self.state_candidates.add(canonical_n)

        canonical_states = {
            self.resolver_module.canonical_state(x)
            for x in self.state_candidates
            if clean_value(x)
        }
        for canonical in canonical_states:
            if canonical:
                self.state_candidates.add(re.sub(r"[^A-Z0-9]", "", canonical))

    def address_state_candidates(self, address: str):
        norm = normalized_text(address)
        if not norm:
            return set()
        padded = f" {norm} "
        compact = re.sub(r"[^A-Z0-9]", "", norm)
        found = set()

        for state in self.state_candidates:
            if len(state) < 3:
                continue
            state_compact = re.sub(r"[^A-Z0-9]", "", state)
            if f" {state} " in padded or (state_compact and state_compact in compact and len(state_compact) >= 5):
                canonical = self.resolver_module.canonical_state(state)
                if canonical:
                    found.add(canonical)
        return found

    def state_from_pin(self, pin: str):
        states = {self.resolver_module.canonical_state(x) for x in self.pin_states.get(pin, set()) if clean_value(x)}
        states.discard("")
        return next(iter(states)) if len(states) == 1 else ""

    def valid_pin(self, pin: str) -> bool:
        return bool(pin and pin in self.pin_rows)

    def pin_candidates_from_address(self, address: str, state: str = ""):
        norm = clean_value(address)
        candidates = set()

        for value in re.findall(r"(?<!\d)\d{6}(?!\d)", norm):
            if self.valid_pin(value):
                candidates.add(value)

        for match in re.finditer(r"\b(?:PO|P\.O\.|POST OFFICE|POST)\s*[-.:]?\s*([^,;]+)", norm, flags=re.I):
            office = normalized_text(match.group(1))
            for pin in self.office_to_pins.get(office, set()):
                candidates.add(pin)

        for segment in re.split(r"[,;]", norm):
            office = normalized_text(segment)
            if office in self.office_to_pins:
                candidates.update(self.office_to_pins[office])

        if state:
            state_c = self.resolver_module.canonical_state(state)
            candidates = {pin for pin in candidates if self.state_from_pin(pin) == state_c}
        return candidates

    def city_candidates_from_address(self, address: str, state: str = ""):
        norm = normalized_text(address)
        if not norm or not self.ulb_names:
            return set()
        tokens = set(norm.split())
        possible = set()
        for token in tokens:
            possible.update(self.ulb_token_index.get(token, set()))
        padded = f" {norm} "
        matches = set()
        for name in possible:
            if f" {name} " not in padded:
                continue
            matches.add(name)
        resolved = set()
        for candidate in matches:
            result = self.resolver.resolve_ulb(candidate, state=state)
            if result.status == "PASS":
                resolved.add(result.match_value)
        return resolved

    def resolve(self, address, city, state, pin):
        original_city = clean_value(city)
        original_state = clean_value(state)
        original_pin = clean_value(pin)

        current_state = self.resolver_module.canonical_state(original_state)
        current_pin = normalized_pin(original_pin)

        city_as_state = self.resolver_module.canonical_state(original_city)
        if city_as_state:
            original_city = ""
            
        state_changed = False
        city_changed = False
        pin_changed = False
        remarks = []

        # 1) Resolve State
        pin_state = self.state_from_pin(current_pin) if self.valid_pin(current_pin) else ""
        address_states = self.address_state_candidates(address)

        if not current_state:
            if pin_state and (not address_states or pin_state in address_states):
                current_state = pin_state
                state_changed = True
                remarks.append(f"state auto resolved from null->{current_state}")
            elif len(address_states) == 1:
                current_state = next(iter(address_states))
                state_changed = True
                remarks.append(f"state auto resolved from null->{current_state}")
        else:
            if pin_state and current_state != pin_state:
                current_state = pin_state
                state_changed = True
                remarks.append(f"state auto corrected {original_state}->{current_state}")
            elif not pin_state and len(address_states) == 1 and current_state != next(iter(address_states)):
                current_state = next(iter(address_states))
                state_changed = True
                remarks.append(f"state auto corrected {original_state}->{current_state}")

        # 2) Resolve PIN
        pin_candidates = self.pin_candidates_from_address(address, current_state)
        if not current_pin or not self.valid_pin(current_pin):
            if len(pin_candidates) == 1:
                new_pin = next(iter(pin_candidates))
                if new_pin != current_pin:
                    current_pin = new_pin
                    pin_changed = True
                    remarks.append(f"pin auto corrected {original_pin or 'null'}->{current_pin}")

        # 3) Resolve City (Respect the cleaner's extracted city, do not overwrite!)
        current_city = original_city
        city_candidates = self.city_candidates_from_address(address, current_state)
        
        if not current_city and len(city_candidates) == 1:
            current_city = next(iter(city_candidates))
            city_changed = True
            remarks.append(f"city missing and resolved null->{current_city}")

        if not state_changed and original_state and current_state:
            if self.resolver_module.canonical_state(original_state) == self.resolver_module.canonical_state(current_state):
                pass

        if pin_changed and is_decimal_only_pin_change(original_pin, current_pin):
            remarks = [r for r in remarks if not r.startswith("pin auto ")]
            pin_changed = False

        # --- EXPLICIT MISSING FIELDS AUDIT REMARK ---
        missing_fields = []
        addr_clean = clean_value(address).upper()
        
        if not addr_clean or addr_clean in {"", "-", "--", "0", "0.0", "NA", "N/A", "NULL", "NONE"}:
            missing_fields.append("Complete Address")
            
        if not current_city: missing_fields.append("City")
        if not current_state: missing_fields.append("State")
        if not current_pin: missing_fields.append("Pincode")

        if missing_fields:
            remarks.append(f"Missing: {', '.join(missing_fields)}")

        return {
            "city": current_city,
            "state": current_state,
            "pin": current_pin,
            "remarks": remarks,
            "changed": bool(state_changed or city_changed or pin_changed or missing_fields),
        }