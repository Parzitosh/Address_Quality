import re
from pathlib import Path
import pandas as pd
import argparse


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "SMFG - Query result.csv"

OUTPUT_FILE = BASE_DIR / "Cleaned_Address_Data.csv"
REPORT_FILE = BASE_DIR / "Cleaning_Report.txt"

REQUIRED_COLUMNS = [
    "Full Address",
    "City",
    "Pincode",
    "State",
    "Lead Code",
]

STATE_ALIASES = {
    "ANDHRA PRADESH": "ANDHRA PRADESH",
    "ARUNACHAL PRADESH": "ARUNACHAL PRADESH",
    "ASSAM": "ASSAM",
    "BIHAR": "BIHAR",
    "CHHATTISGARH": "CHHATTISGARH",
    "GOA": "GOA",
    "GUJARAT": "GUJARAT",
    "HARYANA": "HARYANA",
    "HIMACHAL PRADESH": "HIMACHAL PRADESH",
    "JHARKHAND": "JHARKHAND",
    "KARNATAKA": "KARNATAKA",
    "KERALA": "KERALA",
    "MADHYA PRADESH": "MADHYA PRADESH",
    "MAHARASHTRA": "MAHARASHTRA",
    "MANIPUR": "MANIPUR",
    "MEGHALAYA": "MEGHALAYA",
    "MIZORAM": "MIZORAM",
    "NAGALAND": "NAGALAND",
    "ODISHA": "ODISHA",
    "ORISSA": "ODISHA",
    "PUNJAB": "PUNJAB",
    "RAJASTHAN": "RAJASTHAN",
    "SIKKIM": "SIKKIM",
    "TAMIL NADU": "TAMIL NADU",
    "TELANGANA": "TELANGANA",
    "TRIPURA": "TRIPURA",
    "UTTAR PRADESH": "UTTAR PRADESH",
    "UTTARAKHAND": "UTTARAKHAND",
    "WEST BENGAL": "WEST BENGAL",
    "DELHI": "DELHI",
    "JAMMU AND KASHMIR": "JAMMU AND KASHMIR",
    "LADAKH": "LADAKH",
    "CHANDIGARH": "CHANDIGARH",
    "PUDUCHERRY": "PUDUCHERRY",
}

STATE_ABBREVIATIONS = {
    "UP": "UTTAR PRADESH",
    "MP": "MADHYA PRADESH",
    "RJ": "RAJASTHAN",
    "HR": "HARYANA",
    "PB": "PUNJAB",
    "HP": "HIMACHAL PRADESH",
    "UK": "UTTARAKHAND",
    "CG": "CHHATTISGARH",
    "WB": "WEST BENGAL",
    "KA": "KARNATAKA",
    "TN": "TAMIL NADU",
    "TS": "TELANGANA",
    "AP": "ANDHRA PRADESH",
    "TG": "TELANGANA",
    "GJ": "GUJARAT",
    "MH": "MAHARASHTRA",
    "BR": "BIHAR",
    "JH": "JHARKHAND",
    "KL": "KERALA",
    "OR": "ODISHA",
    "OD": "ODISHA",
}

PLACEHOLDER_VALUES = {
    "",
    "N/A",
    "NA",
    "NIL",
    "NONE",
    "NULL",
    "UNKNOWN",
    "-",
    "--",
    "NOT AVAILABLE",
    "NOT PROVIDED",
    "NOT APPLICABLE",
    "UNAVAILABLE",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_text(value):
    """Normalize whitespace and common separators without deleting content."""
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip().upper()
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)

    # Normalize separator spacing.
    text = re.sub(r"\s*,\s*", ",", text)
    text = re.sub(r"\s*;\s*", ",", text)
    text = re.sub(r"\s*\|\s*", ",", text)
    text = re.sub(r",+", ",", text)

    return text.strip(" ,;")


def normalize_for_matching(value):
    """Canonical form used only for matching/comparison."""
    text = clean_text(value)
    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_pin(value):
    """Return a 6-digit PIN when one is clearly present."""
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip()

    if re.fullmatch(r"\d{6}", text):
        return text

    matches = re.findall(r"(?<!\d)(\d{6})(?!\d)", text)
    return matches[-1] if matches else ""


def canonical_state(value):
    """Normalize full state names and common abbreviations."""
    text = normalize_for_matching(value)

    if text in STATE_ABBREVIATIONS:
        return STATE_ABBREVIATIONS[text]

    return STATE_ALIASES.get(text, text)


# Precompute state matching structures once. The previous implementation
# ran ~50 separate regex searches for every address.
_STATE_VARIANT_TO_CANONICAL = {}
for _state_value in list(STATE_ALIASES.keys()) + list(STATE_ABBREVIATIONS.keys()):
    _STATE_VARIANT_TO_CANONICAL[normalize_for_matching(_state_value)] = canonical_state(_state_value)

_STATE_VARIANT_PATTERN = re.compile(
    r"(?<![A-Z0-9])(?:"
    + "|".join(
        re.escape(x)
        for x in sorted(_STATE_VARIANT_TO_CANONICAL, key=len, reverse=True)
        if x
    )
    + r")(?![A-Z0-9])",
    flags=re.I,
)

_STATE_VARIANTS_CACHE = {}
_STATE_END_PATTERN_CACHE = {}


def is_meaningful(value):
    text = normalize_for_matching(value)
    return bool(text) and text not in PLACEHOLDER_VALUES


def phrase_pattern(value):
    """Regex for an exact standalone phrase after normalization."""
    normalized = normalize_for_matching(value)
    if not normalized:
        return ""
    return r"\b" + r"\s+".join(re.escape(x) for x in normalized.split()) + r"\b"


def count_phrase(text, phrase):
    pattern = phrase_pattern(phrase)
    if not pattern:
        return 0
    return len(re.findall(pattern, normalize_for_matching(text), flags=re.I))


# ============================================================
# STRUCTURED FIELD REPAIR
# ============================================================

def repair_structured_fields(city, state, pincode):
    """
    Repair obvious formatting/cross-column contamination.

    This is deliberately deterministic:
    - no fuzzy city guessing
    - no row deletion
    - no address-to-city guessing
    """
    city_clean = clean_text(city)
    state_clean = canonical_state(state)
    pin_clean = normalize_pin(pincode)

    # If State is embedded at the end of City, remove it.
    if state_clean and city_clean:
        state_pattern = phrase_pattern(state_clean)
        if state_pattern:
            city_clean = re.sub(
                rf"(?:,|\s)+{state_pattern}\s*$",
                "",
                normalize_for_matching(city_clean),
                flags=re.I,
            ).strip()

    # Recover state if the State column is empty but City ends in a
    # recognized state/abbreviation.
    if not state_clean and city_clean:
        city_norm = normalize_for_matching(city_clean)
        candidates = sorted(
            list(STATE_ALIASES.keys()) + list(STATE_ABBREVIATIONS.keys()),
            key=len,
            reverse=True,
        )
        for candidate in candidates:
            candidate_n = normalize_for_matching(candidate)
            if re.search(rf"\b{re.escape(candidate_n)}\s*$", city_norm):
                state_clean = canonical_state(candidate)
                city_clean = re.sub(
                    rf"\s+{re.escape(candidate_n)}\s*$",
                    "",
                    city_norm,
                    flags=re.I,
                ).strip()
                break

    # If Pincode contains text + a valid PIN, keep only the PIN.
    if not pin_clean:
        # No valid 6-digit PIN means do not invent one.
        pin_clean = ""

    # If City is empty and Pincode contains city text plus PIN, recover
    # only when the text portion is clearly non-numeric.
    if not city_clean and pincode is not None and not pd.isna(pincode):
        pin_text = str(pincode).strip()
        pin_match = re.search(r"(?<!\d)\d{6}(?!\d)", pin_text)
        if pin_match:
            before = pin_text[:pin_match.start()].strip(" ,-/")
            before = clean_text(before)
            if before and is_meaningful(before):
                city_clean = before

    return city_clean, state_clean, pin_clean


# ============================================================
# ADDRESS METADATA REMOVAL
# ============================================================

def remove_trailing_metadata(text, state="", pincode=""):
    """Remove state/PIN only when they are clearly metadata at the end."""
    text = clean_text(text)

    pin = normalize_pin(pincode)
    state_n = canonical_state(state)

    if pin:
        text = re.sub(
            rf"(?:,|\s)*{re.escape(pin)}\s*$",
            "",
            text,
            flags=re.I,
        )

    if state_n:
        pattern = phrase_pattern(state_n)
        if pattern:
            text = re.sub(
                rf"(?:,|\s)+{pattern}\s*$",
                "",
                text,
                flags=re.I,
            )

    # Remove known state variants (full name + abbreviations) only at the end.
    for alias in state_variants(state):
        pattern = phrase_pattern(alias)
        if pattern:
            text = re.sub(
                rf"(?:,|\s)+{pattern}\s*$",
                "",
                text,
                flags=re.I,
            )

    # Handle PIN + state and state + PIN combinations that may remain.
    if pin and state_n:
        sp = phrase_pattern(state_n)
        if sp:
            text = re.sub(
                rf"(?:,|\s)+{sp}(?:,|\s)+{re.escape(pin)}\s*$",
                "",
                text,
                flags=re.I,
            )
            text = re.sub(
                rf"(?:,|\s)+{re.escape(pin)}(?:,|\s)+{sp}\s*$",
                "",
                text,
                flags=re.I,
            )

    return clean_text(text)


def _collapse_repeated_token_blocks(tokens, max_block_size=12):
    """
    Collapse adjacent repetitions of a contiguous token block.

    Examples:
        A B A B             -> A B
        A B C A B C         -> A B C
        A B C A B C A B C   -> A B C

    The longest block is tested first. Matching is based on normalized
    tokens, while the original token spelling is retained in the output.
    Only adjacent repetitions are collapsed; non-adjacent occurrences are
    left untouched to avoid deleting legitimate address content.
    """
    if not tokens:
        return tokens, 0

    normalized = [normalize_for_matching(t) for t in tokens]
    removed = 0
    changed = True

    while changed and len(tokens) >= 2:
        changed = False
        n = len(tokens)
        upper = min(max_block_size, n // 2)

        for block_size in range(upper, 1, -1):
            found = False
            i = 0
            while i + (2 * block_size) <= len(tokens):
                left = normalized[i:i + block_size]
                right = normalized[i + block_size:i + 2 * block_size]

                if left and left == right:
                    # Remove all immediately repeated copies, keeping the
                    # first copy.
                    copies = 2
                    end = i + copies * block_size
                    while (
                        end + block_size <= len(tokens)
                        and normalized[end:end + block_size] == left
                    ):
                        copies += 1
                        end += block_size

                    del tokens[i + block_size:end]
                    del normalized[i + block_size:end]
                    removed += (copies - 1) * block_size
                    changed = True
                    found = True
                    break

                i += 1

            if found:
                break

    return tokens, removed


def remove_immediate_duplicate_tokens(text):
    """
    Remove adjacent duplicate tokens and repeated contiguous blocks.

    This is intentionally conservative: only exact normalized token blocks
    that repeat immediately are collapsed.
    """
    tokens = clean_text(text).replace(",", " , ").split()

    # Preserve comma tokens as hard boundaries. Run block collapse separately
    # inside each comma-delimited segment.
    result = []
    for segment in _split_on_commas(tokens):
        cleaned_segment, _ = _collapse_repeated_token_blocks(segment)

        # Also collapse single-token adjacency such as AGRA AGRA. Run this
        # after block collapse because a larger pattern can expose a new
        # immediate duplicate at its boundary.
        deduped = []
        for token in cleaned_segment:
            if (
                deduped
                and normalize_for_matching(deduped[-1])
                == normalize_for_matching(token)
            ):
                continue
            deduped.append(token)

        result.extend(deduped)
        result.append(",")

    if result and result[-1] == ",":
        result.pop()

    return clean_text(" ".join(result))


def _split_on_commas(tokens):
    segment = []
    for token in tokens:
        if token == ",":
            if segment:
                yield segment
                segment = []
        else:
            segment.append(token)
    if segment:
        yield segment


def remove_duplicate_metadata_occurrences(text, city="", state=""):
    """
    Remove later duplicate city/state occurrences only when exact.

    Important:
    We do not remove the only occurrence of a city. This protects
    meaningful phrases such as AGRA ROAD / AGRA COLONY.
    """
    text = clean_text(text)

    city_n = normalize_for_matching(city)
    if city_n and len(city_n) >= 3:
        pattern = phrase_pattern(city_n)
        matches = list(re.finditer(pattern, normalize_for_matching(text), flags=re.I))

        if len(matches) >= 2:
            seen = 0

            def repl(match):
                nonlocal seen
                seen += 1
                return match.group(0) if seen == 1 else " "

            text = re.sub(pattern, repl, text, flags=re.I)

    # State duplication is safe only when it is an exact standalone phrase.
    state_n = canonical_state(state)
    if state_n and len(state_n) >= 4:
        pattern = phrase_pattern(state_n)
        matches = list(re.finditer(pattern, normalize_for_matching(text), flags=re.I))

        if len(matches) >= 2:
            seen = 0

            def repl_state(match):
                nonlocal seen
                seen += 1
                return match.group(0) if seen == 1 else " "

            text = re.sub(pattern, repl_state, text, flags=re.I)

    return clean_text(text)


# ============================================================
# DELIMITER-FREE ADDRESS CLEANING
# ============================================================

def create_address_segments(address, city="", state="", pincode=""):
    """
    Clean a delimiter-free address and introduce conservative separators
    around known structured geographic metadata.

    This does NOT try to infer arbitrary word boundaries. It uses only
    exact values supplied in the structured columns.
    """
    working = clean_text(address)

    if not working:
        return ""

    # Remove PIN and State metadata from the address body.
    working = remove_trailing_metadata(
        working,
        state=state,
        pincode=pincode,
    )

    working = remove_immediate_duplicate_tokens(working)
    working = remove_duplicate_metadata_occurrences(
        working,
        city=city,
        state=state,
    )

    # Separate the last exact City occurrence. If City occurs only once,
    # this simply creates a logical boundary around it.
    city_n = normalize_for_matching(city)

    if city_n and len(city_n) >= 3:
        pattern = phrase_pattern(city_n)
        matches = list(re.finditer(pattern, normalize_for_matching(working), flags=re.I))

        if matches:
            match = matches[-1]
            start, end = match.span()

            working = (
                working[:start].rstrip(" ,")
                + ", "
                + working[start:end]
                + working[end:]
            )

    # Clean separators after segmentation.
    working = re.sub(r"\s*,\s*", ", ", working)
    working = re.sub(r",\s*,+", ", ", working)
    working = re.sub(r"\s+", " ", working)

    return working.strip(" ,")


# ============================================================
# MAIN ADDRESS CLEANER
# ============================================================

def clean_address(address, city="", state="", pincode=""):
    """
    Main cleaner for both delimiter-based and delimiter-free addresses.

    Structured City/State/Pincode are treated as metadata. They are NOT
    blindly deleted from meaningful phrases inside the address.
    """
    if address is None or pd.isna(address):
        return ""

    text = clean_text(address)

    if not text:
        return ""

    has_delimiter = bool(re.search(r"[,;|]", text))

    if not has_delimiter:
        return create_address_segments(
            address=text,
            city=city,
            state=state,
            pincode=pincode,
        )

    text = remove_trailing_metadata(
        text,
        state=state,
        pincode=pincode,
    )

    text = remove_immediate_duplicate_tokens(text)
    text = remove_duplicate_metadata_occurrences(
        text,
        city=city,
        state=state,
    )

    return clean_text(text)


# ============================================================

# ============================================================
# MULTI-ADDRESS RAW INPUT EXTRACTION
# ============================================================

ADDRESS_OUTPUTS = {
    "Current Address": "Cleaned_Current_Address_Data.csv",
    "Office Address": "Cleaned_Office_Address_Data.csv",
    "Alternate Address": "Cleaned_Alternate_Address_Data.csv",
}

STATE_MATCH_ORDER = sorted(
    list(STATE_ALIASES.keys()) + list(STATE_ABBREVIATIONS.keys()),
    key=len,
    reverse=True,
)

def find_pincode_in_address(address, valid_pins=None):
    """Extract the best 6-digit PIN from raw free text."""
    text = clean_text(address)
    matches = re.findall(r"(?<!\d)(\d{6})(?!\d)", text)
    if not matches:
        return ""

    if valid_pins:
        valid = [p for p in matches if p in valid_pins]
        if valid:
            return valid[-1]

    return matches[-1]


def find_state_in_address(address):
    """Extract the latest standalone state/abbreviation with one regex pass."""
    text = normalize_for_matching(address)
    if not text:
        return ""

    matches = list(_STATE_VARIANT_PATTERN.finditer(text))
    if not matches:
        return ""

    return _STATE_VARIANT_TO_CANONICAL.get(
        matches[-1].group(0).upper(),
        "",
    )


def state_variants(canonical):
    """Return cached full-name/abbreviation variants for a canonical state."""
    target = canonical_state(canonical)
    if not target:
        return []

    cached = _STATE_VARIANTS_CACHE.get(target)
    if cached is not None:
        return cached

    variants = [
        value
        for value, mapped in _STATE_VARIANT_TO_CANONICAL.items()
        if mapped == target
    ]
    variants.append(normalize_for_matching(target))
    variants = sorted(set(variants), key=len, reverse=True)
    _STATE_VARIANTS_CACHE[target] = variants
    return variants


def remove_state_variants(text, state):
    working = normalize_for_matching(text)
    variants = state_variants(state)
    if not variants:
        return working.strip(" ,")

    key = canonical_state(state)
    pattern = _STATE_END_PATTERN_CACHE.get(key)
    if pattern is None:
        pattern = re.compile(
            r"(?<![A-Z0-9])(?:"
            + "|".join(re.escape(x) for x in variants)
            + r")(?![A-Z0-9])",
            flags=re.I,
        )
        _STATE_END_PATTERN_CACHE[key] = pattern

    return re.sub(r"\s+", " ", pattern.sub(" ", working)).strip(" ,")


def strip_metadata_from_text(text, state="", pincode=""):
    """Remove only clearly identified state/PIN metadata from free text."""
    working = clean_text(text)

    pin = normalize_pin(pincode)
    if pin:
        working = re.sub(
            rf"(?<!\d){re.escape(pin)}(?!\d)",
            " ",
            working,
            flags=re.I,
        )

    if state:
        working = remove_state_variants(working, state)

    working = re.sub(r"\s+", " ", working).strip(" ,")
    return working


def looks_like_city_candidate(value, state="", pincode=""):
    """
    Conservative city/town candidate test.

    We do not fuzzy-invent a city. A candidate must be a meaningful geographic
    phrase and must not obviously be a premise/route fragment.
    """
    value = clean_text(value).strip(" ,")
    if not value:
        return False

    normalized = normalize_for_matching(value)
    if not normalized or normalized in PLACEHOLDER_VALUES:
        return False

    if pincode and normalized == normalize_for_matching(pincode):
        return False

    if state and normalized == normalize_for_matching(state):
        return False

    # Don't select obvious premise-only segments.
    if re.search(
        r"\b(?:HNO|HOUSE|FLAT|PLOT|SHOP|ROAD|RD|STREET|ST|LANE|"
        r"MARG|GALI|NEAR|OPP|OPPOSITE|BEHIND|BESIDE|COLONY|"
        r"NAGAR|SOCIETY|APARTMENT|BUILDING|TOWER|COMPLEX)\b",
        normalized,
    ):
        return False

    # A candidate that is just a short number is not a city.
    if re.fullmatch(r"\d+", normalized):
        return False

    return True


def extract_city_from_address(address, state="", pincode="", postal_offices=None):
    """
    Infer City/Town conservatively from the raw address.

    Priority:
      1. Segment containing PIN, after removing State.
      2. Nearby trailing geographic segment.
      3. India Post office name for the PIN when explicitly present.
      4. Last plausible segment.

    This is extraction, not fuzzy city guessing.
    """
    text = clean_text(address)
    if not text:
        return ""

    state_n = normalize_for_matching(state)
    pin = normalize_pin(pincode)

    # First use comma/semicolon/pipe boundaries.
    segments = [
        x.strip(" ,")
        for x in re.split(r"[,;|]+", text)
        if x.strip(" ,")
    ]

    # Search from the end because Indian addresses usually append geography.
    candidates = []

    for seg in reversed(segments):
        candidate = clean_text(seg)

        # Remove PIN and state from this segment.
        if pin:
            candidate = re.sub(
                rf"(?<!\d){re.escape(pin)}(?!\d)",
                " ",
                candidate,
                flags=re.I,
            )

        if state:
            candidate = remove_state_variants(candidate, state)

        candidate = re.sub(r"\s+", " ", candidate).strip(" ,-/")

        if looks_like_city_candidate(candidate, state, pin):
            candidates.append(candidate)

    if candidates:
        # Prefer a short geographic segment near the end.
        return candidates[0]

    # Delimiter-free fallback: use postal-office names only when explicitly
    # present in the address and associated with the extracted PIN.
    if postal_offices and pin:
        address_n = normalize_for_matching(text)
        office_candidates = []
        for office in postal_offices.get(pin, []):
            office_n = normalize_for_matching(office)
            if len(office_n) < 3:
                continue
            pattern = rf"(?<![A-Z0-9]){re.escape(office_n)}(?![A-Z0-9])"
            if re.search(pattern, address_n, flags=re.I):
                office_candidates.append(office)
        if office_candidates:
            office_candidates.sort(
                key=lambda x: len(normalize_for_matching(x)),
                reverse=True,
            )
            return clean_text(office_candidates[0])

    # Final conservative fallback for delimiter-free input:
    # remove metadata and take the final plausible phrase.
    body = strip_metadata_from_text(text, state, pin)
    words = body.split()
    if not words:
        return ""

    # Avoid returning an entire long premise as a city.
    # Look at the last 1-4 tokens and select the shortest plausible suffix.
    for n in range(1, min(4, len(words)) + 1):
        candidate = " ".join(words[-n:])
        if looks_like_city_candidate(candidate, state, pin):
            return candidate

    return ""


def build_postal_office_index():
    """Load a lightweight PIN -> Post Office-name index when available."""
    master_path = BASE_DIR / "masters" / "india_post_master.csv"
    if not master_path.exists():
        return {}

    try:
        master = pd.read_csv(
            master_path,
            usecols=["officename", "pincode"],
            dtype=str,
        )
    except Exception:
        return {}

    index = {}
    for pin, group in master.groupby("pincode", dropna=False):
        pin_n = normalize_pin(pin)
        if not pin_n:
            continue

        offices = []
        for office in group["officename"].fillna("").astype(str):
            office = clean_text(office)
            if office and office not in offices:
                offices.append(office)

        index[pin_n] = offices

    return index


def remove_standalone_pincode_segments(text, pincode):
    """Remove a PIN only when it occupies its own delimiter-separated segment."""
    pin = normalize_pin(pincode)
    if not pin:
        return clean_text(text)

    parts = [p.strip(" ,") for p in re.split(r"[,;|]+", clean_text(text))]
    kept = [p for p in parts if normalize_for_matching(p) != pin]
    return ",".join(p for p in kept if p)


def clean_single_address(address, city, state, pincode, postal_offices):
    """
    Produce the same cleaned-address structure used by address_quality.py,
    but derive City/State/PIN directly from the raw address when absent.
    """
    raw = clean_text(address)

    pin = find_pincode_in_address(raw)
    state_value = find_state_in_address(raw)

    # If structured extraction finds no state, preserve an empty state.
    city_value = extract_city_from_address(
        raw,
        state=state_value,
        pincode=pin,
        postal_offices=postal_offices,
    )

    # Remove an extracted PIN when it appears as its own metadata segment
    # anywhere in the address (not when embedded in a premise identifier).
    raw_body = remove_standalone_pincode_segments(raw, pin)

    # Clean address body using the extracted metadata.
    cleaned = clean_address(
        address=raw_body,
        city=city_value,
        state=state_value,
        pincode=pin,
    )

    return {
        "Clean Full Address": cleaned,
        "Clean City": city_value,
        "Clean Pincode": pin,
        "Clean State": state_value,
    }


def process_address_column(df, address_column, output_filename, postal_offices, clean_cache=None):
    """
    Process one of Current/Office/Alternate Address columns.

    LAN is preserved exactly. No row deduplication is performed.
    """
    output_rows = []
    if clean_cache is None:
        clean_cache = {}

    try:
        from tqdm import tqdm
        iterator = tqdm(
            df[["LAN", address_column]].itertuples(index=False, name=None),
            total=len(df),
            desc=f"Cleaning {address_column}",
            unit="row",
            dynamic_ncols=True,
        )
    except ImportError:
        iterator = df[["LAN", address_column]].itertuples(index=False, name=None)

    for row in iterator:
        lan, raw_address = row

        cache_key = clean_text(raw_address)
        cleaned = clean_cache.get(cache_key)
        if cleaned is None:
            cleaned = clean_single_address(
                address=raw_address,
                city="",
                state="",
                pincode="",
                postal_offices=postal_offices,
            )
            if len(clean_cache) < 300000:
                clean_cache[cache_key] = cleaned

        output_rows.append({
            "LAN": lan,
            "Clean Full Address": cleaned["Clean Full Address"],
            "Clean City": cleaned["Clean City"],
            "Clean Pincode": cleaned["Clean Pincode"],
            "Clean State": cleaned["Clean State"],
            "Original Address": clean_text(raw_address),
        })

    result = pd.DataFrame(output_rows)

    output_path = BASE_DIR / output_filename
    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"  Created: {output_path.name} ({len(result):,} rows)")
    return output_path


def process_multi_address_file(input_file=None):
    if input_file:
        input_file = Path(input_file)
        if not input_file.is_absolute():
            input_file = BASE_DIR / input_file
    else:
        candidates = [
            BASE_DIR / "BOBCARD - All Address.csv",
            BASE_DIR / "NON RURAL - Query result.csv",
        ]
        input_file = next((p for p in candidates if p.exists()), None)

    if input_file is None or not input_file.exists():
        raise FileNotFoundError(
            "No multi-address raw CSV found. Put the file in the project "
            "folder or pass --input <file>."
        )

    df = pd.read_csv(
        input_file,
        dtype=str,
        keep_default_na=False,
    )

    required = ["LAN", "Current Address", "Office Address"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    address_columns = [
        c for c in ["Current Address", "Office Address", "Alternate Address"]
        if c in df.columns
    ]

    if len(address_columns) < 2:
        raise ValueError(
            "At least two address columns are required."
        )

    postal_offices = build_postal_office_index()

    print("=" * 70)
    print("MULTI-ADDRESS CLEANING")
    print("=" * 70)
    print(f"Input file : {input_file.name}")
    print(f"Records    : {len(df):,}")
    print(f"Addresses  : {', '.join(address_columns)}")
    print()

    created = []
    # Shared across Current/Office/Alternate so identical raw addresses are
    # cleaned only once. LAN rows are still preserved one-for-one.
    clean_cache = {}

    for address_column in address_columns:
        print(f"Processing {address_column}...")
        created.append(
            process_address_column(
                df,
                address_column,
                ADDRESS_OUTPUTS[address_column],
                postal_offices,
                clean_cache=clean_cache,
            )
        )

    print()
    print("Cleaning completed successfully.")
    print("Rows are preserved; no deduplication was performed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean multi-address customer data")
    parser.add_argument("--input", help="Raw multi-address CSV path")
    args = parser.parse_args()
    process_multi_address_file(args.input)
