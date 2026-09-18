import re
from pathlib import Path
import pandas as pd
import argparse
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_FILE = BASE_DIR / "Cleaned_Address_Data.csv"
REPORT_FILE = BASE_DIR / "Cleaning_Report.txt"

REQUIRED_COLUMNS = [
    "Full Address", "City", "Pincode", "State", "Lead Code",
]

STATE_ALIASES = {
    # Full names & common typos/compacted strings
    "CHHATISGARH": "CHHATTISGARH", "MADHYAPRADESH": "MADHYA PRADESH", 
    "UTTARPRADESH": "UTTAR PRADESH", "HIMACHALPRADESH": "HIMACHAL PRADESH", 
    "ANDHRAPRADESH": "ANDHRA PRADESH", "ARUNACHALPRADESH": "ARUNACHAL PRADESH", 
    "JAMMUANDKASHMIR": "JAMMU AND KASHMIR", "WESTBENGAL": "WEST BENGAL", 
    "TAMILNADU": "TAMIL NADU", "UTTARAKHAND": "UTTARAKHAND", 
    "PUDUCHERRY": "PUDUCHERRY", "ORISSA": "ODISHA", "PONDICHERRY": "PUDUCHERRY",
    "ANDHRA PRADESH": "ANDHRA PRADESH", "ARUNACHAL PRADESH": "ARUNACHAL PRADESH", 
    "ASSAM": "ASSAM", "BIHAR": "BIHAR", "CHHATTISGARH": "CHHATTISGARH", 
    "GOA": "GOA", "GUJARAT": "GUJARAT", "HARYANA": "HARYANA", 
    "HIMACHAL PRADESH": "HIMACHAL PRADESH", "JHARKHAND": "JHARKHAND", 
    "KARNATAKA": "KARNATAKA", "KERALA": "KERALA", "MADHYA PRADESH": "MADHYA PRADESH", 
    "MAHARASHTRA": "MAHARASHTRA", "MANIPUR": "MANIPUR", "MEGHALAYA": "MEGHALAYA", 
    "MIZORAM": "MIZORAM", "NAGALAND": "NAGALAND", "ODISHA": "ODISHA", 
    "PUNJAB": "PUNJAB", "RAJASTHAN": "RAJASTHAN", "SIKKIM": "SIKKIM", 
    "TAMIL NADU": "TAMIL NADU", "TELANGANA": "TELANGANA", "TRIPURA": "TRIPURA", 
    "UTTAR PRADESH": "UTTAR PRADESH", "DELHI": "DELHI", "JAMMU AND KASHMIR": "JAMMU AND KASHMIR", 
    "LADAKH": "LADAKH", "CHANDIGARH": "CHANDIGARH"
}

STATE_ABBREVIATIONS = {
    "UP": "UTTAR PRADESH", "MP": "MADHYA PRADESH", "RJ": "RAJASTHAN", 
    "HR": "HARYANA", "PB": "PUNJAB", "HP": "HIMACHAL PRADESH", "UK": "UTTARAKHAND", 
    "CG": "CHHATTISGARH", "WB": "WEST BENGAL", "KA": "KARNATAKA", "TN": "TAMIL NADU", 
    "TS": "TELANGANA", "TG": "TELANGANA", "AP": "ANDHRA PRADESH", "GJ": "GUJARAT", 
    "MH": "MAHARASHTRA", "BR": "BIHAR", "JH": "JHARKHAND", "KL": "KERALA", 
    "OR": "ODISHA", "OD": "ODISHA", "DL": "DELHI", "AS": "ASSAM", "GA": "GOA", 
    "JK": "JAMMU AND KASHMIR", "J&K": "JAMMU AND KASHMIR", "J AND K": "JAMMU AND KASHMIR", 
    "PY": "PUDUCHERRY", "UT": "LADAKH"
}

PLACEHOLDER_VALUES = {
    "", "N/A", "NA", "NIL", "NONE", "NULL", "UNKNOWN",
    "-", "--", "NOT AVAILABLE", "NOT PROVIDED", "NOT APPLICABLE", "UNAVAILABLE",
    "0", "0.0"
}

# ============================================================
# BASIC HELPERS
# ============================================================

def clean_text(value):
    if pd.isna(value): return ""
    text = str(value).strip().upper()
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*,\s*", ",", text)
    text = re.sub(r"\s*;\s*", ",", text)
    text = re.sub(r"\s*\|\s*", ",", text)
    text = re.sub(r",+", ",", text)
    return text.strip(" ,;")

def normalize_for_matching(value):
    text = clean_text(value)
    text = re.sub(r"[^A-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_pin(value):
    if value is None or pd.isna(value): return ""
    text = str(value).strip()
    if re.fullmatch(r"\d{6}", text): return text
    matches = re.findall(r"(?<!\d)(\d{6})(?!\d)", text)
    return matches[-1] if matches else ""

def canonical_state(value):
    text = normalize_for_matching(value)
    if text in STATE_ABBREVIATIONS: return STATE_ABBREVIATIONS[text]
    return STATE_ALIASES.get(text, text)

_STATE_VARIANT_TO_CANONICAL = {}
for _state_value in list(STATE_ALIASES.keys()) + list(STATE_ABBREVIATIONS.keys()):
    _STATE_VARIANT_TO_CANONICAL[normalize_for_matching(_state_value)] = canonical_state(_state_value)

_STATE_VARIANT_PATTERN = re.compile(
    r"(?<![A-Z0-9])(?:" + "|".join(re.escape(x) for x in sorted(_STATE_VARIANT_TO_CANONICAL, key=len, reverse=True) if x) + r")(?![A-Z0-9])", flags=re.I,
)

_STATE_VARIANTS_CACHE = {}
_STATE_END_PATTERN_CACHE = {}

def is_meaningful(value):
    text = normalize_for_matching(value)
    return bool(text) and text not in PLACEHOLDER_VALUES

def phrase_pattern(value):
    normalized = normalize_for_matching(value)
    if not normalized: return ""
    return r"\b" + r"\s+".join(re.escape(x) for x in normalized.split()) + r"\b"

def state_variants(canonical):
    target = canonical_state(canonical)
    if not target: return []
    cached = _STATE_VARIANTS_CACHE.get(target)
    if cached is not None: return cached
    variants = [value for value, mapped in _STATE_VARIANT_TO_CANONICAL.items() if mapped == target]
    variants.append(normalize_for_matching(target))
    variants = sorted(set(variants), key=len, reverse=True)
    _STATE_VARIANTS_CACHE[target] = variants
    return variants

def remove_state_variants(text, state):
    working = normalize_for_matching(text)
    variants = state_variants(state)
    if not variants: return working.strip(" ,")
    key = canonical_state(state)
    pattern = _STATE_END_PATTERN_CACHE.get(key)
    if pattern is None:
        pattern = re.compile(r"(?<![A-Z0-9])(?:" + "|".join(re.escape(x) for x in variants) + r")(?![A-Z0-9])", flags=re.I)
        _STATE_END_PATTERN_CACHE[key] = pattern
    return re.sub(r"\s+", " ", pattern.sub(" ", working)).strip(" ,")

def strip_metadata_from_text(text, state="", pincode=""):
    working = clean_text(text)
    pin = normalize_pin(pincode)
    if pin: working = re.sub(rf"(?<!\d){re.escape(pin)}(?!\d)", " ", working, flags=re.I)
    if state: working = remove_state_variants(working, state)
    working = re.sub(r"\s+", " ", working).strip(" ,")
    return working

def looks_like_city_candidate(value, state="", pincode=""):
    value = clean_text(value).strip(" ,")
    if not value: return False
    normalized = normalize_for_matching(value)
    if not normalized or normalized in PLACEHOLDER_VALUES: return False
    if pincode and normalized == normalize_for_matching(pincode): return False
    if state and normalized == normalize_for_matching(state): return False
    if re.search(r"\b(?:HNO|HOUSE|FLAT|PLOT|SHOP|ROAD|RD|STREET|ST|LANE|MARG|GALI|NEAR|OPP|OPPOSITE|BEHIND|BESIDE|COLONY|NAGAR|SOCIETY|APARTMENT|BUILDING|TOWER|COMPLEX)\b", normalized): return False
    if re.fullmatch(r"\d+", normalized): return False
    return True

def extract_city_from_address(address, state="", pincode="", postal_offices=None, ulb_token_index=None):
    """Infers City directly from raw address utilizing delimiters and the LGD ULB Master."""
    text = clean_text(address)
    if not text: return ""
    state_n = normalize_for_matching(state)
    pin = normalize_pin(pincode)
    segments = [x.strip(" ,") for x in re.split(r"[,;|]+", text) if x.strip(" ,")]
    candidates = []
    
    # 1. Delimiter-based extraction
    for seg in reversed(segments):
        candidate = clean_text(seg)
        if pin: candidate = re.sub(rf"(?<!\d){re.escape(pin)}(?!\d)", " ", candidate, flags=re.I)
        if state: candidate = remove_state_variants(candidate, state)
        candidate = re.sub(r"\s+", " ", candidate).strip(" ,-/")
        if looks_like_city_candidate(candidate, state, pin): candidates.append(candidate)
    
    if candidates: return candidates[0]
    
    # 2. ULB Master Index Extraction (Extracts hidden cities like AMRELI from free text!)
    if ulb_token_index:
        norm_addr = normalize_for_matching(text)
        tokens = set(norm_addr.split())
        possible = set()
        for token in tokens:
            possible.update(ulb_token_index.get(token, set()))
            
        padded = f" {norm_addr} "
        matches = []
        for name in possible:
            if f" {name} " in padded:
                if state and name == normalize_for_matching(state): continue
                matches.append(name)
        if matches:
            return clean_text(max(matches, key=len))

    # 3. Postal Office Extraction Fallback
    if postal_offices and pin:
        address_n = normalize_for_matching(text)
        office_candidates = []
        for office in postal_offices.get(pin, []):
            office_n = normalize_for_matching(office)
            if len(office_n) < 3: continue
            pattern = rf"(?<![A-Z0-9]){re.escape(office_n)}(?![A-Z0-9])"
            if re.search(pattern, address_n, flags=re.I): office_candidates.append(office)
        if office_candidates:
            office_candidates.sort(key=lambda x: len(normalize_for_matching(x)), reverse=True)
            return clean_text(office_candidates[0])

    # 4. Suffix Fallback
    body = strip_metadata_from_text(text, state, pin)
    words = body.split()
    if not words: return ""
    for n in range(1, min(4, len(words)) + 1):
        candidate = " ".join(words[-n:])
        if looks_like_city_candidate(candidate, state, pin): return candidate
    return ""

def build_postal_office_index():
    master_path = BASE_DIR / "masters" / "india_post_master.csv"
    if not master_path.exists(): return {}
    try: master = pd.read_csv(master_path, usecols=["officename", "pincode"], dtype=str)
    except Exception: return {}
    index = {}
    for pin, group in master.groupby("pincode", dropna=False):
        pin_n = normalize_pin(pin)
        if not pin_n: continue
        offices = []
        for office in group["officename"].fillna("").astype(str):
            office = clean_text(office)
            if office and office not in offices: offices.append(office)
        index[pin_n] = offices
    return index

def build_ulb_token_index():
    """Loads the LGD ULB Master for precise city extraction directly in the cleaner."""
    master_path = BASE_DIR / "masters" / "lgd_ulb_master.csv"
    if not master_path.exists(): return {}
    try:
        master = pd.read_csv(master_path, dtype=str)
        col = [c for c in ["Local Body Name (In English)", "Localbody Name (In English)", "Local Body Name", "Localbody Name"] if c in master.columns]
        if not col: return {}
        ulb_token_index = defaultdict(set)
        for name in master[col[0]].fillna("").astype(str):
            name = normalize_for_matching(name)
            if len(name) >= 3:
                for token in name.split():
                    if len(token) >= 3: ulb_token_index[token].add(name)
        return dict(ulb_token_index)
    except Exception:
        return {}

def remove_standalone_pincode_segments(text, pincode):
    pin = normalize_pin(pincode)
    if not pin: return clean_text(text)
    parts = [p.strip(" ,") for p in re.split(r"[,;|]+", clean_text(text))]
    kept = [p for p in parts if normalize_for_matching(p) != pin]
    return ",".join(p for p in kept if p)

def remove_duplicate_metadata_occurrences(text, city="", state=""):
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

def _collapse_repeated_token_blocks(tokens, max_block_size=12):
    if not tokens: return tokens, 0
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
                    copies = 2
                    end = i + copies * block_size
                    while end + block_size <= len(tokens) and normalized[end:end + block_size] == left:
                        copies += 1
                        end += block_size
                    del tokens[i + block_size:end]
                    del normalized[i + block_size:end]
                    removed += (copies - 1) * block_size
                    changed = True
                    found = True
                    break
                i += 1
            if found: break
    return tokens, removed

def _split_on_commas(tokens):
    segment = []
    for token in tokens:
        if token == ",":
            if segment:
                yield segment
                segment = []
        else: segment.append(token)
    if segment: yield segment

def remove_immediate_duplicate_tokens(text):
    tokens = clean_text(text).replace(",", " , ").split()
    result = []
    for segment in _split_on_commas(tokens):
        cleaned_segment, _ = _collapse_repeated_token_blocks(segment)
        deduped = []
        for token in cleaned_segment:
            if deduped and normalize_for_matching(deduped[-1]) == normalize_for_matching(token): continue
            deduped.append(token)
        result.extend(deduped)
        result.append(",")
    if result and result[-1] == ",": result.pop()
    return clean_text(" ".join(result))

def clean_address(address, city="", state="", pincode=""):
    if address is None or pd.isna(address): return ""
    text = clean_text(address)
    if not text: return ""
    text = remove_standalone_pincode_segments(text, pincode)
    text = strip_metadata_from_text(text, state, pincode)
    text = remove_immediate_duplicate_tokens(text)
    text = remove_duplicate_metadata_occurrences(text, city=city, state=state)
    return clean_text(text)

def clean_single_address(address, city, state, pincode, postal_offices, ulb_index):
    raw = clean_text(address)
    pin = find_pincode_in_address(raw)
    
    # Extract state only if it matches exactly
    state_n = ""
    matches = list(_STATE_VARIANT_PATTERN.finditer(normalize_for_matching(raw)))
    if matches: state_n = _STATE_VARIANT_TO_CANONICAL.get(matches[-1].group(0).upper(), "")

    city_value = extract_city_from_address(raw, state=state_n, pincode=pin, postal_offices=postal_offices, ulb_token_index=ulb_index)
    
    raw_body = remove_standalone_pincode_segments(raw, pin)
    cleaned = clean_address(address=raw_body, city=city_value, state=state_n, pincode=pin)
    
    return {
        "Clean Full Address": cleaned, "Clean City": city_value, "Clean Pincode": pin, "Clean State": state_n,
    }

def process_address_column(df, address_column, output_filename, postal_offices, ulb_index, clean_cache=None):
    output_rows = []
    if clean_cache is None: clean_cache = {}
    
    try:
        from tqdm import tqdm
        iterator = tqdm(df[["LAN", address_column]].itertuples(index=False, name=None), total=len(df), desc=f"Cleaning {address_column}", unit="row")
    except ImportError:
        iterator = df[["LAN", address_column]].itertuples(index=False, name=None)

    for row in iterator:
        lan, raw_address = row
        cache_key = clean_text(raw_address)
        cleaned = clean_cache.get(cache_key)
        if cleaned is None:
            cleaned = clean_single_address(address=raw_address, city="", state="", pincode="", postal_offices=postal_offices, ulb_index=ulb_index)
            if len(clean_cache) < 300000: clean_cache[cache_key] = cleaned
        output_rows.append({
            "LAN": lan, "Clean Full Address": cleaned["Clean Full Address"], "Clean City": cleaned["Clean City"],
            "Clean Pincode": cleaned["Clean Pincode"], "Clean State": cleaned["Clean State"], "Original Address": clean_text(raw_address),
        })

    result = pd.DataFrame(output_rows)
    output_path = BASE_DIR / output_filename
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  Created: {output_path.name} ({len(result):,} rows)")
    return output_path

def process_multi_address_file(input_file=None):
    if input_file:
        input_file = Path(input_file)
        if not input_file.is_absolute(): input_file = BASE_DIR / input_file
    else:
        candidates = [BASE_DIR / "Standardized_Input.csv"]
        input_file = next((p for p in candidates if p.exists()), None)
    if input_file is None or not input_file.exists():
        raise FileNotFoundError("No multi-address raw CSV found. Run lender_router.py first.")

    df = pd.read_csv(input_file, dtype=str, keep_default_na=False)
    address_columns = [c for c in ["Current Address", "Office Address", "Alternate Address"] if c in df.columns]

    postal_offices = build_postal_office_index()
    ulb_index = build_ulb_token_index()

    print("=" * 70)
    print("MULTI-ADDRESS CLEANING")
    print("=" * 70)
    print(f"Input file : {input_file.name}")
    print(f"Records    : {len(df):,}")
    print(f"Addresses  : {', '.join(address_columns)}\n")

    ADDRESS_OUTPUTS = {
        "Current Address": "Cleaned_Current_Address_Data.csv",
        "Office Address": "Cleaned_Office_Address_Data.csv",
        "Alternate Address": "Cleaned_Alternate_Address_Data.csv",
    }

    clean_cache = {}
    for address_column in address_columns:
        print(f"Processing {address_column}...")
        process_address_column(df, address_column, ADDRESS_OUTPUTS[address_column], postal_offices, ulb_index, clean_cache=clean_cache)

    print("\nCleaning completed successfully.")

def find_pincode_in_address(address, valid_pins=None):
    text = clean_text(address)
    matches = re.findall(r"(?<!\d)(\d{6})(?!\d)", text)
    if not matches: return ""
    if valid_pins:
        valid = [p for p in matches if p in valid_pins]
        if valid: return valid[-1]
    return matches[-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean multi-address customer data")
    parser.add_argument("--input", help="Raw multi-address CSV path")
    args = parser.parse_args()
    process_multi_address_file(args.input)