import re
from dataclasses import dataclass
import pandas as pd

PASS, FAIL, AMBIGUOUS, NOT_CHECKED, NOT_APPLICABLE = "PASS", "FAIL", "AMBIGUOUS", "NOT_CHECKED", "NOT_APPLICABLE"

@dataclass
class CheckResult:
    status: str
    score_fraction: float
    reason: str = ""
    severity: str = "INFO"
    auto_correctable: bool = False

GARBAGE_TOKENS = {
    "", "-", "--", "NA", "N/A", "NULL", "NONE", "UNKNOWN",
    "NOT AVAILABLE", "NOT PROVIDED", "CALL CUSTOMER", "ASK CUSTOMER",
    "XXXX", "XXXXXX", "0", "0.0"
}

STATE_ALIASES = {
    "CHHATISGARH": "CHHATTISGARH", "MADHYAPRADESH": "MADHYA PRADESH", 
    "UTTARPRADESH": "UTTAR PRADESH", "HIMACHALPRADESH": "HIMACHAL PRADESH", 
    "ANDHRAPRADESH": "ANDHRA PRADESH", "ARUNACHALPRADESH": "ARUNACHAL PRADESH", 
    "JAMMUANDKASHMIR": "JAMMU AND KASHMIR", "WESTBENGAL": "WEST BENGAL", 
    "TAMILNADU": "TAMIL NADU", "UTTARAKHAND": "UTTARAKHAND", 
    "PUDUCHERRY": "PUDUCHERRY", "ORISSA": "ODISHA", "PONDICHERRY": "PUDUCHERRY",
    "HR": "HARYANA", "UP": "UTTAR PRADESH", "MP": "MADHYA PRADESH",
    "HP": "HIMACHAL PRADESH", "RJ": "RAJASTHAN", "PB": "PUNJAB",
    "WB": "WEST BENGAL", "UK": "UTTARAKHAND", "CG": "CHHATTISGARH",
    "AP": "ANDHRA PRADESH", "TS": "TELANGANA", "TG": "TELANGANA",
    "MH": "MAHARASHTRA", "KA": "KARNATAKA", "TN": "TAMIL NADU",
    "GJ": "GUJARAT", "DL": "DELHI", "KL": "KERALA", "BR": "BIHAR",
    "JH": "JHARKHAND", "AS": "ASSAM", "OD": "ODISHA",
    "GA": "GOA", "JK": "JAMMU AND KASHMIR", "J&K": "JAMMU AND KASHMIR",
    "J AND K": "JAMMU AND KASHMIR", "J K": "JAMMU AND KASHMIR",
    "PY": "PUDUCHERRY", "UT": "LADAKH"
}

def clean_text(value) -> str:
    if pd.isna(value): return ""
    if isinstance(value, float) and value.is_integer(): value = int(value)
    return re.sub(r"\s+", " ", str(value).strip())

def normalize_text(value) -> str:
    value = clean_text(value).upper()
    value = re.sub(r"[^A-Z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()

def normalize_pin(value) -> str:
    value = clean_text(value)
    digits = re.sub(r"\D", "", value)
    return digits if len(digits) == 6 else ""

def canonical_state(value) -> str:
    s = normalize_text(value)
    return STATE_ALIASES.get(s, s)

def is_meaningful_text(value) -> bool:
    text = normalize_text(value)
    return bool(text and text not in GARBAGE_TOKENS and re.search(r"[A-Z0-9]", text))

def validate_address_present(value):
    if is_meaningful_text(value): return CheckResult(PASS, 1)
    return CheckResult(FAIL, 0, "No meaningful address was provided.", "CRITICAL")

def validate_pin_format(value):
    raw = clean_text(value)
    if re.fullmatch(r"\d{6}", raw): return CheckResult(PASS, 1)
    if len(re.sub(r"\D", "", raw)) == 6 and raw: return CheckResult(PASS, 1, "PIN formatting can be normalized.", "LOW", True)
    return CheckResult(FAIL, 0, "PIN is not a valid 6-digit value.", "HIGH")

def validate_pin_exists(value, india_post_index):
    pin = normalize_pin(value)
    if not pin: return CheckResult(FAIL, 0, "PIN is invalid or missing.", "CRITICAL")
    if pin in india_post_index.get("pin_rows", {}): return CheckResult(PASS, 1)
    return CheckResult(FAIL, 0, f"PIN {pin} was not found in the India Post master.", "CRITICAL")

def pin_state_match(value, state, india_post_index):
    pin = normalize_pin(value)
    st = canonical_state(state)
    if not pin or not st: return CheckResult(NOT_CHECKED, 0.5)
    states = india_post_index.get("pin_states", {}).get(pin)
    if states is None: return CheckResult(FAIL, 0, "PIN could not be validated.", "CRITICAL")
    if st in states: return CheckResult(PASS, 1)
    if len(states) > 1: return CheckResult(AMBIGUOUS, 0.5, "PIN has multiple geographic state representations.", "MEDIUM")
    return CheckResult(FAIL, 0, "PIN is valid but is inconsistent with the State provided.", "CRITICAL")

def detect_garbage(value, city="", state="", pincode=""):
    text = normalize_text(value)
    if not text or text in GARBAGE_TOKENS: return CheckResult(FAIL, 0, "Address is blank or contains only placeholder/garbage text.", "CRITICAL")
    address_tokens = text.split()
    city_n = normalize_text(city)
    state_n = canonical_state(state)
    pin_n = normalize_pin(pincode)
    metadata_tokens = set(city_n.split()) | set(state_n.split())
    if pin_n: metadata_tokens.add(pin_n)
    meaningful_non_metadata = [t for t in address_tokens if t not in metadata_tokens]
    if len(address_tokens) <= 4 and not meaningful_non_metadata: return CheckResult(FAIL, 0, "Address text contains only duplicated or metadata values and no usable location detail.", "CRITICAL")
    if city_n and len(address_tokens) == 2 and all(t in city_n.split() for t in address_tokens): return CheckResult(FAIL, 0, "Address text contains only the city value and no premise/locality detail.", "CRITICAL")
    hits = []
    for token in GARBAGE_TOKENS:
        if not token: continue
        if re.search(rf"(?<![A-Z0-9]){re.escape(token)}(?![A-Z0-9])", text): hits.append(token)
    if hits: return CheckResult(AMBIGUOUS, 0.5, "Address contains some placeholder/noise text.", "MEDIUM")
    return CheckResult(PASS, 1)

def _pattern_check(value, patterns, fail_reason, severity="MEDIUM"):
    text = normalize_text(value)
    if any(re.search(pattern, text) for pattern in patterns): return CheckResult(PASS, 1)
    return CheckResult(FAIL, 0, fail_reason, severity)

def detect_premise(address):
    text = clean_text(address).upper()
    if not text: return CheckResult(FAIL, 0, "No clear house/flat/plot/building identifier was detected.")
    patterns = [
        r"\bH\s*\.?\s*N\s*\.?\s*O\s*\.?\s*[-:]?\s*[A-Z0-9][A-Z0-9/-]*\b",
        r"\bHOUSE\s*(?:NO|NUMBER)?\s*[-.:]?\s*[A-Z0-9][A-Z0-9/-]*\b",
        r"\bFLAT\s*(?:NO|NUMBER)?\s*[-.:]?\s*[A-Z0-9][A-Z0-9/-]*\b",
        r"\bROOM\s*(?:NO|NUMBER)?\s*[-.:]?\s*[A-Z0-9][A-Z0-9/-]*\b",
        r"\bPLOT\s*(?:NO|NUMBER)?\s*[-.:]?\s*[A-Z0-9][A-Z0-9/-]*\b",
        r"\bDOOR\s*(?:NO|NUMBER)?\s*[-.:]?\s*[A-Z0-9][A-Z0-9/-]*\b",
        r"\bSHOP\s*(?:NO|NUMBER)?\s*[-.:]?\s*[A-Z0-9][A-Z0-9/-]*\b",
        r"\bUNIT\s*(?:NO|NUMBER)?\s*[-.:]?\s*[A-Z0-9][A-Z0-9/-]*\b",
        r"\bKHASRA\s*(?:NO|NUMBER)?\s*[-.:]?\s*[A-Z0-9][A-Z0-9/-]*\b",
        r"\bSURVEY\s*(?:NO|NUMBER)?\s*[-.:]?\s*[A-Z0-9][A-Z0-9/-]*\b",
        r"(?<![A-Z0-9])[A-Z]{1,3}\s*-?\s*\d{1,6}[A-Z0-9/-]*(?![A-Z0-9])",
        r"(?<![A-Z0-9])[A-Z0-9]{1,6}(?:\s*-?\s*[A-Z0-9]{1,6})?(?:\s*/\s*[A-Z0-9]{1,6})+(?:\s*-\s*[A-Z0-9]+)*(?![A-Z0-9])",
        r"(?<![A-Z0-9])\d{1,6}\s*/\s*\d{1,6}(?:\s*/\s*\d{1,6})?(?![A-Z0-9])",
        r"(?<![A-Z0-9])\d{1,6}[A-Z](?:[-/]\d{1,6}[A-Z0-9]*)?(?![A-Z0-9])",
        r"^\s*\d{1,6}(?:\s*[,-])",
    ]
    if any(re.search(pattern, text) for pattern in patterns): return CheckResult(PASS, 1)
    return CheckResult(FAIL, 0, "No clear house/flat/plot/building identifier was detected.")

def detect_building(address):
    return _pattern_check(address, [
        r"\b(?:APARTMENTS?|TOWER|COMPLEX|BUILDING|RESIDENCY|RESIDENCIES|HEIGHTS|PLAZA|ARCADE|BHAVAN|BUNGALOW|VILLA|VILLAS|SHED|GODOWN|WAREHOUSE|UNIT|INDUSTRIAL\s+ESTATE|SHOPPING\s+(?:CENTRE|CENTER|COMPLEX)|OFFICE\s+(?:COMPLEX|BLOCK))\b",
        r"\b(?:SOCIETY|CO\.?\s*OP\.?\s*HOUSING)\b",
    ], "No clear building/property identifier was detected.", "LOW")

def detect_street(address):
    return _pattern_check(address, [
        r"\b(?:ROAD|RD|STREET|ST|LANE|LN|MARG|SECTOR|CROSS|CROSSING|CHOWK|CHOWKI|CHORAHA|GALI|HIGHWAY|NH|EXPRESSWAY|BYPASS|EXTENSION|EXTN|PHASE|MAIN)\b",
    ], "No clear street/road/sector information was detected.")

def detect_locality(address, city=""):
    return _pattern_check(f"{address} {city}", [
        r"\b(?:COLONY|NAGAR|MOHALLA|AREA|BAZAR|BAZAAR|LOCALITY|ENCLAVE|VIHAR|PARK|TOWN|VILLAGE|VILL)\b",
    ], "No clear locality/area information was detected.")

def detect_anchor(address, city="", state="", district=""):
    text = normalize_text(f"{address} {city} {state} {district}")
    anchors = ["VILLAGE", "VILL", "PO", "POST OFFICE", "DIST", "DISTRICT", "CITY", "TOWN", "COLONY", "NAGAR", "SECTOR", "ROAD", "WARD"]
    if any(x in text for x in anchors): return CheckResult(PASS, 1)
    return CheckResult(FAIL, 0, "No strong geographic anchor was detected.", "HIGH")