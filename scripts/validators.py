import re
from dataclasses import dataclass

import pandas as pd


# ============================================================
# STATUS CONSTANTS
# ============================================================

PASS = "PASS"
FAIL = "FAIL"
AMBIGUOUS = "AMBIGUOUS"
NOT_CHECKED = "NOT_CHECKED"
NOT_APPLICABLE = "NOT_APPLICABLE"


# ============================================================
# CHECK RESULT
# ============================================================

@dataclass
class CheckResult:

    status: str
    score_fraction: float
    reason: str = ""
    severity: str = "INFO"
    auto_correctable: bool = False


# ============================================================
# GARBAGE / PLACEHOLDER VALUES
# ============================================================

GARBAGE_TOKENS = {
    "",
    "-",
    "--",
    "NA",
    "N/A",
    "NULL",
    "NONE",
    "UNKNOWN",
    "NOT AVAILABLE",
    "NOT PROVIDED",
    "CALL CUSTOMER",
    "ASK CUSTOMER",
    "XXXX",
    "XXXXXX",
}


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_text(value) -> str:

    value = clean_text(value).upper()

    value = re.sub(
        r"[^A-Z0-9\s]",
        " ",
        value
    )

    return re.sub(
        r"\s+",
        " ",
        value
    ).strip()


def normalize_pin(value) -> str:

    value = clean_text(value)

    digits = re.sub(
        r"\D",
        "",
        value
    )

    return digits if len(digits) == 6 else ""


# ============================================================
# STATE NORMALIZATION
# ============================================================

def canonical_state(value) -> str:

    s = normalize_text(value)

    aliases = {

        "HR": "HARYANA",
        "UP": "UTTAR PRADESH",
        "MP": "MADHYA PRADESH",
        "HP": "HIMACHAL PRADESH",
        "RJ": "RAJASTHAN",
        "PB": "PUNJAB",
        "WB": "WEST BENGAL",
        "UK": "UTTARAKHAND",
        "CG": "CHHATTISGARH",

        "J AND K": "JAMMU AND KASHMIR",
        "J K": "JAMMU AND KASHMIR",
        "J&K": "JAMMU AND KASHMIR",

        "AP": "ANDHRA PRADESH",
        "TS": "TELANGANA",
    }

    return aliases.get(
        s,
        s
    )


# ============================================================
# BASIC TEXT VALIDATION
# ============================================================

def is_meaningful_text(value) -> bool:

    text = normalize_text(value)

    return bool(
        text
        and text not in GARBAGE_TOKENS
        and re.search(r"[A-Z0-9]", text)
    )


def validate_address_present(value):

    if is_meaningful_text(value):

        return CheckResult(
            PASS,
            1
        )

    return CheckResult(
        FAIL,
        0,
        "No meaningful address was provided.",
        "CRITICAL"
    )


# ============================================================
# PIN FORMAT
# ============================================================

def validate_pin_format(value):

    raw = clean_text(value)

    # Perfectly formatted PIN
    if re.fullmatch(
        r"\d{6}",
        raw
    ):

        return CheckResult(
            PASS,
            1
        )

    # Six digits but formatting contains spaces,
    # decimal representation etc.
    if len(
        re.sub(
            r"\D",
            "",
            raw
        )
    ) == 6 and raw:

        return CheckResult(
            PASS,
            1,
            "PIN formatting can be normalized.",
            "LOW",
            True
        )

    return CheckResult(
        FAIL,
        0,
        "PIN is not a valid 6-digit value.",
        "HIGH"
    )


# ============================================================
# INDIA POST — PIN EXISTENCE
# ============================================================

def validate_pin_exists(
    value,
    india_post_index
):
    """
    FAST VERSION.

    OLD:
        Scanned 165K India Post rows for every customer.

    NEW:
        Dictionary lookup.
    """

    pin = normalize_pin(value)

    if not pin:

        return CheckResult(
            FAIL,
            0,
            "PIN is invalid or missing.",
            "CRITICAL"
        )

    pin_rows = india_post_index.get(
        "pin_rows",
        {}
    )

    if pin in pin_rows:

        return CheckResult(
            PASS,
            1
        )

    return CheckResult(
        FAIL,
        0,
        f"PIN {pin} was not found in the India Post master.",
        "CRITICAL"
    )


# ============================================================
# INDIA POST — PIN ↔ STATE
# ============================================================

def pin_state_match(
    value,
    state,
    india_post_index
):
    """
    FAST VERSION.

    Uses:
        pin_states[PIN]

    instead of scanning the full India Post dataframe.
    """

    pin = normalize_pin(value)
    st = canonical_state(state)

    if not pin or not st:

        return CheckResult(
            NOT_CHECKED,
            0.5
        )

    pin_states = india_post_index.get(
        "pin_states",
        {}
    )

    states = pin_states.get(
        pin
    )

    if states is None:

        return CheckResult(
            FAIL,
            0,
            "PIN could not be validated.",
            "CRITICAL"
        )

    if st in states:

        return CheckResult(
            PASS,
            1
        )

    if len(states) > 1:

        return CheckResult(
            AMBIGUOUS,
            0.5,
            "PIN has multiple geographic state representations.",
            "MEDIUM"
        )

    return CheckResult(
        FAIL,
        0,
        "PIN is valid but is inconsistent with the State provided.",
        "CRITICAL"
    )


# ============================================================
# INDIA POST — PIN ↔ DISTRICT
# ============================================================

def pin_district_match(
    value,
    district,
    india_post_index
):
    """
    This is prepared for the next geographic-resolution phase.

    It is already optimized using dictionary lookup.
    """

    pin = normalize_pin(value)
    dist = normalize_text(district)

    if not pin or not dist:

        return CheckResult(
            NOT_CHECKED,
            0.5
        )

    pin_districts = india_post_index.get(
        "pin_districts",
        {}
    )

    districts = pin_districts.get(
        pin
    )

    if districts is None:

        return CheckResult(
            FAIL,
            0,
            "PIN could not be validated.",
            "HIGH"
        )

    if dist in districts:

        return CheckResult(
            PASS,
            1
        )

    if len(districts) > 1:

        return CheckResult(
            AMBIGUOUS,
            0.5,
            "PIN is associated with multiple postal districts.",
            "MEDIUM"
        )

    return CheckResult(
        FAIL,
        0,
        "PIN and District are inconsistent.",
        "HIGH"
    )


# ============================================================
# GARBAGE DETECTION
# ============================================================


def detect_garbage(value, city="", state="", pincode=""):
    """
    Detect true placeholder/empty-content addresses.

    A valid PIN/City/State must not rescue an address whose free-text
    address field contains no usable premise/locality/geographic detail.
    """
    text = normalize_text(value)

    if not text or text in GARBAGE_TOKENS:
        return CheckResult(
            FAIL, 0,
            "Address is blank or contains only placeholder/garbage text.",
            "CRITICAL"
        )

    # Remove punctuation and compare against metadata values.
    address_tokens = text.split()
    city_n = normalize_text(city)
    state_n = canonical_state(state)
    pin_n = normalize_pin(pincode)

    # If the address is only a repeated city/state/PIN value, it is not
    # a usable free-text address.
    metadata_tokens = set(city_n.split()) | set(state_n.split())
    if pin_n:
        metadata_tokens.add(pin_n)

    meaningful_non_metadata = [
        t for t in address_tokens
        if t not in metadata_tokens
    ]

    # Examples: "MEERUT MEERUT", "DELHI DELHI", "250004".
    if len(address_tokens) <= 4 and not meaningful_non_metadata:
        return CheckResult(
            FAIL, 0,
            "Address text contains only duplicated or metadata values and no usable location detail.",
            "CRITICAL"
        )

    # A single city repeated twice is especially weak.
    if (
        city_n
        and len(address_tokens) == 2
        and all(t in city_n.split() for t in address_tokens)
    ):
        return CheckResult(
            FAIL, 0,
            "Address text contains only the city value and no premise/locality detail.",
            "CRITICAL"
        )

    hits = []
    for token in GARBAGE_TOKENS:
        if not token:
            continue
        if " " in token:
            found = re.search(
                rf"(?<![A-Z0-9]){re.escape(token)}(?![A-Z0-9])",
                text,
            )
        else:
            found = re.search(
                rf"(?<![A-Z0-9]){re.escape(token)}(?![A-Z0-9])",
                text,
            )
        if found:
            hits.append(token)

    if hits:
        return CheckResult(
            AMBIGUOUS, 0.5,
            "Address contains some placeholder/noise text.",
            "MEDIUM"
        )

    return CheckResult(PASS, 1)

def _pattern_check(
    value,
    patterns,
    fail_reason,
    severity="MEDIUM"
):

    text = normalize_text(value)

    if any(
        re.search(pattern, text)
        for pattern in patterns
    ):

        return CheckResult(
            PASS,
            1
        )

    return CheckResult(
        FAIL,
        0,
        fail_reason,
        severity
    )


# ============================================================
# PREMISE
# ============================================================


def detect_premise(address):
    """
    High-recall Indian premise identifier detection.

    Covers common forms such as:
        A-13
        D-43
        H-338
        E-49/C-24
        23/5
        391/20
        12/48
        HNO. 767
        HOUSE NO 333
        PLOT NO 601
        FLAT 12
        KHASRA NO 121/2
        SURVEY NO 45
    """
    text = clean_text(address).upper()

    if not text:
        return CheckResult(
            FAIL, 0,
            "No clear house/flat/plot/building identifier was detected."
        )

    patterns = [
        # Explicit labels.
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

        # Alphanumeric premise IDs: A-13, D-43, H-338, C-2445.
        r"(?<![A-Z0-9])[A-Z]{1,3}\s*-\s*\d{1,6}[A-Z0-9/-]*(?![A-Z0-9])",

        # Alphanumeric slash IDs: E-49/C-24, RZ-19/C-11-A.
        r"(?<![A-Z0-9])[A-Z]{1,4}\s*-\s*\d{1,6}(?:/[A-Z]{1,4}\s*-\s*\d{1,6})+(?:-[A-Z0-9]+)?",

        # Numeric premise IDs: 23/5, 391/20, 12/48, 121/2.
        r"(?<![A-Z0-9])\d{1,6}\s*/\s*\d{1,6}(?:\s*/\s*\d{1,6})?(?![A-Z0-9])",

        # Number followed by a letter, e.g. 12A.
        r"(?<![A-Z0-9])\d{1,6}[A-Z](?:[-/]\d{1,6}[A-Z0-9]*)?(?![A-Z0-9])",

        # Plain numbered premises where number is at the beginning.
        r"^\s*\d{1,6}(?:\s*[,-])",
    ]

    if any(re.search(pattern, text) for pattern in patterns):
        return CheckResult(PASS, 1)

    return CheckResult(
        FAIL, 0,
        "No clear house/flat/plot/building identifier was detected."
    )

def detect_building(address):

    return _pattern_check(

        address,

        [

            r"\b(?:APARTMENTS?|TOWER|COMPLEX|BUILDING|RESIDENCY|RESIDENCIES|"
            r"HEIGHTS|PLAZA|ARCADE|BHAVAN|BUNGALOW|VILLA|VILLAS|"
            r"SHOPPING\s+(?:CENTRE|CENTER|COMPLEX)|OFFICE\s+(?:COMPLEX|BLOCK))\b",
            r"\b(?:SOCIETY|CO\.?\s*OP\.?\s*HOUSING)\b",

        ],

        "No clear building/property identifier was detected.",

        "LOW"
    )


# ============================================================
# STREET
# ============================================================

def detect_street(address):

    return _pattern_check(

        address,

        [

            r"\b(?:ROAD|RD|STREET|ST|LANE|LN|MARG|SECTOR|CROSS|"
            r"CROSSING|CHOWK|CHOWKI|CHORAHA|GALI|HIGHWAY|NH|"
            r"EXPRESSWAY|BYPASS|EXTENSION|EXTN|PHASE|MAIN)\b",

        ],

        "No clear street/road/sector information was detected."
    )


# ============================================================
# LOCALITY
# ============================================================

def detect_locality(
    address,
    city=""
):

    return _pattern_check(

        f"{address} {city}",

        [

            r"\b(?:COLONY|NAGAR|MOHALLA|AREA|BAZAR|BAZAAR|LOCALITY|ENCLAVE|VIHAR|PARK|TOWN|VILLAGE|VILL)\b",

        ],

        "No clear locality/area information was detected."
    )


# ============================================================
# GEOGRAPHIC ANCHOR
# ============================================================

def detect_anchor(
    address,
    city="",
    state="",
    district=""
):

    text = normalize_text(
        f"{address} {city} {state} {district}"
    )

    anchors = [

        "VILLAGE",
        "VILL",
        "PO",
        "POST OFFICE",
        "DIST",
        "DISTRICT",
        "CITY",
        "TOWN",
        "COLONY",
        "NAGAR",
        "SECTOR",
        "ROAD",
        "WARD",
    ]

    if any(
        x in text
        for x in anchors
    ):

        return CheckResult(
            PASS,
            1
        )

    return CheckResult(
        FAIL,
        0,
        "No strong geographic anchor was detected.",
        "HIGH"
    )