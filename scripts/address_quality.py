from pathlib import Path
import sys
import time
import re
import json
from collections import OrderedDict

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))

from master_loader import load_masters, prepare_indexes

from entity_resolution import (
    GeographicResolver,
    normalize_entity,
    canonical_state,
)

from validators import (
    PASS,
    FAIL,
    AMBIGUOUS,
    validate_address_present,
    validate_pin_format,
    validate_pin_exists,
    pin_state_match,
    detect_garbage,
    normalize_pin,
)

from scoring import (
    calculate_score,
    classify_score,
)

from remarks import build_remarks


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = ROOT / "Cleaned_Address_Data.csv"

MASTERS_DIR = ROOT / "masters"

OUTPUT_FILE = (
    ROOT
    / "output"
    / "address_quality_v3.csv"
)


# ============================================================
# COLUMN FINDER
# ============================================================

def col(df, names, required=False):

    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for name in names:

        if name.lower() in lookup:
            return lookup[name.lower()]

    if required:

        raise ValueError(
            f"Missing required column. "
            f"Expected one of: {names}"
        )

    return None


# ============================================================
# BASIC TEXT HELPERS
# ============================================================

def safe_text(value):

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def norm(value):

    return normalize_entity(
        safe_text(value)
    )


def extract_locality_candidate(address):
    """Extract an explicitly labeled locality/area candidate for P1 evidence."""
    text = safe_text(address)
    if not text:
        return ""
    patterns = (
        r"\bLOCALITY\s*[-.:]?\s*([^,;]+)",
        r"\bAREA\s*[-.:]?\s*([^,;]+)",
        r"\bMOHALLA(?:H)?\s*[-.:]?\s*([^,;]+)",
    )
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.I)
        if m and m.group(1).strip():
            return m.group(1).strip()
    return ""

def build_address_metadata_json(
    *,
    address,
    city,
    state,
    pin,
    address_type,
    extracted,
    resolved,
    statuses,
    component_confidence,
    routing,
):
    """Create a compact, auditable structured representation of one address."""
    payload = {
        "input": {
            "address": safe_text(address),
            "city": safe_text(city),
            "state": safe_text(state),
            "pincode": normalize_pin(pin),
        },
        "address_type": address_type,
        "extracted_entities": {k: safe_text(v) for k, v in extracted.items()},
        "resolved_entities": resolved,
        "validation": statuses,
        "component_confidence": component_confidence,
        "routing": routing,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


# ============================================================
# ADDRESS SEGMENTS
# ============================================================

def address_segments(address):

    text = safe_text(address)

    if not text:
        return []

    # Normalize separators.
    text = text.replace(
        ";",
        ","
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    segments = []

    for part in text.split(","):

        part = re.sub(
            r"\s+",
            " ",
            part.strip()
        )

        if part:
            segments.append(part)

    return segments


# ============================================================
# LABEL EXTRACTION
# ============================================================


def extract_labeled_entities(address):
    """
    Extract explicit administrative labels from common Indian address
    writing styles. Labels need not be separated by commas.
    """
    text = safe_text(address)

    result = {
        "village": "",
        "post_office": "",
        "district": "",
        "subdistrict": "",
    }

    if not text:
        return result

    # Stop at the next comma/semicolon or end of string.
    patterns = {
        "village": [
            r"\bVILL(?:AGE)?\s*[-.:]?\s*([^,;]+)",
            r"\bGRAM\s*[-.:]?\s*([^,;]+)",
            r"\bGAON\s*[-.:]?\s*([^,;]+)",
        ],
        "post_office": [
            r"\bP\.?\s*O\.?\s*[-.:]?\s*([^,;]+)",
            r"\bPOST\s+OFFICE\s*[-.:]?\s*([^,;]+)",
            r"\bPOST\s*[-.:]?\s*([^,;]+)",
        ],
        "district": [
            r"\bDIST(?:RICT)?\s*[-.:]?\s*([^,;]+)",
            r"\bDT\s*[-.:]?\s*([^,;]+)",
        ],
        "subdistrict": [
            r"\bTEHSIL\s*[-.:]?\s*([^,;]+)",
            r"\bTAHSIL\s*[-.:]?\s*([^,;]+)",
            r"\bTALUK(?:A)?\s*[-.:]?\s*([^,;]+)",
            r"\bTALUKAM?\s*[-.:]?\s*([^,;]+)",
            r"\bMANDAL\s*[-.:]?\s*([^,;]+)",
            r"\bBLOCK\s*[-.:]?\s*([^,;]+)",
            r"\bSUB[- ]?DISTRICT\s*[-.:]?\s*([^,;]+)",
        ],
    }

    for key, pats in patterns.items():
        for pattern in pats:
            m = re.search(pattern, text, flags=re.I)
            if m:
                value = m.group(1).strip()
                if value:
                    result[key] = value
                    break

    return result

def is_geographic_candidate_segment(
    segment,
    city="",
    state="",
    pin="",
):

    value = norm(segment)

    if not value:
        return False

    # Very long segments are usually complete premise
    # descriptions rather than village/PO names.
    if len(value) > 80:
        return False

    # Ignore pure PIN.
    if re.fullmatch(
        r"\d{6}",
        value
    ):
        return False

    # Ignore state.
    if value == canonical_state(state):
        return False

    # Ignore city when searching village.
    if city and value == norm(city):
        return False

    # Ignore obvious premise-only fragments.
    if re.search(
        r"\b(?:NEAR|OPP|OPPOSITE|BEHIND|BESIDE|"
        r"ROAD|RD|STREET|ST|LANE|MARG|"
        r"COLONY|NAGAR|SOCIETY|APARTMENT|"
        r"FLAT|HOUSE|HNO|PLOT|SHOP)\b",
        value,
    ):

        # Do not immediately reject.
        # A village may itself contain NAGAR/COLONY.
        pass

    return True


# ============================================================
# INFER VILLAGE FROM UNLABELED ADDRESS
# ============================================================


def infer_village(
    address,
    city,
    state,
    district,
    pin,
    resolver,
):
    """
    Fast village inference.

    Performance rule:
      - exact segment lookup first
      - token-index candidate lookup second
      - never scan/sort the district's entire village list per row
      - fuzzy resolution is delegated to GeographicResolver only when needed
    """
    address_text = safe_text(address)
    if not address_text:
        return ""

    # 1) Explicit/literal comma-separated segments.
    for segment in address_segments(address_text):
        if not is_geographic_candidate_segment(segment, city, state, pin):
            continue
        candidate = norm(segment)
        if candidate in resolver.village_by_name:
            result = resolver.resolve_village(
                candidate, state=state, district=district, pin=pin
            )
            if result.status == PASS:
                return result.match_value

    # 2) Token index -> small candidate set.
    address_norm = norm(address_text)
    if not address_norm:
        return ""

    state_n = norm(state)
    district_n = norm(district)
    pin_n = normalize_pin(pin)

    candidate_names = resolver._token_candidate_names(
        address_norm,
        resolver.village_token_index,
    )

    # Do not materialize the entire district village list here.
    # The token index already gives a small candidate set, and the
    # resolver applies district/state/PIN context when resolving each
    # candidate.

    # Use padded containment rather than compiling a regex for every candidate.
    padded = f" {address_norm} "
    for name in candidate_names:
        if not name or len(name) < 3:
            continue
        if f" {name} " in padded:
            result = resolver.resolve_village(
                name, state=state_n, district=district_n, pin=pin_n
            )
            if result.status == PASS:
                return result.match_value

    return ""

def infer_subdistrict(
    address,
    city,
    state,
    district,
    pin,
    resolver,
):
    """
    Fast subdistrict inference using exact lookup + token index.
    Avoids scanning all ~7K subdistrict names for every customer.
    """
    address_text = safe_text(address)
    if not address_text:
        return ""

    # 1) Exact segment lookup.
    for segment in address_segments(address_text):
        candidate = norm(segment)
        if not candidate:
            continue
        result = resolver.resolve_subdistrict(
            candidate,
            state=state,
            district=district,
            pin=pin,
        )
        if result.status == PASS:
            return result.match_value

    # 2) Token-index candidates only.
    address_norm = norm(address_text)
    candidates = resolver._token_candidate_names(
        address_norm,
        resolver.subdistrict_token_index,
    )

    district_n = norm(district)
    if district_n:
        bounded = resolver.subdistrict_by_district.get(district_n, {})
        if isinstance(bounded, dict) and bounded:
            allowed = set(bounded.keys())
            candidates = [n for n in candidates if n in allowed]

    padded = f" {address_norm} "
    for name in candidates:
        if name and len(name) >= 3 and f" {name} " in padded:
            result = resolver.resolve_subdistrict(
                name,
                state=state,
                district=district_n,
                pin=pin,
            )
            if result.status == PASS:
                return result.match_value

    return ""

def infer_post_office(
    address,
    pin,
    resolver,
):
    """
    Fast India Post PO inference.

    PIN -> PO names is already indexed, so only the offices belonging
    to this PIN are considered. Sorted PO lists are cached on the resolver.
    """
    pin_n = normalize_pin(pin)
    if not pin_n:
        return ""

    offices = resolver.pin_post_offices.get(pin_n, set())
    if not offices:
        return ""

    # Cache normalized/sorted names by PIN.
    cache = getattr(resolver, "_sorted_po_cache", None)
    if cache is None:
        cache = {}
        resolver._sorted_po_cache = cache

    if pin_n not in cache:
        normalized = {
            norm(x) for x in offices if norm(x)
        }
        cache[pin_n] = sorted(normalized, key=len, reverse=True)

    offices_sorted = cache[pin_n]
    address_norm = norm(address)
    if not address_norm:
        return ""

    # Explicit PO label first.
    for pattern in (
        r"\bP\.?\s*O\.?\s*[-.:]?\s*([^,;]+)",
        r"\bPOST\s+OFFICE\s*[-.:]?\s*([^,;]+)",
        r"\bPOST\s*[-.:]?\s*([^,;]+)",
    ):
        m = re.search(pattern, safe_text(address), flags=re.I)
        if m:
            candidate = norm(m.group(1))
            if candidate in offices_sorted:
                return candidate

    padded = f" {address_norm} "
    for office in offices_sorted:
        if office and f" {office} " in padded:
            return office

    return ""

def infer_district(
    address,
    city,
    state,
    pin,
    resolver,
):
    """
    Fast district inference.

    Prefer City -> exact district resolution. If that fails, inspect
    address segments only. No global district scan is performed here.
    """
    if city:
        result = resolver.resolve_district(
            city,
            state=state,
            pin=pin,
        )
        if result.status == PASS:
            return result.match_value

    for segment in address_segments(address):
        candidate = norm(segment)
        if not candidate:
            continue
        result = resolver.resolve_district(
            candidate,
            state=state,
            pin=pin,
        )
        if result.status == PASS:
            return result.match_value

    return ""

def enhanced_premise(address):
    text = safe_text(address).upper()

    if not text:
        return False

    patterns = [
        # Explicit labels.
        r"\bH\.?\s*N\.?\s*O\.?\s*[-:.]?\s*[A-Z0-9][A-Z0-9/-]*\b",
        r"\bHOUSE\s*(?:NO|NUMBER)?\s*[-:.]?\s*[A-Z0-9][A-Z0-9/-]*\b",
        r"\bFLAT\s*(?:NO|NUMBER)?\s*[-:.]?\s*[A-Z0-9][A-Z0-9/-]*\b",
        r"\bROOM\s*(?:NO|NUMBER)?\s*[-:.]?\s*[A-Z0-9][A-Z0-9/-]*\b",
        r"\bPLOT\s*(?:NO|NUMBER)?\s*[-:.]?\s*[A-Z0-9][A-Z0-9/-]*\b",
        r"\bDOOR\s*(?:NO|NUMBER)?\s*[-:.]?\s*[A-Z0-9][A-Z0-9/-]*\b",
        r"\bSHOP\s*(?:NO|NUMBER)?\s*[-:.]?\s*[A-Z0-9][A-Z0-9/-]*\b",
        r"\bUNIT\s*(?:NO|NUMBER)?\s*[-:.]?\s*[A-Z0-9][A-Z0-9/-]*\b",
        r"\bKHASRA\s*(?:NO|NUMBER)?\s*[-:.]?\s*[A-Z0-9][A-Z0-9/-]*\b",
        r"\bSURVEY\s*(?:NO|NUMBER)?\s*[-:.]?\s*[A-Z0-9][A-Z0-9/-]*\b",

        # A-13, D-43, H-338, C-2445.
        r"(?<![A-Z0-9])[A-Z]{1,3}\s*-\s*\d{1,6}[A-Z0-9/-]*(?![A-Z0-9])",

        # E-49/C-24, RZ-19/C-11-A.
        r"(?<![A-Z0-9])[A-Z]{1,4}\s*-\s*\d{1,6}(?:/[A-Z]{1,4}\s*-\s*\d{1,6})+(?:-[A-Z0-9]+)?",

        # 23/5, 391/20, 12/48, 121/2.
        r"(?<![A-Z0-9])\d{1,6}\s*/\s*\d{1,6}(?:\s*/\s*\d{1,6})?(?![A-Z0-9])",

        # 12A, 302E, etc.
        r"(?<![A-Z0-9])\d{1,6}[A-Z](?:[-/]\d{1,6}[A-Z0-9]*)?(?![A-Z0-9])",

        # Leading plain house number: "502 PUNEET...", "25, ...".
        r"^\s*\d{1,6}\s*(?:[, -])",
        r"^\s*\d{1,6}\s+[A-Z]",
    ]

    return any(re.search(p, text) for p in patterns)

def enhanced_building(
    address,
):

    text = safe_text(address).upper()

    if not text:
        return False

    patterns = [

        r"\bBUILDING\b",

        r"\bAPARTMENT\b",

        r"\bAPPARTMENT\b",

        r"\bBUNGALOW\b",

        r"\bRESIDENCY\b",

        r"\bRESIDENCE\b",

        r"\bTOWER\b",

        r"\bBLOCK\b",

        r"\bCOMPLEX\b",

        r"\bSOCIETY\b",

        r"\bHOSTEL\b",

        r"\bSCHOOL\b",

        r"\bCOLLEGE\b",

        r"\bHOSPITAL\b",

        r"\bHOTEL\b",

        r"\bGARMENTS\b",

        r"\bSHOP\b",

        r"\bOFFICE\b",

        r"\bFACTORY\b",

        r"\bMALL\b",

        r"\bMARKET\b",

    ]

    return any(
        re.search(
            pattern,
            text,
        )
        for pattern in patterns
    )


# ============================================================
# ENHANCED STREET DETECTION
# ============================================================

def enhanced_street(
    address,
):

    text = safe_text(address).upper()

    if not text:
        return False

    patterns = [

        r"\bROAD\b",
        r"\bRD\b",
        r"\bSTREET\b",
        r"\bST\b",
        r"\bLANE\b",
        r"\bMARG\b",
        r"\bPATH\b",
        r"\bGALI\b",
        r"\bGALII\b",
        r"\bCHOWK\b",
        r"\bCROSS\b",
        r"\bHIGHWAY\b",
        r"\bHWY\b",
        r"\bRING\s*ROAD\b",
        r"\bMAIN\s*ROAD\b",
        r"\bBYPASS\b",
        r"\bCIRCULAR\s*ROAD\b",
        r"\bSECTOR\b",
        r"\bWARD\b",
    ]

    return any(
        re.search(
            pattern,
            text,
        )
        for pattern in patterns
    )


# ============================================================
# ENHANCED LOCALITY DETECTION
# ============================================================

def enhanced_locality(
    address,
):

    text = safe_text(address).upper()

    if not text:
        return False

    patterns = [

        r"\bCOLONY\b",
        r"\bNAGAR\b",
        r"\bMOHALLA\b",
        r"\bMOHALLAH\b",
        r"\bLOCALITY\b",
        r"\bAREA\b",
        r"\bBASTI\b",
        r"\bPURA\b",
        r"\bDHANI\b",
        r"\bKHEDA\b",
        r"\bTOLA\b",
        r"\bPARA\b",
        r"\bPURAM\b",
        r"\bENCLAVE\b",
        r"\bEXTENSION\b",
        r"\bPHASE\b",
        r"\bSOCIETY\b",
        r"\bVILLAGE\b",
        r"\bVILL\b",
        r"\bGRAM\b",
        r"\bGAON\b",
        r"\bWARD\b",
    ]

    return any(
        re.search(
            pattern,
            text,
        )
        for pattern in patterns
    )


# ============================================================
# ENHANCED GEOGRAPHIC ANCHOR
# ============================================================

def enhanced_anchor(address):
    """Return address-text geographic anchor evidence only.

    Structured City/State/PIN and values inferred from masters are not used
    to manufacture completeness credit. The anchor must be visible in the
    free-text address itself.
    """
    text = safe_text(address).upper()
    if not text:
        return False

    geographic_terms = [
        r"\bVILLAGE\b", r"\bVILL\b", r"\bGRAM\b", r"\bGAON\b",
        r"\bPO\b", r"\bP\s*O\b", r"\bPOST\b", r"\bDIST\b",
        r"\bDISTRICT\b", r"\bROAD\b", r"\bRD\b", r"\bSTREET\b",
        r"\bLANE\b", r"\bGALI\b", r"\bNAGAR\b", r"\bCOLONY\b",
        r"\bMOHALLA\b", r"\bAREA\b", r"\bSECTOR\b", r"\bWARD\b",
        r"\bCHOWK\b", r"\bCHOWRAHA\b", r"\bBASTI\b", r"\bPURA\b",
        r"\bDHANI\b", r"\bENCLAVE\b", r"\bMARG\b", r"\bPATH\b",
        r"\bHIGHWAY\b", r"\bCROSSING\b", r"\bBLOCK\b",
    ]

    return any(re.search(pattern, text) for pattern in geographic_terms)


# ============================================================
# ADDRESS TYPE + COMPONENT EXTRACTION
# ============================================================

# These markers are used only as a routing optimization: clearly urban
# addresses skip expensive rural hierarchy inference.  They must therefore
# be conservative and must not affect the quality score directly.
RURAL_MARKERS = re.compile(
    r"\b(?:VILL(?:AGE)?|GRAM|GAON|KHET|KHEDA|DHANI|TOLA|"
    r"PURA|MAJRA|MAUZA|TEHSIL|TAHSIL|TALUKA?|MANDAL|"
    r"SUB[- ]?DISTRICT|FARM HOUSE|FARMS)\b",
    re.I,
)

URBAN_MARKERS = re.compile(
    r"\b(?:FLAT|APARTMENT|RESIDENCY|RESIDENCE|TOWER|"
    r"SECTOR|SHOPPING|COMPLEX|SOCIETY|ENCLAVE|COLONY|NAGAR|"
    r"STREET|ROAD|LANE|MARG|GALI|CHOWK|WARD|MARKET|MALL|"
    r"OFFICE|PLOT|HOUSE|HNO|DOOR|UNIT|SHOP)\b",
    re.I,
)

def classify_address_type(address, city=""):
    """Classify only for inference routing; UNKNOWN is intentionally common."""
    text = safe_text(address)
    if not text:
        return "UNKNOWN"
    if RURAL_MARKERS.search(text):
        return "RURAL"
    if URBAN_MARKERS.search(text):
        return "URBAN"
    return "UNKNOWN"


def extract_components(address, city=""):
    """Run completeness detectors once and reuse their evidence."""
    premise = bool(enhanced_premise(address))
    building = bool(enhanced_building(address))
    street = bool(enhanced_street(address))
    locality = bool(enhanced_locality(address))
    return {
        "premise": premise,
        "building": building,
        "street": street,
        "locality": locality,
        "anchor": bool(street or locality),
    }


# ============================================================
# GEO COMPONENT
# ============================================================

def geo_component(
    status,
):

    if status == PASS:
        return 1.0

    if status == AMBIGUOUS:
        return 0.5

    if status == "NOT_CHECKED":
        return 0.5

    return 0.0


# ============================================================
# PIN ↔ DISTRICT
# ============================================================

def check_pin_district_evidence(
    address,
    pin,
    indexes,
):
    """
    Infer district(s) from the India Post PIN master and check whether
    any India Post district associated with that PIN is explicitly present
    in the cleaned address.

    This is NOT a customer-supplied District validation.

    PASS:
        A district associated with the PIN is explicitly present in the
        cleaned address.

    NOT_FOUND:
        The PIN has district information, but none of those district names
        occurs in the address. This is neutral, not a contradiction.

    AMBIGUOUS:
        More than one India Post district associated with the PIN is found
        in the address.

    NOT_CHECKED:
        PIN/district reference data is unavailable.
    """
    pin = normalize_pin(pin)
    address_norm = norm(address)

    if not pin or not address_norm:
        return "NOT_CHECKED", 0.5, []

    pin_districts = sorted(
        {
            norm(x)
            for x in indexes.get("pin_districts", {}).get(pin, set())
            if norm(x)
        },
        key=len,
        reverse=True,
    )

    if not pin_districts:
        return "NOT_CHECKED", 0.5, []

    matched = []

    for district in pin_districts:
        if re.search(
            rf"(?<![A-Z0-9]){re.escape(district)}(?![A-Z0-9])",
            address_norm,
        ):
            matched.append(district)

    if len(matched) == 1:
        return "PASS", 1.0, matched

    if len(matched) > 1:
        return "AMBIGUOUS", 0.5, matched

    # Absence is not a mismatch because District is not customer-supplied.
    return "NOT_FOUND", 0.5, []


def india_post_districts_for_pin(pin, indexes):
    pin = normalize_pin(pin)

    if not pin:
        return ""

    districts = sorted(
        {
            str(x).strip()
            for x in indexes.get("pin_districts", {}).get(pin, set())
            if str(x).strip()
        }
    )

    return "; ".join(districts)


# ============================================================
# LEGACY DISTRICT ↔ STATE CHECK
# ============================================================
#
# Intentionally retained only as a compatibility helper.
# It is no longer used in V5 scoring or QC because District is not
# customer-supplied in the input dataset.
# ============================================================

def check_district_state(
    pin,
    district,
    state,
    indexes,
):
    return "NOT_CHECKED", 0.5


# ============================================================
# VILLAGE HIERARCHY SCORE
# ============================================================


def village_hierarchy_score(
    village_res,
    village_supplied,
):
    if not village_supplied:
        # Village is not a mandatory component for an urban address.
        return 1.0

    if village_res.status == PASS:
        return 1.0

    if village_res.status == AMBIGUOUS:
        return 0.5

    return 0.0


def get_village_hierarchy_status(village_res, village_supplied):
    if not village_supplied:
        return "NOT_CHECKED"
    if village_res.status == PASS:
        return PASS
    if village_res.status == AMBIGUOUS:
        return AMBIGUOUS
    return FAIL


def po_city_status(po_res, ulb_res, city, post_office_supplied):
    if post_office_supplied:
        if po_res.status == PASS:
            return PASS
        if po_res.status == AMBIGUOUS:
            return AMBIGUOUS
        return FAIL
    if not city:
        return "NOT_CHECKED"
    if ulb_res.status == PASS:
        return PASS
    if ulb_res.status == AMBIGUOUS:
        return AMBIGUOUS
    return "NOT_CHECKED"


def po_city_score(
    po_res,
    ulb_res,
    city,
    post_office_supplied,
):
    """Score postal/city evidence without treating missing optional fields as failures."""
    if post_office_supplied:
        if po_res.status == PASS:
            return 1.0
        if po_res.status == AMBIGUOUS:
            return 0.5
        return 0.0

    if city:
        if ulb_res.status == PASS:
            return 1.0
        if ulb_res.status == AMBIGUOUS:
            return 0.5
        # City may be a town/district name rather than an LGD ULB.
        # Without positive reference evidence, remain neutral.
        return 0.5

    return 0.5

def preflight_input_guard(df):
    """Fail fast when an input schema/value problem would invalidate the run."""
    address_c = col(
        df,
        [
            "Full Address", "full_address", "address",
            "Clean Full Address", "clean full address", "clean_address",
        ],
        True,
    )
    pin_c = col(
        df,
        [
            "Pincode", "PIN", "pincode",
            "Clean Pincode", "clean pincode", "clean_pincode",
        ],
        True,
    )
    state_c = col(
        df,
        [
            "State", "state",
            "Clean State", "clean state", "clean_state",
        ],
        True,
    )
    city_c = col(
        df,
        [
            "City", "city",
            "Clean City", "clean city", "clean_city",
        ],
    )

    def non_empty_count(series):
        return int(series.map(safe_text).ne("").sum())

    address_non_empty = non_empty_count(df[address_c])
    pin_non_empty = non_empty_count(df[pin_c])
    state_non_empty = non_empty_count(df[state_c])

    normalized_pins = df[pin_c].map(normalize_pin)
    valid_pin_count = int(normalized_pins.ne("").sum())

    print("\n      INPUT DATA GUARD:")
    print(f"      Address column : {address_c}")
    print(f"      City column    : {city_c}")
    print(f"      PIN column     : {pin_c}")
    print(f"      State column   : {state_c}")
    print(f"      Non-empty addr : {address_non_empty:,}/{len(df):,}")
    print(f"      Non-empty PIN  : {pin_non_empty:,}/{len(df):,}")
    print(f"      Valid 6-digit  : {valid_pin_count:,}/{len(df):,}")

    failures = []

    if len(df) == 0:
        failures.append("Input file contains zero customer records.")

    if address_non_empty == 0 and len(df) > 0:
        failures.append(
            f"Address column '{address_c}' is 100% empty."
        )

    if state_non_empty == 0 and len(df) > 0:
        failures.append(
            f"State column '{state_c}' is 100% empty."
        )

    # PIN is a critical branch. A zero-valid-PIN batch is always a pipeline
    # failure for this dataset, not a legitimate address-quality outcome.
    if valid_pin_count == 0 and len(df) > 0:
        failures.append(
            f"Pincode column '{pin_c}' contains 0 valid 6-digit PINs. "
            "This indicates an input-schema or normalization failure."
        )

    if failures:
        raise RuntimeError(
            "\n\nINPUT PIPELINE GUARD FAILED:\n- "
            + "\n- ".join(failures)
            + "\n\nNo validation output was written."
        )

    # Catch catastrophic partial corruption too. This is intentionally a
    # high threshold so legitimate bad customer PINs are still measurable.
    invalid_pin_rate = 1.0 - (valid_pin_count / len(df)) if len(df) else 1.0
    if invalid_pin_rate >= 0.50:
        raise RuntimeError(
            f"\n\nINPUT PIPELINE GUARD FAILED: {invalid_pin_rate:.1%} of PINs "
            f"in '{pin_c}' are invalid/empty after normalization. "
            "This is too high to safely run the downstream geography engine."
        )

    return {
        "address_c": address_c,
        "city_c": city_c,
        "pin_c": pin_c,
        "state_c": state_c,
        "valid_pin_count": valid_pin_count,
    }


def main():

    started = time.perf_counter()

    print("=" * 70)
    print(
        "ADDRESS QUALITY VALIDATION - P1"
    )
    print(
        "HIGH-RECALL EXTRACTION + SCORE/REVIEW SEPARATION"
    )
    print("=" * 70)

    # ========================================================
    # INPUT
    # ========================================================

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Cleaned customer input not found:\n"
            f"{INPUT_FILE}\n\n"
            f"Run clean_addresses.py first."
        )

    print(
        "\n[1/6] Loading cleaned customer data..."
    )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    print(
        f"      Customer records: "
        f"{len(df):,}"
    )

    # Fail before loading the 800K+ reference records if the input schema
    # itself is broken.
    preflight = preflight_input_guard(df)

    # ========================================================
    # MASTERS
    # ========================================================

    print(
        "\n[2/6] Loading masters..."
    )

    masters_started = time.perf_counter()

    masters = load_masters(
        str(MASTERS_DIR)
    )

    print(
        f"      India Post : "
        f"{len(masters['india_post']):,}"
    )

    print(
        f"      District   : "
        f"{len(masters['district']):,}"
    )

    print(
        f"      Subdistrict: "
        f"{len(masters['subdistrict']):,}"
    )

    print(
        f"      Village    : "
        f"{len(masters['village']):,}"
    )

    print(
        f"      ULB        : "
        f"{len(masters['ulb']):,}"
    )

    print(
        f"      Load time  : "
        f"{time.perf_counter() - masters_started:.2f} sec"
    )

    # ========================================================
    # INDEXES
    # ========================================================

    print(
        "\n[3/6] Building indexes..."
    )

    index_started = time.perf_counter()

    indexes = prepare_indexes(
        masters
    )

    print(
        f"      PINs indexed: "
        f"{len(indexes['pin_rows']):,}"
    )

    print(
        f"      Index time  : "
        f"{time.perf_counter() - index_started:.2f} sec"
    )

    resolver = GeographicResolver(
        masters,
        indexes,
    )

    # ========================================================
    # CUSTOMER COLUMNS
    # ========================================================

    address_c = col(
        df,
        [
            "Full Address",
            "full_address",
            "address",
            "Clean Full Address",
            "clean full address",
            "clean_address",
        ],
        True,
    )

    city_c = col(
        df,
        [
            "City",
            "city",
            "Clean City",
            "clean city",
            "clean_city",
        ],
    )

    pin_c = col(
        df,
        [
            "Pincode",
            "PIN",
            "pincode",
            "Clean Pincode",
            "clean pincode",
            "clean_pincode",
        ],
        True,
    )

    state_c = col(
        df,
        [
            "State",
            "state",
            "Clean State",
            "clean state",
            "clean_state",
        ],
        True,
    )

    district_c = col(
        df,
        [
            "District",
            "district",
        ],
    )

    subdistrict_c = col(
        df,
        [
            "Subdistrict",
            "subdistrict",
            "Sub-District",
            "sub_district",
        ],
    )

    village_c = col(
        df,
        [
            "Village",
            "village",
        ],
    )

    po_c = col(
        df,
        [
            "Post Office",
            "post_office",
            "PO",
            "po",
        ],
    )

    print(
        "\n      Detected columns:"
    )

    print(
        f"      Address      : {address_c}"
    )

    print(
        f"      City         : {city_c}"
    )

    print(
        f"      PIN          : {pin_c}"
    )

    print(
        f"      State        : {state_c}"
    )

    print(
        f"      District     : {district_c}"
    )

    print(
        f"      Subdistrict  : {subdistrict_c}"
    )

    print(
        f"      Village      : {village_c}"
    )

    print(
        f"      Post Office  : {po_c}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print(
        "\n[4/6] Resolving geography + validating addresses..."
    )

    validation_started = time.perf_counter()

    out = []

    # ========================================================
    # PERFORMANCE CACHE
    # ========================================================
    # Many customer rows share the same address/geography. Cache the
    # expensive geography preparation + resolution by normalized input.
    MAX_CACHE_ENTRIES = 50000
    geo_cache = OrderedDict()
    inference_cache = OrderedDict()
    po_cache = OrderedDict()
    locality_cache = OrderedDict()

    total = len(df)

    # ========================================================
    # PROCESS ROWS
    # ========================================================

    records = df.to_dict(orient="records")

    for i, r in enumerate(records, start=1):

        address = r[address_c]

        city = (
            r[city_c]
            if city_c
            else ""
        )

        pin = r[pin_c]

        state = r[state_c]

        # ----------------------------------------------------
        # Explicit entities
        # ----------------------------------------------------

        explicit = (
            extract_labeled_entities(
                address
            )
        )

        supplied_district = (
            safe_text(
                r[district_c]
            )
            if district_c
            else explicit["district"]
        )

        supplied_subdistrict = (
            safe_text(
                r[subdistrict_c]
            )
            if subdistrict_c
            else explicit["subdistrict"]
        )

        supplied_village = (
            safe_text(
                r[village_c]
            )
            if village_c
            else explicit["village"]
        )

        supplied_po = (
            safe_text(
                r[po_c]
            )
            if po_c
            else explicit["post_office"]
        )

        # Preserve whether hierarchy fields were actually supplied by the
        # customer/address text. Inferred master values are enrichment and
        # must not later be described as customer corrections.
        explicit_flags = {
            "district": bool(supplied_district),
            "subdistrict": bool(supplied_subdistrict),
            "village": bool(supplied_village),
            "post_office": bool(supplied_po),
        }

        # ----------------------------------------------------
        # Normalize blank values
        # ----------------------------------------------------

        if not supplied_district:
            supplied_district = ""

        if not supplied_subdistrict:
            supplied_subdistrict = ""

        if not supplied_village:
            supplied_village = ""

        if not supplied_po:
            supplied_po = ""

        # ----------------------------------------------------
        # Fast geography inference with caching
        # Rural hierarchy is resolved only for rural/unknown addresses.
        # Clearly urban records do not pay the village/subdistrict inference cost.
        # ----------------------------------------------------
        address_type = classify_address_type(address, city)

        inference_key = (
            norm(address),
            norm(city),
            canonical_state(state),
            normalize_pin(pin),
            norm(supplied_district),
            norm(supplied_subdistrict),
            norm(supplied_village),
            norm(supplied_po),
        )

        cached_inference = inference_cache.get(inference_key)

        if cached_inference is not None:
            inference_cache.move_to_end(inference_key)
            supplied_district, supplied_subdistrict, supplied_village, supplied_po = cached_inference
        else:
            if not supplied_district:
                supplied_district = infer_district(
                    address, city, state, pin, resolver
                )

            if address_type != "URBAN" and not supplied_subdistrict:
                supplied_subdistrict = infer_subdistrict(
                    address, city, state, supplied_district, pin, resolver
                )

            if address_type != "URBAN" and not supplied_village:
                supplied_village = infer_village(
                    address, city, state, supplied_district, pin, resolver
                )

            if not supplied_po:
                supplied_po = infer_post_office(
                    address, pin, resolver
                )

            inference_cache[inference_key] = (
                supplied_district,
                supplied_subdistrict,
                supplied_village,
                supplied_po,
            )
            inference_cache.move_to_end(inference_key)
            if len(inference_cache) > MAX_CACHE_ENTRIES:
                inference_cache.popitem(last=False)

        # ====================================================
        # BASIC VALIDATION
        # ====================================================

        address_present = (
            validate_address_present(
                address
            )
        )

        pin_fmt = (
            validate_pin_format(
                pin
            )
        )

        pin_ex = (
            validate_pin_exists(
                pin,
                indexes,
            )
        )

        ps = (
            pin_state_match(
                pin,
                state,
                indexes,
            )
        )

        garbage = (
            detect_garbage(
                address,
                city=city,
                state=state,
                pincode=pin,
            )
        )

        # ====================================================
        # SINGLE-PASS COMPONENT EXTRACTION
        # ====================================================
        # The enhanced detectors are now the canonical completeness layer.
        # Baseline validators are not executed again, eliminating duplicate
        # regex work on every customer row.
        components = extract_components(address, city)

        premise_pass = components["premise"]
        building_pass = components["building"]
        street_pass = components["street"]
        locality_pass = components["locality"]
        # Address-anchor evidence must come from the customer address text.
        # Derived master values must not manufacture completeness credit.
        anchor_pass = bool(enhanced_anchor(address))

        # Deterministic rule confidence for auditability. These values are
        # heuristic evidence-strength indicators, not statistical calibration
        # probabilities and do not alter the score.
        component_confidence = {
            "premise_confidence": 0.90 if premise_pass else 0.0,
            "building_confidence": 0.90 if building_pass else 0.0,
            "street_confidence": 0.90 if street_pass else 0.0,
            "locality_confidence": 0.90 if locality_pass else 0.0,
        }

        # GEOGRAPHIC RESOLUTION
        # ====================================================

        geo_key = (
            norm(address),
            norm(city),
            canonical_state(state),
            norm(supplied_district),
            norm(supplied_subdistrict),
            norm(supplied_village),
            norm(supplied_po),
            normalize_pin(pin),
        )

        geo = geo_cache.get(geo_key)

        if geo is not None:
            geo_cache.move_to_end(geo_key)

        if geo is None:
            geo = resolver.resolve_address(
                address=address,
                city=city,
                state=state,
                district=supplied_district,
                subdistrict=supplied_subdistrict,
                village=supplied_village,
                post_office=supplied_po,
                pin=pin,
            )
            geo_cache[geo_key] = geo
            geo_cache.move_to_end(geo_key)
            if len(geo_cache) > MAX_CACHE_ENTRIES:
                geo_cache.popitem(last=False)

        district_res = geo[
            "district"
        ]

        subdistrict_res = geo[
            "subdistrict"
        ]

        village_res = geo[
            "village"
        ]

        po_res = geo[
            "post_office"
        ]

        ulb_res = geo[
            "ulb"
        ]

        # P1 locality evidence: only explicitly labeled locality/area values
        # are sent to LGD fuzzy resolution. Private colony/street names are
        # not expected to exist in LGD and remain NOT_CHECKED.
        locality_candidate = extract_locality_candidate(address)
        locality_key = (
            norm(locality_candidate),
            canonical_state(state),
            norm(district_res.match_value),
            norm(subdistrict_res.match_value),
            normalize_pin(pin),
        )
        locality_res = locality_cache.get(locality_key)
        if locality_res is not None:
            locality_cache.move_to_end(locality_key)
        else:
            locality_res = resolver.resolve_locality(
                locality_candidate,
                state=state,
                district=district_res.match_value,
                subdistrict=subdistrict_res.match_value,
                pin=pin,
            )
            locality_cache[locality_key] = locality_res
            locality_cache.move_to_end(locality_key)
            if len(locality_cache) > MAX_CACHE_ENTRIES:
                locality_cache.popitem(last=False)

        # ====================================================
        # GEO COMPONENTS
        # ====================================================

        normalized_pin = normalize_pin(
            pin
        )

        # ----------------------------------------------------
        # PIN ↔ DISTRICT
        #
        # FIX:
        # Directly compare supplied/resolved district with
        # India Post PIN district set.
        # ----------------------------------------------------

        effective_district = (
            district_res.match_value
            if district_res.status
            in {
                PASS,
                AMBIGUOUS,
            }
            else supplied_district
        )

        pin_district_status, pin_district_fraction, district_matches = (
            check_pin_district_evidence(
                address,
                normalized_pin,
                indexes,
            )
        )

        # District-State mismatch is deliberately not evaluated because
        # District is not a customer-supplied field.
        district_state_status = "NOT_CHECKED"
        district_state_fraction = 0.5

        india_post_district = india_post_districts_for_pin(
            normalized_pin,
            indexes,
        )

        # ----------------------------------------------------
        # Village hierarchy
        # ----------------------------------------------------

        village_hierarchy_fraction = (
            village_hierarchy_score(
                village_res,
                bool(supplied_village),
            )
        )

        # ----------------------------------------------------
        # PO / City
        # ----------------------------------------------------

        po_city_fraction = (
            po_city_score(
                po_res,
                ulb_res,
                city,
                bool(supplied_po),
            )
        )

        village_hierarchy_status = get_village_hierarchy_status(
            village_res,
            bool(supplied_village),
        )

        po_city_status_value = po_city_status(
            po_res,
            ulb_res,
            city,
            bool(supplied_po),
        )

        # ====================================================
        # GEO SCORE
        # ====================================================

        # V5 geographic model:
        #
        # PIN-State remains a direct customer-field validation.
        # PIN-District is now only positive address evidence because
        # District is not present in the customer input.
        # District-State is removed from customer quality scoring.
        #
        # The remaining geographic weight is retained through
        # administrative/entity evidence rather than invented mismatch
        # penalties.
        district_evidence_fraction = pin_district_fraction

        # Administrative entity evidence:
        # a resolved district from City/address is useful corroboration,
        # but absence is neutral.
        if district_res.status == PASS:
            admin_entity_fraction = 1.0
        elif district_res.status == AMBIGUOUS:
            admin_entity_fraction = 0.5
        else:
            admin_entity_fraction = 0.5

        geo_fraction = (
            ps.score_fraction * 7
            + district_evidence_fraction * 3
            + admin_entity_fraction * 2
            + village_hierarchy_fraction * 5
            + po_city_fraction * 3
            + anchor_pass * 5
        ) / 25

        # ====================================================
        # PREMISE COMPLETENESS
        # ====================================================

        # Rural address adjustment:
        #
        # If village/PO/district geography is established,
        # lack of HNO should not make the address invalid.
        #
        # We give partial premise credit for a geographically
        # identifiable rural address.

        if premise_pass:

            premise_identifier_fraction = 1.0

        elif (
            village_res.status == PASS
            or supplied_village
        ):

            premise_identifier_fraction = 0.5

        else:

            premise_identifier_fraction = 0.0

        building_fraction = (
            1.0
            if building_pass
            else 0.0
        )

        # Premise specificity.
        if (
            premise_pass
            or building_pass
            or village_res.status == PASS
            or supplied_village
        ):

            specificity_fraction = 1.0

        else:

            specificity_fraction = 0.0

        premise_fraction = (

            premise_identifier_fraction * 10

            + building_fraction * 5

            + specificity_fraction * 5

        ) / 20

        # ====================================================
        # STREET / LOCALITY
        # ====================================================

        street_fraction = (

            (
                1.0
                if street_pass
                else 0.0
            ) * 8

            +

            (
                1.0
                if locality_pass
                else 0.0
            ) * 7

            +

            (
                1.0
                if anchor_pass
                else 0.0
            ) * 5

        ) / 20

        # ====================================================
        # STANDARDIZATION
        # ====================================================

        if garbage.status == FAIL:

            standard_fraction = 0.0

        elif garbage.status == AMBIGUOUS:

            standard_fraction = 0.5

        else:

            standard_fraction = (
                0.9
                if pin_fmt.auto_correctable
                else 1.0
            )

        # ====================================================
        # GEO CONFIDENCE
        # ====================================================

        # This is ADMINISTRATIVE geography confidence.
        #
        # It is NOT rooftop confidence.
        #
        # India Post coordinates represent postal offices,
        # not customer properties.

        # Administrative geography confidence.
        #
        # This does NOT mean rooftop/property-level geocoding.
        # Full confidence is possible when the supplied PIN, district,
        # state and at least one local hierarchy/entity are consistent.

        geo_confidence_fraction = 0.0

        local_geo_evidence = (
            village_res.status == PASS
            or subdistrict_res.status == PASS
            or po_res.status == PASS
            or ulb_res.status == PASS
        )

        # Administrative confidence is based on independent customer/PIN
        # evidence. District-State is intentionally NOT used because District
        # is not a customer-supplied field in this dataset.
        if ps.status == PASS and local_geo_evidence:
            geo_confidence_fraction = 1.0
        elif ps.status == PASS and pin_district_status == PASS:
            geo_confidence_fraction = 0.85
        elif ps.status == PASS:
            geo_confidence_fraction = 0.75
        elif ps.status == AMBIGUOUS:
            geo_confidence_fraction = 0.5

        # ====================================================
        # FINAL SCORE
        # ====================================================

        scores = calculate_score({

            "pin":
                pin_ex.score_fraction,

            "geo":
                geo_fraction,

            "premise":
                premise_fraction,

            "street_locality":
                street_fraction,

            "standardization":
                standard_fraction,

            "geo_confidence":
                geo_confidence_fraction,
        })

        # Keep the raw additive score for auditability.  Hard-invalid
        # inputs are not allowed to remain in a score band such as
        # USABLE/GOOD merely because other fields happened to score well.
        raw_quality_score = scores["quality_score"]

        # ====================================================
        # HARD INVALID vs MANUAL REVIEW
        # ====================================================
        #
        # Hard invalid is reserved for truly unusable input:
        #   - no meaningful address text
        #   - invalid/nonexistent PIN
        #   - explicit garbage/metadata-only address
        #
        # Geographic contradictions remain a separate review flag.

        hard_invalid = (
            address_present.status == FAIL
            or pin_ex.status == FAIL
            or garbage.status == FAIL
        )

        # Preserve the raw score but cap the publishable quality score
        # below the INVALID threshold so quality_class and quality_score
        # remain mathematically consistent.
        if hard_invalid:
            scores["quality_score"] = min(
                raw_quality_score,
                39.99,
            )

        quality_class = classify_score(
            scores["quality_score"]
        )

        # Operational routing is deliberately separate from quality_class.
        # Hard invalids always reject; otherwise score bands route the work.
        if hard_invalid or scores["quality_score"] < 60:
            operational_queue = "fatal_rejects"
        elif scores["quality_score"] >= 85:
            operational_queue = "auto_dispatch"
        else:
            operational_queue = "manual_review"

        # ====================================================
        # QC REVIEW REASONS
        # ====================================================
        # These are review triggers, not score/class overrides. Only
        # customer-supplied contradictions are elevated to QC review.
        review_reasons = []

        if ps.status == FAIL:
            review_reasons.append("PIN_STATE_MISMATCH")
        elif ps.status == AMBIGUOUS:
            review_reasons.append("PIN_STATE_AMBIGUOUS")

        if explicit_flags["village"]:
            if village_res.status == FAIL:
                review_reasons.append("VILLAGE_UNVALIDATED")
            elif village_res.status == AMBIGUOUS:
                review_reasons.append("VILLAGE_AMBIGUOUS")

        if explicit_flags["subdistrict"]:
            if subdistrict_res.status == FAIL:
                review_reasons.append("SUBDISTRICT_UNVALIDATED")
            elif subdistrict_res.status == AMBIGUOUS:
                review_reasons.append("SUBDISTRICT_AMBIGUOUS")

        # If an explicit PO was supplied, a failed/ambiguous PO resolution
        # is a locality verification issue. Inferred PO failures are neutral.
        if explicit_flags["post_office"]:
            if po_res.status == FAIL:
                review_reasons.append("POST_OFFICE_UNVALIDATED")
            elif po_res.status == AMBIGUOUS:
                review_reasons.append("POST_OFFICE_AMBIGUOUS")


        if locality_candidate:
            if locality_res.status == FAIL:
                review_reasons.append("LOCALITY_UNVALIDATED")
            elif locality_res.status == AMBIGUOUS:
                review_reasons.append("LOCALITY_AMBIGUOUS")

        review_reason = "; ".join(review_reasons)
        review_flag = bool(review_reasons)

        if review_flag and operational_queue == "auto_dispatch":
            operational_queue = "manual_review"

        # ====================================================
        # CORRECTION SUGGESTION
        # ====================================================

        normalized_state = (
            canonical_state(
                state
            )
        )

        corrections = []

        # Corrections are reserved for actual defects or verification needs.
        # Optional missing building/street/locality detail is NOT a correction.
        if garbage.status == FAIL:
            corrections.append(
                "Recapture a complete premise and locality/road address"
            )
        elif pin_ex.status == FAIL:
            corrections.append(
                "Verify and correct the 6-digit PIN"
            )
        elif ps.status == FAIL:
            corrections.append(
                "Verify State against the supplied PIN"
            )

        if "VILLAGE_UNVALIDATED" in review_reasons:
            corrections.append("Verify Village against LGD hierarchy")
        elif "VILLAGE_AMBIGUOUS" in review_reasons:
            corrections.append("Confirm the correct Village name")

        if "SUBDISTRICT_UNVALIDATED" in review_reasons:
            corrections.append("Verify Subdistrict against LGD hierarchy")
        elif "SUBDISTRICT_AMBIGUOUS" in review_reasons:
            corrections.append("Confirm the correct Subdistrict")

        if "POST_OFFICE_UNVALIDATED" in review_reasons:
            corrections.append("Verify Post Office against the PIN")
        elif "POST_OFFICE_AMBIGUOUS" in review_reasons:
            corrections.append("Confirm the correct Post Office")

        # Keep the field human-readable and bounded.
        correction = "; ".join(corrections[:4])

        # ====================================================
        # REMARKS
        # ====================================================

        primary, detailed, action = (
            build_remarks(

                quality_class,

                pin_exists=
                    pin_ex.status,

                pin_state_match=
                    ps.status,

                premise_present=
                    premise_pass,

                building_present=
                    building_pass,

                street_present=
                    street_pass,

                locality_present=
                    locality_pass,

                anchor_present=
                    anchor_pass,

                garbage_detected=
                    garbage.status != PASS,

                critical_issue=
                    hard_invalid,

                review_flag=
                    review_flag,

                review_reason=
                    review_reason,

                correction_suggestion=
                    correction,
            )
        )

        # No correction text for a clean ACCEPT outcome. Optional enrichment
        # may still retain actionable suggestions.
        if action in {"ACCEPT", "ACCEPT – OPTIONAL ENRICHMENT"}:
            correction = ""

        # ====================================================
        # OUTPUT ROW
        # ====================================================

        x = dict(r)

        x.update({

            "address_type":
                address_type,

            "district_explicit_in_input": explicit_flags["district"],
            "subdistrict_explicit_in_input": explicit_flags["subdistrict"],
            "village_explicit_in_input": explicit_flags["village"],
            "post_office_explicit_in_input": explicit_flags["post_office"],

            "address_metadata_json":
                build_address_metadata_json(
                    address=address,
                    city=city,
                    state=state,
                    pin=normalized_pin,
                    address_type=address_type,
                    extracted={
                        "district": supplied_district,
                        "subdistrict": supplied_subdistrict,
                        "village": supplied_village,
                        "post_office": supplied_po,
                    },
                    resolved={
                        "district": district_res.match_value,
                        "subdistrict": subdistrict_res.match_value,
                        "village": village_res.match_value,
                        "post_office": po_res.match_value,
                        "ulb": ulb_res.match_value,
                        "locality": locality_res.match_value,
                    },
                    statuses={
                        "pin_state": ps.status,
                        "village": village_res.status,
                        "subdistrict": subdistrict_res.status,
                        "post_office": po_res.status,
                        "pin_district_evidence": pin_district_status,
                        "locality": locality_res.status,
                    },
                    component_confidence=component_confidence,
                    routing={
                        "quality_class": quality_class,
                        "quality_score": scores["quality_score"],
                        "qc_review_flag": review_flag,
                        "qc_review_reason": review_reason,
                        "operational_queue": operational_queue,
                    },
                ),

            "operational_queue":
                operational_queue,

            "normalized_pincode":
                normalized_pin,

            "normalized_state":
                normalized_state,

            # ------------------------------------------------
            # Derived India Post geography
            # ------------------------------------------------
            "india_post_district":
                india_post_district,

            "district_found_in_address":
                "; ".join(district_matches),

            "district_in_address_status":
                pin_district_status,

            # ------------------------------------------------
            # Extracted entities
            # ------------------------------------------------

            "extracted_district":
                supplied_district,

            "extracted_subdistrict":
                supplied_subdistrict,

            "extracted_village":
                supplied_village,

            "extracted_post_office":
                supplied_po,

            # ------------------------------------------------
            # Basic validation
            # ------------------------------------------------

            "address_present_status":
                address_present.status,

            "pin_format_status":
                pin_fmt.status,

            "pin_exists_status":
                pin_ex.status,

            "pin_state_match_status":
                ps.status,

            "garbage_status":
                garbage.status,

            # ------------------------------------------------
            # Enhanced completeness
            # ------------------------------------------------

            "premise_status":
                (
                    PASS
                    if premise_pass
                    else FAIL
                ),

            "building_status":
                (
                    PASS
                    if building_pass
                    else FAIL
                ),

            "street_status":
                (
                    PASS
                    if street_pass
                    else FAIL
                ),

            "locality_status":
                (
                    PASS
                    if locality_pass
                    else FAIL
                ),

            "geographic_anchor_status":
                (
                    PASS
                    if anchor_pass
                    else FAIL
                ),

            # ------------------------------------------------
            # Resolved district
            # ------------------------------------------------

            "resolved_district":
                district_res.match_value,

            "district_resolution_status":
                district_res.status,

            "district_resolution_score":
                round(
                    district_res.match_score,
                    4,
                ),

            "district_resolution_source":
                district_res.source,

            # ------------------------------------------------
            # Resolved subdistrict
            # ------------------------------------------------

            "resolved_subdistrict":
                subdistrict_res.match_value,

            "subdistrict_resolution_status":
                subdistrict_res.status,

            "subdistrict_resolution_score":
                round(
                    subdistrict_res.match_score,
                    4,
                ),

            "subdistrict_resolution_source":
                subdistrict_res.source,

            # ------------------------------------------------
            # Resolved village
            # ------------------------------------------------

            "resolved_village":
                village_res.match_value,

            "village_resolution_status":
                village_res.status,

            "village_resolution_score":
                round(
                    village_res.match_score,
                    4,
                ),

            "village_resolution_source":
                village_res.source,

            "resolved_village_district":
                village_res.district,

            "resolved_village_subdistrict":
                village_res.subdistrict,

            # ------------------------------------------------
            # Resolved PO
            # ------------------------------------------------

            "resolved_post_office":
                po_res.match_value,

            "post_office_resolution_status":
                po_res.status,

            "post_office_resolution_score":
                round(
                    po_res.match_score,
                    4,
                ),

            "post_office_resolution_source":
                po_res.source,

            # ------------------------------------------------
            # Resolved ULB
            # ------------------------------------------------

            "resolved_ulb":
                ulb_res.match_value,

            "ulb_resolution_status":
                ulb_res.status,

            "ulb_resolution_score":
                round(
                    ulb_res.match_score,
                    4,
                ),

            "ulb_resolution_source":
                ulb_res.source,

            "locality_candidate":
                locality_candidate,

            "locality_resolution_status":
                locality_res.status,

            "locality_resolution_score":
                round(locality_res.match_score, 4),

            "locality_resolution_source":
                locality_res.source,

            "resolved_locality":
                locality_res.match_value,

            # ------------------------------------------------
            # Geo checks
            # ------------------------------------------------

            "geo_district_state_status":
                district_state_status,

            "geo_pin_district_status":
                pin_district_status,

            "geo_village_hierarchy_status":
                village_hierarchy_status,

            "geo_village_hierarchy_score":
                round(village_hierarchy_fraction, 4),

            "geo_po_city_status":
                po_city_status_value,

            "geo_po_city_score":
                round(po_city_fraction, 4),

            "geo_anchor_status":
                PASS if anchor_pass else FAIL,

            # ------------------------------------------------
            # Component evidence confidence
            # ------------------------------------------------

            **component_confidence,

            "geo_confidence_basis":
                "ADMINISTRATIVE_GEOGRAPHY_ONLY",

            # ------------------------------------------------
            # Scores
            # ------------------------------------------------

            **scores,

            # Audit fields: preserve the additive score even when a genuine
            # critical input causes the publishable score to be capped.
            "raw_quality_score": raw_quality_score,
            "hard_invalid": hard_invalid,

            # ------------------------------------------------
            # QC
            # ------------------------------------------------

            "quality_class":
                quality_class,

            "qc_review_flag":
                review_flag,

            "qc_review_reason":
                review_reason,

            "primary_qc_remark":
                primary,

            "detailed_qc_remark":
                detailed,

            "qc_action":
                action,

            "correction_suggestion":
                correction,
        })

        out.append(x)

        # ====================================================
        # PROGRESS
        # ====================================================

        if (
            i % 100 == 0
            or i == total
        ):

            elapsed = (
                time.perf_counter()
                - validation_started
            )

            rate = (
                i / elapsed
                if elapsed
                else 0
            )

            eta = (
                (total - i) / rate
                if rate
                else 0
            )

            print(
                f"      Progress: "
                f"{i:,}/{total:,} "
                f"({i / total:.1%}) "
                f"| {rate:,.0f} rows/sec "
                f"| ETA: {eta:.1f} sec"
            )

    # ========================================================
    # PERFORMANCE SUMMARY
    # ========================================================

    print(
        f"      Inference cache keys: {len(inference_cache):,}"
    )
    print(
        f"      Geography cache keys: {len(geo_cache):,}"
    )
    print(f"      Locality cache keys: {len(locality_cache):,}")
    print(    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print(
        "\n[5/6] Writing output..."
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = pd.DataFrame(
        out
    )

    # ========================================================
    # OUTPUT INVARIANTS
    # ========================================================
    # A catastrophic normalization failure must never be published as a
    # legitimate quality result.
    if "normalized_pincode" not in result.columns:
        raise RuntimeError(
            "OUTPUT INVARIANT FAILED: normalized_pincode column is missing."
        )

    normalized_pin_valid = result["normalized_pincode"].map(normalize_pin)
    normalized_pin_valid_count = int(normalized_pin_valid.ne("").sum())

    if normalized_pin_valid_count == 0 and len(result) > 0:
        raise RuntimeError(
            "OUTPUT INVARIANT FAILED: normalized_pincode is invalid/empty "
            "for every output row. Output was NOT written. "
            "This indicates a broken PIN pipeline."
        )

    output_invalid_pin_rate = (
        1.0 - normalized_pin_valid_count / len(result)
        if len(result)
        else 1.0
    )

    if output_invalid_pin_rate >= 0.50:
        raise RuntimeError(
            f"OUTPUT INVARIANT FAILED: {output_invalid_pin_rate:.1%} of "
            "normalized_pincode values are invalid/empty. Output was NOT "
            "written because this indicates a pipeline-level failure."
        )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n[6/6] Summary"
    )

    print(
        f"      Records processed: "
        f"{len(result):,}"
    )

    print(
        f"      Total runtime    : "
        f"{time.perf_counter() - started:.2f} sec"
    )

    print(
        f"      Output           : "
        f"{OUTPUT_FILE}"
    )

    print(
        "\n      Quality class distribution:"
    )

    print(
        result[
            "quality_class"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print(
        "\n      Resolution status distribution:"
    )

    for column in [

        "district_resolution_status",

        "village_resolution_status",

        "post_office_resolution_status",

        "ulb_resolution_status",

    ]:

        print(
            f"\n      {column}:"
        )

        print(
            result[column]
            .value_counts(
                dropna=False
            )
            .to_string()
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()