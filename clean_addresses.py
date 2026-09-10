import re
from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "NON MFI - Query result.csv"

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

    # Remove known full state aliases only at the end.
    for alias in sorted(STATE_ALIASES, key=len, reverse=True):
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
# FLAGS / REPORTING
# ============================================================

def has_redundancy(original, cleaned):
    """True only when address content was actually removed, not just reformatted."""
    return count_removed_components(original, cleaned) > 0


def count_removed_components(original, cleaned):
    """
    Approximate count of removed address components using normalized
    token multiset comparison. It is a reporting metric, not a validator.
    """
    original_tokens = normalize_for_matching(original).split()
    cleaned_tokens = normalize_for_matching(cleaned).split()

    from collections import Counter

    before = Counter(original_tokens)
    after = Counter(cleaned_tokens)

    removed = 0
    for token, count in before.items():
        removed += max(0, count - after.get(token, 0))

    return removed


def detect_garbage(text):
    """
    Conservative garbage/metadata-only detection.

    Returns True only for obvious non-address content.
    """
    if not is_meaningful(text):
        return False

    norm = normalize_for_matching(text)

    # Obvious placeholders / repeated single geographic metadata.
    if norm in PLACEHOLDER_VALUES:
        return True

    words = norm.split()

    # Examples such as "MEERUT MEERUT".
    if len(words) == 2 and words[0] == words[1]:
        return True

    # Purely geographic metadata is not a useful premise address.
    if len(words) <= 3:
        geo_words = {
            "INDIA",
            "UP",
            "MP",
            "RJ",
            "HR",
            "PB",
            "DELHI",
        }
        if all(w in geo_words for w in words):
            return True

    return False


# ============================================================
# INPUT / OUTPUT
# ============================================================

def find_input_file():
    if INPUT_FILE.exists():
        return INPUT_FILE

    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

def validate_columns(df):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )


def process_file():
    input_file = find_input_file()

    df = pd.read_csv(
        input_file,
        dtype=str,
        keep_default_na=False,
    )

    validate_columns(df)

    original_rows = len(df)

    # --------------------------------------------------------
    # Repair structured fields first.
    # --------------------------------------------------------
    repaired = df.apply(
        lambda row: repair_structured_fields(
            row["City"],
            row["State"],
            row["Pincode"],
        ),
        axis=1,
        result_type="expand",
    )

    repaired.columns = [
        "_Clean City",
        "_Clean State",
        "_Clean Pincode",
    ]

    df["_Clean City"] = repaired["_Clean City"]
    df["_Clean State"] = repaired["_Clean State"]
    df["_Clean Pincode"] = repaired["_Clean Pincode"]

    # --------------------------------------------------------
    # Clean address AFTER structured fields are repaired.
    # --------------------------------------------------------
    df["_Clean Full Address"] = df.apply(
        lambda row: clean_address(
            address=row["Full Address"],
            city=row["_Clean City"],
            state=row["_Clean State"],
            pincode=row["_Clean Pincode"],
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # Build required output columns.
    # --------------------------------------------------------
    output = pd.DataFrame({
        "Lead Code": df["Lead Code"],
        "Clean Full Address": df["_Clean Full Address"],
        "Clean City": df["_Clean City"],
        "Clean Pincode": df["_Clean Pincode"],
        "Clean State": df["_Clean State"],
    })

    output["Address Empty"] = ~output["Clean Full Address"].map(is_meaningful)
    output["City Empty"] = ~output["Clean City"].map(is_meaningful)
    output["State Empty"] = ~output["Clean State"].map(is_meaningful)
    output["Pincode Empty"] = output["Clean Pincode"].eq("")
    output["Pincode Length Valid"] = output["Clean Pincode"].str.fullmatch(
        r"\d{6}",
        na=False,
    )
    output["Pincode Starts With Zero"] = output["Clean Pincode"].str.startswith(
        "0",
        na=False,
    )

    output["Address Duplicate Components Removed"] = df.apply(
        lambda row: has_redundancy(
            row["Full Address"],
            row["_Clean Full Address"],
        ),
        axis=1,
    )

    output["Address Redundancy Cleaned"] = output[
        "Address Duplicate Components Removed"
    ]

    output["Original Full Address"] = df["Full Address"]
    output["Original City"] = df["City"]
    output["Original Pincode"] = df["Pincode"]
    output["Original State"] = df["State"]

    # Keep exact requested 17-column order.
    output = output[
        [
            "Lead Code",
            "Clean Full Address",
            "Clean City",
            "Clean Pincode",
            "Clean State",
            "Address Empty",
            "City Empty",
            "State Empty",
            "Pincode Empty",
            "Pincode Length Valid",
            "Pincode Starts With Zero",
            "Address Duplicate Components Removed",
            "Address Redundancy Cleaned",
            "Original Full Address",
            "Original City",
            "Original Pincode",
            "Original State",
        ]
    ]

    # --------------------------------------------------------
    # IMPORTANT: NEVER deduplicate rows or Lead Codes.
    # Duplicate addresses can legitimately belong to different leads.
    # --------------------------------------------------------
    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # Report.
    # --------------------------------------------------------
    total_removed_components = sum(
        count_removed_components(o, c)
        for o, c in zip(
            output["Original Full Address"],
            output["Clean Full Address"],
        )
    )

    rows_with_structured_changes = (
        (
            output["Clean City"].map(normalize_for_matching)
            != output["Original City"].map(normalize_for_matching)
        )
        | (
            output["Clean State"].map(canonical_state)
            != output["Original State"].map(canonical_state)
        )
        | (
            output["Clean Pincode"].map(normalize_pin)
            != output["Original Pincode"].map(normalize_pin)
        )
    ).sum()

    report = [
        "ADDRESS CLEANING REPORT",
        "=" * 60,
        f"Input file: {input_file.name}",
        f"Output file: {OUTPUT_FILE.name}",
        f"Input rows: {original_rows}",
        f"Output rows: {len(output)}",
        f"Rows removed: {original_rows - len(output)}",
        "Duplicate records removed: 0",
        "Duplicate Lead Codes / rows removed: 0",
        f"Rows with structured-field changes: {rows_with_structured_changes}",
        f"Empty Full Address: {int(output['Address Empty'].sum())}",
        f"Empty City: {int(output['City Empty'].sum())}",
        f"Empty State: {int(output['State Empty'].sum())}",
        f"Empty Pincode: {int(output['Pincode Empty'].sum())}",
        f"Invalid Pincode Length: {int((~output['Pincode Length Valid']).sum())}",
        f"Pincodes Starting With Zero: {int(output['Pincode Starts With Zero'].sum())}",
        (
            "Address records with redundant components cleaned: "
            f"{int(output['Address Redundancy Cleaned'].sum())}"
        ),
        f"Total address components removed: {total_removed_components}",
    ]

    REPORT_FILE.write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print("\n".join(report))
    print("\nCleaning completed successfully.")


if __name__ == "__main__":
    process_file()
