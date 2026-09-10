from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Iterable

import pandas as pd


# ============================================================
# MASTER LOADER
# ============================================================
#
# Loads:
#   1. India Post master
#   2. LGD District master
#   3. LGD Subdistrict master
#   4. LGD Village master
#   5. LGD ULB master
#
# Then prepares fast lookup indexes for:
#   - PIN
#   - State
#   - District
#   - Post Office
#   - Subdistrict
#   - Village
#   - ULB
#
# IMPORTANT:
# This loader is intentionally tolerant of different
# column naming conventions in LGD exports.
#
# Example:
#   "District Name (In English)"
#   "district_name_in_english"
#   "District Name"
#
# can all resolve to the same internal field.
# ============================================================


# ============================================================
# GENERAL NORMALIZATION
# ============================================================

def clean_string(value) -> str:
    """
    Basic string cleanup.

    Does NOT perform geographic aliasing such as:
        Gurgaon -> Gurugram

    That belongs in entity_resolution.py.
    """
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip().upper()

    text = text.replace("&", " AND ")

    # Replace punctuation with spaces.
    text = re.sub(r"[^A-Z0-9\s]", " ", text)

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_pin(value) -> str:
    """
    Normalize PIN to a six-digit string where possible.
    """
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()

    # Handle Excel-style numeric PIN:
    # 122001.0 -> 122001
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".")[0]

    digits = re.sub(r"\D", "", text)

    if len(digits) == 6:
        return digits

    return digits


# ============================================================
# COLUMN NORMALIZATION
# ============================================================

def normalize_column_name(value) -> str:
    """
    Convert arbitrary column names into a stable internal form.

    Example:
        District Name (In English)
        ->
        district_name_in_english
    """
    text = str(value).strip().lower()

    text = text.replace("&", " and ")

    text = re.sub(r"[^a-z0-9]+", "_", text)

    text = re.sub(r"_+", "_", text)

    return text.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize dataframe column names.

    We retain the dataframe structure but make column access
    predictable.
    """

    rename_map = {}

    for column in df.columns:
        normalized = normalize_column_name(column)
        rename_map[column] = normalized

    df = df.rename(columns=rename_map)

    return df


# ============================================================
# FLEXIBLE COLUMN FINDER
# ============================================================

def find_column(
    df: pd.DataFrame,
    candidates: Iterable[str],
    required: bool = False,
    dataframe_name: str = "",
):
    """
    Find a dataframe column using multiple possible names.

    Matching happens after column-name normalization.
    """

    normalized_lookup = {
        normalize_column_name(column): column
        for column in df.columns
    }

    for candidate in candidates:

        candidate_normalized = normalize_column_name(candidate)

        if candidate_normalized in normalized_lookup:
            return normalized_lookup[candidate_normalized]

    if required:

        available = ", ".join(str(c) for c in df.columns)

        raise ValueError(
            f"\nRequired column not found in {dataframe_name or 'dataframe'}.\n"
            f"Expected one of:\n"
            f"  {list(candidates)}\n\n"
            f"Available columns:\n"
            f"  {available}\n"
        )

    return None


# ============================================================
# STANDARDIZE DISTRICT MASTER
# ============================================================

def standardize_district_master(df: pd.DataFrame) -> pd.DataFrame:

    df = normalize_columns(df)

    district_code_col = find_column(
        df,
        [
            "District Code",
            "district_code",
            "districtcode",
        ],
        required=True,
        dataframe_name="District master",
    )

    district_name_col = find_column(
        df,
        [
            "District Name (In English)",
            "district_name_in_english",
            "District Name",
            "district_name",
            "district",
        ],
        required=True,
        dataframe_name="District master",
    )

    # Rename to stable internal names.
    df = df.rename(
        columns={
            district_code_col: "district_code",
            district_name_col: "district_name",
        }
    )

    df["district_code"] = (
        df["district_code"]
        .astype(str)
        .str.strip()
    )

    df["district_name"] = (
        df["district_name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["district_name_clean"] = df["district_name"].map(
        clean_string
    )

    return df


# ============================================================
# STANDARDIZE SUBDISTRICT MASTER
# ============================================================

def standardize_subdistrict_master(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = normalize_columns(df)

    district_code_col = find_column(
        df,
        [
            "District code",
            "District Code",
            "district_code",
            "districtcode",
        ],
        required=True,
        dataframe_name="Subdistrict master",
    )

    district_name_col = find_column(
        df,
        [
            "District Name",
            "District Name (In English)",
            "district_name",
            "district_name_in_english",
        ],
        required=True,
        dataframe_name="Subdistrict master",
    )

    subdistrict_code_col = find_column(
        df,
        [
            "Subdistrict Code",
            "Sub-District Code",
            "subdistrict_code",
            "sub_district_code",
        ],
        required=True,
        dataframe_name="Subdistrict master",
    )

    subdistrict_name_col = find_column(
        df,
        [
            "Subdistrict Name (In English)",
            "Sub-District Name (In English)",
            "Subdistrict Name",
            "Sub-District Name",
            "subdistrict_name",
            "sub_district_name",
        ],
        required=True,
        dataframe_name="Subdistrict master",
    )

    df = df.rename(
        columns={
            district_code_col: "district_code",
            district_name_col: "district_name",
            subdistrict_code_col: "subdistrict_code",
            subdistrict_name_col: "subdistrict_name",
        }
    )

    df["district_code"] = (
        df["district_code"]
        .astype(str)
        .str.strip()
    )

    df["district_name"] = (
        df["district_name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["subdistrict_code"] = (
        df["subdistrict_code"]
        .astype(str)
        .str.strip()
    )

    df["subdistrict_name"] = (
        df["subdistrict_name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["district_name_clean"] = df["district_name"].map(
        clean_string
    )

    df["subdistrict_name_clean"] = df[
        "subdistrict_name"
    ].map(clean_string)

    return df


# ============================================================
# STANDARDIZE VILLAGE MASTER
# ============================================================

def standardize_village_master(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = normalize_columns(df)

    district_code_col = find_column(
        df,
        [
            "District Code",
            "district_code",
            "districtcode",
        ],
        required=True,
        dataframe_name="Village master",
    )

    district_name_col = find_column(
        df,
        [
            "District Name",
            "District Name (In English)",
            "district_name",
            "district_name_in_english",
        ],
        required=True,
        dataframe_name="Village master",
    )

    subdistrict_code_col = find_column(
        df,
        [
            "Sub-District Code",
            "Subdistrict Code",
            "sub_district_code",
            "subdistrict_code",
        ],
        required=True,
        dataframe_name="Village master",
    )

    subdistrict_name_col = find_column(
        df,
        [
            "Sub-District Name",
            "Subdistrict Name",
            "Sub-District Name (In English)",
            "Subdistrict Name (In English)",
            "sub_district_name",
            "subdistrict_name",
        ],
        required=True,
        dataframe_name="Village master",
    )

    village_code_col = find_column(
        df,
        [
            "Village Code",
            "village_code",
            "villagecode",
        ],
        required=True,
        dataframe_name="Village master",
    )

    village_name_col = find_column(
        df,
        [
            "Village Name (In English)",
            "Village Name",
            "village_name_in_english",
            "village_name",
        ],
        required=True,
        dataframe_name="Village master",
    )

    df = df.rename(
        columns={
            district_code_col: "district_code",
            district_name_col: "district_name",
            subdistrict_code_col: "subdistrict_code",
            subdistrict_name_col: "subdistrict_name",
            village_code_col: "village_code",
            village_name_col: "village_name",
        }
    )

    df["district_code"] = (
        df["district_code"]
        .astype(str)
        .str.strip()
    )

    df["district_name"] = (
        df["district_name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["subdistrict_code"] = (
        df["subdistrict_code"]
        .astype(str)
        .str.strip()
    )

    df["subdistrict_name"] = (
        df["subdistrict_name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["village_code"] = (
        df["village_code"]
        .astype(str)
        .str.strip()
    )

    df["village_name"] = (
        df["village_name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Compatibility aliases expected by entity_resolution.py
    df["village_name_in_english"] = df["village_name"]

    df["district_name_clean"] = df[
        "district_name"
    ].map(clean_string)

    df["subdistrict_name_clean"] = df[
        "subdistrict_name"
    ].map(clean_string)

    df["village_name_clean"] = df[
        "village_name"
    ].map(clean_string)

    return df


# ============================================================
# STANDARDIZE ULB MASTER
# ============================================================

def standardize_ulb_master(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = normalize_columns(df)

    ulb_code_col = find_column(
        df,
        [
            "Local Body Code",
            "Localbody Code",
            "local_body_code",
            "localbody_code",
        ],
        required=True,
        dataframe_name="ULB master",
    )

    ulb_name_col = find_column(
        df,
        [
            "Local Body Name (In English)",
            "Localbody Name (In English)",
            "Local Body Name",
            "Localbody Name",
            "local_body_name",
            "localbody_name",
        ],
        required=True,
        dataframe_name="ULB master",
    )

    ulb_type_col = find_column(
        df,
        [
            "Localbody Type Name",
            "Local Body Type Name",
            "localbody_type_name",
            "local_body_type_name",
        ],
        required=False,
        dataframe_name="ULB master",
    )

    df = df.rename(
        columns={
            ulb_code_col: "ulb_code",
            ulb_name_col: "ulb_name",
        }
    )

    if ulb_type_col:
        df = df.rename(
            columns={
                ulb_type_col: "ulb_type"
            }
        )
    else:
        df["ulb_type"] = ""

    df["ulb_code"] = (
        df["ulb_code"]
        .astype(str)
        .str.strip()
    )

    df["ulb_name"] = (
        df["ulb_name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["ulb_type"] = (
        df["ulb_type"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["ulb_name_clean"] = df[
        "ulb_name"
    ].map(clean_string)

    return df


# ============================================================
# STANDARDIZE INDIA POST MASTER
# ============================================================

def standardize_india_post_master(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = normalize_columns(df)

    pin_col = find_column(
        df,
        [
            "pincode",
            "pin",
            "pin_code",
        ],
        required=True,
        dataframe_name="India Post master",
    )

    state_col = find_column(
        df,
        [
            "statename",
            "state_name",
            "state",
        ],
        required=True,
        dataframe_name="India Post master",
    )

    district_col = find_column(
        df,
        [
            "district",
            "districtname",
            "district_name",
        ],
        required=True,
        dataframe_name="India Post master",
    )

    office_col = find_column(
        df,
        [
            "officename",
            "office_name",
            "post_office",
            "postoffice",
        ],
        required=True,
        dataframe_name="India Post master",
    )

    df = df.rename(
        columns={
            pin_col: "pincode",
            state_col: "statename",
            district_col: "district",
            office_col: "officename",
        }
    )

    df["pincode_clean"] = df["pincode"].map(
        clean_pin
    )

    df["statename_clean"] = df["statename"].map(
        clean_string
    )

    df["district_clean"] = df["district"].map(
        clean_string
    )

    df["officename_clean"] = df["officename"].map(
        clean_string
    )

    return df


# ============================================================
# LOAD CSV / XLS / XML-SPREADSHEET FILE
# ============================================================

def load_table(path: Path) -> pd.DataFrame:
    """
    Load a master file.

    LGD files commonly have .xls extension but contain
    XML SpreadsheetML rather than a conventional binary XLS.
    """

    suffix = path.suffix.lower()

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    if suffix == ".csv":

        return pd.read_csv(
            path,
            low_memory=False,
        )

    # --------------------------------------------------------
    # Excel / XML Spreadsheet
    # --------------------------------------------------------

    if suffix in {".xls", ".xlsx", ".xlsm"}:

        try:

            return pd.read_excel(
                path,
                dtype=str,
            )

        except Exception:

            # ------------------------------------------------
            # XML Spreadsheet fallback
            # ------------------------------------------------

            try:

                tables = pd.read_xml(
                    path,
                    xpath=".//Row",
                )

                if tables is not None and not tables.empty:
                    return tables

            except Exception:
                pass

            # Last attempt using XML Spreadsheet parsing.
            return parse_excel_xml(path)

    raise ValueError(
        f"Unsupported master file format: {path}"
    )


# ============================================================
# XML SPREADSHEET PARSER
# ============================================================

def parse_excel_xml(path: Path) -> pd.DataFrame:
    """
    Parse Microsoft XML Spreadsheet 2003 files.

    LGD sometimes provides files with .xls extension which
    are actually XML Spreadsheet files.
    """

    import xml.etree.ElementTree as ET

    tree = ET.parse(path)
    root = tree.getroot()

    # Microsoft SpreadsheetML namespaces.
    namespaces = {
        "ss": "urn:schemas-microsoft-com:office:spreadsheet",
    }

    rows = []

    for row in root.findall(
        ".//ss:Worksheet/ss:Table/ss:Row",
        namespaces,
    ):

        values = []

        for cell in row.findall(
            "ss:Cell",
            namespaces,
        ):

            data = cell.find(
                "ss:Data",
                namespaces,
            )

            if data is None:
                values.append("")
            else:
                values.append(
                    data.text or ""
                )

        rows.append(values)

    if not rows:
        raise ValueError(
            f"No SpreadsheetML rows found in:\n{path}"
        )

    # First row is normally the header.
    headers = rows[0]

    data_rows = rows[1:]

    # Handle missing trailing cells.
    width = len(headers)

    normalized_rows = []

    for row in data_rows:

        if len(row) < width:
            row = row + [""] * (
                width - len(row)
            )

        elif len(row) > width:
            row = row[:width]

        normalized_rows.append(row)

    return pd.DataFrame(
        normalized_rows,
        columns=headers,
    )


# ============================================================
# FIND MASTER FILE
# ============================================================

def find_master_file(
    directory: Path,
    keyword: str,
) -> Path:

    candidates = []

    for path in directory.rglob("*"):

        if not path.is_file():
            continue

        name = path.name.lower()

        if keyword.lower() in name:

            if path.suffix.lower() in {
                ".xls",
                ".xlsx",
                ".xlsm",
                ".csv",
            }:
                candidates.append(path)

    if not candidates:

        raise FileNotFoundError(
            f"\nCould not find master file containing "
            f"'{keyword}' inside:\n{directory}"
        )

    # Prefer exact-looking names.
    candidates.sort(
        key=lambda p: (
            len(p.name),
            str(p),
        )
    )

    return candidates[0]


# ============================================================
# LOAD ALL MASTERS
# ============================================================

def load_masters(
    masters_dir: str,
) -> Dict[str, pd.DataFrame]:

    root = Path(masters_dir)

    if not root.exists():

        raise FileNotFoundError(
            f"Masters directory does not exist:\n{root}"
        )

    # --------------------------------------------------------
    # India Post
    # --------------------------------------------------------

    india_post_path = find_master_file(
        root,
        "india_post",
    )

    india_post = load_table(
        india_post_path
    )

    india_post = standardize_india_post_master(
        india_post
    )

    # --------------------------------------------------------
    # LGD files
    # --------------------------------------------------------

    district_path = find_master_file(
        root,
        "district",
    )

    subdistrict_path = find_master_file(
        root,
        "subdistrict",
    )

    village_path = find_master_file(
        root,
        "village",
    )

    ulb_path = find_master_file(
        root,
        "ulb",
    )

    district = standardize_district_master(
        load_table(district_path)
    )

    subdistrict = standardize_subdistrict_master(
        load_table(subdistrict_path)
    )

    village = standardize_village_master(
        load_table(village_path)
    )

    ulb = standardize_ulb_master(
        load_table(ulb_path)
    )

    return {
        "india_post": india_post,
        "district": district,
        "subdistrict": subdistrict,
        "village": village,
        "ulb": ulb,
    }


# ============================================================
# GENERIC INDEX BUILDER
# ============================================================

def _dict_of_indices(
    df: pd.DataFrame,
    key_column: str,
) -> Dict[str, List[int]]:
    """
    Build:

        normalized_value -> [row_index, row_index, ...]

    We intentionally retain ALL row indexes.

    Duplicate geographic names are valid and important.
    They must become AMBIGUOUS unless context resolves them.
    """

    result: Dict[str, List[int]] = {}

    for idx, value in df[key_column].items():

        key = clean_string(value)

        if not key:
            continue

        result.setdefault(
            key,
            [],
        ).append(idx)

    return result


# ============================================================
# COMPOSITE INDEX BUILDER
# ============================================================

def _composite_index(
    df: pd.DataFrame,
    columns: List[str],
) -> Dict[str, List[int]]:
    """
    Build indexes using multiple columns.

    Example:

        district + subdistrict + village
    """

    result: Dict[str, List[int]] = {}

    for idx, row in df.iterrows():

        values = []

        for column in columns:

            value = clean_string(
                row.get(column, "")
            )

            values.append(value)

        key = "|".join(values)

        if key.strip("|"):

            result.setdefault(
                key,
                [],
            ).append(idx)

    return result


# ============================================================
# TOKEN INDEX
# ============================================================

def build_token_index(
    names: Iterable[str],
) -> Dict[str, set]:
    """
    Build:

        TOKEN -> set(names)

    This lets the resolver narrow fuzzy searches without
    scanning the entire geographic master.
    """

    index: Dict[str, set] = {}

    for name in names:

        name = clean_string(name)

        if not name:
            continue

        tokens = set(
            token
            for token in name.split()
            if len(token) >= 2
        )

        for token in tokens:

            index.setdefault(
                token,
                set(),
            ).add(name)

    return index


# ============================================================
# PREPARE INDIA POST INDEXES
# ============================================================

def prepare_india_post_index(
    india_post: pd.DataFrame,
) -> Dict[str, object]:

    pin_rows: Dict[str, List[int]] = {}

    pin_states: Dict[str, set] = {}

    pin_districts: Dict[str, set] = {}

    pin_post_offices: Dict[str, set] = {}

    for idx, row in india_post.iterrows():

        pin = clean_pin(
            row.get("pincode_clean", "")
        )

        if not pin:
            continue

        state = clean_string(
            row.get("statename_clean", "")
        )

        district = clean_string(
            row.get("district_clean", "")
        )

        office = clean_string(
            row.get("officename_clean", "")
        )

        # -----------------------------------------------
        # PIN -> rows
        # -----------------------------------------------

        pin_rows.setdefault(
            pin,
            [],
        ).append(idx)

        # -----------------------------------------------
        # PIN -> states
        # -----------------------------------------------

        if state:

            pin_states.setdefault(
                pin,
                set(),
            ).add(state)

        # -----------------------------------------------
        # PIN -> districts
        # -----------------------------------------------

        if district:

            pin_districts.setdefault(
                pin,
                set(),
            ).add(district)

        # -----------------------------------------------
        # PIN -> post offices
        # -----------------------------------------------

        if office:

            pin_post_offices.setdefault(
                pin,
                set(),
            ).add(office)

    return {
        "pin_rows": pin_rows,
        "pin_states": pin_states,
        "pin_districts": pin_districts,
        "pin_post_offices": pin_post_offices,
    }


# ============================================================
# PREPARE LGD INDEXES
# ============================================================

def prepare_lgd_indexes(
    masters: Dict[str, pd.DataFrame],
) -> Dict[str, object]:

    district = masters["district"]

    subdistrict = masters["subdistrict"]

    village = masters["village"]

    ulb = masters["ulb"]

    # ========================================================
    # DISTRICT INDEX
    # ========================================================

    district_by_name = _dict_of_indices(
        district,
        "district_name_clean",
    )

    # ========================================================
    # SUBDISTRICT INDEX
    # ========================================================

    subdistrict_by_name = _dict_of_indices(
        subdistrict,
        "subdistrict_name_clean",
    )

    # ========================================================
    # VILLAGE INDEX
    # ========================================================

    village_by_name = _dict_of_indices(
        village,
        "village_name_clean",
    )

    # ========================================================
    # ULB INDEX
    # ========================================================

    ulb_by_name = _dict_of_indices(
        ulb,
        "ulb_name_clean",
    )

    # ========================================================
    # VILLAGE BY DISTRICT
    # ========================================================

    village_by_district = {}

    for idx, row in village.iterrows():

        district_name = clean_string(
            row.get(
                "district_name_clean",
                "",
            )
        )

        village_name = clean_string(
            row.get(
                "village_name_clean",
                "",
            )
        )

        if not district_name or not village_name:
            continue

        village_by_district.setdefault(
            district_name,
            {},
        )

        village_by_district[
            district_name
        ].setdefault(
            village_name,
            [],
        ).append(idx)

    # ========================================================
    # VILLAGE BY SUBDISTRICT
    # ========================================================

    village_by_subdistrict = {}

    for idx, row in village.iterrows():

        subdistrict_name = clean_string(
            row.get(
                "subdistrict_name_clean",
                "",
            )
        )

        village_name = clean_string(
            row.get(
                "village_name_clean",
                "",
            )
        )

        if not subdistrict_name or not village_name:
            continue

        village_by_subdistrict.setdefault(
            subdistrict_name,
            {},
        )

        village_by_subdistrict[
            subdistrict_name
        ].setdefault(
            village_name,
            [],
        ).append(idx)

    # ========================================================
    # VILLAGE BY DISTRICT + SUBDISTRICT
    # ========================================================

    village_by_district_subdistrict = {}

    for idx, row in village.iterrows():

        district_name = clean_string(
            row.get(
                "district_name_clean",
                "",
            )
        )

        subdistrict_name = clean_string(
            row.get(
                "subdistrict_name_clean",
                "",
            )
        )

        village_name = clean_string(
            row.get(
                "village_name_clean",
                "",
            )
        )

        if (
            not district_name
            or not subdistrict_name
            or not village_name
        ):
            continue

        key = (
            district_name,
            subdistrict_name,
        )

        village_by_district_subdistrict.setdefault(
            key,
            {},
        )

        village_by_district_subdistrict[
            key
        ].setdefault(
            village_name,
            [],
        ).append(idx)

    # ========================================================
    # SUBDISTRICT BY DISTRICT
    # ========================================================

    subdistrict_by_district = {}

    for idx, row in subdistrict.iterrows():

        district_name = clean_string(
            row.get(
                "district_name_clean",
                "",
            )
        )

        subdistrict_name = clean_string(
            row.get(
                "subdistrict_name_clean",
                "",
            )
        )

        if (
            not district_name
            or not subdistrict_name
        ):
            continue

        subdistrict_by_district.setdefault(
            district_name,
            {},
        )

        subdistrict_by_district[
            district_name
        ].setdefault(
            subdistrict_name,
            [],
        ).append(idx)

    # ========================================================
    # TOKEN INDEXES
    # ========================================================

    village_tokens = build_token_index(
        village_by_name.keys()
    )

    district_tokens = build_token_index(
        district_by_name.keys()
    )

    subdistrict_tokens = build_token_index(
        subdistrict_by_name.keys()
    )

    ulb_tokens = build_token_index(
        ulb_by_name.keys()
    )

    # ========================================================
    # RETURN
    # ========================================================

    return {

        # Basic entity indexes
        "district_by_name":
            district_by_name,

        "subdistrict_by_name":
            subdistrict_by_name,

        "village_by_name":
            village_by_name,

        "ulb_by_name":
            ulb_by_name,

        # Hierarchical indexes
        "village_by_district":
            village_by_district,

        "village_by_subdistrict":
            village_by_subdistrict,

        "village_by_district_subdistrict":
            village_by_district_subdistrict,

        "subdistrict_by_district":
            subdistrict_by_district,

        # Token indexes
        "village_tokens":
            village_tokens,

        "district_tokens":
            district_tokens,

        "subdistrict_tokens":
            subdistrict_tokens,

        "ulb_tokens":
            ulb_tokens,
    }


# ============================================================
# PREPARE ALL INDEXES
# ============================================================

def prepare_indexes(
    masters: Dict[str, pd.DataFrame],
) -> Dict[str, object]:

    indexes = {}

    # India Post indexes
    indexes.update(
        prepare_india_post_index(
            masters["india_post"]
        )
    )

    # LGD indexes
    indexes.update(
        prepare_lgd_indexes(
            masters
        )
    )

    return indexes


# ============================================================
# DEBUG / STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    SCRIPT_DIR = Path(__file__).resolve().parent

    ROOT = SCRIPT_DIR.parent

    MASTERS_DIR = ROOT / "masters"

    print("=" * 70)
    print("MASTER LOADER TEST")
    print("=" * 70)

    print(
        f"\nMasters directory:\n{MASTERS_DIR}"
    )

    print("\n[1] Loading masters...")

    masters = load_masters(
        str(MASTERS_DIR)
    )

    print(
        f"India Post : "
        f"{len(masters['india_post']):,}"
    )

    print(
        f"District   : "
        f"{len(masters['district']):,}"
    )

    print(
        f"Subdistrict: "
        f"{len(masters['subdistrict']):,}"
    )

    print(
        f"Village    : "
        f"{len(masters['village']):,}"
    )

    print(
        f"ULB        : "
        f"{len(masters['ulb']):,}"
    )

    print("\n[2] Building indexes...")

    indexes = prepare_indexes(
        masters
    )

    print(
        f"PINs indexed: "
        f"{len(indexes['pin_rows']):,}"
    )

    print(
        f"District names: "
        f"{len(indexes['district_by_name']):,}"
    )

    print(
        f"Subdistrict names: "
        f"{len(indexes['subdistrict_by_name']):,}"
    )

    print(
        f"Village names: "
        f"{len(indexes['village_by_name']):,}"
    )

    print(
        f"ULB names: "
        f"{len(indexes['ulb_by_name']):,}"
    )

    print(
        f"Village token index: "
        f"{len(indexes['village_tokens']):,}"
    )

    print("\n[3] Sample columns:")

    print(
        "\nDistrict:"
    )

    print(
        masters["district"].columns.tolist()
    )

    print(
        "\nVillage:"
    )

    print(
        masters["village"].columns.tolist()
    )

    print("\nMASTER LOADER TEST COMPLETE")