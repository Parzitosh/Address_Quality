import pandas as pd
from pathlib import Path

# ============================================================
# CONFIGURATION & FILE PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"  # Drop raw files here
OUTPUT_FILE = BASE_DIR / "Standardized_Input.csv"

# ============================================================
# LENDER MAPPING DEFINITIONS
# ============================================================
# 'signature': The unique columns needed to identify the lender.
# Lists under 'current', 'office', 'alternate' will be safely concatenated.

LENDER_CONFIGS = {
    "AGRIM": {
        "signature": ["Loan No.", "Current Address", "Permanent Address"],
        "lan": ["Loan No."],
        "current": ["Current Address"],
        "alternate": ["Permanent Address"],
        "office": ["Employer Address"] # Handled safely if it exists 'sometimes'
    },
    "ARCIL": {
        "signature": ["Loan No", "Primary Address", "Alternate Address"],
        "lan": ["Loan No"],
        "current": ["Primary Address", "Primary Address City", "Primary Address State", "Primary Address Pincode"],
        "alternate": ["Alternate Address", "Alternate Address City", "Alternate Address State", "Alternate Address Pincode"],
        "office": []
    },
    "BOBCARD": {
        "signature": ["LAN", "ADD_LINE_1", "CITY", "POSTAL_CODE"],
        "lan": ["LAN"],
        "current": ["ADD_LINE_1", "ADD_LINE_2", "ADD_LINE_3", "ADD_LINE_4", "CITY", "STATE", "POSTAL_CODE"],
        "office": ["EMP_ADDR_1", "EMP_ADDR_2", "EMP_ADDR_3", "EMP_CITY", "EMP_STATE", "EMP_PIN"],
        "alternate": ["ALT_ADDR_1", "ALT_ADDR_2", "ALT_ADDR_3", "ALT_ADDR_4", "ALT_CITY", "ALT_PIN"]
    },
    "CASHE_NACL": {
        "signature": ["Partner Loan ID", "Addresses"],
        "lan": ["Partner Loan ID"],
        "current": ["Addresses"],
        "alternate": [],
        "office": []
    },
    "CHINMAY_FINLEASE": {
        "signature": ["Loan ID", "Address 1", "Address 2", "Address 3", "Address 4"],
        "lan": ["Loan ID"],
        "current": ["Address 1", "Address 2", "Address 3", "Address 4"], # Assuming these are lines of one address
        "alternate": [],
        "office": []
    },
    "HDBFS_PL": {
        "signature": ["Loan Number", "RESIDENCE ADDRESS", "OFFICE ADDRESS"],
        "lan": ["Loan Number"],
        "current": ["RESIDENCE ADDRESS"],
        "office": ["OFFICE ADDRESS"],
        "alternate": ["CoApp_Address"]
    },
    "HOME_CREDIT": {
        "signature": ["CONTRACT_NO", "CONTACT_ADDRESS_AF", "PERMANENT_ADDRESS_AF"],
        "lan": ["CONTRACT_NO"],
        "current": ["CONTACT_ADDRESS_AF"],
        "alternate": ["PERMANENT_ADDRESS_AF"],
        "office": []
    },
    "IARC": {
        "signature": ["ACCOUNT_NUMBER", "ADDRESS", "PIN CODE", "STATE", "CITY"],
        "lan": ["ACCOUNT_NUMBER"],
        "current": ["ADDRESS", "CITY", "STATE", "PIN CODE"],
        "alternate": [],
        "office": []
    },
    "SMFG": {
        "signature": ["LAN", "App_Full_Address", "Coapp_Full_Address"],
        "lan": ["LAN"],
        "current": ["App_Full_Address"],
        "alternate": ["Coapp_Full_Address"],
        "office": []
    },
    "TATA_CAPITAL": {
        "signature": ["Ac No", "RESIDENCE ADDRESS", "PINCODE"],
        "lan": ["Ac No"],
        "current": ["RESIDENCE ADDRESS", "PINCODE"],
        "alternate": [],
        "office": []
    },
    "FEDERAL_BANK": {
        "signature": ["Account", "Address Line1", "City", "State", "Pin"],
        "lan": ["Account"],
        "current": ["Address Line1", "Address Line2", "Address Line3", "City", "State", "Pin"],
        "alternate": ["Perm Address Line1", "Perm Address Line2", "Perm Address Line3", "Perm City", "Perm State", "Perm Pin"],
        "office": ["Additional Address"]
    },
    "CUSTOM": {
            "signature": ["LAN", "Address"],
            "lan": ["LAN"],
            "current": ["Address"],
            "alternate": [],
            "office": []
        }
}

# ============================================================
# ROUTER LOGIC
# ============================================================

def concat_columns(row, cols):
    """Safely concatenates multiple columns into a single comma-separated string."""
    parts = []
    for col in cols:
        if col in row.index and pd.notna(row[col]):
            val = str(row[col]).strip()
            # Ignore nan strings, nulls, and pure zeros
            if val and val.lower() not in {"nan", "null", "none", "0", "0.0"}:
                parts.append(val)
    return ", ".join(parts) if parts else ""

def identify_lender(df_columns):
    """Detects the lender by checking if their unique signature columns exist."""
    col_set = set(df_columns)
    for lender, config in LENDER_CONFIGS.items():
        if set(config["signature"]).issubset(col_set):
            return lender, config
    return None, None

def process_file(file_path):
    print(f"Reading raw file: {file_path.name}")
    
    # Read file handling both CSV and Excel
    if file_path.suffix.lower() == '.csv':
        df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
    elif file_path.suffix.lower() in {'.xls', '.xlsx'}:
        df = pd.read_excel(file_path, dtype=str, keep_default_na=False)
    else:
        print(f"Unsupported file format: {file_path.name}")
        return

    # 1. Sniff the schema to identify lender
    lender_name, config = identify_lender(df.columns)
    
    if not lender_name:
        raise ValueError(f"Could not identify lender format for file: {file_path.name}. Please check column names.")
        
    print(f"Detected Lender Format: {lender_name}")

    # 2. Map and concatenate columns based on config
    standardized_df = pd.DataFrame()
    
    standardized_df["LAN"] = df.apply(lambda row: concat_columns(row, config["lan"]), axis=1)
    standardized_df["Current Address"] = df.apply(lambda row: concat_columns(row, config["current"]), axis=1)
    standardized_df["Office Address"] = df.apply(lambda row: concat_columns(row, config["office"]), axis=1)
    standardized_df["Alternate Address"] = df.apply(lambda row: concat_columns(row, config["alternate"]), axis=1)

    # Wrap LAN in quotes if needed to prevent Excel scientific notation
    standardized_df["LAN"] = "'" + standardized_df["LAN"].astype(str) + "'"

    # Drop rows where LAN is completely empty
    standardized_df = standardized_df[standardized_df["LAN"].str.replace("'", "") != ""]

    # 3. Export to standardized file
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    standardized_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    
    print(f"Successfully routed and standardized {len(standardized_df)} records.")
    print(f"Output saved to: {OUTPUT_FILE.name}")

if __name__ == "__main__":
    # Create input directory if it doesn't exist
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Automatically process the first file found in the input directory
    input_files = list(INPUT_DIR.glob("*.*"))
    valid_files = [f for f in input_files if f.suffix.lower() in {'.csv', '.xlsx', '.xls'}]
    
    if valid_files:
        process_file(valid_files[0])
    else:
        print(f"No valid CSV/Excel files found in the '{INPUT_DIR.name}' folder.")