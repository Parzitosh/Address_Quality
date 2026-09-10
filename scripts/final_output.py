import pandas as pd
from pathlib import Path

# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = Path("output/address_quality_v3.csv")
OUTPUT_FILE = Path("output/field_visit_address_file.csv")


# ============================================================
# LOAD ADDRESS QUALITY OUTPUT
# ============================================================

df = pd.read_csv(INPUT_FILE)


# ============================================================
# FINAL FIELD-VISIT OUTPUT
# ============================================================

field_visit_df = pd.DataFrame({
    "Full Address": df["Clean Full Address"],
    "City": df["Clean City"],
    "Pincode": df["Clean Pincode"],
    "State": df["Clean State"],
    "District": df["resolved_district"],
    "Lead Code": df["Lead Code"],
})


# ============================================================
# FIELD VISIT DECISIONS
# ============================================================
#
# Score >= 60:
#   USABLE / GOOD / EXCELLENT
#   → Field visit possible
#
# Score < 60:
#   REQUIRES_CALL / INVALID
#   → Field visit not possible
#
# ============================================================

field_visit_df["Field Visit Possible"] = (
    pd.to_numeric(df["quality_score"], errors="coerce")
    .ge(60)
    .map({
        True: "YES",
        False: "NO"
    })
)


# ============================================================
# SAVE FINAL FILE
# ============================================================

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

field_visit_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"Field visit file created: {OUTPUT_FILE}")
print(f"Total records: {len(field_visit_df)}")
print(
    field_visit_df["Field Visit Possible"]
    .value_counts()
)