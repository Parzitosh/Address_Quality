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

scores = pd.to_numeric(df["quality_score"], errors="coerce")
hard_invalid = df["hard_invalid"] if "hard_invalid" in df.columns else pd.Series(False, index=df.index)
review_flag = df["qc_review_flag"] if "qc_review_flag" in df.columns else pd.Series(False, index=df.index)

# Stratified operational routing.
# - fatal_rejects: hard-invalid or score < 60
# - manual_review: score 60-84.99, or any QC review flag
# - auto_dispatch: score >= 85 with no critical/review flag
field_visit_df["operational_queue"] = "manual_review"
field_visit_df.loc[hard_invalid.astype(bool) | scores.lt(60), "operational_queue"] = "fatal_rejects"
field_visit_df.loc[scores.ge(85) & ~hard_invalid.astype(bool) & ~review_flag.astype(bool), "operational_queue"] = "auto_dispatch"

field_visit_df["Field Visit Possible"] = field_visit_df["operational_queue"].isin(
    ["auto_dispatch", "manual_review"]
).map({True: "YES", False: "NO"})


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
print("\nOperational queue:")
print(field_visit_df["operational_queue"].value_counts())
print("\nField visit:")
print(field_visit_df["Field Visit Possible"].value_counts())