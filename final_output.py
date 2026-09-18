from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"

QUALITY_FILES = {
    "Current Address": OUTPUT_DIR / "address_quality_current.csv",
    "Office Address": OUTPUT_DIR / "address_quality_office.csv",
    "Alternate Address": OUTPUT_DIR / "address_quality_alternate.csv",
}

FINAL_OUTPUT = OUTPUT_DIR / "final_address_quality.csv"
AUDIT_OUTPUT = OUTPUT_DIR / "final_address_quality_audit.csv"

NO_RESOLUTION = "didn't needed any resolving"


def clean_value(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def normalized_pin(value: str) -> str:
    text = clean_value(value)
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".")[0]
    digits = re.sub(r"\D", "", text)
    return digits if len(digits) == 6 else ""


def combine_address(row) -> str:
    address = clean_value(row.get("Clean Full Address", ""))
    city = clean_value(row.get("Clean City", ""))
    state = clean_value(row.get("Clean State", ""))
    pin = normalized_pin(row.get("Clean Pincode", ""))
    parts = [address]
    existing = " ".join(parts).upper()
    for value in [city, state, pin]:
        if value and value.upper() not in existing:
            parts.append(value)
            existing = " ".join(parts).upper()
    return ", ".join(x for x in parts if x)


def load_quality(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required quality output not found: {path}. "
            f"Run run_address_quality.py first."
        )

    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    required = {
        "LAN",
        "quality_class",
        "raw_quality_score",
        "Clean Full Address",
        "Clean City",
        "Clean State",
        "Clean Pincode",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Required columns missing from {path}: {sorted(missing)}"
        )

    return df


def validate_alignment(base, other, base_name, other_name):
    if len(base) != len(other):
        raise ValueError(
            f"{base_name} and {other_name} row counts do not match."
        )

    if not base["LAN"].reset_index(drop=True).equals(
        other["LAN"].reset_index(drop=True)
    ):
        raise ValueError(
            f"LAN ordering does not match between "
            f"{base_name} and {other_name} outputs."
        )


def build_audit_remark(df):
    # Prefer the audit/resolution information already emitted by the
    # upstream quality pipeline, if present. Otherwise do not invent a
    # resolution event.
    audit_candidates = [
        "auto_resolution_remark",
        "audit remark",
        "audit_remark",
        "resolution_remark",
        "qc_resolution_remark",
    ]

    for column in audit_candidates:
        if column in df.columns:
            return df[column].apply(
                lambda x: clean_value(x) or NO_RESOLUTION
            )

    return pd.Series(NO_RESOLUTION, index=df.index)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # IMPORTANT ARCHITECTURE:
    # final_output.py is a downstream assembler only.
    # It NEVER runs address_quality.py and NEVER performs State/City/PIN
    # auto-resolution. Those steps belong upstream.
    current = load_quality(QUALITY_FILES["Current Address"])

    office = (
        load_quality(QUALITY_FILES["Office Address"])
        if QUALITY_FILES["Office Address"].exists()
        else None
    )

    alternate = (
        load_quality(QUALITY_FILES["Alternate Address"])
        if QUALITY_FILES["Alternate Address"].exists()
        else None
    )

    if office is not None:
        validate_alignment(
            current, office, "Current Address", "Office Address"
        )

    if alternate is not None:
        validate_alignment(
            current, alternate, "Current Address", "Alternate Address"
        )

    result = pd.DataFrame({
        "LAN": current["LAN"],
        "Clean Current Address": current.apply(combine_address, axis=1),
        "Current Quality Class": current["quality_class"],
        "Current Raw Quality Score": current["raw_quality_score"],
    })

    audit = pd.DataFrame({"LAN": current["LAN"]})
    audit["CURRENT_ADDRESS"] = current.apply(combine_address, axis=1)
    audit["CURRENT_ADDRESS_AUDIT_REMARK"] = build_audit_remark(current)
    audit["CURRENT_QUALITY_CLASS"] = current["quality_class"]

    if office is not None:
        result["Clean Office Address"] = office.apply(
            combine_address, axis=1
        )
        result["Office Quality Class"] = office["quality_class"]
        result["Office Raw Quality Score"] = office["raw_quality_score"]

        audit["OFFICE_ADDRESS"] = office.apply(
            combine_address, axis=1
        )
        audit["OFFICE_ADDRESS_AUDIT_REMARK"] = build_audit_remark(office)
        audit["OFFICE_QUALITY_CLASS"] = office["quality_class"]
    else:
        audit["OFFICE_ADDRESS"] = ""
        audit["OFFICE_ADDRESS_AUDIT_REMARK"] = NO_RESOLUTION
        audit["OFFICE_QUALITY_CLASS"] = ""

    if alternate is not None:
        result["Clean Alternate Address"] = alternate.apply(
            combine_address, axis=1
        )
        result["Alternate Quality Class"] = alternate["quality_class"]
        result["Alternate Raw Quality Score"] = alternate["raw_quality_score"]

        audit["ALTERNATE_ADDRESS"] = alternate.apply(
            combine_address, axis=1
        )
        audit["ALTERNATE_ADDRESS_AUDIT_REMARK"] = build_audit_remark(alternate)
        audit["ALTERNATE_QUALITY_CLASS"] = alternate["quality_class"]
    else:
        audit["ALTERNATE_ADDRESS"] = ""
        audit["ALTERNATE_ADDRESS_AUDIT_REMARK"] = NO_RESOLUTION
        audit["ALTERNATE_QUALITY_CLASS"] = ""

    result.to_csv(FINAL_OUTPUT, index=False, encoding="utf-8-sig")
    audit.to_csv(AUDIT_OUTPUT, index=False, encoding="utf-8-sig")

    print("=" * 70)
    print("FINAL ADDRESS QUALITY OUTPUT")
    print("=" * 70)
    print(f"Records: {len(result):,}")
    print(f"Final  : {FINAL_OUTPUT}")
    print(f"Audit  : {AUDIT_OUTPUT}")
    print()
    print("final_output.py consumed existing run_address_quality outputs only.")
    print("No quality engine was re-run.")
    print("No State/City/PIN resolution was performed here.")


if __name__ == "__main__":
    main()
