import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent

QUALITY_FILES = {
    "Current Address": ROOT / "output" / "address_quality_current.csv",
    "Office Address": ROOT / "output" / "address_quality_office.csv",
    "Alternate Address": ROOT / "output" / "address_quality_alternate.csv",
}

OUTPUT_FILE = ROOT / "output" / "final_address_quality.csv"


def clean_value(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def combine_address(row):
    address = clean_value(row.get("Clean Full Address", ""))
    city = clean_value(row.get("Clean City", ""))
    state = clean_value(row.get("Clean State", ""))
    pin = clean_value(row.get("Clean Pincode", ""))

    parts = [address]

    # Avoid duplicating metadata if the cleaner has already retained it.
    existing = " ".join(parts).upper()

    for value in [city, state, pin]:
        if not value:
            continue

        if value.upper() not in existing:
            parts.append(value)
            existing = " ".join(parts).upper()

    return ", ".join(x for x in parts if x)


def load_quality(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Quality output not found: {path}\n"
            "Run run_address_quality.py first."
        )

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
    )

    if "LAN" not in df.columns:
        raise ValueError(f"LAN column missing from {path}")

    # These are required for the final output.
    required_quality_columns = [
        "quality_class",
        "raw_quality_score",
    ]

    missing = [
        col for col in required_quality_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Required quality columns missing from {path}: {missing}"
        )

    return df


def validate_alignment(base, other, base_name, other_name):
    if len(other) != len(base):
        raise ValueError(
            f"{base_name} and {other_name} row counts do not match."
        )

    if not other["LAN"].reset_index(drop=True).equals(
        base["LAN"].reset_index(drop=True)
    ):
        raise ValueError(
            f"LAN ordering does not match between "
            f"{base_name} and {other_name} outputs."
        )


def main():
    # ---------------------------------------------------------
    # LOAD QUALITY OUTPUTS
    # ---------------------------------------------------------

    current = load_quality(
        QUALITY_FILES["Current Address"]
    )

    office = load_quality(
        QUALITY_FILES["Office Address"]
    )

    validate_alignment(
        current,
        office,
        "Current Address",
        "Office Address",
    )

    # ---------------------------------------------------------
    # BUILD FINAL OUTPUT
    # ---------------------------------------------------------

    result = pd.DataFrame({
        "LAN": current["LAN"],

        # Current Address
        "Clean Current Address": current.apply(
            combine_address,
            axis=1,
        ),
        "Current Quality Class": current["quality_class"],
        "Current Raw Quality Score": current["raw_quality_score"],

        # Office Address
        "Clean Office Address": office.apply(
            combine_address,
            axis=1,
        ),
        "Office Quality Class": office["quality_class"],
        "Office Raw Quality Score": office["raw_quality_score"],
    })

    # ---------------------------------------------------------
    # ALTERNATE ADDRESS
    # ---------------------------------------------------------

    if QUALITY_FILES["Alternate Address"].exists():

        alternate = load_quality(
            QUALITY_FILES["Alternate Address"]
        )

        validate_alignment(
            current,
            alternate,
            "Current Address",
            "Alternate Address",
        )

        result["Clean Alternate Address"] = alternate.apply(
            combine_address,
            axis=1,
        )

        result["Alternate Quality Class"] = (
            alternate["quality_class"]
        )

        result["Alternate Raw Quality Score"] = (
            alternate["raw_quality_score"]
        )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print("=" * 70)
    print("FINAL MULTI-ADDRESS OUTPUT")
    print("=" * 70)

    print(f"Records: {len(result):,}")
    print(f"Output : {OUTPUT_FILE}")

    print("\nColumns:")
    for column in result.columns:
        print(f"  - {column}")

    print("\nQuality columns included:")
    print("  - Current Quality Class")
    print("  - Current Raw Quality Score")
    print("  - Office Quality Class")
    print("  - Office Raw Quality Score")

    if "Alternate Quality Class" in result.columns:
        print("  - Alternate Quality Class")
        print("  - Alternate Raw Quality Score")


if __name__ == "__main__":
    main()