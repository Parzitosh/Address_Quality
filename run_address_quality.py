from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = ROOT / "scripts"
MASTERS_DIR = ROOT / "masters"
OUTPUT_DIR = ROOT / "output"
AUTO_RESOLVER_PATH = ROOT / "auto_resolver.py"

INPUTS = {
    "current": ROOT / "Cleaned_Current_Address_Data.csv",
    "office": ROOT / "Cleaned_Office_Address_Data.csv",
    "alternate": ROOT / "Cleaned_Alternate_Address_Data.csv",
}
OUTPUTS = {
    "current": OUTPUT_DIR / "address_quality_current.csv",
    "office": OUTPUT_DIR / "address_quality_office.csv",
    "alternate": OUTPUT_DIR / "address_quality_alternate.csv",
}

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_module(name, path):
    if not path.exists():
        raise FileNotFoundError(f"Required module not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_file(df, auto, auto_module):
    required = {"LAN", "Clean Full Address", "Clean City", "Clean State", "Clean Pincode"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required cleaned-address columns missing: {sorted(missing)}")

    remarks = []
    for idx, row in df.iterrows():
        result = auto.resolve(
            row.get("Clean Full Address", ""),
            row.get("Clean City", ""),
            row.get("Clean State", ""),
            row.get("Clean Pincode", ""),
        )
        df.at[idx, "Clean City"] = result["city"]
        df.at[idx, "Clean State"] = result["state"]
        df.at[idx, "Clean Pincode"] = result["pin"]
        remarks.append(
            "; ".join(result["remarks"])
            if result["remarks"]
            else auto_module.NO_RESOLUTION
        )

    df["auto_resolution_remark"] = remarks
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    master_loader = load_module("master_loader_runtime", SCRIPT_DIR / "master_loader.py")
    resolver_module = load_module("entity_resolution_runtime", SCRIPT_DIR / "entity_resolution.py")
    auto_module = load_module("auto_resolver_runtime", AUTO_RESOLVER_PATH)

    masters = master_loader.load_masters(str(MASTERS_DIR))
    indexes = master_loader.prepare_indexes(masters)
    resolver = resolver_module.GeographicResolver(masters, indexes)
    auto = auto_module.AutoResolver(masters, indexes, resolver, resolver_module)

    print("=" * 70)
    print("MULTI-ADDRESS AUTO-RESOLUTION + QUALITY CHECK")
    print("=" * 70)

    for name, input_file in INPUTS.items():
        if not input_file.exists():
            print(f"SKIPPED {name.upper()}: {input_file.name} not found.")
            continue

        output_file = OUTPUTS[name]
        df = pd.read_csv(input_file, dtype=str, keep_default_na=False)
        original_count = len(df)

        resolved = resolve_file(df, auto, auto_module)
        if len(resolved) != original_count:
            raise RuntimeError(
                f"Row-count invariant failed for {name}: "
                f"{original_count} -> {len(resolved)}"
            )

        tmp = OUTPUT_DIR / f"_resolved_{name}_input.csv"
        resolved.to_csv(tmp, index=False, encoding="utf-8-sig")

        quality = load_module(
            f"address_quality_{name}_runtime",
            SCRIPT_DIR / "address_quality.py",
        )
        quality.INPUT_FILE = tmp
        quality.OUTPUT_FILE = output_file
        quality.main()

        print(f"Completed {name}: {original_count:,} rows")

    print("All available address quality files processed.")
    print("Auto-resolution occurs before the existing quality engine.")
    print("No row deduplication is performed.")


if __name__ == "__main__":
    main()
