from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = ROOT / "scripts"

INPUTS = {
    "current": ROOT / "Cleaned_Current_Address_Data.csv",
    "office": ROOT / "Cleaned_Office_Address_Data.csv",
    "alternate": ROOT / "Cleaned_Alternate_Address_Data.csv",
}

OUTPUTS = {
    "current": ROOT / "output" / "address_quality_current.csv",
    "office": ROOT / "output" / "address_quality_office.csv",
    "alternate": ROOT / "output" / "address_quality_alternate.csv",
}

def load_quality_module():
    path = SCRIPT_DIR / "address_quality.py"
    if not path.exists():
        raise FileNotFoundError(f"address_quality.py not found: {path}")

    spec = importlib.util.spec_from_file_location("address_quality_batch_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def main():
    quality = load_quality_module()

    print("=" * 70)
    print("MULTI-ADDRESS QUALITY CHECK")
    print("=" * 70)

    for name, input_file in INPUTS.items():
        if not input_file.exists():
            print(f"\nSKIPPED {name.upper()}: {input_file.name} not found.")
            continue

        output_file = OUTPUTS[name]
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print("\n" + "-" * 70)
        print(f"ADDRESS TYPE: {name.upper()}")
        print(f"INPUT : {input_file}")
        print(f"OUTPUT: {output_file}")
        print("-" * 70)

        # IMPORTANT:
        # Only orchestration/path variables are changed here.
        # The address-quality validation/scoring logic itself is untouched.
        quality.INPUT_FILE = input_file
        quality.OUTPUT_FILE = output_file
        quality.main()

    print("\nAll available address quality files processed.")

if __name__ == "__main__":
    main()
