import pandas as pd
from pathlib import Path

from load_data import load_csv, DataLoadError


def main():
    # Create a temporary test CSV
    test_path = Path("temp_test.csv")

    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": ["x", "y", "z"],
        }
    )

    df.to_csv(test_path, index=False)

    try:
        # Test successful load
        result = load_csv(test_path)

        print("✅ LOAD SUCCESS")
        print("Source:", result.source)
        print("Shape:", result.df.shape)
        print(result.df.head())

    except Exception as e:
        print("❌ LOAD FAILED:", e)

    finally:
        # Clean up test file
        if test_path.exists():
            test_path.unlink()


if __name__ == "__main__":
    main()