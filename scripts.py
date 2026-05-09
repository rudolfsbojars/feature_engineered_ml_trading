import joblib
import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    root = Path(input("Ievadi ceļu uz mapes sakni: ").strip())
    results = {"base": [], "base_plus_rsi_and_ema": [], "full": [], "random": []}

    for folder in sorted(root.rglob("*")):
        if not folder.is_dir():
            continue
        for fs in results.keys():
            pkl_path = folder / f"{fs}.pkl"
            if pkl_path.exists():
                try:
                    package = joblib.load(pkl_path)
                    model = package["model"]
                    features = package["features"]
                    importance = dict(zip(features, model.feature_importances_))
                    results[fs].append(importance)
                    print(f"OK: {pkl_path}")
                except Exception as e:
                    print(f"KĻŪDA: {pkl_path} — {e}")

    print("\n=== VIDĒJAIS FEATURE IMPORTANCE ===")
    for fs, records in results.items():
        if not records:
            print(f"{fs}: nav datu")
            continue
        df = pd.DataFrame(records).fillna(0)
        avg = df.mean().sort_values(ascending=False)
        print(f"\n--- {fs} ---")
        print(avg.to_string())
        avg.to_csv(f"feature_importance_{fs}.csv")
        print(f"Saglabāts: feature_importance_{fs}.csv")