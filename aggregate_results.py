import os
import json
import argparse
from collections import defaultdict
 
 
BACKTEST_FILES = {
    "base":                  "backtest_result_base.json",
    "base_plus_rsi_and_ema": "backtest_result_base_plus_rsi_and_ema.json",
    "full":                  "backtest_result.json",
}
 
CLASSIFICATION_FILE = "results.json"
 
 
def safe_get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
        if d is None:
            return default
    return d
 
 
def is_date_range(name: str) -> bool:
    parts = name.split("_")
    return len(parts) == 2 and len(parts[0]) == 10 and len(parts[1]) == 10
 
 
def extract_backtest(data: dict) -> dict:
    trades = data.get("trades", {})
    total  = safe_get(trades, "total", "total", default=0)
    won    = safe_get(trades, "won",   "total", default=0)
    return {
        "total_trades": total,
        "win_rate":     round(won / total, 4) if total else None,
        "net_pnl":      safe_get(trades, "pnl", "net", "total"),
        "avg_pnl":      safe_get(trades, "pnl", "net", "average"),
        "max_drawdown": safe_get(data, "drawdown", "max", "drawdown"),
        "sharpe":       safe_get(data, "sharpe", "sharperatio"),
        "mc_mean":      safe_get(data, "monte_carlo", "mc_mean"),
        "mc_std":       safe_get(data, "monte_carlo", "mc_std"),
        "mc_min":       safe_get(data, "monte_carlo", "mc_min"),
        "mc_max":       safe_get(data, "monte_carlo", "mc_max"),
        "rtot":         safe_get(data, "returns", "rtot"),
    }
 
 
def extract_classification(data: dict) -> dict:
    out = {}
    for model, metrics in data.items():
        out[model] = {
            "accuracy":  metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall":    metrics.get("recall"),
            "f1":        metrics.get("f1"),
            "roc_auc":   metrics.get("roc_auc"),
            "log_loss":  metrics.get("log_loss"),
        }
    return out
 
 
def average_dicts(records: list) -> dict:
    if not records:
        return {}
    result = {}
    for k in records[0].keys():
        vals = [r[k] for r in records if r.get(k) is not None]
        result[k] = round(sum(vals) / len(vals), 6) if vals else None
    return result
 
 
def load_period(folder: str) -> dict:
    period = {"backtest": {}, "classification": {}}
 
    for model_key, filename in BACKTEST_FILES.items():
        fp = os.path.join(folder, filename)
        if os.path.exists(fp):
            with open(fp) as f:
                period["backtest"][model_key] = extract_backtest(json.load(f))
 
    clf_path = os.path.join(folder, CLASSIFICATION_FILE)
    if os.path.exists(clf_path):
        with open(clf_path) as f:
            period["classification"] = extract_classification(json.load(f))
 
    return period
 
 
def find_best_worst(periods: list, data_key: str, model: str, metric: str) -> dict:
    scored = []
    for p in periods:
        val = p.get(data_key, {}).get(model, {}).get(metric)
        if val is not None:
            scored.append((p["period"], val))
 
    if not scored:
        return {"best": None, "worst": None}
 
    scored.sort(key=lambda x: x[1])
    return {
        "best":  {"period": scored[-1][0], "value": round(scored[-1][1], 6)},
        "worst": {"period": scored[0][0],  "value": round(scored[0][1], 6)},
    }
 
 
def aggregate_periods(periods: list) -> dict:
    bt_by_model  = defaultdict(list)
    clf_by_model = defaultdict(list)
 
    for p in periods:
        for model, metrics in p["backtest"].items():
            bt_by_model[model].append(metrics)
        for model, metrics in p["classification"].items():
            clf_by_model[model].append(metrics)
 
    backtest_avg       = {m: average_dicts(v) for m, v in bt_by_model.items()}
    classification_avg = {m: average_dicts(v) for m, v in clf_by_model.items()}
 
    all_models = set(list(bt_by_model.keys()) + list(clf_by_model.keys()))
    highlights = {}
    for model in all_models:
        highlights[model] = {
            "backtest": {
                "by_net_pnl":      find_best_worst(periods, "backtest",       model, "net_pnl"),
                "by_win_rate":     find_best_worst(periods, "backtest",       model, "win_rate"),
                "by_max_drawdown": find_best_worst(periods, "backtest",       model, "max_drawdown"),
            },
            "classification": {
                "by_roc_auc":      find_best_worst(periods, "classification", model, "roc_auc"),
                "by_f1":           find_best_worst(periods, "classification", model, "f1"),
            },
        }
 
    return {
        "n_periods":          len(periods),
        "periods":            [p["period"] for p in periods],
        "backtest_avg":       backtest_avg,
        "classification_avg": classification_avg,
        "highlights":         highlights,
        "backtest_all":       dict(bt_by_model),
        "classification_all": dict(clf_by_model),
    }
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Instrument folder, e.g. path/to/BTC_1H")
    args = parser.parse_args()
 
    folder     = args.folder.rstrip("/\\")
    instrument = os.path.basename(folder)
    groups     = defaultdict(list)
 
    for entry in sorted(os.scandir(folder), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        name = entry.name
 
        if name.lower().startswith("4year"):
            for sub in sorted(os.scandir(entry.path), key=lambda e: e.name):
                if sub.is_dir() and is_date_range(sub.name):
                    p = load_period(sub.path)
                    p["period"] = sub.name
                    groups["4years"].append(p)
 
        elif is_date_range(name):
            p = load_period(entry.path)
            p["period"] = name 
            groups["1year"].append(p)
 
    results = {}
    for group_label, periods in groups.items():
        print(f"{group_label}: {len(periods)} periods found")
        results[group_label] = aggregate_periods(periods)
 
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(folder)),
        f"{instrument}_results.json"
    )
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
 
    print(f"\nSaved -> {out_path}")
 
 
if __name__ == "__main__":
    main()