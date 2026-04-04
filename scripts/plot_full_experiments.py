#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

ROOT = Path(__file__).resolve().parents[1]
CSV_ROOT = ROOT / "outputs" / "csv"
RUN_CSV = CSV_ROOT / "local_run_metrics.csv"
LEGACY_REF = CSV_ROOT / "legacy_overall_reference.csv"
PLOT_ROOT = ROOT / "outputs" / "plots" / "paper_full"
PLOT_TABLE_JSON = CSV_ROOT / "paper_full_plot_tables.json"
PLOT_TABLE_PY = CSV_ROOT / "paper_full_plot_tables.py"
PLOT_TABLE_SOURCES = CSV_ROOT / "paper_full_plot_sources.csv"

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"]
rcParams["font.size"] = 8
rcParams["axes.labelsize"] = 9
rcParams["axes.titlesize"] = 9
rcParams["xtick.labelsize"] = 8
rcParams["ytick.labelsize"] = 8
rcParams["legend.fontsize"] = 6

COLORS = {
    "ApxIChase": "#5B9BD5",
    "HeuIChase": "#ED7D31",
    "GNNExplainer": "#70AD47",
    "PGExplainer": "#E15759",
    "Exhaustive": "#A682B3",
}
HATCHES = {
    "ApxIChase": "///",
    "HeuIChase": "\\\\\\",
    "GNNExplainer": "xxx",
    "PGExplainer": "...",
    "Exhaustive": "+++",
}
MARKERS = {
    "ApxIChase": "^",
    "HeuIChase": "s",
    "GNNExplainer": "o",
    "PGExplainer": "x",
    "Exhaustive": "D",
}
LEGEND_LABELS = {
    "ApxIChase": "ApxC",
    "HeuIChase": "HeuC",
    "GNNExplainer": "GEX",
    "PGExplainer": "PGX",
    "Exhaustive": "Exh",
}
METHOD_MAP = {
    "ApxC": "ApxIChase",
    "HeuC": "HeuIChase",
    "GEX": "GNNExplainer",
    "PGX": "PGExplainer",
    "Exh": "Exhaustive",
}
METHOD_ORDER = ["ApxIChase", "HeuIChase", "GNNExplainer", "PGExplainer", "Exhaustive"]
FIG_WIDTH = 3.5
FIG_HEIGHT = 2.6


def load_runs() -> pd.DataFrame:
    df = pd.read_csv(RUN_CSV)
    if df.empty:
        raise ValueError(f"No run data in {RUN_CSV}")
    return df


def load_legacy() -> pd.DataFrame:
    if not LEGACY_REF.exists():
        return pd.DataFrame(columns=["figure", "x", "method", "value"])
    return pd.read_csv(LEGACY_REF)


def exp_rows(df: pd.DataFrame, experiment_name: str) -> pd.DataFrame:
    out = df[(df["experiment_name"] == experiment_name) & (df["status"] == "completed")].copy()
    out["legacy_method"] = out["method"].map(METHOD_MAP)
    return out


def first_available_experiment(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if ((df["experiment_name"] == name) & (df["status"] == "completed")).any():
            return name
    return None


def override_overall(
    legacy: pd.DataFrame,
    runs: pd.DataFrame,
    figure: str,
    experiment_name: str,
    metric_col: str,
    datasets: Iterable[str],
) -> pd.DataFrame:
    base = legacy[legacy["figure"] == figure].copy()
    if base.empty:
        base = pd.DataFrame(columns=["figure", "x_col", "x", "method", "value", "source", "note", "metric"])

    updates = exp_rows(runs, experiment_name)
    updates = updates[updates["dataset"].isin(list(datasets))]
    if updates.empty:
        return base

    override_rows: List[Dict[str, object]] = []
    for _, row in updates.iterrows():
        override_rows.append(
            {
                "figure": figure,
                "x_col": "Dataset",
                "x": row["dataset"],
                "method": row["legacy_method"],
                "value": row[metric_col],
                "source": "new_local",
                "note": experiment_name,
                "metric": metric_col,
            }
        )
    overrides = pd.DataFrame(override_rows)
    key_cols = ["figure", "x", "method"]
    dedup = {(r["figure"], r["x"], r["method"]) for _, r in overrides.iterrows()}
    base = base[[tuple(row[c] for c in key_cols) not in dedup for _, row in base.iterrows()]]
    return pd.concat([base, overrides], ignore_index=True)


def series_from_runs(df: pd.DataFrame, experiment_name: str, x_col: str, y_col: str, method_subset: Iterable[str] | None = None) -> pd.DataFrame:
    sub = exp_rows(df, experiment_name)
    if method_subset is not None:
        sub = sub[sub["legacy_method"].isin(list(method_subset))]
    keep = [x_col, y_col, "legacy_method"]
    if x_col == "model_key":
        sub = sub.copy()
        sub["model_key"] = sub["model_key"].map({"gcn2": "GCN_2", "gat2": "GAT_2", "sage2": "SAGE_2", "gcn1": "GCN_1", "gcn3": "GCN_3"}).fillna(sub["model_key"])
    return sub[keep].rename(columns={"legacy_method": "method", y_col: "value", x_col: "x"})


def plot_bar_chart(df: pd.DataFrame, x_values: List[str], ylabel: str, xlabel: str, filename: str, use_log: bool = False) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    width = 0.13
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    for i, method in enumerate(methods):
        offset = (i - len(methods) / 2 + 0.5) * width
        vals = []
        for x in x_values:
            row = df[(df["x"] == x) & (df["method"] == method)]
            vals.append(float(row["value"].iloc[0]) if not row.empty else np.nan)
        ax.bar(
            np.arange(len(x_values)) + offset,
            vals,
            width,
            label=LEGEND_LABELS.get(method, method),
            color=COLORS[method],
            hatch=HATCHES[method],
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels(x_values)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="gray")
    ax.set_axisbelow(True)
    if use_log:
        ax.set_yscale("log")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    plt.savefig(PLOT_ROOT / filename, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close()


def plot_line_chart(df: pd.DataFrame, x_values: List, ylabel: str, xlabel: str, filename: str, use_log: bool = False) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    for method in methods:
        sub = df[df["method"] == method].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("x")
        ax.plot(
            sub["x"],
            sub["value"],
            marker=MARKERS[method],
            color=COLORS[method],
            label=LEGEND_LABELS.get(method, method),
            linewidth=1.3,
            markersize=5.5,
            markerfacecolor="none",
            markeredgecolor=COLORS[method],
            markeredgewidth=1.3,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_values)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="gray")
    ax.set_axisbelow(True)
    if use_log:
        ax.set_yscale("log")
    ax.legend(loc="upper center", frameon=False, fontsize=7, ncol=min(5, len(methods)))
    plt.savefig(PLOT_ROOT / filename, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close()


def to_python_scalar(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value


def export_plot_table(
    figure_key: str,
    df: pd.DataFrame,
    x_values: List,
    x_label: str,
    plot_tables: Dict[str, Dict[str, List]],
    source_rows: List[Dict[str, object]],
) -> None:
    if df.empty:
        return
    table: Dict[str, List] = {x_label: [to_python_scalar(x) for x in x_values]}
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    for method in methods:
        values = []
        for x in x_values:
            row = df[(df["x"] == x) & (df["method"] == method)]
            value = row["value"].iloc[0] if not row.empty else np.nan
            values.append(to_python_scalar(value))
            if not row.empty:
                source = row["source"].iloc[0] if "source" in row.columns else "new_local"
                note = row["note"].iloc[0] if "note" in row.columns else ""
            else:
                source = "missing"
                note = ""
            source_rows.append(
                {
                    "figure": figure_key,
                    "x_label": x_label,
                    "x": x,
                    "method": method,
                    "source": source,
                    "note": note,
                    "value": np.nan if pd.isna(value) else value,
                }
            )
        table[method] = values
    plot_tables[figure_key] = table


def save_plot_tables(plot_tables: Dict[str, Dict[str, List]], source_rows: List[Dict[str, object]]) -> None:
    serializable = {}
    for key, table in plot_tables.items():
        serializable[key] = {}
        for col, values in table.items():
            serializable[key][col] = [None if (isinstance(v, float) and np.isnan(v)) else v for v in values]
    PLOT_TABLE_JSON.write_text(json.dumps(serializable, indent=2))

    with PLOT_TABLE_PY.open("w") as fh:
        fh.write("import numpy as np\n\n")
        for key in sorted(plot_tables):
            fh.write(f"{key} = {{\n")
            table = plot_tables[key]
            for col, values in table.items():
                rendered = []
                for value in values:
                    if isinstance(value, str):
                        rendered.append(repr(value))
                    elif isinstance(value, float) and np.isnan(value):
                        rendered.append("np.nan")
                    else:
                        rendered.append(repr(value))
                fh.write(f'    "{col}": [{", ".join(rendered)}],\n')
            fh.write("}\n\n")

    pd.DataFrame(source_rows).to_csv(PLOT_TABLE_SOURCES, index=False)


def main() -> None:
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    legacy = load_legacy()
    plot_tables: Dict[str, Dict[str, List]] = {}
    source_rows: List[Dict[str, object]] = []
    overall_source = first_available_experiment(runs, ["full_overall_efficiency", "full_overall_coverage", "full_overall_fidelity"])
    sigma_source = first_available_experiment(runs, ["full_runtime_vs_sigma_dblp", "full_coverage_vs_sigma_dblp"])
    l_source = first_available_experiment(runs, ["full_runtime_vs_L_dblp", "full_coverage_vs_L_dblp", "full_fidelity_vs_L_dblp"])
    incompleteness_source = first_available_experiment(runs, ["full_runtime_vs_incompleteness_dblp", "full_coverage_vs_incompleteness_dblp"])

    # Overall figures merged with legacy.
    if overall_source:
        overall_runtime = override_overall(legacy, runs, "figure_1", overall_source, "runtime_total", ["MUTAG", "Cora", "DBLP"])
        xvals = ["MUTAG", "ATLAS", "Cora", "DBLP", "BAShape"]
        export_plot_table("figure_1", overall_runtime, xvals, "Dataset", plot_tables, source_rows)
        plot_bar_chart(overall_runtime, xvals, "Total Run time (sec)", "Dataset", "figure_1_overall_efficiency.png", use_log=True)

        overall_coverage = override_overall(legacy, runs, "figure_7", overall_source, "coverage", ["MUTAG", "Cora", "DBLP"])
        export_plot_table("figure_7", overall_coverage, xvals, "Dataset", plot_tables, source_rows)
        plot_bar_chart(overall_coverage, xvals, "Coverage", "Dataset", "figure_7_overall_coverage.png", use_log=False)

        overall_fidelity = override_overall(legacy, runs, "figure_12", overall_source, "one_minus_fidelity_minus", ["MUTAG", "Cora", "DBLP"])
        export_plot_table("figure_12", overall_fidelity, xvals, "Dataset", plot_tables, source_rows)
        plot_bar_chart(overall_fidelity, xvals, "1 - Fidelity$^{-}$", "Dataset", "figure_12_overall_fidelity.png", use_log=False)

    # DBLP paper figures.
    if sigma_source:
        xvals = [10, 20, 30, 40, 50]
        df = series_from_runs(runs, sigma_source, "sigma_requested", "runtime_total")
        export_plot_table("figure_2", df, xvals, "|Σ|", plot_tables, source_rows)
        plot_line_chart(df, xvals, "Total Run time (sec)", "|Σ|", "figure_2_runtime_vs_constraint_size.png", use_log=True)
        df = series_from_runs(runs, sigma_source, "sigma_requested", "coverage")
        export_plot_table("figure_9", df, xvals, "Constraint_Size", plot_tables, source_rows)
        plot_line_chart(df, xvals, "Coverage", "|Σ|", "figure_9_coverage_vs_constraint_size.png", use_log=False)
    if ((runs["experiment_name"] == "full_runtime_vs_target_ratio_dblp") & (runs["status"] == "completed")).any():
        xvals = [0.01, 0.02, 0.03, 0.04, 0.05]
        df = series_from_runs(runs, "full_runtime_vs_target_ratio_dblp", "target_ratio", "runtime_total")
        export_plot_table("figure_3", df, xvals, "Target_Ratio", plot_tables, source_rows)
        plot_line_chart(df, xvals, "Total Run time (sec)", "Target Node Ratio", "figure_3_runtime_vs_target_ratio.png", use_log=True)
    if l_source:
        xvals = [1, 2, 3]
        df = series_from_runs(runs, l_source, "L", "runtime_total")
        export_plot_table("figure_4", df, xvals, "L", plot_tables, source_rows)
        plot_line_chart(df, xvals, "Total Run time (sec)", "L", "figure_4_runtime_vs_hops.png", use_log=True)
        df = series_from_runs(runs, l_source, "L", "coverage")
        export_plot_table("figure_10", df, xvals, "L", plot_tables, source_rows)
        plot_line_chart(df, xvals, "Coverage", "L", "figure_10_coverage_vs_hops.png", use_log=False)
        df = series_from_runs(runs, l_source, "L", "one_minus_fidelity_minus")
        export_plot_table("figure_20", df, xvals, "L", plot_tables, source_rows)
        plot_line_chart(df, xvals, "1 - Fidelity$^{-}$", "L", "figure_20_fidelity_vs_hops.png", use_log=False)
    if incompleteness_source:
        xvals = [0.05, 0.10, 0.15, 0.20]
        df = series_from_runs(runs, incompleteness_source, "incompleteness", "runtime_total")
        export_plot_table("figure_5", df, xvals, "Incompleteness", plot_tables, source_rows)
        plot_line_chart(df, xvals, "Total Run time (sec)", "Incompleteness", "figure_5_runtime_vs_incompleteness.png", use_log=True)
        df = series_from_runs(runs, incompleteness_source, "incompleteness", "coverage")
        export_plot_table("figure_11", df, xvals, "Incompleteness", plot_tables, source_rows)
        plot_line_chart(df, xvals, "Coverage", "Incompleteness", "figure_11_coverage_vs_incompleteness.png", use_log=False)
    if ((runs["experiment_name"] == "full_runtime_vs_model_dblp") & (runs["status"] == "completed")).any():
        xvals = ["GCN_2", "GAT_2", "SAGE_2"]
        df = series_from_runs(runs, "full_runtime_vs_model_dblp", "model_key", "runtime_total")
        export_plot_table("figure_6", df, xvals, "GNN_Type", plot_tables, source_rows)
        plot_bar_chart(df, xvals, "Total Run time (sec)", "GNN Architecture", "figure_6_runtime_gnn_types.png", use_log=True)

    if ((runs["experiment_name"] == "full_coverage_vs_k_dblp") & (runs["status"] == "completed")).any():
        xvals = [1, 2, 4, 6, 8]
        df = series_from_runs(runs, "full_coverage_vs_k_dblp", "k", "coverage")
        export_plot_table("figure_8", df, xvals, "k", plot_tables, source_rows)
        plot_line_chart(df, xvals, "Coverage", "k", "figure_8_coverage_vs_budget.png", use_log=False)

    gamma_methods = ["ApxIChase", "HeuIChase", "Exhaustive"]
    if ((runs["experiment_name"] == "full_fidelity_vs_gamma_dblp") & (runs["status"] == "completed")).any():
        xvals = [0.2, 0.4, 0.6, 0.8, 1.0]
        df = series_from_runs(runs, "full_fidelity_vs_gamma_dblp", "gamma", "one_minus_fidelity_minus", gamma_methods)
        export_plot_table("figure_17", df, xvals, "gamma", plot_tables, source_rows)
        plot_line_chart(df, xvals, "1 - Fidelity$^{-}$", "Gamma", "figure_17_fidelity_vs_gamma.png", use_log=False)
    if ((runs["experiment_name"] == "full_conciseness_vs_gamma_dblp") & (runs["status"] == "completed")).any():
        xvals = [0.2, 0.4, 0.6, 0.8, 1.0]
        df = series_from_runs(runs, "full_conciseness_vs_gamma_dblp", "gamma", "conciseness", gamma_methods)
        export_plot_table("figure_18", df, xvals, "gamma", plot_tables, source_rows)
        plot_line_chart(df, xvals, "Conciseness", "Gamma", "figure_18_conciseness_vs_gamma.png", use_log=False)
    if ((runs["experiment_name"] == "full_coverage_vs_gamma_dblp") & (runs["status"] == "completed")).any():
        xvals = [0.2, 0.4, 0.6, 0.8, 1.0]
        df = series_from_runs(runs, "full_coverage_vs_gamma_dblp", "gamma", "coverage", gamma_methods)
        export_plot_table("figure_19", df, xvals, "gamma", plot_tables, source_rows)
        plot_line_chart(df, xvals, "Coverage", "Gamma", "figure_19_coverage_vs_gamma.png", use_log=False)

    save_plot_tables(plot_tables, source_rows)
    print(f"Saved plots under {PLOT_ROOT}")


if __name__ == "__main__":
    main()
