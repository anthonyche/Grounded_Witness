#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import rcParams

from plot_full_experiments import (
    COLORS,
    FIG_HEIGHT,
    FIG_WIDTH,
    HATCHES,
    LEGEND_LABELS,
    MARKERS,
    METHOD_ORDER,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_JSON = ROOT / "scripts" / "manual_paper_tables.json"
PLOT_ROOT = ROOT / "outputs" / "plots" / "paper_manual"
PLOT_TABLE_JSON = ROOT / "outputs" / "csv" / "manual_paper_plot_tables.json"
PLOT_TABLE_PY = ROOT / "outputs" / "csv" / "manual_paper_plot_tables.py"
PLOT_TABLE_SOURCES = ROOT / "outputs" / "csv" / "manual_paper_plot_sources.csv"

METHOD_ALIASES = {
    "apxchase": "ApxIChase",
    "apxc": "ApxIChase",
    "heuchase": "HeuIChase",
    "heuc": "HeuIChase",
    "gnnexplainer": "GNNExplainer",
    "gex": "GNNExplainer",
    "pgexplainer": "PGExplainer",
    "pgx": "PGExplainer",
    "exhaustive": "Exhaustive",
    "exh": "Exhaustive",
}

rcParams["mathtext.fontset"] = "stix"

FIGURE_OVERRIDES: Dict[str, Dict[str, object]] = {
    "figure_1": {
        "figsize": (FIG_WIDTH, FIG_HEIGHT),
        "legend_ncol": 5,
        "legend_bbox": (0.5, 0.98),
        "legend_loc": "upper center",
        "legend_order": ["ApxIChase", "HeuIChase", "GNNExplainer", "PGExplainer", "Exhaustive"],
        "headroom_log": 6.0,
    },
    "figure_2": {
        "xlabel": r"$|\Sigma|$",
        "legend_ncol": 3,
        "legend_bbox": (0.5, 0.985),
        "legend_loc": "upper center",
        "headroom_log": 3.0,
    },
    "figure_3": {"legend_ncol": 3, "legend_bbox": (0.5, 0.985), "legend_loc": "upper center", "headroom_log": 3.0},
    "figure_4": {"legend_ncol": 3, "legend_bbox": (0.5, 0.985), "legend_loc": "upper center", "headroom_log": 3.2},
    "figure_5": {"legend_ncol": 3, "legend_bbox": (0.5, 0.985), "legend_loc": "upper center", "headroom_log": 3.0},
    "figure_6": {"figsize": (FIG_WIDTH, FIG_HEIGHT), "legend_ncol": 5, "legend_bbox": (0.5, 0.98), "legend_loc": "upper center", "headroom_log": 4.0},
    "figure_7": {
        "figsize": (FIG_WIDTH, FIG_HEIGHT),
        "legend_ncol": 5,
        "legend_bbox": (0.5, 0.98),
        "legend_loc": "upper center",
        "ylim": [0.0, 1.10],
        "yticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    },
    "figure_8": {"legend_ncol": 3, "legend_bbox": (0.5, 0.985), "legend_loc": "upper center", "headroom_linear": 1.35},
    "figure_9": {
        "xlabel": r"$\gamma$",
        "legend_ncol": 3,
        "legend_bbox": (0.5, 0.985),
        "legend_loc": "upper center",
        "headroom_linear": 1.35,
    },
    "figure_10": {"legend_ncol": 3, "legend_bbox": (0.5, 0.985), "legend_loc": "upper center", "headroom_linear": 1.35},
    "figure_11": {"legend_ncol": 3, "legend_bbox": (0.5, 0.985), "legend_loc": "upper center", "headroom_linear": 1.35},
    "figure_12": {
        "figsize": (FIG_WIDTH, FIG_HEIGHT),
        "legend_ncol": 5,
        "legend_bbox": (0.5, 0.985),
        "legend_loc": "upper center",
        "ylabel": "Fidelity score",
        "ylim": [0.0, 1.10],
        "yticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    },
    "figure_13": {"legend_ncol": 3, "legend_bbox": (0.5, 0.985), "legend_loc": "upper center", "headroom_log": 3.0},
    "figure_14": {"legend_ncol": 3, "legend_bbox": (0.5, 0.985), "legend_loc": "upper center", "headroom_log": 3.0},
    "figure_15": {
        "legend_ncol": 3,
        "legend_bbox": (0.5, 0.985),
        "legend_loc": "upper center",
        "headroom_log": 3.4,
        "timeout": {"method": "Exhaustive", "point_x": "20", "cross_x": ["4", "6", "8", "10"]},
    },
    "figure_16": {
        "legend_ncol": 3,
        "legend_bbox": (0.5, 0.985),
        "legend_loc": "upper center",
        "headroom_log": 3.4,
        "xlabel": "Query Load(# Target Nodes)",
        "timeout": {"method": "Exhaustive", "point_x": "100", "cross_x": ["200", "300", "400", "500"]},
    },
    "figure_17": {
        "xlabel": r"$\gamma$",
        "ylabel": "Fidelity score",
        "filename": "figure_17_fidelity_vs_gamma.png",
        "legend_ncol": 3,
        "legend_bbox": (0.5, 0.985),
        "legend_loc": "upper center",
        "ylim": [0.7, 1.05],
        "yticks": [0.7, 0.8, 0.9, 1.0],
    },
    "figure_18": {
        "xlabel": r"$\gamma$",
        "legend_ncol": 3,
        "legend_bbox": (0.5, 0.985),
        "legend_loc": "upper center",
        "ylim": [0.0, 1.10],
        "yticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    },
    "figure_19": {
        "xlabel": r"$|\Sigma|$",
        "legend_ncol": 3,
        "legend_bbox": (0.5, 0.985),
        "legend_loc": "upper center",
        "headroom_linear": 1.35,
    },
    "figure_20": {
        "ylabel": "Fidelity score",
        "legend_ncol": 3,
        "legend_bbox": (0.5, 0.985),
        "legend_loc": "upper center",
        "ylim": [0.45, 1.05],
    },
}


def canonical_method(name: str) -> str:
    key = str(name).strip().lower()
    return METHOD_ALIASES.get(key, str(name))


def load_specs() -> Dict[str, Dict[str, object]]:
    with DATA_JSON.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def to_float(value):
    if value is None:
        return np.nan
    return float(value)


def figure_frame(spec: Dict[str, object]) -> pd.DataFrame:
    x_values = [str(x) for x in spec["x"]]
    rows: List[Dict[str, object]] = []
    for raw_method, values in spec["series"].items():
        method = canonical_method(raw_method)
        for x, value in zip(x_values, values):
            rows.append(
                {
                    "x": x,
                    "method": method,
                    "value": to_float(value),
                    "source": "manual_table",
                    "note": spec.get("xlabel", ""),
                }
            )
    return pd.DataFrame(rows)


def _legend_handles_ordered(handles_map: Dict[str, object], figure_key: str) -> tuple[list, list]:
    override = FIGURE_OVERRIDES.get(figure_key, {})
    order = override.get("legend_order", [m for m in METHOD_ORDER if m in handles_map])
    methods = [m for m in order if m in handles_map]
    handles = [handles_map[m] for m in methods]
    labels = [LEGEND_LABELS.get(m, m) for m in methods]
    return handles, labels


def _style_for_figure(figure_key: str, spec: Dict[str, object]) -> Dict[str, object]:
    merged = dict(spec)
    merged.update(FIGURE_OVERRIDES.get(figure_key, {}))
    return merged


def _apply_legend(ax, handles_map: Dict[str, object], figure_key: str) -> None:
    handles, labels = _legend_handles_ordered(handles_map, figure_key)
    override = FIGURE_OVERRIDES.get(figure_key, {})
    if not handles:
        return
    ax.legend(
        handles,
        labels,
        loc=str(override.get("legend_loc", "upper center")),
        bbox_to_anchor=override.get("legend_bbox", (0.5, 0.985)),
        frameon=False,
        fontsize=6.8,
        ncol=int(override.get("legend_ncol", min(5, len(handles)))),
        columnspacing=1.0,
        handletextpad=0.5,
        borderaxespad=0.0,
    )


def _reserve_inner_headroom(ax, df: pd.DataFrame, figure_key: str, use_log: bool) -> None:
    finite = df["value"].dropna().astype(float)
    if finite.empty:
        return
    override = FIGURE_OVERRIDES.get(figure_key, {})
    explicit_ylim = override.get("ylim")
    if explicit_ylim is not None:
        ax.set_ylim(float(explicit_ylim[0]), float(explicit_ylim[1]))
        return
    ymax = float(finite.max())
    ymin = float(finite.min())
    if use_log:
        current_low, current_high = ax.get_ylim()
        ax.set_ylim(current_low, max(current_high, ymax * float(override.get("headroom_log", 3.0))))
    else:
        current_low, current_high = ax.get_ylim()
        span = max(ymax - min(0.0, ymin), 1e-9)
        proposed = ymax + span * (float(override.get("headroom_linear", 1.28)) - 1.0)
        ax.set_ylim(current_low, max(current_high, proposed))


def _timeout_overlay(ax, positions: np.ndarray, x_values: List[str], df: pd.DataFrame, figure_key: str, handles_map: Dict[str, object], use_log: bool) -> None:
    override = FIGURE_OVERRIDES.get(figure_key, {})
    timeout = override.get("timeout")
    if not timeout:
        return

    finite_values = df["value"].dropna().astype(float)
    if finite_values.empty:
        return
    ymax = float(finite_values.max())
    top_y = ymax * (1.25 if use_log else 1.08)
    x_index = {x: idx for idx, x in enumerate(x_values)}
    timeout_method = str(timeout.get("method", "Exhaustive"))

    point_x = str(timeout["point_x"])
    line_pos = []
    if point_x in x_index:
        line_pos.append(positions[x_index[point_x]])
        ax.scatter(
            [positions[x_index[point_x]]],
            [top_y],
            marker="D",
            s=28,
            facecolors="none",
            edgecolors=COLORS["Exhaustive"],
            linewidths=1.2,
            zorder=5,
        )

    cross_x = [str(x) for x in timeout.get("cross_x", [])]
    cross_pos = [positions[x_index[x]] for x in cross_x if x in x_index]
    line_pos.extend(cross_pos)
    if len(line_pos) >= 2:
        ax.plot(
            sorted(line_pos),
            [top_y] * len(line_pos),
            linestyle="--",
            linewidth=1.1,
            color=COLORS["Exhaustive"],
            zorder=4,
        )
    if cross_pos:
        ax.scatter(
            cross_pos,
            [top_y] * len(cross_pos),
            marker="x",
            s=42,
            color="red",
            linewidths=1.2,
            zorder=5,
        )
    handles_map[timeout_method] = Line2D(
        [0],
        [0],
        color=COLORS["Exhaustive"],
        linestyle="--",
        linewidth=1.1,
        marker="D",
        markerfacecolor="none",
        markeredgecolor=COLORS["Exhaustive"],
        markeredgewidth=1.2,
        markersize=5,
    )

    current_ylim = ax.get_ylim()
    upper = max(current_ylim[1], top_y * (1.15 if use_log else 1.08))
    ax.set_ylim(current_ylim[0], upper)


def _apply_ticks(ax, figure_key: str) -> None:
    override = FIGURE_OVERRIDES.get(figure_key, {})
    yticks = override.get("yticks")
    if yticks is not None:
        ax.set_yticks(list(yticks))


def plot_bar_chart(df: pd.DataFrame, x_values: List[str], ylabel: str, xlabel: str, filename: str, use_log: bool = False, figure_key: str = "") -> None:
    if df.empty:
        return
    override = FIGURE_OVERRIDES.get(figure_key, {})
    fig, ax = plt.subplots(figsize=override.get("figsize", (FIG_WIDTH, FIG_HEIGHT)))
    width = 0.13
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    positions = np.arange(len(x_values))
    handles_map: Dict[str, object] = {}
    for i, method in enumerate(methods):
        offset = (i - len(methods) / 2 + 0.5) * width
        vals = []
        for x in x_values:
            row = df[(df["x"] == x) & (df["method"] == method)]
            vals.append(float(row["value"].iloc[0]) if not row.empty else np.nan)
        bars = ax.bar(
            positions + offset,
            vals,
            width,
            color=COLORS[method],
            hatch=HATCHES[method],
            edgecolor="black",
            linewidth=0.5,
        )
        handles_map[method] = Patch(
            facecolor=COLORS[method],
            hatch=HATCHES[method],
            edgecolor="black",
            linewidth=0.5,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(x_values)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="gray")
    ax.set_axisbelow(True)
    if use_log:
        ax.set_yscale("log")
    _reserve_inner_headroom(ax, df, figure_key, use_log)
    _apply_ticks(ax, figure_key)
    _apply_legend(ax, handles_map, figure_key)
    plt.savefig(PLOT_ROOT / filename, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close()


def plot_line_chart(df: pd.DataFrame, x_values: List[str], ylabel: str, xlabel: str, filename: str, use_log: bool = False, figure_key: str = "") -> None:
    if df.empty:
        return
    override = FIGURE_OVERRIDES.get(figure_key, {})
    fig, ax = plt.subplots(figsize=override.get("figsize", (FIG_WIDTH, FIG_HEIGHT)))
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    x_index = {x: idx for idx, x in enumerate(x_values)}
    positions = np.arange(len(x_values))
    handles_map: Dict[str, object] = {}
    for method in methods:
        sub = df[df["method"] == method].copy()
        if sub.empty:
            continue
        ordered = []
        values = []
        for x in x_values:
            row = sub[sub["x"] == x]
            if row.empty:
                ordered.append(x_index[x])
                values.append(np.nan)
            else:
                ordered.append(x_index[x])
                values.append(float(row["value"].iloc[0]))
        (line,) = ax.plot(
            ordered,
            values,
            marker=MARKERS[method],
            color=COLORS[method],
            linewidth=1.3,
            markersize=5.5,
            markerfacecolor="none",
            markeredgecolor=COLORS[method],
            markeredgewidth=1.3,
        )
        handles_map[method] = line
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(x_values)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="gray")
    ax.set_axisbelow(True)
    if use_log:
        ax.set_yscale("log")
    _reserve_inner_headroom(ax, df, figure_key, use_log)
    _timeout_overlay(ax, positions, x_values, df, figure_key, handles_map, use_log)
    _apply_ticks(ax, figure_key)
    _apply_legend(ax, handles_map, figure_key)
    plt.savefig(PLOT_ROOT / filename, dpi=300, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close()


def export_plot_table(
    figure_key: str,
    df: pd.DataFrame,
    x_values: List[str],
    x_label: str,
    plot_tables: Dict[str, Dict[str, List]],
    source_rows: List[Dict[str, object]],
) -> None:
    table: Dict[str, List] = {x_label: list(x_values)}
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    for method in methods:
        values: List[object] = []
        for x in x_values:
            row = df[(df["x"] == x) & (df["method"] == method)]
            value = row["value"].iloc[0] if not row.empty else np.nan
            values.append(None if pd.isna(value) else float(value))
            source_rows.append(
                {
                    "figure": figure_key,
                    "x_label": x_label,
                    "x": x,
                    "method": method,
                    "source": "manual_table",
                    "note": "",
                    "value": np.nan if pd.isna(value) else float(value),
                }
            )
        table[method] = values
    plot_tables[figure_key] = table


def save_plot_tables(plot_tables: Dict[str, Dict[str, List]], source_rows: List[Dict[str, object]]) -> None:
    PLOT_TABLE_JSON.write_text(json.dumps(plot_tables, indent=2), encoding="utf-8")
    with PLOT_TABLE_PY.open("w", encoding="utf-8") as handle:
        handle.write("manual_paper_plot_tables = ")
        json.dump(plot_tables, handle, indent=2)
        handle.write("\n")
    pd.DataFrame(source_rows).to_csv(PLOT_TABLE_SOURCES, index=False)


def main() -> None:
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    PLOT_TABLE_JSON.parent.mkdir(parents=True, exist_ok=True)
    specs = load_specs()
    plot_tables: Dict[str, Dict[str, List]] = {}
    source_rows: List[Dict[str, object]] = []

    for figure_key in sorted(specs.keys(), key=lambda s: int(s.split("_")[1])):
        spec = _style_for_figure(figure_key, specs[figure_key])
        df = figure_frame(spec)
        x_values = [str(x) for x in spec["x"]]
        export_plot_table(figure_key, df, x_values, str(spec["xlabel"]), plot_tables, source_rows)
        if spec["kind"] == "bar":
            plot_bar_chart(
                df,
                x_values,
                str(spec["ylabel"]),
                str(spec["xlabel"]),
                str(spec["filename"]),
                use_log=bool(spec.get("use_log", False)),
                figure_key=figure_key,
            )
        else:
            plot_line_chart(
                df,
                x_values,
                str(spec["ylabel"]),
                str(spec["xlabel"]),
                str(spec["filename"]),
                use_log=bool(spec.get("use_log", False)),
                figure_key=figure_key,
            )

    save_plot_tables(plot_tables, source_rows)
    print(f"Saved manual paper plots under {PLOT_ROOT}")


if __name__ == "__main__":
    main()
