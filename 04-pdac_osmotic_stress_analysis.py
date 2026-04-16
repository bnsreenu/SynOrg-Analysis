"""
pdac_osmotic_stress_analysis.py
--------------------------------
Morphology analysis: PDAC isotonic vs hypertonic (osmotic stress).

Scientific goal: validate that synthetic PDAC organoids correctly reproduce
the morphological response to osmotic stress -- primarily chromatin compaction
(CV_chromatin decrease in hypertonic) with minimal shape changes, consistent
with published real-organoid observations.

Analysis is conducted at the organoid level (one row per organoid = median
of all cells) to respect the non-independence of cells within an organoid.
Statistical comparison uses Mann-Whitney U with FDR correction, appropriate
for n=5 per condition.

Input:
    output/features/pdac_isotonic_*_features.csv
    output/features/pdac_hypertonic_*_features.csv

Output:
    analysis/pdac_osmotic_stress/
        combined_cells.csv
        organoid_summaries.csv
        statistical_results.csv
        fig1_cell_counts.png
        fig2_feature_histograms.png
        fig3_violin_organoid_level.png
        fig4_effect_sizes.png
        fig5_radial_profiles.png
        fig6_feature_correlations.png

Usage:
    python pdac_osmotic_stress_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from scipy.stats import mannwhitneyu, shapiro, levene
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import warnings
import re

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES_DIR = "output/features"
OUTPUT_DIR   = "analysis/pdac_osmotic_stress"

CONDITION_A_PREFIX = "pdac_isotonic"
CONDITION_A_LABEL  = "Isotonic"

CONDITION_B_PREFIX = "pdac_hypertonic"
CONDITION_B_LABEL  = "Hypertonic"

# All morphology + intensity features to compare.
# Topology features are excluded -- osmotic stress affects cell biology,
# not packing architecture, so topology is not the relevant signal here.
MORPHOLOGY_FEATURES = [
    "cell_volume_um3",
    "nuclei_volume_um3",
    "cell_elongation",
    "cell_roundedness",
    "CV_chromatin",
    "avg_intensity_nuclear",
    "std_intensity_nuclear",
    "max_intensity_nuclear",
    "prolate_ratio",
    "oblate_ratio",
    "major_axis_um",
    "medium_axis_um",
    "minor_axis_um",
]

# Topology features -- included in summaries for completeness but
# not the primary focus of this analysis.
TOPOLOGY_FEATURES = [
    "n_nuclei_neighbors",
    "distance_to_neighbors_mean_um",
    "local_density_per_mm3",
    "crystal_distance_um",
    "radial_dist_norm",
]

ALL_FEATURES = MORPHOLOGY_FEATURES + TOPOLOGY_FEATURES

ALPHA = 0.05

# Nature Methods style
NM_STYLE = {
    "font.family":       "Arial",
    "font.size":         7,
    "axes.titlesize":    8,
    "axes.labelsize":    7,
    "xtick.labelsize":   6,
    "ytick.labelsize":   6,
    "legend.fontsize":   6,
    "axes.linewidth":    0.75,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
}

COL_A   = "#4878CF"   # isotonic -- blue
COL_B   = "#D65F5F"   # hypertonic -- red

plt.rcParams.update(NM_STYLE)

COND_A  = CONDITION_A_LABEL
COND_B  = CONDITION_B_LABEL
PALETTE = {COND_A: COL_A, COND_B: COL_B}

# ============================================================
# END CONFIGURATION
# ============================================================


def _log(msg):
    print(msg, flush=True)


def _organoid_id(path: Path) -> str:
    stem = path.stem.replace("_features", "")
    m = re.match(r"(.+?_seed\d+)", stem)
    return m.group(1) if m else stem


def _condition(path: Path) -> str | None:
    s = path.stem.lower()
    if s.startswith(CONDITION_A_PREFIX.lower()): return COND_A
    if s.startswith(CONDITION_B_PREFIX.lower()): return COND_B
    return None


def _cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = np.sqrt(((na - 1) * np.var(a, ddof=1) +
                      (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


def _effect_label(d):
    ad = abs(d)
    if ad < 0.2:  return "negligible"
    if ad < 0.5:  return "small"
    if ad < 0.8:  return "medium"
    return "large"


def _save(fig, name, out_dir):
    p = out_dir / name
    fig.savefig(str(p))
    plt.close(fig)
    _log(f"  Saved: {p.name}")


# ── data loading ─────────────────────────────────────────────────────

def load_all_csvs(features_dir: Path) -> pd.DataFrame:
    _log("\n[1/6] Loading feature CSVs...")
    frames = []
    counts = {COND_A: 0, COND_B: 0}

    for path in sorted(features_dir.glob("*.csv")):
        cond = _condition(path)
        if cond is None:
            continue
        oid = _organoid_id(path)
        df  = pd.read_csv(str(path))
        df["organoid_id"] = oid
        df["condition"]   = cond
        frames.append(df)
        counts[cond] += 1
        _log(f"  {cond:12s}  {oid}  ({len(df)} cells)")

    if not frames:
        raise FileNotFoundError(
            f"No matching CSVs for prefixes "
            f"'{CONDITION_A_PREFIX}' / '{CONDITION_B_PREFIX}' in {features_dir}")

    combined = pd.concat(frames, ignore_index=True)
    _log(f"\n  {COND_A}: {counts[COND_A]} organoid(s)")
    _log(f"  {COND_B}: {counts[COND_B]} organoid(s)")
    _log(f"  Total cells: {len(combined)}")

    if counts[COND_A] < 2 or counts[COND_B] < 2:
        _log("\n  WARNING: fewer than 2 organoids per condition. "
             "Stats will run but p-values are not meaningful with n<2.")
    return combined


def compute_organoid_summaries(cells: pd.DataFrame) -> pd.DataFrame:
    _log("\n[2/6] Computing per-organoid summaries (median)...")
    present = [f for f in ALL_FEATURES if f in cells.columns]
    agg = (cells.groupby(["organoid_id", "condition"])[present]
                .median().reset_index())
    counts = cells.groupby("organoid_id").size().rename("n_cells")
    agg = agg.merge(counts, on="organoid_id")
    _log(f"  {len(agg)} organoid summaries  |  {len(present)} features")
    return agg


# ── statistics ───────────────────────────────────────────────────────

def run_statistics(summaries: pd.DataFrame) -> pd.DataFrame:
    _log("\n[3/6] Statistical comparison (Mann-Whitney U + FDR)...")
    present = [f for f in ALL_FEATURES if f in summaries.columns]
    grp_a   = summaries[summaries["condition"] == COND_A]
    grp_b   = summaries[summaries["condition"] == COND_B]
    rows    = []

    for feat in present:
        a = grp_a[feat].dropna()
        b = grp_b[feat].dropna()
        p = np.nan
        if len(a) >= 2 and len(b) >= 2:
            _, p = mannwhitneyu(a, b, alternative="two-sided")
        d = _cohens_d(a.values, b.values)
        if np.isfinite(d) and abs(d) > 50:
            _log(f"  WARNING: |d|={abs(d):.0f} for '{feat}' -- degenerate. Setting NaN.")
            d = np.nan
        rows.append({
            "feature":           feat,
            f"median_{COND_A}":  float(a.median()) if len(a) else np.nan,
            f"median_{COND_B}":  float(b.median()) if len(b) else np.nan,
            "cohens_d":          d,
            "effect_size":       _effect_label(d) if pd.notna(d) else "?",
            "p_value":           p,
        })

    df = pd.DataFrame(rows)

    valid = df["p_value"].notna()
    if valid.sum() > 0:
        reject, p_fdr, _, _ = multipletests(
            df.loc[valid, "p_value"].values, alpha=ALPHA, method="fdr_bh")
        df.loc[valid, "p_fdr"]      = p_fdr
        df.loc[valid, "significant"] = reject
    else:
        df["p_fdr"] = np.nan
        df["significant"] = False

    df["significant"] = df["significant"].fillna(False).astype(bool)

    def _sig(row):
        if pd.isna(row.get("p_fdr")): return "?"
        p = row["p_fdr"]
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < ALPHA: return "*"
        return "ns"

    df["sig_label"] = df.apply(_sig, axis=1)
    df = df.sort_values("cohens_d", key=lambda s: s.abs(),
                        ascending=False, na_position="last").reset_index(drop=True)

    n_sig = df["significant"].sum()
    _log(f"  FDR (BH) at alpha={ALPHA}")
    _log(f"  Significant: {n_sig} / {len(df)}")
    _log("\n  Top 5 by |Cohen's d|:")
    for _, row in df.head(5).iterrows():
        d_s = f"{row['cohens_d']:+.3f}" if pd.notna(row["cohens_d"]) else "NaN"
        p_s = f"{row['p_value']:.3e}"   if pd.notna(row["p_value"])  else "NaN"
        _log(f"    {row['feature']:<38s}  d={d_s}  p={p_s}  {row['sig_label']}")
    return df


# ── figures ──────────────────────────────────────────────────────────

def fig1_cell_counts(summaries: pd.DataFrame, out_dir: Path):
    _log("  fig1: cell counts")
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for cond, col in [(COND_A, COL_A), (COND_B, COL_B)]:
        sub = summaries[summaries["condition"] == cond].sort_values("organoid_id")
        x   = np.arange(len(sub))
        offset = 0 if cond == COND_A else len(summaries[summaries["condition"] == COND_A]) + 0.6
        ax.bar(x + offset, sub["n_cells"], color=col, width=0.7, label=cond)
    ax.set_ylabel("Cells per organoid")
    ax.set_title("Segmented cell count")
    ax.legend(frameon=False)
    plt.tight_layout()
    _save(fig, "fig1_cell_counts.png", out_dir)


def fig2_feature_histograms(cells: pd.DataFrame, out_dir: Path):
    """Cell-level histograms -- useful for seeing full distributions."""
    _log("  fig2: feature histograms (cell level)")
    feats  = [f for f in MORPHOLOGY_FEATURES if f in cells.columns]
    ncols  = 3
    nrows  = int(np.ceil(len(feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 2.5, nrows * 2.2))
    axes = np.array(axes).flatten()

    for idx, feat in enumerate(feats):
        ax = axes[idx]
        for cond, col in [(COND_A, COL_A), (COND_B, COL_B)]:
            data = cells[cells["condition"] == cond][feat].dropna()
            ax.hist(data, bins=40, color=col, alpha=0.6,
                    density=True, label=cond, edgecolor="none")
        ax.set_xlabel(feat.replace("_um", " (µm)")
                         .replace("_um3", " (µm³)")
                         .replace("_", " "), fontsize=5.5)
        ax.set_ylabel("Density", fontsize=5.5)
        ax.set_title(feat.replace("_", " ")[:30], fontsize=6)
        ax.tick_params(labelsize=5)

    for ax in axes[len(feats):]:
        ax.set_visible(False)

    handles = [mlines.Line2D([], [], color=c, lw=4, label=l)
               for l, c in [(COND_A, COL_A), (COND_B, COL_B)]]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Feature distributions (cell level): {COND_A} vs {COND_B}",
                 fontsize=8, y=1.01)
    plt.tight_layout()
    _save(fig, "fig2_feature_histograms.png", out_dir)


def fig3_violin_organoid_level(summaries: pd.DataFrame,
                                stats: pd.DataFrame,
                                out_dir: Path):
    """Violin + organoid-level points for all morphology features."""
    _log("  fig3: organoid-level violins")
    feats = [f for f in MORPHOLOGY_FEATURES
             if f in summaries.columns][:12]
    ncols = 4
    nrows = int(np.ceil(len(feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 1.8, nrows * 2.4))
    axes = np.array(axes).flatten()

    for idx, feat in enumerate(feats):
        ax  = axes[idx]
        row = stats[stats["feature"] == feat]
        sig = row.iloc[0]["sig_label"] if not row.empty else "?"
        d   = row.iloc[0]["cohens_d"]  if not row.empty else np.nan

        for j, (cond, col) in enumerate([(COND_A, COL_A), (COND_B, COL_B)]):
            vals = summaries[summaries["condition"] == cond][feat].dropna()
            if len(vals) > 2:
                vp = ax.violinplot([vals], positions=[j],
                                   widths=0.55, showmedians=False,
                                   showextrema=False)
                for body in vp["bodies"]:
                    body.set_facecolor(col)
                    body.set_alpha(0.4)
                    body.set_edgecolor("none")
            ax.scatter([j] * len(vals), vals, color=col,
                       s=20, zorder=3, linewidths=0)
            ax.hlines(vals.median(), j - 0.2, j + 0.2,
                      color=col, lw=1.5, zorder=4)

        if sig not in ("ns", "?") and not summaries[feat].empty:
            y_max = summaries[feat].max()
            y_rng = summaries[feat].max() - summaries[feat].min()
            ax.plot([0, 1], [y_max + 0.08 * y_rng] * 2, color="k", lw=0.8)
            ax.text(0.5, y_max + 0.10 * y_rng, sig,
                    ha="center", va="bottom", fontsize=7)

        d_str = f"d={d:+.2f}" if pd.notna(d) else ""
        ax.set_xticks([0, 1])
        ax.set_xticklabels([COND_A, COND_B], rotation=30, ha="right")
        ax.set_ylabel(feat.replace("_um3", " (µm³)")
                         .replace("_um", " (µm)")
                         .replace("_", " "), fontsize=5.5)
        ax.set_title(d_str, fontsize=6, pad=2)

    for ax in axes[len(feats):]:
        ax.set_visible(False)

    fig.suptitle(f"Morphology features -- organoid medians: {COND_A} vs {COND_B}",
                 fontsize=8, y=1.01)
    plt.tight_layout()
    _save(fig, "fig3_violin_organoid_level.png", out_dir)


def fig4_effect_sizes(stats: pd.DataFrame, out_dir: Path):
    """Horizontal bar chart of Cohen's d for all morphology features."""
    _log("  fig4: effect sizes")
    df = stats[stats["feature"].isin(MORPHOLOGY_FEATURES)].copy()
    df = df.dropna(subset=["cohens_d"])
    df = df.sort_values("cohens_d", ascending=True)

    short = [f.replace("_um3", "").replace("_um", "")
              .replace("_", " ")[:32]
              for f in df["feature"]]
    bar_colours = [COL_A if d > 0 else COL_B for d in df["cohens_d"]]

    fig, ax = plt.subplots(figsize=(4.5, max(2.5, len(df) * 0.35)))
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["cohens_d"], color=bar_colours, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short, fontsize=5.5)
    ax.axvline(0,    color="k",    lw=0.8)
    ax.axvline(0.5,  color="grey", lw=0.6, ls="--", alpha=0.6)
    ax.axvline(-0.5, color="grey", lw=0.6, ls="--", alpha=0.6)
    ax.axvline(0.8,  color="grey", lw=0.6, ls=":",  alpha=0.6)
    ax.axvline(-0.8, color="grey", lw=0.6, ls=":",  alpha=0.6)

    # significance markers
    for i, (_, row) in enumerate(df.iterrows()):
        if row["sig_label"] not in ("ns", "?"):
            x_pos = row["cohens_d"] + (0.05 if row["cohens_d"] >= 0 else -0.05)
            ax.text(x_pos, i, row["sig_label"],
                    va="center", ha="left" if row["cohens_d"] >= 0 else "right",
                    fontsize=6)

    ax.set_xlabel("Cohen's d  (positive = higher in " + COND_A + ")")
    ax.set_title(f"Effect sizes: {COND_A} vs {COND_B}")

    legend_h = [
        mlines.Line2D([], [], color=COL_A, lw=5, alpha=0.85,
                      label=f"Higher in {COND_A}"),
        mlines.Line2D([], [], color=COL_B, lw=5, alpha=0.85,
                      label=f"Higher in {COND_B}"),
    ]
    ax.legend(handles=legend_h, frameon=False, fontsize=5, loc="lower right")
    plt.tight_layout()
    _save(fig, "fig4_effect_sizes.png", out_dir)


def fig5_radial_profiles(cells: pd.DataFrame, out_dir: Path):
    """
    Feature vs radial position -- key for PDAC organoids.
    Shows how each morphology feature changes from core to periphery
    within each condition, and whether the two conditions differ radially.
    """
    _log("  fig5: radial profiles")
    if "radial_dist_norm" not in cells.columns:
        _log("  Skipped (radial_dist_norm not in data)")
        return

    # Key features for radial analysis
    rad_feats = [f for f in [
        "CV_chromatin", "cell_volume_um3", "n_nuclei_neighbors",
        "cell_elongation", "prolate_ratio", "avg_intensity_nuclear",
    ] if f in cells.columns]

    bins = np.linspace(0, 1.1, 12)
    labels = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]

    ncols = 3
    nrows = int(np.ceil(len(rad_feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 2.5, nrows * 2.2))
    axes = np.array(axes).flatten()

    for idx, feat in enumerate(rad_feats):
        ax = axes[idx]
        for cond, col in [(COND_A, COL_A), (COND_B, COL_B)]:
            sub = cells[cells["condition"] == cond].copy()
            sub["rbin"] = pd.cut(sub["radial_dist_norm"], bins=bins, labels=labels)
            profile = sub.groupby("rbin")[feat].median()
            ax.plot(profile.index.astype(float), profile.values,
                    color=col, lw=1.2, marker="o", ms=3, label=cond)
        ax.axvline(0.5,  color="grey", lw=0.6, ls="--", alpha=0.5)
        ax.axvline(0.75, color="grey", lw=0.6, ls="--", alpha=0.5)
        ax.set_xlabel("Radial position (0=centre, 1=surface)", fontsize=5.5)
        ax.set_ylabel(feat.replace("_", " ")[:25], fontsize=5.5)
        ax.set_title(feat.replace("_", " ")[:28], fontsize=6)
        ax.tick_params(labelsize=5)
        ax.legend(frameon=False, fontsize=5)

    for ax in axes[len(rad_feats):]:
        ax.set_visible(False)

    fig.suptitle("Radial feature profiles", fontsize=8, y=1.01)
    plt.tight_layout()
    _save(fig, "fig5_radial_profiles.png", out_dir)


def fig6_feature_correlations(summaries: pd.DataFrame, out_dir: Path):
    """Correlation heatmap at organoid level."""
    _log("  fig6: feature correlations")
    present = [f for f in ALL_FEATURES if f in summaries.columns]
    corr    = summaries[present].corr()
    short   = [f.replace("_per_mm3", "/mm³").replace("_um", "")
                .replace("_", " ")[:25] for f in present]
    n = len(present)
    fig, ax = plt.subplots(figsize=(max(4, n * 0.38), max(3.5, n * 0.38)))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Pearson r")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=60, ha="right", fontsize=4.5)
    ax.set_yticklabels(short, fontsize=4.5)
    ax.set_title("Feature correlations (organoid medians)", fontsize=7)
    plt.tight_layout()
    _save(fig, "fig6_feature_correlations.png", out_dir)


# ── main ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  PDAC Osmotic Stress -- Morphology Analysis")
    print("=" * 60)

    features_dir = Path(FEATURES_DIR)
    out_dir      = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells     = load_all_csvs(features_dir)
    cells.to_csv(out_dir / "combined_cells.csv", index=False)

    summaries = compute_organoid_summaries(cells)
    summaries.to_csv(out_dir / "organoid_summaries.csv", index=False)

    stats = run_statistics(summaries)
    stats.to_csv(out_dir / "statistical_results.csv", index=False)

    _log("\n[4/6] Generating figures...")
    fig1_cell_counts(summaries, out_dir)
    fig2_feature_histograms(cells, out_dir)
    fig3_violin_organoid_level(summaries, stats, out_dir)
    fig4_effect_sizes(stats, out_dir)
    fig5_radial_profiles(cells, out_dir)
    fig6_feature_correlations(summaries, out_dir)

    print("\n" + "=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)
    sig = stats[stats["significant"]]
    print(f"\n  Significant features (FDR BH): {len(sig)} / {len(stats)}")
    if not sig.empty:
        print("  Top significant morphology features:")
        for _, row in sig[sig["feature"].isin(MORPHOLOGY_FEATURES)].head(5).iterrows():
            higher = COND_A if row["cohens_d"] > 0 else COND_B
            print(f"    {row['feature']:<38s}  "
                  f"d={row['cohens_d']:+.2f}  "
                  f"p={row['p_value']:.2e}  "
                  f"higher in {higher}")
    print(f"\n  Outputs written to: {out_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
