"""
hmecyst_topology_analysis.py
-----------------------------
Spatial topology analysis: HMECyst control vs cyst-forming organoids.

Scientific goal: validate that synthetic organoids produced by the generator
can be used as simulation tools -- demonstrating that topology features
reliably discriminate tissue architecture (solid spheroid vs hollow shell)
in the same way as published real-organoid analyses.

Input:
    output/features/hmecyst_control_*_features.csv   (one file per organoid)
    output/features/hmecyst_cyst_*_features.csv      (one file per organoid)

Output:
    analysis/hmecyst_control_vs_cyst/
        combined_cells.csv          all cells with organoid_id column
        organoid_summaries.csv      one row per organoid (median of each feature)
        statistical_results.csv     per-feature Mann-Whitney + Cohen's d
        fig1_cell_counts.png        cells per organoid per condition
        fig2_radial_distribution.png radial position histograms
        fig3_topology_violins.png   top topology features per organoid
        fig4_spatial_scatter.png    2D centroid scatter coloured by zone
        fig5_feature_correlations.png correlation heatmap
        fig6_roc_curves.png         classifier ROC (organoid-level)
        fig7_feature_importance.png consensus feature importance

Usage:
    python hmecyst_topology_analysis.py

Edit the CONFIGURATION section below before running.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from pathlib import Path
import warnings
import re

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

FEATURES_DIR = "output/features"          # where segment_organoids.py wrote CSVs
OUTPUT_DIR   = "analysis/hmecyst_control_vs_cyst"

# Condition A prefix -- any CSV whose stem starts with this = condition A
CONDITION_A_PREFIX = "hmecyst_control"
CONDITION_A_LABEL  = "Control"

# Condition B prefix
CONDITION_B_PREFIX = "hmecyst_cyst"
CONDITION_B_LABEL  = "Cyst"

# Topology features to analyse -- these are the columns from the pipeline
# that capture spatial organisation rather than morphology or intensity.
# We include the spatial coordinate features too since zone and radial_dist_norm
# are ground truth for the cyst shell architecture.
#
# NOTE ON RADIUS_UM: topology features were pre-computed in segment_organoids.py
# using RADIUS_UM defined there. For HMECyst cells (mean radius ~13 um,
# inter-cell distance ~26 um) you need RADIUS_UM >= 35 in segment_organoids.py
# to reliably capture neighbours. Re-run segmentation with RADIUS_UM = 40
# if n_nuclei_neighbors is near 0 for most cells.
TOPOLOGY_FEATURES = [
    "n_nuclei_neighbors",
    "nb_nuclei_neighbors_ripley",
    "distance_to_neighbors_mean_um",
    "distance_to_neighbors_min_um",
    "distance_to_neighbors_max_um",
    "local_density_per_mm3",
    "crystal_distance_um",
    "radial_dist_norm",
]

# Additional morphology features to include in the organoid summary
# (not topology per se but useful context for the manuscript)
MORPHOLOGY_FEATURES = [
    "cell_volume_um3",
    "nuclei_volume_um3",
    "cell_elongation",
    "cell_roundedness",
    "CV_chromatin",
    "major_axis_um",
    "medium_axis_um",
    "minor_axis_um",
    "prolate_ratio",
    "oblate_ratio",
]

ALL_FEATURES = TOPOLOGY_FEATURES + MORPHOLOGY_FEATURES

# Statistical significance threshold.
# FDR (Benjamini-Hochberg) correction is used -- more appropriate than
# Bonferroni when n per condition is small (n=5 gives minimum Mann-Whitney
# p=0.0079, making Bonferroni for 18 features literally unreachable).
ALPHA = 0.05

# Nature Methods style -- minimal, clean
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

# Colours -- consistent throughout
COL_A   = "#4878CF"   # control -- blue
COL_B   = "#D65F5F"   # cyst -- red
COL_MAP = None        # set after labels are known

# ============================================================
# END CONFIGURATION
# ============================================================

plt.rcParams.update(NM_STYLE)

COND_A = CONDITION_A_LABEL
COND_B = CONDITION_B_LABEL
PALETTE = {COND_A: COL_A, COND_B: COL_B}


# ── helpers ──────────────────────────────────────────────────────────

def _log(msg):
    print(msg, flush=True)


def _organoid_id_from_path(path: Path) -> str:
    """Short readable ID from filename, e.g. 'hmecyst_control_seed42'."""
    stem = path.stem.replace("_features", "")
    # Keep up to the timestamp (drop the long datetime suffix)
    m = re.match(r"(.+?_seed\d+)", stem)
    return m.group(1) if m else stem


def _condition_from_path(path: Path,
                          prefix_a: str, prefix_b: str,
                          label_a: str, label_b: str) -> str | None:
    stem = path.stem.lower()
    if stem.startswith(prefix_a.lower()):
        return label_a
    if stem.startswith(prefix_b.lower()):
        return label_b
    return None


def _cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = np.sqrt(((na - 1) * a.std(ddof=1)**2 +
                      (nb - 1) * b.std(ddof=1)**2) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else 0.0


def _sig_label(p, threshold):
    if p < threshold / 100:
        return "***"
    if p < threshold / 10:
        return "**"
    if p < threshold:
        return "*"
    return "ns"


def _save(fig, name, out_dir):
    path = out_dir / name
    fig.savefig(str(path))
    plt.close(fig)
    _log(f"  Saved: {path.name}")


# ── step 1: load data ────────────────────────────────────────────────

def load_all_csvs(features_dir: Path,
                  prefix_a: str, prefix_b: str,
                  label_a: str, label_b: str) -> pd.DataFrame:
    _log("\n[1/7] Loading feature CSVs...")

    csvs = sorted(features_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {features_dir}")

    frames = []
    counts = {label_a: 0, label_b: 0}

    for path in csvs:
        cond = _condition_from_path(path, prefix_a, prefix_b, label_a, label_b)
        if cond is None:
            continue
        oid = _organoid_id_from_path(path)
        df  = pd.read_csv(str(path))
        df["organoid_id"] = oid
        df["condition"]   = cond
        frames.append(df)
        counts[cond] += 1
        _log(f"  {cond:10s}  {oid}  ({len(df)} cells)")

    if not frames:
        raise ValueError(
            f"No matching CSVs found for prefixes "
            f"'{prefix_a}' / '{prefix_b}' in {features_dir}"
        )

    combined = pd.concat(frames, ignore_index=True)

    _log(f"\n  {label_a}: {counts[label_a]} organoid(s)")
    _log(f"  {label_b}: {counts[label_b]} organoid(s)")
    _log(f"  Total cells: {len(combined)}")

    if counts[label_a] < 2 or counts[label_b] < 2:
        _log("\n  WARNING: fewer than 2 organoids per condition. "
             "Statistical tests will run but p-values are not meaningful "
             "with n < 2. Generate more replicates for a valid comparison.")

    return combined


# ── step 2: organoid-level summaries ────────────────────────────────

def compute_organoid_summaries(cells: pd.DataFrame,
                                features: list[str]) -> pd.DataFrame:
    _log("\n[2/7] Computing per-organoid summaries (median of each feature)...")

    # Only keep features that are present
    present = [f for f in features if f in cells.columns]
    missing = set(features) - set(present)
    if missing:
        _log(f"  Note: features not in CSV (skipped): {missing}")

    agg = (cells
           .groupby(["organoid_id", "condition"])[present]
           .median()
           .reset_index())

    # Also store cell count per organoid
    counts = cells.groupby("organoid_id").size().rename("n_cells")
    agg = agg.merge(counts, on="organoid_id")

    _log(f"  {len(agg)} organoid summaries  |  {len(present)} features")
    return agg


# ── step 3: statistical comparison ──────────────────────────────────

def run_statistics(summaries: pd.DataFrame,
                   features: list[str],
                   label_a: str, label_b: str) -> pd.DataFrame:
    _log("\n[3/7] Statistical comparison (Mann-Whitney U + FDR correction)...")

    present = [f for f in features if f in summaries.columns]
    grp_a   = summaries[summaries["condition"] == label_a]
    grp_b   = summaries[summaries["condition"] == label_b]
    results = []

    for feat in present:
        a = grp_a[feat].dropna()
        b = grp_b[feat].dropna()

        if len(a) < 2 or len(b) < 2:
            p = np.nan
        else:
            _, p = mannwhitneyu(a, b, alternative="two-sided")

        d = _cohens_d(a, b)

        # Guard against numerical overflow in Cohen's d when a feature has
        # near-zero variance in one group (e.g. local_density or n_neighbours
        # when RADIUS_UM was too small in segmentation so most values are 0).
        if np.isfinite(d) and abs(d) > 50:
            _log(f"  WARNING: |Cohen's d| = {abs(d):.0f} for '{feat}' "
                 f"-- degenerate distribution (all-zero?). Setting d=NaN. "
                 f"If this is a neighbour feature, re-run segmentation with "
                 f"larger RADIUS_UM.")
            d = np.nan

        results.append({
            "feature":           feat,
            f"median_{label_a}": float(a.median()) if len(a) else np.nan,
            f"median_{label_b}": float(b.median()) if len(b) else np.nan,
            "cohens_d":          d,
            "p_value":           p,
        })

    df = pd.DataFrame(results)

    # FDR (Benjamini-Hochberg) correction on non-NaN p-values.
    # With n=5 per group the minimum achievable Mann-Whitney p is 0.0079.
    # Bonferroni at 18 features requires p < 0.0028 -- unreachable. FDR is
    # the standard alternative for small-n multiple testing.
    valid = df["p_value"].notna()
    if valid.sum() > 0:
        reject, p_fdr, _, _ = multipletests(
            df.loc[valid, "p_value"].values,
            alpha=ALPHA,
            method="fdr_bh",
        )
        df.loc[valid, "p_fdr"]      = p_fdr
        df.loc[valid, "significant"] = reject
    else:
        df["p_fdr"]       = np.nan
        df["significant"] = False

    df["significant"] = df["significant"].fillna(False).astype(bool)

    def _fdr_label(row):
        if pd.isna(row["p_fdr"]):
            return "?"
        p = row["p_fdr"]
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < ALPHA: return "*"
        return "ns"

    df["sig_label"] = df.apply(_fdr_label, axis=1)

    df = df.sort_values("cohens_d", key=lambda s: s.abs(), ascending=False,
                        na_position="last").reset_index(drop=True)

    n_sig = df["significant"].sum()
    _log(f"  FDR (Benjamini-Hochberg) correction at alpha={ALPHA}")
    _log(f"  Significant features (FDR < {ALPHA}): {n_sig} / {len(df)}")
    _log("\n  Top 5 by |Cohen's d|:")
    for _, row in df.head(5).iterrows():
        d_str = f"{row['cohens_d']:+.3f}" if pd.notna(row["cohens_d"]) else "   NaN"
        p_str = f"{row['p_value']:.3e}"   if pd.notna(row["p_value"])  else "    NaN"
        _log(f"    {row['feature']:<40s}  d={d_str}  "
             f"p_raw={p_str}  {row['sig_label']}")

    return df


# ── plots ────────────────────────────────────────────────────────────

def fig1_cell_counts(summaries: pd.DataFrame, out_dir: Path):
    """Bar plot: cells per organoid, grouped by condition."""
    _log("  fig1: cell counts per organoid")

    orgs_a = summaries[summaries["condition"] == COND_A].sort_values("organoid_id")
    orgs_b = summaries[summaries["condition"] == COND_B].sort_values("organoid_id")

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    x_a = np.arange(len(orgs_a))
    x_b = np.arange(len(orgs_b)) + len(orgs_a) + 0.6

    ax.bar(x_a, orgs_a["n_cells"], color=COL_A, width=0.7, label=COND_A)
    ax.bar(x_b, orgs_b["n_cells"], color=COL_B, width=0.7, label=COND_B)

    all_x    = list(x_a) + list(x_b)
    all_labs = ([o.split("_seed")[1][:2] + "s" + str(i+1)
                 for i, o in enumerate(orgs_a["organoid_id"])] +
                [o.split("_seed")[1][:2] + "s" + str(i+1)
                 for i, o in enumerate(orgs_b["organoid_id"])])

    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labs, rotation=45, ha="right")
    ax.set_ylabel("Cells per organoid")
    ax.set_title("Segmented cell count per organoid")
    ax.legend(frameon=False)

    _save(fig, "fig1_cell_counts.png", out_dir)


def fig2_radial_distribution(cells: pd.DataFrame, out_dir: Path):
    """Histogram of radial_dist_norm per condition -- key validation for cyst."""
    _log("  fig2: radial distribution")

    if "radial_dist_norm" not in cells.columns:
        _log("  Skipped (radial_dist_norm not in data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5), sharey=False)

    for ax, cond, col in zip(axes, [COND_A, COND_B], [COL_A, COL_B]):
        data = cells[cells["condition"] == cond]["radial_dist_norm"].dropna()
        ax.hist(data, bins=40, color=col, alpha=0.85, edgecolor="none",
                density=True)
        ax.axvline(0.78, color="grey", lw=0.8, ls="--",
                   label="lumen boundary (0.78)")
        ax.set_xlabel("Radial position (0=centre, 1=surface)")
        ax.set_ylabel("Density")
        ax.set_title(cond)
        ax.legend(frameon=False, fontsize=5)

    fig.suptitle("Radial distribution of cells", y=1.02)
    plt.tight_layout()
    _save(fig, "fig2_radial_distribution.png", out_dir)


def fig3_topology_violins(summaries: pd.DataFrame,
                           stats: pd.DataFrame,
                           top_n: int,
                           out_dir: Path):
    """
    Violin plots for top topology features.
    Each point = one organoid median.
    Significance brackets from organoid-level Mann-Whitney.
    """
    _log("  fig3: topology feature violins")

    # select top N topology features by |Cohen's d|
    topo_stats = stats[stats["feature"].isin(TOPOLOGY_FEATURES)].head(top_n)
    feats = topo_stats["feature"].tolist()
    if not feats:
        _log("  Skipped (no topology features in stats)")
        return

    ncols = min(4, len(feats))
    nrows = int(np.ceil(len(feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 1.6, nrows * 2.2))
    axes = np.array(axes).flatten()

    for idx, feat in enumerate(feats):
        ax = axes[idx]
        row = topo_stats[topo_stats["feature"] == feat].iloc[0]

        for j, (cond, col) in enumerate([(COND_A, COL_A), (COND_B, COL_B)]):
            vals = summaries[summaries["condition"] == cond][feat].dropna()
            # violin body
            if len(vals) > 2:
                vp = ax.violinplot([vals], positions=[j],
                                   widths=0.55, showmedians=False,
                                   showextrema=False)
                for body in vp["bodies"]:
                    body.set_facecolor(col)
                    body.set_alpha(0.45)
                    body.set_edgecolor("none")
            # individual organoid points
            ax.scatter([j] * len(vals), vals, color=col, s=18,
                       zorder=3, linewidths=0)
            # median line
            ax.hlines(vals.median(), j - 0.2, j + 0.2,
                      color=col, lw=1.5, zorder=4)

        # significance bracket
        if not np.isnan(row["p_value"]):
            y_max = summaries[feat].max()
            y_rng = summaries[feat].max() - summaries[feat].min()
            br_y  = y_max + 0.08 * y_rng
            ax.plot([0, 1], [br_y, br_y], color="black", lw=0.8)
            ax.text(0.5, br_y + 0.02 * y_rng, row["sig_label"],
                    ha="center", va="bottom", fontsize=7)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([COND_A, COND_B], rotation=30, ha="right")
        ax.set_ylabel(feat.replace("_um", " (µm)")
                         .replace("_per_mm3", " (/mm³)")
                         .replace("_", " "), fontsize=5.5)
        ax.set_title(f"d = {row['cohens_d']:+.2f}", fontsize=6, pad=2)

    for ax in axes[len(feats):]:
        ax.set_visible(False)

    fig.suptitle(f"Top {len(feats)} topology features -- organoid medians",
                 y=1.01, fontsize=8)
    plt.tight_layout()
    _save(fig, "fig3_topology_violins.png", out_dir)


def fig4_spatial_scatter(cells: pd.DataFrame, out_dir: Path):
    """
    2D XY scatter of cell centroids coloured by zone.
    One panel per organoid (up to 6 shown), two rows: control / cyst.
    Validates that cyst organoids show the hollow shell architecture.
    """
    _log("  fig4: spatial scatter coloured by zone")

    coord_cols = {"centroid_x_rel_um", "centroid_y_rel_um", "zone"}
    if not coord_cols.issubset(cells.columns):
        _log("  Skipped (centroid columns not in data)")
        return

    zone_colours = {
        "core":         "#4878CF",
        "intermediate": "#6ACC65",
        "periphery":    "#D65F5F",
    }

    for cond, col_prefix in [(COND_A, COL_A), (COND_B, COL_B)]:
        orgs = sorted(cells[cells["condition"] == cond]["organoid_id"].unique())
        show = orgs[:6]
        ncols = min(3, len(show))
        nrows = int(np.ceil(len(show) / ncols))

        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(ncols * 2.2, nrows * 2.2),
                                  squeeze=False)

        for idx, oid in enumerate(show):
            ax   = axes[idx // ncols][idx % ncols]
            data = cells[cells["organoid_id"] == oid]
            for zone, zc in zone_colours.items():
                sub = data[data["zone"] == zone]
                ax.scatter(sub["centroid_x_rel_um"], sub["centroid_y_rel_um"],
                           c=zc, s=4, alpha=0.6, linewidths=0, label=zone)
            ax.set_aspect("equal")
            ax.set_title(oid.split("_seed")[1][:5], fontsize=6)
            ax.set_xlabel("x (µm)", fontsize=5)
            ax.set_ylabel("y (µm)", fontsize=5)
            ax.tick_params(labelsize=5)

        for idx in range(len(show), nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        legend_elems = [Line2D([0], [0], marker="o", color="w",
                                markerfacecolor=zc, markersize=5, label=z)
                         for z, zc in zone_colours.items()]
        fig.legend(handles=legend_elems, loc="lower center",
                   ncol=3, frameon=False, fontsize=5,
                   bbox_to_anchor=(0.5, -0.02))

        fig.suptitle(f"Cell spatial distribution -- {cond}", fontsize=8)
        plt.tight_layout()
        fname = f"fig4_spatial_scatter_{cond.lower().replace(' ', '_')}.png"
        _save(fig, fname, out_dir)


def fig5_feature_correlations(summaries: pd.DataFrame, out_dir: Path):
    """Correlation heatmap across all features at organoid level."""
    _log("  fig5: feature correlations")

    present = [f for f in ALL_FEATURES if f in summaries.columns]
    corr = summaries[present].corr()

    short = [f.replace("distance_to_neighbors_", "nbr_dist_")
              .replace("_per_mm3", "/mm³")
              .replace("_um", "")
              .replace("_", " ")[:28]
              for f in present]

    n = len(present)
    fig, ax = plt.subplots(figsize=(max(4, n * 0.38), max(3.5, n * 0.38)))

    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1,
                   aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Pearson r")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=60, ha="right", fontsize=4.5)
    ax.set_yticklabels(short, fontsize=4.5)
    ax.set_title("Feature correlation matrix (organoid medians)", fontsize=7)

    plt.tight_layout()
    _save(fig, "fig5_feature_correlations.png", out_dir)


def fig6_roc_curves(summaries: pd.DataFrame,
                     stats: pd.DataFrame,
                     out_dir: Path):
    """
    Classifier ROC curves at organoid level.
    Uses leave-one-out cross-validation (appropriate for small n).
    Features: all topology features that are significant or top 7 by |d|.
    Three classifiers: Logistic Regression, Random Forest (small),
    and a single-feature benchmark (top feature by |d|).
    """
    _log("  fig6: classifier ROC curves")

    present = [f for f in TOPOLOGY_FEATURES if f in summaries.columns]
    if not present:
        _log("  Skipped (no topology features available)")
        return

    X_df = summaries[present].fillna(summaries[present].median())
    X    = X_df.values
    y    = (summaries["condition"] == COND_B).astype(int).values
    n    = len(summaries)

    if n < 4:
        _log("  Skipped ROC (fewer than 4 organoids total -- need more replicates)")
        return

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # leave-one-out predictions
    cv = StratifiedKFold(n_splits=min(n, 5), shuffle=True, random_state=42)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000,
                                                   class_weight="balanced",
                                                   random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=50,
                                                       max_depth=3,
                                                       class_weight="balanced",
                                                       random_state=42),
    }

    fig, ax = plt.subplots(figsize=(3.2, 3.2))

    for name, clf in classifiers.items():
        col = COL_A if "Logistic" in name else COL_B
        probas = np.zeros(n)
        for train_idx, test_idx in cv.split(X_sc, y):
            clf.fit(X_sc[train_idx], y[train_idx])
            probas[test_idx] = clf.predict_proba(X_sc[test_idx])[:, 1]
        fpr, tpr, _ = roc_curve(y, probas)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=1.2,
                label=f"{name} (AUC={roc_auc:.2f})")

    # single-feature benchmark -- top topology feature by |d|
    top_feat_row = stats[stats["feature"].isin(present)].head(1)
    if not top_feat_row.empty:
        tf = top_feat_row.iloc[0]["feature"]
        x1 = summaries[tf].fillna(summaries[tf].median()).values.reshape(-1, 1)
        x1s = scaler.fit_transform(x1)
        bf_probas = np.zeros(n)
        bf_clf = LogisticRegression(max_iter=1000, class_weight="balanced",
                                    random_state=42)
        for train_idx, test_idx in cv.split(x1s, y):
            bf_clf.fit(x1s[train_idx], y[train_idx])
            bf_probas[test_idx] = bf_clf.predict_proba(x1s[test_idx])[:, 1]
        fpr1, tpr1, _ = roc_curve(y, bf_probas)
        auc1 = auc(fpr1, tpr1)
        ax.plot(fpr1, tpr1, color="grey", lw=1.0, ls="--",
                label=f"Top feature only ({tf[:18]})\n(AUC={auc1:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.7, label="Random (AUC=0.50)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Classifier ROC\n{COND_A} vs {COND_B}  (organoid level)")
    ax.legend(frameon=False, fontsize=5, loc="lower right")

    plt.tight_layout()
    _save(fig, "fig6_roc_curves.png", out_dir)


def fig7_feature_importance(summaries: pd.DataFrame,
                              stats: pd.DataFrame,
                              out_dir: Path):
    """
    Horizontal bar chart: consensus feature importance across
    Logistic Regression and Random Forest, ranked by |Cohen's d|.
    Points are coloured by which condition has the higher median.
    """
    _log("  fig7: feature importance")

    present = [f for f in ALL_FEATURES if f in summaries.columns]
    if len(present) < 2:
        _log("  Skipped (insufficient features)")
        return

    X = summaries[present].fillna(summaries[present].median()).values
    y = (summaries["condition"] == COND_B).astype(int).values

    if len(summaries) < 4:
        _log("  Skipped feature importance (need >= 4 organoids)")
        return

    scaler  = StandardScaler()
    X_sc    = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced",
                             random_state=42)
    rf = RandomForestClassifier(n_estimators=100, max_depth=4,
                                 class_weight="balanced", random_state=42)
    lr.fit(X_sc, y)
    rf.fit(X_sc, y)

    lr_imp = np.abs(lr.coef_[0])
    rf_imp = rf.feature_importances_

    # normalise to [0,1] then average
    lr_norm = lr_imp / (lr_imp.max() + 1e-9)
    rf_norm = rf_imp / (rf_imp.max() + 1e-9)
    consensus = (lr_norm + rf_norm) / 2

    imp_df = pd.DataFrame({
        "feature":    present,
        "consensus":  consensus,
        "lr":         lr_norm,
        "rf":         rf_norm,
    }).sort_values("consensus", ascending=True)

    # colour bar by which condition is higher
    bar_colours = []
    for f in imp_df["feature"]:
        stat_row = stats[stats["feature"] == f]
        if stat_row.empty:
            bar_colours.append("grey")
            continue
        d = stat_row.iloc[0]["cohens_d"]
        bar_colours.append(COL_A if d > 0 else COL_B)

    short_names = [f.replace("distance_to_neighbors_", "nbr_dist_")
                    .replace("_per_mm3", "/mm³")
                    .replace("_um", " (µm)")
                    .replace("_", " ")
                    for f in imp_df["feature"]]

    fig, ax = plt.subplots(figsize=(4.5, max(2.5, len(present) * 0.3)))
    y_pos = np.arange(len(imp_df))
    ax.barh(y_pos, imp_df["consensus"], color=bar_colours, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_names, fontsize=5.5)
    ax.set_xlabel("Consensus importance (normalised)")
    ax.set_title("Feature importance\n(LR + RF consensus)")

    legend_elems = [
        Line2D([0], [0], color=COL_A, lw=5, alpha=0.85,
               label=f"Higher in {COND_A}"),
        Line2D([0], [0], color=COL_B, lw=5, alpha=0.85,
               label=f"Higher in {COND_B}"),
    ]
    ax.legend(handles=legend_elems, frameon=False, fontsize=5,
              loc="lower right")

    plt.tight_layout()
    _save(fig, "fig7_feature_importance.png", out_dir)


# ── main ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  HMECyst Control vs Cyst -- Topology Analysis")
    print("=" * 60)

    features_dir = Path(FEATURES_DIR)
    out_dir      = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    cells = load_all_csvs(
        features_dir,
        CONDITION_A_PREFIX, CONDITION_B_PREFIX,
        COND_A, COND_B,
    )
    cells.to_csv(out_dir / "combined_cells.csv", index=False)

    # 2. Organoid summaries
    summaries = compute_organoid_summaries(cells, ALL_FEATURES)
    summaries.to_csv(out_dir / "organoid_summaries.csv", index=False)

    # 3. Statistics
    stats = run_statistics(summaries, ALL_FEATURES, COND_A, COND_B)
    stats.to_csv(out_dir / "statistical_results.csv", index=False)

    # 4-10. Figures
    _log("\n[4/7] Generating figures...")
    fig1_cell_counts(summaries, out_dir)
    fig2_radial_distribution(cells, out_dir)
    fig3_topology_violins(summaries, stats, top_n=8, out_dir=out_dir)
    fig4_spatial_scatter(cells, out_dir)
    fig5_feature_correlations(summaries, out_dir)
    fig6_roc_curves(summaries, stats, out_dir)
    fig7_feature_importance(summaries, stats, out_dir)

    # Summary print
    print("\n" + "=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)
    n_org = summaries.groupby("condition").size()
    for cond, n in n_org.items():
        print(f"  {cond}: {n} organoid(s)")

    sig = stats[stats["significant"]]
    print(f"\n  Significant features (FDR BH): {len(sig)} / {len(stats)}")
    if not sig.empty:
        print("  Top significant topology features:")
        topo_sig = sig[sig["feature"].isin(TOPOLOGY_FEATURES)]
        for _, row in topo_sig.head(5).iterrows():
            higher = COND_A if row["cohens_d"] > 0 else COND_B
            print(f"    {row['feature']:<40s}  "
                  f"d={row['cohens_d']:+.2f}  "
                  f"p={row['p_value']:.2e}  "
                  f"higher in {higher}")

    print(f"\n  Outputs written to: {out_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()