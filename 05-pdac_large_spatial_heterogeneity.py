"""
pdac_large_spatial_heterogeneity.py
-------------------------------------
Spatial heterogeneity analysis of a single large PDAC organoid.

Scientific goal: validate that the large synthetic PDAC organoid produces
a biologically realistic continuous radial gradient rather than discrete
abrupt subpopulations -- consistent with published observations in real
primary PDAC organoids showing a continuous spectrum from core to periphery.

Key analyses:
  1. Radial gradient profiles -- how each feature changes from core to surface
  2. Shape heterogeneity (prolate/oblate) -- 3D shape distribution of cells
  3. Unsupervised clustering (k-means) with zone validation -- do discovered
     clusters map to radial zones (core / intermediate / periphery)?
  4. PCA coloured by zone -- continuous vs discrete spectrum
  5. Feature importance for predicting radial zone

Input:
    output/features/pdac_large_clustering_*_features.csv
    (uses the first file found if multiple exist; for a genuine
    single-organoid analysis, generate one organoid)

Output:
    analysis/pdac_large_heterogeneity/
        fig1_radial_profiles.png
        fig2_prolate_oblate_scatter.png
        fig3_pca_by_zone.png
        fig4_cluster_selection.png
        fig5_clusters_pca.png
        fig6_cluster_zone_heatmap.png
        fig7_cluster_feature_violins.png
        fig8_feature_importance.png
        cell_data_with_clusters.csv
        statistical_results_anova.csv

Usage:
    python pdac_large_spatial_heterogeneity.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from pathlib import Path
import warnings
import re

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

FEATURES_DIR  = "output/features"
OUTPUT_DIR    = "analysis/pdac_large_heterogeneity"

# Prefix to identify the large PDAC organoid CSV(s).
# If multiple files match, they are concatenated and an organoid_id
# column distinguishes them. For true single-organoid analysis,
# generate one file.
CONDITION_PREFIX = "pdac_large_clustering"

# k-means: set to None to auto-select via silhouette (tests k=2..MAX_K)
N_CLUSTERS = 3
MAX_K      = 8

# Features used for clustering and PCA.
# We use the full morphology + intensity + topology set.
CLUSTER_FEATURES = [
    "cell_volume_um3",
    "nuclei_volume_um3",
    "cell_elongation",
    "cell_roundedness",
    "CV_chromatin",
    "avg_intensity_nuclear",
    "std_intensity_nuclear",
    "prolate_ratio",
    "oblate_ratio",
    "major_axis_um",
    "medium_axis_um",
    "minor_axis_um",
    "n_nuclei_neighbors",
    "local_density_per_mm3",
    "crystal_distance_um",
    "distance_to_neighbors_mean_um",
]

# Nature Methods style
NM_STYLE = {
    "font.family":        "Arial",
    "font.size":          7,
    "axes.titlesize":     8,
    "axes.labelsize":     7,
    "xtick.labelsize":    6,
    "ytick.labelsize":    6,
    "legend.fontsize":    6,
    "axes.linewidth":     0.75,
    "xtick.major.width":  0.75,
    "ytick.major.width":  0.75,
    "xtick.major.size":   3,
    "ytick.major.size":   3,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
}

ZONE_COLOURS = {
    "core":         "#4878CF",
    "intermediate": "#6ACC65",
    "periphery":    "#D65F5F",
}
CLUSTER_CMAP = plt.cm.get_cmap("tab10")

plt.rcParams.update(NM_STYLE)

# ============================================================
# END CONFIGURATION
# ============================================================


def _log(msg):
    print(msg, flush=True)


def _save(fig, name, out_dir):
    p = out_dir / name
    fig.savefig(str(p))
    plt.close(fig)
    _log(f"  Saved: {p.name}")


# ── data loading ─────────────────────────────────────────────────────

def load_data(features_dir: Path) -> pd.DataFrame:
    _log("\n[1/8] Loading feature CSVs...")
    csvs = [p for p in sorted(features_dir.glob("*.csv"))
            if p.stem.lower().startswith(CONDITION_PREFIX.lower())]

    if not csvs:
        raise FileNotFoundError(
            f"No CSVs matching prefix '{CONDITION_PREFIX}' in {features_dir}")

    frames = []
    for path in csvs:
        stem = path.stem.replace("_features", "")
        m    = re.match(r"(.+?_seed\d+)", stem)
        oid  = m.group(1) if m else stem
        df   = pd.read_csv(str(path))
        df["organoid_id"] = oid
        frames.append(df)
        _log(f"  {oid}  ({len(df)} cells)")

    cells = pd.concat(frames, ignore_index=True)
    _log(f"  Total: {len(cells)} cells across {len(frames)} organoid(s)")

    if "zone" not in cells.columns:
        raise ValueError("'zone' column not found. Re-run extract_features.py.")
    if "radial_dist_norm" not in cells.columns:
        raise ValueError("'radial_dist_norm' column not found. "
                         "Re-run extract_features.py.")
    return cells


# ── preprocessing ────────────────────────────────────────────────────

def prepare_features(cells: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Impute missing values and standardize. Returns (X_scaled, feat_names, X_raw).
    """
    present = [f for f in CLUSTER_FEATURES if f in cells.columns]
    missing = set(CLUSTER_FEATURES) - set(present)
    if missing:
        _log(f"  Note: features missing from CSV (skipped): {missing}")

    X_raw = cells[present].values
    imp   = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_raw)
    sc    = StandardScaler()
    X_sc  = sc.fit_transform(X_imp)
    return X_sc, present, X_imp


# ── figures ──────────────────────────────────────────────────────────

def fig1_radial_profiles(cells: pd.DataFrame, out_dir: Path):
    """
    Median feature value binned by radial position.
    Key validation: shows a continuous gradient rather than abrupt steps.
    """
    _log("  fig1: radial profiles")
    feat_list = [f for f in [
        "CV_chromatin", "cell_volume_um3", "n_nuclei_neighbors",
        "cell_elongation", "prolate_ratio", "avg_intensity_nuclear",
        "local_density_per_mm3", "crystal_distance_um",
    ] if f in cells.columns]

    bins   = np.linspace(0, 1.1, 14)
    mids   = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
    cells  = cells.copy()
    cells["rbin"] = pd.cut(cells["radial_dist_norm"], bins=bins, labels=mids)

    ncols = 4
    nrows = int(np.ceil(len(feat_list) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.2))
    axes = np.array(axes).flatten()

    for idx, feat in enumerate(feat_list):
        ax      = axes[idx]
        profile = cells.groupby("rbin")[feat].median()
        sem     = cells.groupby("rbin")[feat].sem()
        x       = profile.index.astype(float)
        ax.plot(x, profile.values, color="#2C5F8A", lw=1.5, marker="o", ms=3)
        ax.fill_between(x,
                        profile.values - sem.values,
                        profile.values + sem.values,
                        alpha=0.2, color="#2C5F8A")
        for v, label in [(0.5, "core|int"), (0.75, "int|periph")]:
            ax.axvline(v, color="grey", lw=0.6, ls="--", alpha=0.6)
        ax.set_xlabel("Radial position", fontsize=5.5)
        ax.set_ylabel(feat.replace("_", " ")[:22], fontsize=5.5)
        ax.set_title(feat.replace("_", " ")[:28], fontsize=6)
        ax.tick_params(labelsize=5)

    for ax in axes[len(feat_list):]:
        ax.set_visible(False)

    fig.suptitle("Radial feature gradients -- large PDAC organoid",
                 fontsize=8, y=1.01)
    plt.tight_layout()
    _save(fig, "fig1_radial_profiles.png", out_dir)


def fig2_prolate_oblate_scatter(cells: pd.DataFrame, out_dir: Path):
    """
    Prolate vs oblate scatter coloured by radial zone.
    Validates that cell shape varies continuously with position.
    """
    _log("  fig2: prolate/oblate scatter")
    if "prolate_ratio" not in cells.columns or "oblate_ratio" not in cells.columns:
        _log("  Skipped (prolate/oblate not in data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.8))

    # left: coloured by zone
    ax = axes[0]
    for zone, col in ZONE_COLOURS.items():
        sub = cells[cells["zone"] == zone]
        ax.scatter(sub["prolate_ratio"], sub["oblate_ratio"],
                   c=col, s=3, alpha=0.4, linewidths=0, label=zone)
    ax.set_xlabel("Prolate ratio")
    ax.set_ylabel("Oblate ratio")
    ax.set_title("Shape by zone")
    ax.legend(frameon=False, fontsize=5, markerscale=2)

    # right: coloured by radial_dist_norm (continuous)
    ax = axes[1]
    sc = ax.scatter(cells["prolate_ratio"], cells["oblate_ratio"],
                    c=cells["radial_dist_norm"], cmap="RdYlBu_r",
                    s=3, alpha=0.5, linewidths=0, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="Radial position")
    ax.set_xlabel("Prolate ratio")
    ax.set_ylabel("Oblate ratio")
    ax.set_title("Shape by radial position (continuous)")

    fig.suptitle("3D shape heterogeneity", fontsize=8)
    plt.tight_layout()
    _save(fig, "fig2_prolate_oblate_scatter.png", out_dir)


def fig3_pca_by_zone(X_sc: np.ndarray, X_pca: np.ndarray,
                     cells: pd.DataFrame, out_dir: Path):
    """
    PCA coloured by radial zone.
    A continuous spectrum (not discrete blobs) validates the continuous
    gradient biology we encoded in the large PDAC preset.
    X_pca is pre-computed in main() and shared across all PCA figures
    so coordinates are consistent.
    """
    _log("  fig3: PCA coloured by zone and radial position")
    pca   = PCA(n_components=2, random_state=42)
    pca.fit(X_sc)                          # fit only, to get explained variance
    ev    = pca.explained_variance_ratio_  # X_pca already computed in main()

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

    # zone colours
    ax = axes[0]
    for zone, col in ZONE_COLOURS.items():
        mask = cells["zone"] == zone
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=col, s=3, alpha=0.5, linewidths=0, label=zone)
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.set_title("PCA -- by zone")
    ax.legend(frameon=False, fontsize=5, markerscale=2)

    # continuous radial position
    ax = axes[1]
    sc = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                    c=cells["radial_dist_norm"].values,
                    cmap="RdYlBu_r", s=3, alpha=0.5, linewidths=0,
                    vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, shrink=0.8, label="Radial position")
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.set_title("PCA -- continuous radial position")

    total_ev = ev.sum() * 100
    fig.suptitle(f"PCA of {len(CLUSTER_FEATURES)} features "
                 f"({total_ev:.1f}% variance)", fontsize=8)
    plt.tight_layout()
    _save(fig, "fig3_pca_by_zone.png", out_dir)


def fig4_cluster_selection(X_sc: np.ndarray, out_dir: Path) -> int:
    """
    Elbow + silhouette to choose k. Returns recommended k.
    If N_CLUSTERS is set in config, that value is used directly and
    this figure is still produced for the record.
    """
    _log("  fig4: cluster selection (elbow + silhouette)")
    k_range  = range(2, MAX_K + 1)
    inertias = []
    silhs    = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X_sc)
        inertias.append(km.inertia_)
        silhs.append(silhouette_score(X_sc, lbl))

    best_k = k_range[int(np.argmax(silhs))]
    chosen = N_CLUSTERS if N_CLUSTERS is not None else best_k

    fig, axes = plt.subplots(1, 2, figsize=(6, 2.5))

    axes[0].plot(k_range, inertias, "bo-", lw=1.2, ms=5)
    axes[0].axvline(chosen, color="red", lw=0.8, ls="--",
                    label=f"chosen k={chosen}")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow"); axes[0].legend(frameon=False, fontsize=5)

    axes[1].plot(k_range, silhs, "go-", lw=1.2, ms=5)
    axes[1].axvline(chosen, color="red", lw=0.8, ls="--",
                    label=f"chosen k={chosen}")
    axes[1].axhline(0.5, color="grey", lw=0.6, ls=":", alpha=0.6)
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette"); axes[1].legend(frameon=False, fontsize=5)

    plt.tight_layout()
    _save(fig, "fig4_cluster_selection.png", out_dir)

    _log(f"  Best k by silhouette: {best_k}  |  Using k={chosen}")
    return chosen


def run_kmeans(X_sc: np.ndarray, cells: pd.DataFrame,
               k: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Fit k-means and attach cluster labels to cells DataFrame."""
    km  = KMeans(n_clusters=k, random_state=42, n_init=10)
    lbl = km.fit_predict(X_sc)
    sil = silhouette_score(X_sc, lbl)
    cells = cells.copy()
    cells["cluster"] = lbl

    _log(f"  k-means k={k}  silhouette={sil:.3f}")
    for c in range(k):
        n = (lbl == c).sum()
        _log(f"    Cluster {c}: {n} cells ({100*n/len(lbl):.1f}%)")

    if sil < 0.3:
        _log("  NOTE: silhouette < 0.3 -- clusters capture a continuous "
             "gradient, not discrete populations. This is the expected "
             "biology for the large PDAC organoid.")
    return cells, lbl


def fig5_clusters_pca(X_sc: np.ndarray, X_pca: np.ndarray,
                      cells: pd.DataFrame, k: int, out_dir: Path):
    """PCA coloured by k-means cluster assignment."""
    _log("  fig5: clusters on PCA")
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    for c in range(k):
        mask = cells["cluster"] == c
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[CLUSTER_CMAP(c / max(k-1, 1))],
                   s=3, alpha=0.6, linewidths=0, label=f"Cluster {c}")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"k-means clusters (k={k})")
    ax.legend(frameon=False, fontsize=5, markerscale=2)
    plt.tight_layout()
    _save(fig, "fig5_clusters_pca.png", out_dir)


def fig6_cluster_zone_heatmap(cells: pd.DataFrame, k: int, out_dir: Path):
    """
    Crosstab: cluster vs radial zone.
    Validates whether k-means clusters correspond to anatomical zones.
    A smooth diagonal pattern = continuous gradient captured by clustering.
    """
    _log("  fig6: cluster vs zone heatmap")
    ct = pd.crosstab(cells["cluster"], cells["zone"],
                     normalize="index")
    # reorder zones sensibly
    zone_order = [z for z in ["core", "intermediate", "periphery"]
                  if z in ct.columns]
    ct = ct[zone_order]

    fig, ax = plt.subplots(figsize=(3.5, max(2.5, k * 0.55)))
    im = ax.imshow(ct.values, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Fraction of cluster cells")

    ax.set_xticks(range(len(zone_order)))
    ax.set_xticklabels(zone_order)
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"Cluster {c}" for c in range(k)])

    for i in range(k):
        for j in range(len(zone_order)):
            ax.text(j, i, f"{ct.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=6,
                    color="white" if ct.values[i, j] > 0.6 else "black")

    ax.set_title("Cluster composition by radial zone")
    ax.set_xlabel("Radial zone")
    ax.set_ylabel("k-means cluster")
    plt.tight_layout()
    _save(fig, "fig6_cluster_zone_heatmap.png", out_dir)


def fig7_cluster_feature_violins(cells: pd.DataFrame,
                                  feat_list: list[str],
                                  k: int, out_dir: Path):
    """Violin plots of top features per cluster."""
    _log("  fig7: cluster feature violins")
    feats = feat_list[:8]
    ncols = 4
    nrows = int(np.ceil(len(feats) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 2.0, nrows * 2.4))
    axes = np.array(axes).flatten()

    for idx, feat in enumerate(feats):
        ax = axes[idx]
        data_per_cluster = [cells[cells["cluster"] == c][feat].dropna().values
                            for c in range(k)]
        cols = [CLUSTER_CMAP(c / max(k-1, 1)) for c in range(k)]

        for c, (vals, col) in enumerate(zip(data_per_cluster, cols)):
            if len(vals) > 2:
                vp = ax.violinplot([vals], positions=[c],
                                   widths=0.6, showmedians=False,
                                   showextrema=False)
                for body in vp["bodies"]:
                    body.set_facecolor(col)
                    body.set_alpha(0.6)
                    body.set_edgecolor("none")
            ax.scatter([c] * min(len(vals), 200),
                       vals[:200] if len(vals) > 200 else vals,
                       color=col, s=1.5, alpha=0.3, linewidths=0)
            if len(vals) > 0:
                ax.hlines(np.median(vals), c - 0.22, c + 0.22,
                          color=col, lw=1.5, zorder=4)

        ax.set_xticks(range(k))
        ax.set_xticklabels([f"C{c}" for c in range(k)])
        ax.set_ylabel(feat.replace("_", " ")[:22], fontsize=5.5)
        ax.set_title(feat.replace("_", " ")[:28], fontsize=6)
        ax.tick_params(labelsize=5)

    for ax in axes[len(feats):]:
        ax.set_visible(False)

    fig.suptitle(f"Feature distributions per cluster (k={k})",
                 fontsize=8, y=1.01)
    plt.tight_layout()
    _save(fig, "fig7_cluster_feature_violins.png", out_dir)


def anova_per_feature(cells: pd.DataFrame,
                      feat_list: list[str], k: int) -> pd.DataFrame:
    """One-way ANOVA for each feature across clusters. Returns ranked DataFrame."""
    rows = []
    for feat in feat_list:
        groups = [cells[cells["cluster"] == c][feat].dropna().values
                  for c in range(k)]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            continue
        f_stat, p = stats.f_oneway(*groups)
        # eta-squared (proportion of variance explained by cluster)
        all_vals = np.concatenate(groups)
        ss_total = np.sum((all_vals - all_vals.mean()) ** 2)
        ss_btw   = sum(len(g) * (np.mean(g) - all_vals.mean()) ** 2 for g in groups)
        eta2     = ss_btw / ss_total if ss_total > 0 else 0
        rows.append({"feature": feat, "F_stat": f_stat, "p_value": p,
                     "eta_squared": eta2})
    return (pd.DataFrame(rows)
              .sort_values("eta_squared", ascending=False)
              .reset_index(drop=True))


def fig8_feature_importance(cells: pd.DataFrame,
                             feat_list: list[str], k: int,
                             out_dir: Path) -> list[str]:
    """
    Random Forest importance for predicting cluster membership.
    Also returns the top features list for fig7.
    """
    _log("  fig8: feature importance")
    X   = SimpleImputer(strategy="median").fit_transform(cells[feat_list].values)
    X_s = StandardScaler().fit_transform(X)
    y   = cells["cluster"].values

    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                 class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    rf.fit(X_s, y)

    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(rf, X_s, y, cv=cv, scoring="accuracy")
    _log(f"  RF CV accuracy: {acc.mean():.3f} +/- {acc.std():.3f}")

    imp_df = (pd.DataFrame({"feature": feat_list,
                             "importance": rf.feature_importances_})
                .sort_values("importance", ascending=True))

    short = [f.replace("_per_mm3", "/mm³").replace("_um", "")
              .replace("_", " ")[:30]
              for f in imp_df["feature"]]

    fig, ax = plt.subplots(figsize=(4.5, max(2.5, len(feat_list) * 0.28)))
    ax.barh(range(len(imp_df)), imp_df["importance"],
            color="#2C5F8A", alpha=0.85)
    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels(short, fontsize=5)
    ax.set_xlabel("Feature importance (Gini)")
    ax.set_title(f"RF importance for cluster prediction (CV acc={acc.mean():.2f})")
    plt.tight_layout()
    _save(fig, "fig8_feature_importance.png", out_dir)

    top = imp_df.sort_values("importance", ascending=False)["feature"].tolist()
    return top


# ── main ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Large PDAC -- Spatial Heterogeneity Analysis")
    print("=" * 60)

    features_dir = Path(FEATURES_DIR)
    out_dir      = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. load
    cells = load_data(features_dir)

    # 2. prepare
    _log("\n[2/8] Preparing features...")
    X_sc, feat_names, _ = prepare_features(cells)
    _log(f"  Using {len(feat_names)} features, {len(cells)} cells")

    # 3. PCA (used by multiple figures)
    _log("\n[3/8] Running PCA...")
    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sc)
    ev    = pca.explained_variance_ratio_
    _log(f"  PC1={ev[0]*100:.1f}%  PC2={ev[1]*100:.1f}%  "
         f"total={ev.sum()*100:.1f}%")

    # 4. figures that don't need clustering
    _log("\n[4/8] Generating pre-clustering figures...")
    fig1_radial_profiles(cells, out_dir)
    fig2_prolate_oblate_scatter(cells, out_dir)
    fig3_pca_by_zone(X_sc, X_pca, cells, out_dir)

    # 5. cluster selection
    _log("\n[5/8] Cluster selection...")
    k = fig4_cluster_selection(X_sc, out_dir)

    # 6. k-means
    _log(f"\n[6/8] k-means clustering (k={k})...")
    cells, lbl = run_kmeans(X_sc, cells, k)

    # 7. cluster figures
    _log("\n[7/8] Generating cluster figures...")
    fig5_clusters_pca(X_sc, X_pca, cells, k, out_dir)
    fig6_cluster_zone_heatmap(cells, k, out_dir)

    # ANOVA to rank features by discriminative power
    anova_df = anova_per_feature(cells, feat_names, k)
    anova_df.to_csv(out_dir / "statistical_results_anova.csv", index=False)
    top_feats = anova_df.head(8)["feature"].tolist()

    fig7_cluster_feature_violins(cells, top_feats, k, out_dir)

    _log("\n[8/8] Feature importance...")
    rf_top = fig8_feature_importance(cells, feat_names, k, out_dir)

    # 8. save annotated cells
    cells.to_csv(out_dir / "cell_data_with_clusters.csv", index=False)

    # ── summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ANALYSIS COMPLETE")
    print("=" * 60)

    sil = silhouette_score(X_sc, lbl)
    print(f"\n  Clusters: {k}  |  Silhouette: {sil:.3f}")
    if sil < 0.3:
        print("  Interpretation: continuous gradient (no discrete populations)")
        print("  -- consistent with expected large PDAC biology")
    elif sil < 0.5:
        print("  Interpretation: weak-to-moderate cluster separation")
    else:
        print("  Interpretation: clear distinct subpopulations")

    print(f"\n  Top discriminative features (ANOVA eta-squared):")
    for _, row in anova_df.head(5).iterrows():
        print(f"    {row['feature']:<38s}  "
              f"eta2={row['eta_squared']:.3f}  "
              f"p={row['p_value']:.2e}")

    print(f"\n  Outputs written to: {out_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()