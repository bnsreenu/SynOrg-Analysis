"""
extract_features.py
--------------------
Standalone feature extraction for synthetic organoid segmentation results.

Reads saved mask TIFFs from output/segmentation/ and raw OME-TIFFs from
input/images/, extracts per-cell features, and writes one CSV per organoid
to output/features/.

Run this AFTER segment_organoids.py. Because segmentation and feature
extraction are now separate, you can freely change any feature parameter
(RADIUS_UM, COMPUTE_DISTANCE_TO_BORDER, metadata, etc.) and re-run in
minutes without touching Cellpose.

Directory layout expected:
  input/
    images/                  raw OME-TIFF files (for DAPI intensity features)
  output/
    segmentation/
      <stem>_cell_masks_filtered.tif
      <stem>_nuclei_masks_filtered.tif

Output written to:
  output/
    features/
      <stem>_features.csv

Usage -- process everything:
  python extract_features.py

Usage -- process only specific condition prefixes:
  Edit PROCESS_STEMS in the CONFIGURATION section below, e.g.:
    PROCESS_STEMS = ["hmecyst_control", "hmecyst_cyst"]
  Leave as None to process all available stems.
"""

import numpy as np
import tifffile as tiff
import pandas as pd
from skimage import measure
from scipy import ndimage
from scipy.spatial import cKDTree
from pathlib import Path
import warnings
import time
import sys

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION  --  the only section you need to edit
# ============================================================

INPUT_DIR  = "input/"
OUTPUT_DIR = "output/"

# Process only files whose stem starts with one of these prefixes.
# Set to None to process every organoid in output/segmentation/.
# Examples:
#   PROCESS_STEMS = None                              # all organoids
#   PROCESS_STEMS = ["hmecyst_control"]               # one condition
#   PROCESS_STEMS = ["hmecyst_control", "hmecyst_cyst"]  # two conditions

PROCESS_STEMS = ["pdac_isotonic", "pdac_hypertonic"]

# ---- Metadata written to every CSV row ----
# These are per-run labels. For per-organoid metadata, the organoid stem
# is always available in the filename itself.
BATCH  = "Batch1"
SAMPLE = "PDAC_osmotic"
WELL   = "A01"
FIELD  = "s1"

# ---- Voxel parameters ----
# READ_VOXEL_FROM_OME=True reads XY/Z from the OME-XML in the raw image.
# Set to False and fill in values below if OME metadata is absent.
READ_VOXEL_FROM_OME = True
PIXEL_SIZE_XY = 0.414   # um/pixel (fallback)
PIXEL_SIZE_Z  = 1.0     # um/slice  (fallback)

# ---- Topology neighbourhood radius ----
# Cells within this radius (in um) of a given cell centroid are counted
# as neighbours. For HMECyst cells (mean radius ~13 um, inter-cell
# distance ~26 um) use at least 35-40 um.
# For PDAC cells (mean radius ~8 um) 20 um is appropriate.
RADIUS_UM = 20.0

# ---- Distance-to-border features ----
# Adds nuclei_distance_to_border_um and nuclei_distance_to_border_ratio.
# Uses one distance_transform_edt call per organoid (fast).
# Set to False to skip these two features.
COMPUTE_DISTANCE_TO_BORDER = True

# ---- Overwrite existing CSVs? ----
# Set to False to skip organoids that already have a features CSV.
OVERWRITE = True

# ============================================================
# END OF CONFIGURATION
# ============================================================


COLUMN_ORDER_BASE = [
    "Batch", "Sample", "Well", "Field", "Cell_Number",
    "centroid_z_um", "centroid_y_um", "centroid_x_um",
    "centroid_z_rel_um", "centroid_y_rel_um", "centroid_x_rel_um",
    "radial_dist_um", "radial_dist_norm", "zone",
    "cell_volume_um3", "nuclei_volume_um3",
    "cell_elongation", "cell_roundedness",
    "CV_chromatin", "max_intensity_nuclear", "min_intensity_nuclear",
    "avg_intensity_nuclear", "std_intensity_nuclear", "sum_intensity_nuclear",
    "distance_to_neighbors_mean_um", "distance_to_neighbors_min_um",
    "distance_to_neighbors_max_um", "n_nuclei_neighbors",
    "nb_nuclei_neighbors_ripley", "local_density_per_mm3", "crystal_distance_um",
    "major_axis_um", "medium_axis_um", "minor_axis_um",
    "prolate_ratio", "oblate_ratio",
]
COLUMN_ORDER_WITH_BORDER = [
    "Batch", "Sample", "Well", "Field", "Cell_Number",
    "centroid_z_um", "centroid_y_um", "centroid_x_um",
    "centroid_z_rel_um", "centroid_y_rel_um", "centroid_x_rel_um",
    "radial_dist_um", "radial_dist_norm", "zone",
    "cell_volume_um3", "nuclei_volume_um3",
    "nuclei_distance_to_border_um", "nuclei_distance_to_border_ratio",
    "cell_elongation", "cell_roundedness",
    "CV_chromatin", "max_intensity_nuclear", "min_intensity_nuclear",
    "avg_intensity_nuclear", "std_intensity_nuclear", "sum_intensity_nuclear",
    "distance_to_neighbors_mean_um", "distance_to_neighbors_min_um",
    "distance_to_neighbors_max_um", "n_nuclei_neighbors",
    "nb_nuclei_neighbors_ripley", "local_density_per_mm3", "crystal_distance_um",
    "major_axis_um", "medium_axis_um", "minor_axis_um",
    "prolate_ratio", "oblate_ratio",
]


def _log(msg: str, indent: int = 0):
    print("  " * indent + msg, flush=True)


# ── helpers ───────────────────────────────────────────────────────────

def read_voxel_from_ome(img_path: Path):
    """Read physical voxel size from OME-XML metadata."""
    pxy, pz = PIXEL_SIZE_XY, PIXEL_SIZE_Z
    try:
        import xml.etree.ElementTree as ET
        with tiff.TiffFile(str(img_path)) as tf:
            if tf.ome_metadata:
                root = ET.fromstring(tf.ome_metadata)
                ns   = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
                px   = root.find(".//ome:Pixels", ns)
                if px is not None:
                    pxy = float(px.get("PhysicalSizeX", pxy))
                    pz  = float(px.get("PhysicalSizeZ", pz))
    except Exception as e:
        _log(f"Warning: could not read OME voxel sizes ({e}); using config values", 1)
    return pxy, pz


def load_nuclei_channel(img_path: Path) -> np.ndarray:
    """Load only the DAPI/nuclei channel (channel 0) from the raw OME-TIFF."""
    raw = tiff.imread(str(img_path))
    if raw.ndim == 4:
        if raw.shape[0] == 2:
            return raw[0].astype(np.float32)
        elif raw.shape[1] == 2:
            return raw[:, 0, :, :].astype(np.float32)
        else:
            raise ValueError(f"Unexpected shape {raw.shape}")
    raise ValueError(f"Expected 4-D array from {img_path.name}")


def find_raw_image(stem: str, images_dir: Path) -> Path | None:
    """
    Find the raw OME-TIFF corresponding to a mask stem.
    The mask stem is e.g. 'hmecyst_control_seed42_20260408_143345'
    and the raw image is 'hmecyst_control_seed42_20260408_143345.ome.tif'.
    """
    for ext in (".ome.tif", ".tif"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def collect_stems(seg_dir: Path, process_prefixes) -> list[str]:
    """
    Find all unique organoid stems in seg_dir that have both
    cell_masks_filtered and nuclei_masks_filtered TIFFs.
    If process_prefixes is given, only return stems starting with
    one of those prefixes.
    """
    cell_masks = sorted(seg_dir.glob("*_cell_masks_filtered.tif"))
    stems = []
    for p in cell_masks:
        stem = p.name.replace("_cell_masks_filtered.tif", "")
        nuc_mask = seg_dir / f"{stem}_nuclei_masks_filtered.tif"
        if not nuc_mask.exists():
            _log(f"Skipping {stem}: nuclei_masks_filtered.tif not found")
            continue
        if process_prefixes is not None:
            if not any(stem.startswith(pfx) for pfx in process_prefixes):
                continue
        stems.append(stem)
    return stems


# ── feature extraction ────────────────────────────────────────────────

def compute_organoid_distance_transform(cell_masks_filtered, pxy, pz):
    """
    Single distance_transform_edt on the whole organoid mask.
    Returns (Z,Y,X) float32 array of distances in um.
    Much faster than one EDT per cell.
    """
    t0 = time.time()
    organoid_mask = cell_masks_filtered > 0
    dist = ndimage.distance_transform_edt(
        organoid_mask, sampling=[pz, pxy, pxy]
    ).astype(np.float32)
    _log(f"distance_transform_edt done  [{time.time()-t0:.1f}s]", 2)
    return dist


def extract_features(cell_masks_filtered: np.ndarray,
                     nuclei_masks_filtered: np.ndarray,
                     nuclei: np.ndarray,
                     pxy: float,
                     pz: float,
                     organoid_dist: np.ndarray | None) -> pd.DataFrame:
    """Extract per-cell features from pre-computed masks and raw DAPI channel."""

    t0        = time.time()
    voxel_vol = (pxy ** 2) * pz
    vol_sphere_um3 = (4 / 3) * np.pi * (RADIUS_UM ** 3)
    vol_sphere_mm3 = vol_sphere_um3 / 1e9

    # ── 1. regionprops ────────────────────────────────────────────────
    t_step = time.time()
    _log("[1/4] Extracting regionprops...", 1)
    cell_props_all   = measure.regionprops(cell_masks_filtered)
    nuclei_props_all = measure.regionprops(nuclei_masks_filtered,
                                           intensity_image=nuclei)
    cell_props_dict   = {p.label: p for p in cell_props_all}
    nuclei_props_dict = {p.label: p for p in nuclei_props_all}
    cell_ids = np.array([p.label for p in cell_props_all])
    n_cells  = len(cell_ids)
    _log(f"{n_cells} cells  [{time.time()-t_step:.1f}s]", 2)

    # ── 2. centroids + organoid geometry ─────────────────────────────
    t_step = time.time()
    _log("[2/4] Computing centroids and organoid geometry...", 1)
    all_centroids = np.array([
        [p.centroid[0] * pz, p.centroid[1] * pxy, p.centroid[2] * pxy]
        for p in cell_props_all
    ], dtype=np.float64)
    cell_id_to_idx  = {cid: idx for idx, cid in enumerate(cell_ids)}
    organoid_centre = all_centroids.mean(axis=0)
    radial_dists    = np.linalg.norm(all_centroids - organoid_centre, axis=1)
    organoid_radius = float(np.percentile(radial_dists, 95))
    _log(f"Centre (z,y,x): ({organoid_centre[0]:.1f}, {organoid_centre[1]:.1f}, "
         f"{organoid_centre[2]:.1f}) um  |  Radius (p95): {organoid_radius:.1f} um", 2)
    _log(f"Centroids done  [{time.time()-t_step:.1f}s]", 2)

    # ── 3. KDTree for topology ────────────────────────────────────────
    t_step = time.time()
    _log(f"[3/4] Building KDTree (RADIUS_UM={RADIUS_UM} um)...", 1)
    tree          = cKDTree(all_centroids)
    all_neighbors = tree.query_ball_tree(tree, r=RADIUS_UM)
    _log(f"KDTree done  [{time.time()-t_step:.1f}s]", 2)

    # pre-compute nucleus centroid voxel coords for border distance lookup
    if organoid_dist is not None:
        nuc_centroid_vox = {}
        for p in nuclei_props_all:
            nz, ny, nx = p.centroid
            nuc_centroid_vox[p.label] = (
                int(round(nz)), int(round(ny)), int(round(nx)))

    # ── 4. per-cell loop ──────────────────────────────────────────────
    _log("[4/4] Extracting per-cell features...", 1)
    t_loop    = time.time()
    last_log  = time.time()
    all_feats = []

    for i, cell_id in enumerate(cell_ids):
        now = time.time()
        if (i + 1) % 50 == 0 or (now - last_log) > 5:
            elapsed  = now - t_loop
            rate     = (i + 1) / elapsed if elapsed > 0 else 0
            eta      = (n_cells - i - 1) / rate if rate > 0 else 0
            _log(f"  {i+1}/{n_cells} ({100*(i+1)/n_cells:.1f}%)  "
                 f"{rate:.1f} cells/s  ETA {eta:.0f}s", 2)
            last_log = now

        cell_prop = cell_props_dict[cell_id]

        # find nucleus (cheaply: multiply masks in bounding box)
        bb  = cell_prop.bbox          # (z0,y0,x0, z1,y1,x1)
        roi_cell = cell_masks_filtered[bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]]
        roi_nuc  = nuclei_masks_filtered[bb[0]:bb[3], bb[1]:bb[4], bb[2]:bb[5]]
        cell_mask_roi = roi_cell == cell_id
        overlap   = roi_nuc * cell_mask_roi
        nuc_ids   = np.unique(overlap)[1:]
        if len(nuc_ids) == 0:
            continue
        nucleus_id   = nuc_ids[0]
        nucleus_prop = nuclei_props_dict.get(nucleus_id)
        if nucleus_prop is None:
            continue

        feat = {
            "Batch":       BATCH,
            "Sample":      SAMPLE,
            "Well":        WELL,
            "Field":       FIELD,
            "Cell_Number": int(cell_id),
        }

        # ── spatial coordinates ───────────────────────────────────────
        cell_idx         = cell_id_to_idx[cell_id]
        cz_um, cy_um, cx_um = all_centroids[cell_idx]
        feat["centroid_z_um"]     = float(cz_um)
        feat["centroid_y_um"]     = float(cy_um)
        feat["centroid_x_um"]     = float(cx_um)
        feat["centroid_z_rel_um"] = float(cz_um - organoid_centre[0])
        feat["centroid_y_rel_um"] = float(cy_um - organoid_centre[1])
        feat["centroid_x_rel_um"] = float(cx_um - organoid_centre[2])
        feat["radial_dist_um"]    = float(radial_dists[cell_idx])
        rn = float(radial_dists[cell_idx] / organoid_radius
                   if organoid_radius > 0 else 0.0)
        feat["radial_dist_norm"]  = rn
        feat["zone"] = ("core" if rn < 0.50
                         else "intermediate" if rn < 0.75
                         else "periphery")

        # ── volumes ───────────────────────────────────────────────────
        feat["cell_volume_um3"]   = cell_prop.area   * voxel_vol
        feat["nuclei_volume_um3"] = nucleus_prop.area * voxel_vol

        # ── distance to organoid border (vectorized) ──────────────────
        if organoid_dist is not None:
            cz_v, cy_v, cx_v = nuc_centroid_vox.get(nucleus_id, (0, 0, 0))
            cz_v = min(cz_v, organoid_dist.shape[0] - 1)
            cy_v = min(cy_v, organoid_dist.shape[1] - 1)
            cx_v = min(cx_v, organoid_dist.shape[2] - 1)
            d_um  = float(organoid_dist[cz_v, cy_v, cx_v])
            # max distance within this cell -- work in bounding box only
            cell_mask_full = cell_masks_filtered == cell_id
            d_max = float((organoid_dist * cell_mask_full).max())
            feat["nuclei_distance_to_border_um"]    = d_um
            feat["nuclei_distance_to_border_ratio"] = (
                d_um / d_max if d_max > 0 else 0.0)

        # ── shape (regionprops 2D proxy) ──────────────────────────────
        try:
            maj = cell_prop.axis_major_length * pxy
            mnr = cell_prop.axis_minor_length * pxy
            feat["cell_elongation"]  = maj / mnr if mnr > 0 else 1.0
            feat["cell_roundedness"] = mnr / maj if maj > 0 else 1.0
        except Exception:
            feat["cell_elongation"]  = np.nan
            feat["cell_roundedness"] = np.nan

        # ── nuclear intensity (DAPI) ──────────────────────────────────
        nuc_mask_full = nuclei_masks_filtered == nucleus_id
        nuc_int       = nuclei[nuc_mask_full]
        if len(nuc_int) > 0:
            mean_i = float(np.mean(nuc_int))
            std_i  = float(np.std(nuc_int))
            feat["avg_intensity_nuclear"] = mean_i
            feat["std_intensity_nuclear"] = std_i
            feat["max_intensity_nuclear"] = float(np.max(nuc_int))
            feat["min_intensity_nuclear"] = float(np.min(nuc_int))
            feat["sum_intensity_nuclear"] = float(np.sum(nuc_int))
            feat["CV_chromatin"]          = std_i / mean_i if mean_i > 0 else 0.0
        else:
            for k in ("avg_intensity_nuclear", "std_intensity_nuclear",
                      "max_intensity_nuclear", "min_intensity_nuclear",
                      "sum_intensity_nuclear", "CV_chromatin"):
                feat[k] = 0.0

        # ── topology ─────────────────────────────────────────────────
        nbr_idxs = [j for j in all_neighbors[cell_idx] if j != cell_idx]
        n_nbr    = len(nbr_idxs)
        if n_nbr > 0:
            nbr_c = all_centroids[nbr_idxs]
            dists = np.linalg.norm(nbr_c - all_centroids[cell_idx], axis=1)
            feat["distance_to_neighbors_mean_um"] = float(dists.mean())
            feat["distance_to_neighbors_min_um"]  = float(dists.min())
            feat["distance_to_neighbors_max_um"]  = float(dists.max())
        else:
            feat["distance_to_neighbors_mean_um"] = np.nan
            feat["distance_to_neighbors_min_um"]  = np.nan
            feat["distance_to_neighbors_max_um"]  = np.nan

        feat["n_nuclei_neighbors"]         = n_nbr
        feat["nb_nuclei_neighbors_ripley"] = float(n_nbr)
        feat["local_density_per_mm3"]      = n_nbr / vol_sphere_mm3
        feat["crystal_distance_um"] = (
            (vol_sphere_um3 / n_nbr) ** (1 / 3) if n_nbr > 0 else np.nan)

        # ── 3D shape (PCA on voxel coords, bounding-box restricted) ──
        try:
            coords    = np.argwhere(cell_mask_roi)
            coords_um = coords * np.array([pz, pxy, pxy])
            cc        = coords_um - coords_um.mean(axis=0)
            evals     = np.sort(np.linalg.eigvalsh(np.cov(cc.T)))[::-1]
            axes      = 2 * np.sqrt(np.maximum(evals, 0))
            feat["major_axis_um"]  = float(axes[0])
            feat["medium_axis_um"] = float(axes[1]) if len(axes) > 1 else float(axes[0])
            feat["minor_axis_um"]  = float(axes[2]) if len(axes) > 2 else float(axes[0])
            feat["prolate_ratio"]  = (float(axes[0] / axes[1])
                                       if len(axes) > 1 and axes[1] > 0 else 1.0)
            feat["oblate_ratio"]   = (float(axes[1] / axes[2])
                                       if len(axes) > 2 and axes[2] > 0 else 1.0)
        except Exception:
            for k in ("major_axis_um", "medium_axis_um", "minor_axis_um",
                      "prolate_ratio", "oblate_ratio"):
                feat[k] = np.nan

        all_feats.append(feat)

    _log(f"Per-cell loop done  [{time.time()-t_loop:.1f}s]", 2)
    _log(f"Total feature extraction: {time.time()-t0:.1f}s  "
         f"({len(all_feats)} cells)", 1)

    df        = pd.DataFrame(all_feats)
    col_order = (COLUMN_ORDER_WITH_BORDER
                 if organoid_dist is not None else COLUMN_ORDER_BASE)
    col_order = [c for c in col_order if c in df.columns]
    return df[col_order]


# ── per-organoid runner ───────────────────────────────────────────────

def process_one(stem: str,
                seg_dir: Path,
                images_dir: Path,
                features_dir: Path):
    print()
    print("=" * 60)
    print(f"Extracting features: {stem}")
    print("=" * 60)
    t_start = time.time()

    out_csv = features_dir / f"{stem}_features.csv"
    if out_csv.exists() and not OVERWRITE:
        _log(f"Skipping (CSV exists and OVERWRITE=False): {out_csv.name}", 1)
        return

    # ── load masks ───────────────────────────────────────────────────
    t = time.time()
    _log("Loading masks...", 1)
    cell_masks = tiff.imread(
        str(seg_dir / f"{stem}_cell_masks_filtered.tif"))
    nuc_masks  = tiff.imread(
        str(seg_dir / f"{stem}_nuclei_masks_filtered.tif"))
    _log(f"Cell mask: {cell_masks.shape}  "
         f"({int(cell_masks.max())} cells)  [{time.time()-t:.1f}s]", 1)

    # ── load raw DAPI channel ─────────────────────────────────────────
    t = time.time()
    _log("Loading raw DAPI channel...", 1)
    img_path = find_raw_image(stem, images_dir)
    if img_path is None:
        raise FileNotFoundError(
            f"Raw image not found for stem '{stem}' in {images_dir}. "
            f"Expected {stem}.ome.tif or {stem}.tif")
    nuclei_raw = load_nuclei_channel(img_path)
    _log(f"Loaded: {img_path.name}  [{time.time()-t:.1f}s]", 1)

    # ── voxel size ────────────────────────────────────────────────────
    if READ_VOXEL_FROM_OME:
        pxy, pz = read_voxel_from_ome(img_path)
        _log(f"Voxel size: XY={pxy} um  Z={pz} um", 1)
    else:
        pxy, pz = PIXEL_SIZE_XY, PIXEL_SIZE_Z
        _log(f"Voxel size (config): XY={pxy} um  Z={pz} um", 1)

    # ── distance transform ────────────────────────────────────────────
    organoid_dist = None
    if COMPUTE_DISTANCE_TO_BORDER:
        t = time.time()
        _log("Computing organoid distance transform...", 1)
        organoid_dist = compute_organoid_distance_transform(
            cell_masks, pxy, pz)
        _log(f"Distance transform done  [{time.time()-t:.1f}s]", 1)

    # ── feature extraction ────────────────────────────────────────────
    t = time.time()
    _log("Extracting features...", 1)
    df = extract_features(
        cell_masks, nuc_masks, nuclei_raw, pxy, pz, organoid_dist)
    _log(f"Feature extraction done  [{time.time()-t:.1f}s]", 1)

    # ── save ──────────────────────────────────────────────────────────
    df.to_csv(str(out_csv), index=False)
    _log(f"Saved: {out_csv.name}  "
         f"({len(df)} rows, {len(df.columns)-5} features)", 1)

    elapsed = time.time() - t_start
    _log(f"Total: {elapsed:.1f}s  ({elapsed/60:.1f} min)", 1)


# ── batch runner ──────────────────────────────────────────────────────

def run_batch():
    input_dir  = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    images_dir   = input_dir  / "images"
    seg_dir      = output_dir / "segmentation"
    features_dir = output_dir / "features"

    features_dir.mkdir(parents=True, exist_ok=True)

    if not seg_dir.exists():
        print(f"Segmentation directory not found: {seg_dir}")
        print("Run segment_organoids.py first.")
        sys.exit(1)

    stems = collect_stems(seg_dir, PROCESS_STEMS)

    if not stems:
        msg = (f"No matching organoids found in {seg_dir}"
               + (f" for prefixes {PROCESS_STEMS}" if PROCESS_STEMS else ""))
        print(msg)
        sys.exit(1)

    print(f"Found {len(stems)} organoid(s) to process")
    if PROCESS_STEMS:
        print(f"Prefix filter: {PROCESS_STEMS}")
    print(f"RADIUS_UM = {RADIUS_UM} um")
    print(f"COMPUTE_DISTANCE_TO_BORDER = {COMPUTE_DISTANCE_TO_BORDER}")
    print(f"OVERWRITE = {OVERWRITE}")

    ok, failed = 0, []
    for stem in stems:
        try:
            process_one(stem, seg_dir, images_dir, features_dir)
            ok += 1
        except Exception as e:
            print(f"\nERROR processing {stem}: {e}")
            import traceback; traceback.print_exc()
            failed.append(stem)

    print()
    print("=" * 60)
    print(f"Batch complete: {ok}/{len(stems)} succeeded")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print("=" * 60)


if __name__ == "__main__":
    run_batch()
