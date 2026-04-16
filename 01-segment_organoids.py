"""
segment_organoids.py
--------------------
Batch 3D segmentation for synthetic organoid OME-TIFFs.
Runs Cellpose, applies QC filtering, saves masks and plots.

Does NOT extract features -- run extract_features.py separately
after segmentation is complete. This separation means you never
need to re-run the expensive Cellpose step just to change feature
parameters such as RADIUS_UM.

Directory layout:
  input/
    images/           raw OME-TIFF files from the synthetic organoid generator
    labels/
      cell_labels/    optional GT cell masks  (<stem>_labels.ome.tif)
      nucleus_labels/ optional GT nucleus masks (<stem>_nucleus_labels.ome.tif)

Output written to output/:
  segmentation/
    <stem>_cell_masks_filtered.tif
    <stem>_nuclei_masks_filtered.tif
    <stem>_cell_masks_unfiltered.tif
    <stem>_nuclei_masks_unfiltered.tif
    <stem>_segmentation_summary.txt
  plots/
    <stem>_raw_data.png
    <stem>_segmentation_qc.png
    <stem>_accuracy.png          (only when GT labels present)

Usage:
  python segment_organoids.py
"""

import numpy as np
import tifffile as tiff
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cellpose import models
import torch
from skimage import measure, segmentation
from scipy import ndimage
from pathlib import Path
import warnings
import time
import sys

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR  = "input_pdac/"
OUTPUT_DIR = "output_pdac/"

# ---- Voxel parameters ----
# READ_VOXEL_FROM_OME=True reads XY/Z from the OME-XML embedded in the file.
# Set to False and fill in the values below if your files lack OME metadata.
READ_VOXEL_FROM_OME = True
PIXEL_SIZE_XY = 0.414   # um/pixel (fallback if OME metadata absent)
PIXEL_SIZE_Z  = 1.0     # um/slice

# ---- Nuclei segmentation (Cellpose) ----
NUCLEI_DIAMETER           = 30    # pixels
NUCLEI_FLOW_THRESHOLD     = 0.4
NUCLEI_CELLPROB_THRESHOLD = 0.0

# ---- Cell segmentation (Cellpose) ----
CELL_DIAMETER             = 50    # pixels
CELL_FLOW_THRESHOLD       = 0.6
CELL_CELLPROB_THRESHOLD   = 0.0

# ---- Quality-control filtering ----
MIN_CELL_VOLUME_UM3    = 500    # minimum cell volume in um3
MIN_NUCLEUS_VOLUME_UM3 = 100    # minimum nucleus volume in um3
REQUIRE_NUCLEUS        = True   # drop cells without a detected nucleus
MAX_NC_RATIO           = 0.9    # nucleus/cell volume ratio upper limit

# ---- Normalisation percentiles (Cellpose input prep) ----
NORM_LOWER_PERCENTILE = 1
NORM_UPPER_PERCENTILE = 99

# ============================================================
# END OF CONFIGURATION
# ============================================================


def _log(msg: str, indent: int = 0):
    print("  " * indent + msg, flush=True)


# ── image loading ────────────────────────────────────────────────────

def load_ome_tiff(path: Path):
    """
    Load a 2-channel synthetic organoid OME-TIFF.
    Returns (nuclei_ch, cells_ch, pixel_size_xy, pixel_size_z).
    Handles axis orders (C,Z,Y,X) and (Z,C,Y,X).
    """
    raw = tiff.imread(str(path))
    _log(f"Raw shape: {raw.shape}  dtype: {raw.dtype}", 1)

    if raw.ndim == 4:
        if raw.shape[0] == 2:
            nuclei_ch = raw[0].astype(np.float32)
            cells_ch  = raw[1].astype(np.float32)
        elif raw.shape[1] == 2:
            nuclei_ch = raw[:, 0, :, :].astype(np.float32)
            cells_ch  = raw[:, 1, :, :].astype(np.float32)
        else:
            raise ValueError(
                f"Unexpected shape {raw.shape}: C=2 expected in axis 0 or 1")
    else:
        raise ValueError(
            f"Expected 4-D array, got {raw.ndim}-D from {path.name}")

    pxy, pz = PIXEL_SIZE_XY, PIXEL_SIZE_Z
    if READ_VOXEL_FROM_OME:
        try:
            import xml.etree.ElementTree as ET
            with tiff.TiffFile(str(path)) as tf:
                if tf.ome_metadata:
                    root = ET.fromstring(tf.ome_metadata)
                    ns   = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
                    px   = root.find(".//ome:Pixels", ns)
                    if px is not None:
                        pxy = float(px.get("PhysicalSizeX", pxy))
                        pz  = float(px.get("PhysicalSizeZ", pz))
                        _log(f"OME metadata: XY={pxy} um  Z={pz} um", 1)
        except Exception as e:
            _log(f"Warning: could not read OME voxel sizes ({e}); "
                 f"using config values", 1)

    return nuclei_ch, cells_ch, pxy, pz


def load_ground_truth(stem: str, labels_dir: Path):
    """Load optional GT label masks. Returns (cell_gt, nuc_gt) or (None, None)."""
    cell_path = labels_dir / "cell_labels"    / f"{stem}_labels.ome.tif"
    nuc_path  = labels_dir / "nucleus_labels" / f"{stem}_nucleus_labels.ome.tif"
    cell_gt   = tiff.imread(str(cell_path)) if cell_path.exists() else None
    nuc_gt    = tiff.imread(str(nuc_path))  if nuc_path.exists()  else None
    return cell_gt, nuc_gt


def normalize_image(img: np.ndarray) -> np.ndarray:
    p_low  = np.percentile(img, NORM_LOWER_PERCENTILE)
    p_high = np.percentile(img, NORM_UPPER_PERCENTILE)
    out    = np.clip(img, p_low, p_high)
    return ((out - p_low) / (p_high - p_low + 1e-8)).astype(np.float32)


# ── segmentation ─────────────────────────────────────────────────────

def run_segmentation(nuclei_norm, cells_norm, anisotropy, use_gpu):
    """Run Cellpose 3D for nuclei and cells. Returns (nuclei_masks_3d, cell_masks_3d)."""
    Z = nuclei_norm.shape[0]

    t0 = time.time()
    _log("Loading nuclei model...", 1)
    nuc_model = models.CellposeModel(gpu=use_gpu, model_type="nuclei")
    _log(f"Model loaded  [{time.time()-t0:.1f}s]", 2)

    t0 = time.time()
    _log(f"Segmenting nuclei (3D, {Z} slices, diameter={NUCLEI_DIAMETER}px)...", 1)
    nuclei_masks_3d, _ = nuc_model.eval(
        nuclei_norm,
        diameter=NUCLEI_DIAMETER,
        channels=[0, 0],
        do_3D=True,
        z_axis=0,
        anisotropy=anisotropy,
        flow_threshold=NUCLEI_FLOW_THRESHOLD,
        cellprob_threshold=NUCLEI_CELLPROB_THRESHOLD,
    )[:2]
    _log(f"Nuclei done -- {nuclei_masks_3d.max()} detected  [{time.time()-t0:.1f}s]", 1)

    t0 = time.time()
    _log("Loading cell model...", 1)
    cell_model = models.CellposeModel(gpu=use_gpu, model_type="cyto3")
    _log(f"Model loaded  [{time.time()-t0:.1f}s]", 2)

    t0 = time.time()
    img_2ch = np.stack([cells_norm, nuclei_norm], axis=-1)
    _log(f"Segmenting cells (3D, {Z} slices, diameter={CELL_DIAMETER}px, 2-ch)...", 1)
    cell_masks_3d, _ = cell_model.eval(
        img_2ch,
        diameter=CELL_DIAMETER,
        channels=[1, 2],
        do_3D=True,
        z_axis=0,
        channel_axis=3,
        anisotropy=anisotropy,
        flow_threshold=CELL_FLOW_THRESHOLD,
        cellprob_threshold=CELL_CELLPROB_THRESHOLD,
    )[:2]
    _log(f"Cells done -- {cell_masks_3d.max()} detected  [{time.time()-t0:.1f}s]", 1)

    return nuclei_masks_3d, cell_masks_3d


# ── quality control ───────────────────────────────────────────────────

def build_cell_info(cell_masks_3d, nuclei_masks_3d, voxel_volume_um3):
    """
    Fully vectorized: compute cell/nucleus volumes and nucleus-cell
    overlap using np.bincount -- no per-cell array scans.
    """
    t0 = time.time()
    flat_cells  = cell_masks_3d.ravel()
    flat_nuclei = nuclei_masks_3d.ravel()

    max_cell_id = int(cell_masks_3d.max())
    max_nuc_id  = int(nuclei_masks_3d.max())
    cell_vols   = np.bincount(flat_cells,  minlength=max_cell_id + 1)
    nuc_vols    = np.bincount(flat_nuclei, minlength=max_nuc_id  + 1)

    overlap_mask = (flat_cells > 0) & (flat_nuclei > 0)
    oc = flat_cells[overlap_mask].astype(np.int64)
    on = flat_nuclei[overlap_mask].astype(np.int64)
    stride      = max_nuc_id + 1
    pair_counts = np.bincount(oc * stride + on,
                              minlength=(max_cell_id + 1) * stride)
    pair_counts = pair_counts.reshape(max_cell_id + 1, stride)

    best_nuc_idx = np.argmax(pair_counts, axis=1)
    best_nuc_vox = pair_counts[np.arange(max_cell_id + 1), best_nuc_idx]
    has_nucleus  = best_nuc_vox > 0

    cell_info = {}
    for cell_id in np.unique(cell_masks_3d)[1:]:
        cid    = int(cell_id)
        cv_um3 = int(cell_vols[cid]) * voxel_volume_um3
        hn     = bool(has_nucleus[cid])
        nid    = int(best_nuc_idx[cid]) if hn else 0
        nv_um3 = (int(nuc_vols[nid]) * voxel_volume_um3
                  if hn and nid > 0 else 0.0)
        cell_info[cid] = {
            "cell_volume_um3":    cv_um3,
            "nucleus_id":         nid,
            "nucleus_volume_um3": nv_um3,
            "has_nucleus":        hn,
            "nc_ratio":           nv_um3 / cv_um3 if cv_um3 > 0 else 0.0,
        }

    _log(f"Cell info built in {time.time()-t0:.1f}s  ({len(cell_info)} cells)", 2)
    return cell_info


def filter_cells(cell_masks_3d, nuclei_masks_3d, cell_info):
    """
    Apply QC filters. Mask remapping is vectorized via lookup tables.
    Returns (cell_masks_filtered, nuclei_masks_filtered, stats_dict).
    """
    t0 = time.time()
    cells_to_keep = []
    rm_small = rm_no_nuc = rm_nuc_small = rm_nc = 0

    for cid, info in cell_info.items():
        if info["cell_volume_um3"] < MIN_CELL_VOLUME_UM3:
            rm_small += 1;     continue
        if REQUIRE_NUCLEUS and not info["has_nucleus"]:
            rm_no_nuc += 1;    continue
        if info["has_nucleus"] and info["nucleus_volume_um3"] < MIN_NUCLEUS_VOLUME_UM3:
            rm_nuc_small += 1; continue
        if info["has_nucleus"] and info["nc_ratio"] > MAX_NC_RATIO:
            rm_nc += 1;        continue
        cells_to_keep.append(cid)

    max_cell_id = int(cell_masks_3d.max())
    max_nuc_id  = int(nuclei_masks_3d.max())
    cell_lut    = np.zeros(max_cell_id + 1, dtype=np.uint16)
    nuc_lut     = np.zeros(max_nuc_id  + 1, dtype=np.uint16)

    for new_id, old_id in enumerate(cells_to_keep, start=1):
        cell_lut[old_id] = new_id
        nid = cell_info[old_id]["nucleus_id"]
        if nid > 0:
            nuc_lut[nid] = new_id

    cell_masks_filtered   = cell_lut[cell_masks_3d]
    nuclei_masks_filtered = nuc_lut[nuclei_masks_3d]

    stats = {
        "n_before":              len(cell_info),
        "n_after":               int(cell_masks_filtered.max()),
        "removed_too_small":     rm_small,
        "removed_no_nucleus":    rm_no_nuc,
        "removed_nucleus_small": rm_nuc_small,
        "removed_nc_ratio":      rm_nc,
    }
    _log(f"Filtering done in {time.time()-t0:.1f}s", 2)
    return cell_masks_filtered, nuclei_masks_filtered, stats


# ── accuracy vs ground truth ─────────────────────────────────────────

def compute_iou(pred, gt):
    inter = ((pred > 0) & (gt > 0)).sum()
    union = ((pred > 0) | (gt > 0)).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def accuracy_report(cell_f, nuclei_f, cell_gt, nuc_gt):
    rep = {
        "segmented_cells":  int(cell_f.max()),
        "segmented_nuclei": int(nuclei_f.max()),
        "gt_cells":         int(cell_gt.max()) if cell_gt is not None else None,
        "gt_nuclei":        int(nuc_gt.max())  if nuc_gt  is not None else None,
        "cell_iou":         compute_iou(cell_f, cell_gt)  if cell_gt is not None else None,
        "nucleus_iou":      compute_iou(nuclei_f, nuc_gt) if nuc_gt  is not None else None,
    }
    if cell_gt is not None:
        ns, ng = rep["segmented_cells"], rep["gt_cells"]
        rep["cell_count_error_pct"] = 100 * abs(ns - ng) / ng if ng > 0 else None
    if nuc_gt is not None:
        ns, ng = rep["segmented_nuclei"], rep["gt_nuclei"]
        rep["nucleus_count_error_pct"] = 100 * abs(ns - ng) / ng if ng > 0 else None
    return rep


# ── plots ────────────────────────────────────────────────────────────

def save_raw_data_plot(nuclei, cells, stem, plots_dir):
    Z = nuclei.shape[0]
    view_z = Z // 2
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].imshow(nuclei[view_z], cmap="gray")
    axes[0, 0].set_title(f"Nuclei (DAPI)  Z={view_z}", fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(cells[view_z], cmap="gray")
    axes[0, 1].set_title(f"Cells (Actin)  Z={view_z}", fontweight="bold")
    axes[0, 1].axis("off")

    axes[1, 0].hist(nuclei[view_z].ravel(), bins=100, color="royalblue", alpha=0.8)
    axes[1, 0].set_title("Nuclei intensity histogram (mid-slice)")
    axes[1, 0].set_xlabel("Intensity"); axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].hist(cells[view_z].ravel(), bins=100, color="seagreen", alpha=0.8)
    axes[1, 1].set_title("Cells intensity histogram (mid-slice)")
    axes[1, 1].set_xlabel("Intensity"); axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(alpha=0.3)

    fig.suptitle(f"{stem}  --  raw data", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = plots_dir / f"{stem}_raw_data.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"Saved: {out.name}", 2)


def save_segmentation_qc_plot(nuclei_norm, cells_norm,
                               nuclei_masks, cell_masks,
                               stem, plots_dir):
    Z = nuclei_norm.shape[0]
    slices = [Z // 4, Z // 2, 3 * Z // 4]
    fig, axes = plt.subplots(len(slices), 3, figsize=(18, 6 * len(slices)))

    for row, z in enumerate(slices):
        cell_bdry = segmentation.find_boundaries(cell_masks[z],   mode="outer")
        nuc_bdry  = segmentation.find_boundaries(nuclei_masks[z], mode="outer")

        axes[row, 0].imshow(nuclei_norm[z], cmap="gray")
        axes[row, 0].imshow(nuclei_masks[z], alpha=0.45, cmap="jet")
        axes[row, 0].set_title(
            f"Nuclei  Z={z}  ({len(np.unique(nuclei_masks[z]))-1} in slice)",
            fontweight="bold")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(cells_norm[z], cmap="gray")
        axes[row, 1].imshow(cell_masks[z], alpha=0.45, cmap="jet")
        axes[row, 1].set_title(
            f"Cells  Z={z}  ({len(np.unique(cell_masks[z]))-1} in slice)",
            fontweight="bold")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(cells_norm[z], cmap="gray")
        axes[row, 2].contour(cell_bdry, colors="red",  linewidths=1.2, levels=[0.5])
        axes[row, 2].contour(nuc_bdry,  colors="cyan", linewidths=1.2, levels=[0.5])
        axes[row, 2].set_title(f"Outlines  Z={z}  red=cells  cyan=nuclei",
                                fontweight="bold")
        axes[row, 2].axis("off")

    fig.suptitle(f"{stem}  --  segmentation QC", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = plots_dir / f"{stem}_segmentation_qc.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"Saved: {out.name}", 2)


def save_accuracy_plot(accuracy, stem, plots_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, kind, iou_key in zip(
        axes,
        ["cells", "nuclei"],
        ["cell_iou", "nucleus_iou"],
    ):
        gt  = accuracy.get(f"gt_{kind}", 0) or 0
        seg = accuracy.get(f"segmented_{kind}", 0)
        iou = accuracy.get(iou_key, 0) or 0
        bars = ax.bar(["Ground truth", "Segmented"], [gt, seg],
                      color=["steelblue", "darkorange"])
        ax.set_title(f"{kind.capitalize()} count  (IoU={iou:.3f})",
                     fontweight="bold")
        ax.set_ylabel("Count")
        for bar, val in zip(bars, [gt, seg]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1, str(val),
                    ha="center", fontweight="bold")
    fig.suptitle(f"{stem}  --  accuracy vs ground truth",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = plots_dir / f"{stem}_accuracy.png"
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    _log(f"Saved: {out.name}", 2)


# ── save masks + summary ─────────────────────────────────────────────

def save_masks(cell_f, nuclei_f, cell_raw, nuclei_raw, seg_dir, stem):
    for arr, name in [
        (cell_f,     "cell_masks_filtered"),
        (nuclei_f,   "nuclei_masks_filtered"),
        (cell_raw,   "cell_masks_unfiltered"),
        (nuclei_raw, "nuclei_masks_unfiltered"),
    ]:
        t0  = time.time()
        out = seg_dir / f"{stem}_{name}.tif"
        tiff.imwrite(str(out), arr.astype(np.uint16))
        _log(f"Saved {name}  [{time.time()-t0:.1f}s]", 2)


def save_summary(path, stem, shape, pxy, pz, filter_stats, accuracy):
    Z, Y, X = shape
    with open(str(path), "w") as f:
        f.write("=" * 60 + "\n3D ORGANOID SEGMENTATION SUMMARY\n"
                + "=" * 60 + "\n\n")
        f.write(f"File:        {stem}\n")
        f.write(f"Dimensions:  {Z} x {Y} x {X}\n")
        f.write(f"Voxel size:  XY={pxy} um  Z={pz} um\n\n")
        f.write("SEGMENTATION PARAMETERS\n")
        f.write(f"  Cell diameter:             {CELL_DIAMETER} px\n")
        f.write(f"  Cell flow threshold:       {CELL_FLOW_THRESHOLD}\n")
        f.write(f"  Cell cellprob threshold:   {CELL_CELLPROB_THRESHOLD}\n")
        f.write(f"  Nuclei diameter:           {NUCLEI_DIAMETER} px\n")
        f.write(f"  Nuclei flow threshold:     {NUCLEI_FLOW_THRESHOLD}\n")
        f.write(f"  Nuclei cellprob threshold: {NUCLEI_CELLPROB_THRESHOLD}\n")
        f.write(f"  Anisotropy:                {pz/pxy:.3f}\n\n")
        f.write("QUALITY CONTROL\n")
        f.write(f"  Min cell volume:    {MIN_CELL_VOLUME_UM3} um3\n")
        f.write(f"  Min nucleus volume: {MIN_NUCLEUS_VOLUME_UM3} um3\n")
        f.write(f"  Max NC ratio:       {MAX_NC_RATIO}\n\n")
        f.write("FILTERING RESULTS\n")
        f.write(f"  Cells before filtering:  {filter_stats['n_before']}\n")
        f.write(f"  Removed (too small):     {filter_stats['removed_too_small']}\n")
        f.write(f"  Removed (no nucleus):    {filter_stats['removed_no_nucleus']}\n")
        f.write(f"  Removed (nucleus small): {filter_stats['removed_nucleus_small']}\n")
        f.write(f"  Removed (NC ratio high): {filter_stats['removed_nc_ratio']}\n")
        f.write(f"  Cells after filtering:   {filter_stats['n_after']}\n\n")
        if accuracy:
            f.write("ACCURACY vs GROUND TRUTH\n")
            for k, v in accuracy.items():
                f.write(f"  {k}: {v}\n")


# ── per-organoid pipeline ────────────────────────────────────────────

def process_one(img_path, labels_dir, seg_dir, plots_dir, use_gpu):
    stem = img_path.stem.replace(".ome", "")
    print()
    print("=" * 60)
    print(f"Processing: {img_path.name}")
    print("=" * 60)
    t_start = time.time()

    t = time.time()
    _log("Loading image...", 1)
    nuclei_raw, cells_raw, pxy, pz = load_ome_tiff(img_path)
    anisotropy = pz / pxy
    Z, Y, X = nuclei_raw.shape
    _log(f"Shape: {Z}x{Y}x{X}  anisotropy: {anisotropy:.2f}  "
         f"[{time.time()-t:.1f}s]", 1)

    cell_gt, nuc_gt = load_ground_truth(stem, labels_dir)
    if cell_gt is not None: _log("Ground-truth cell labels found", 1)
    if nuc_gt  is not None: _log("Ground-truth nucleus labels found", 1)

    t = time.time()
    _log("Saving raw data plot...", 1)
    save_raw_data_plot(nuclei_raw, cells_raw, stem, plots_dir)
    _log(f"Raw plot done  [{time.time()-t:.1f}s]", 1)

    t = time.time()
    _log("Normalising...", 1)
    nuclei_norm = normalize_image(nuclei_raw)
    cells_norm  = normalize_image(cells_raw)
    _log(f"Normalisation done  [{time.time()-t:.1f}s]", 1)

    t = time.time()
    _log("Running 3D segmentation...", 1)
    nuclei_masks_3d, cell_masks_3d = run_segmentation(
        nuclei_norm, cells_norm, anisotropy, use_gpu)
    _log(f"Segmentation done  [{time.time()-t:.1f}s]", 1)

    t = time.time()
    _log("Saving segmentation QC plot...", 1)
    save_segmentation_qc_plot(nuclei_norm, cells_norm,
                               nuclei_masks_3d, cell_masks_3d,
                               stem, plots_dir)
    _log(f"QC plot done  [{time.time()-t:.1f}s]", 1)

    t = time.time()
    _log("Quality control and filtering...", 1)
    voxel_vol = (pxy ** 2) * pz
    cell_info = build_cell_info(cell_masks_3d, nuclei_masks_3d, voxel_vol)
    cell_f, nuclei_f, filter_stats = filter_cells(
        cell_masks_3d, nuclei_masks_3d, cell_info)
    _log(f"QC done -- {filter_stats['n_after']} cells kept  "
         f"[{time.time()-t:.1f}s]", 1)

    accuracy = None
    if cell_gt is not None or nuc_gt is not None:
        t = time.time()
        _log("Computing accuracy metrics...", 1)
        accuracy = accuracy_report(cell_f, nuclei_f, cell_gt, nuc_gt)
        for k, v in accuracy.items():
            _log(f"{k}: {v}", 2)
        save_accuracy_plot(accuracy, stem, plots_dir)
        _log(f"Accuracy done  [{time.time()-t:.1f}s]", 1)

    t = time.time()
    _log("Saving masks...", 1)
    save_masks(cell_f, nuclei_f, cell_masks_3d, nuclei_masks_3d,
               seg_dir, stem)
    _log(f"Masks saved  [{time.time()-t:.1f}s]", 1)

    save_summary(seg_dir / f"{stem}_segmentation_summary.txt",
                 stem, (Z, Y, X), pxy, pz, filter_stats, accuracy)

    elapsed = time.time() - t_start
    _log(f"Total: {elapsed:.1f}s  ({elapsed/60:.1f} min)", 1)


# ── batch runner ─────────────────────────────────────────────────────

def run_batch():
    input_dir  = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    images_dir = input_dir  / "images"
    labels_dir = input_dir  / "labels"
    seg_dir    = output_dir / "segmentation"
    plots_dir  = output_dir / "plots"

    seg_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    seen, img_files = set(), []
    for p in (sorted(images_dir.glob("*.ome.tif")) +
              sorted(images_dir.glob("*.tif"))):
        if p not in seen:
            seen.add(p)
            img_files.append(p)

    if not img_files:
        print(f"No .tif / .ome.tif files found in {images_dir}")
        sys.exit(1)

    use_gpu = torch.cuda.is_available()
    print(f"GPU available: {use_gpu}")
    print(f"Found {len(img_files)} image(s) to process")

    ok, failed = 0, []
    for img_path in img_files:
        try:
            process_one(img_path, labels_dir, seg_dir, plots_dir, use_gpu)
            ok += 1
        except Exception as e:
            print(f"\nERROR processing {img_path.name}: {e}")
            import traceback; traceback.print_exc()
            failed.append(img_path.name)

    print()
    print("=" * 60)
    print(f"Batch complete: {ok}/{len(img_files)} succeeded")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print("=" * 60)


if __name__ == "__main__":
    run_batch()
