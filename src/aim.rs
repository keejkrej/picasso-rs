//! Adaptive Intersection Maximization (AIM) drift correction.
//!
//! Ported from `picasso.aim` (Ma et al., 2024). Estimates drift directly from localization
//! point clouds by maximizing the intersection count of localizations within `intersect_d`
//! across frame segments.

use crate::types::{Localization, Result};
use marray::np::{Array1, Array2};

/// Drift trajectory returned by [`aim`].
#[derive(Debug, Clone)]
pub struct AimResult {
    /// Undrifted localizations.
    pub locs: Vec<Localization>,
    /// Per-frame x drift trajectory (length `n_frames`).
    pub drift_x: Array1<f64>,
    /// Per-frame y drift trajectory (length `n_frames`).
    pub drift_y: Array1<f64>,
    /// Optional per-frame z drift trajectory (length `n_frames`, in nm).
    pub drift_z: Option<Array1<f64>>,
}

/// Finds the sub-bin shift `(dx, dy)` that maximizes the 2D intersection count between
/// reference points `ref_pts` and target segment points `seg_pts` within search radius `roi_r`
/// using bin width `intersect_d` (`picasso.aim`).
fn estimate_segment_shift_2d(
    ref_pts: &[(f64, f64)],
    seg_pts: &[(f64, f64)],
    intersect_d: f64,
    roi_r: f64,
) -> (f64, f64) {
    if ref_pts.is_empty() || seg_pts.is_empty() {
        return (0.0, 0.0);
    }
    let step = intersect_d.max(0.05);
    let n_bins = ((2.0 * roi_r / step).ceil() as usize).max(3) | 1;
    let half = (n_bins / 2) as isize;
    let mut grid = Array2::<f64>::zeros((n_bins, n_bins));
    let r2_max = roi_r * roi_r;

    for &(rx, ry) in ref_pts {
        for &(sx, sy) in seg_pts {
            let dx = sx - rx;
            let dy = sy - ry;
            if dx * dx + dy * dy <= r2_max {
                let bx = ((dx / step).round() as isize) + half;
                let by = ((dy / step).round() as isize) + half;
                if bx >= 0 && bx < (n_bins as isize) && by >= 0 && by < (n_bins as isize) {
                    grid[[by as usize, bx as usize]] += 1.0;
                }
            }
        }
    }

    // Find peak bin and compute local 3x3 weighted centroid for sub-bin accuracy
    let mut best_val = -1.0;
    let mut best_iy = half as usize;
    let mut best_ix = half as usize;
    for iy in 0..n_bins {
        for ix in 0..n_bins {
            if grid[[iy, ix]] > best_val {
                best_val = grid[[iy, ix]];
                best_iy = iy;
                best_ix = ix;
            }
        }
    }
    if best_val <= 0.0 {
        return (0.0, 0.0);
    }

    let mut sum_w = 0.0;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for ky in best_iy.saturating_sub(1)..=(best_iy + 1).min(n_bins - 1) {
        for kx in best_ix.saturating_sub(1)..=(best_ix + 1).min(n_bins - 1) {
            let w = grid[[ky, kx]];
            sum_w += w;
            sum_x += w * ((kx as isize - half) as f64) * step;
            sum_y += w * ((ky as isize - half) as f64) * step;
        }
    }
    if sum_w > 0.0 {
        (sum_x / sum_w, sum_y / sum_w)
    } else {
        (0.0, 0.0)
    }
}

/// Linearly interpolates segment knot values `(seg_centers, seg_vals)` onto every frame `0..n_frames`.
pub fn interpolate_drift_knots(
    knot_frames: &[f64],
    knot_shifts: &[f64],
    n_frames: usize,
) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(n_frames);
    if knot_frames.is_empty() || n_frames == 0 {
        return out;
    }
    if knot_frames.len() == 1 {
        out.fill(knot_shifts[0]);
        return out;
    }

    for f in 0..n_frames {
        let x = f as f64;
        if x <= knot_frames[0] {
            out[f] = knot_shifts[0];
        } else if x >= *knot_frames.last().unwrap() {
            out[f] = *knot_shifts.last().unwrap();
        } else {
            let mut k = 0;
            while k + 1 < knot_frames.len() && knot_frames[k + 1] < x {
                k += 1;
            }
            let x0 = knot_frames[k];
            let x1 = knot_frames[k + 1];
            let t = if (x1 - x0).abs() > 1e-12 {
                (x - x0) / (x1 - x0)
            } else {
                0.0
            };
            out[f] = knot_shifts[k] * (1.0 - t) + knot_shifts[k + 1] * t;
        }
    }

    // Zero-mean or anchor at frame 0
    let f0 = out[0];
    for v in out.iter_mut() {
        *v -= f0;
    }
    out
}

/// Performs Adaptive Intersection Maximization (AIM) drift correction (`picasso.aim.aim`).
///
/// # Arguments
/// * `locs` - Input localizations.
/// * `segmentation` - Frame interval per temporal segment (e.g., `100` frames).
/// * `intersect_d` - Intersection distance in camera pixels (e.g., `0.15` px ≈ 20 nm).
/// * `roi_r` - Maximum search radius between segments in camera pixels (e.g., `0.5` px ≈ 60 nm).
pub fn aim(
    locs: &[Localization],
    segmentation: usize,
    intersect_d: f64,
    roi_r: f64,
) -> Result<AimResult> {
    if locs.is_empty() {
        return Err("Cannot run AIM on empty localizations".into());
    }
    let max_frame = locs.iter().map(|l| l.frame as usize).max().unwrap_or(0);
    let n_frames = max_frame + 1;
    let seg_len = segmentation.max(1);
    let n_segs = n_frames.div_ceil(seg_len).max(1);

    let mut seg_pts: Vec<Vec<(f64, f64)>> = vec![Vec::new(); n_segs];
    for l in locs {
        let s_idx = (l.frame as usize / seg_len).min(n_segs - 1);
        seg_pts[s_idx].push((l.x, l.y));
    }

    let mut knot_frames = Vec::with_capacity(n_segs);
    let mut knot_dx = vec![0.0f64; n_segs];
    let mut knot_dy = vec![0.0f64; n_segs];

    for s in 0..n_segs {
        let f_start = s * seg_len;
        let f_end = ((s + 1) * seg_len).min(n_frames);
        knot_frames.push(0.5 * ((f_start + f_end.saturating_sub(1)) as f64));
    }

    // Round 1: align each segment to cumulative reference starting from segment 0
    let mut ref_cloud = seg_pts[0].clone();
    for s in 1..n_segs {
        let prev_dx = knot_dx[s - 1];
        let prev_dy = knot_dy[s - 1];
        let shifted_seg: Vec<(f64, f64)> = seg_pts[s]
            .iter()
            .map(|&(x, y)| (x - prev_dx, y - prev_dy))
            .collect();
        let (ddx, ddy) = estimate_segment_shift_2d(&ref_cloud, &shifted_seg, intersect_d, roi_r);
        knot_dx[s] = prev_dx + ddx;
        knot_dy[s] = prev_dy + ddy;
        for &(x, y) in &seg_pts[s] {
            ref_cloud.push((x - knot_dx[s], y - knot_dy[s]));
        }
    }

    // Round 2: refine each segment against the combined undrifted point cloud
    for s in 0..n_segs {
        let shifted_seg: Vec<(f64, f64)> = seg_pts[s]
            .iter()
            .map(|&(x, y)| (x - knot_dx[s], y - knot_dy[s]))
            .collect();
        let (ddx, ddy) = estimate_segment_shift_2d(&ref_cloud, &shifted_seg, intersect_d, roi_r);
        knot_dx[s] += ddx;
        knot_dy[s] += ddy;
    }

    let drift_x = interpolate_drift_knots(&knot_frames, &knot_dx, n_frames);
    let drift_y = interpolate_drift_knots(&knot_frames, &knot_dy, n_frames);

    let mut undrifted = locs.to_vec();
    for l in &mut undrifted {
        let f = (l.frame as usize).min(n_frames - 1);
        l.x -= drift_x[f];
        l.y -= drift_y[f];
    }

    Ok(AimResult {
        locs: undrifted,
        drift_x,
        drift_y,
        drift_z: None,
    })
}
