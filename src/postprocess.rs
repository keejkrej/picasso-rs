//! Post-processing of localizations: spatial indexing, ROI picking, trajectory linking,
//! dark-time kinetics, NeNA localization precision, pair correlation, RCC drift correction,
//! multi-channel alignment, and group properties.
//!
//! Ported from `picasso.postprocess`.

use crate::aim::interpolate_drift_knots;
use crate::imageprocess::{get_image_shift, rcc as image_rcc};
use crate::lib_utils::{is_loc_at, point_in_polygon, point_in_rectangle};
use crate::render::{RenderBlurMethod, default_viewport_2d, render, render_hist};
use crate::types::{Localization, PickShape, Result};
use marray::np::{self, Array1, Array2};
use marray::sp::optimize::minimize;
use marray::sp::spatial::KDTree;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// 2D spatial grid index for fast neighborhood queries (`picasso.postprocess.get_index_blocks`).
#[derive(Debug, Clone)]
pub struct SpatialBlocks {
    pub block_size: f64,
    pub n_blocks_y: usize,
    pub n_blocks_x: usize,
    pub blocks: Vec<Vec<usize>>,
}

impl SpatialBlocks {
    /// Builds a 2D block index over `locs` with grid spacing `block_size`.
    pub fn build(locs: &[Localization], block_size: f64) -> Self {
        let bs = block_size.max(1e-3);
        let max_x = locs.iter().map(|l| l.x).fold(0.0f64, f64::max).max(0.0);
        let max_y = locs.iter().map(|l| l.y).fold(0.0f64, f64::max).max(0.0);
        let n_blocks_x = ((max_x / bs).floor() as usize) + 2;
        let n_blocks_y = ((max_y / bs).floor() as usize) + 2;
        let mut blocks = vec![Vec::new(); n_blocks_y * n_blocks_x];

        for (idx, l) in locs.iter().enumerate() {
            let bx = ((l.x.max(0.0) / bs).floor() as usize).min(n_blocks_x - 1);
            let by = ((l.y.max(0.0) / bs).floor() as usize).min(n_blocks_y - 1);
            blocks[by * n_blocks_x + bx].push(idx);
        }

        Self {
            block_size: bs,
            n_blocks_y,
            n_blocks_x,
            blocks,
        }
    }

    /// Returns indices of all localizations in the 3x3 block neighborhood around `(x, y)`
    /// (`picasso.postprocess.get_block_locs_at`).
    pub fn indices_around(&self, x: f64, y: f64) -> Vec<usize> {
        let bx = (x.max(0.0) / self.block_size).floor() as isize;
        let by = (y.max(0.0) / self.block_size).floor() as isize;
        let mut out = Vec::new();
        for ky in (by - 1).max(0)..=(by + 1).min((self.n_blocks_y - 1) as isize) {
            for kx in (bx - 1).max(0)..=(bx + 1).min((self.n_blocks_x - 1) as isize) {
                let cell = &self.blocks[(ky as usize) * self.n_blocks_x + (kx as usize)];
                out.extend_from_slice(cell);
            }
        }
        out
    }
}

/// Extracts localizations inside each ROI of `pick_shape` (`picasso.postprocess.picked_locs`).
///
/// If `add_group == true`, assigns the pick index `0..n_picks` to `loc.group` and returns a single
/// flattened `Vec<Localization>` (inside a 1-element outer vector); otherwise returns one
/// `Vec<Localization>` per pick region.
pub fn picked_locs(
    locs: &[Localization],
    pick_shape: &PickShape,
    add_group: bool,
) -> Vec<Vec<Localization>> {
    let mut picked_lists: Vec<Vec<Localization>> = Vec::new();

    match pick_shape {
        PickShape::Circles { centers, diameter } => {
            let r = 0.5 * diameter;
            for (g_idx, &(cx, cy)) in centers.iter().enumerate() {
                let mask = is_loc_at(locs, cx, cy, r);
                let mut group_locs = Vec::new();
                for (l, inside) in locs.iter().zip(mask) {
                    if inside {
                        let mut c = l.clone();
                        if add_group {
                            c.group = Some(g_idx as i32);
                        }
                        group_locs.push(c);
                    }
                }
                picked_lists.push(group_locs);
            }
        }
        PickShape::Rectangles { segments, width } => {
            for (g_idx, &(p1, p2)) in segments.iter().enumerate() {
                let mut group_locs = Vec::new();
                for l in locs {
                    if point_in_rectangle(l.x, l.y, p1, p2, *width) {
                        let mut c = l.clone();
                        if add_group {
                            c.group = Some(g_idx as i32);
                        }
                        group_locs.push(c);
                    }
                }
                picked_lists.push(group_locs);
            }
        }
        PickShape::Polygons { polygons } => {
            for (g_idx, poly) in polygons.iter().enumerate() {
                let mut group_locs = Vec::new();
                for l in locs {
                    if point_in_polygon(l.x, l.y, poly) {
                        let mut c = l.clone();
                        if add_group {
                            c.group = Some(g_idx as i32);
                        }
                        group_locs.push(c);
                    }
                }
                picked_lists.push(group_locs);
            }
        }
    }

    if add_group {
        let mut flat: Vec<Localization> = picked_lists.into_iter().flatten().collect();
        flat.sort_by(|a, b| a.group.cmp(&b.group).then_with(|| a.frame.cmp(&b.frame)));
        vec![flat]
    } else {
        picked_lists
    }
}

/// Computes local localization density within radius `r` for each localization
/// (`picasso.postprocess.compute_local_density`) using `marray::sp::spatial::KDTree`.
pub fn compute_local_density(locs: &mut [Localization], r: f64) -> Result<()> {
    if locs.is_empty() {
        return Ok(());
    }
    let has_z = locs.iter().all(|l| l.z.is_some());
    let n = locs.len();
    let dim = if has_z { 3 } else { 2 };
    let mut coords = Array2::<f64>::zeros((n, dim));
    for (i, l) in locs.iter().enumerate() {
        coords[[i, 0]] = l.x;
        coords[[i, 1]] = l.y;
        if has_z {
            coords[[i, 2]] = l.z.unwrap_or(0.0);
        }
    }

    let tree = KDTree::new(&coords);
    for i in 0..n {
        let pt: Vec<f64> = (0..dim).map(|d| coords[[i, d]]).collect();
        let neighbors = tree.query_radius(&pt, r);
        locs[i].density = Some(neighbors.len() as u32);
    }
    Ok(())
}

/// Automatically finds circular picks throughout the field of view whose localization count
/// and dark-time statistics match the provided reference picks (`picasso.postprocess.pick_similar`).
#[allow(clippy::too_many_arguments)]
pub fn pick_similar(
    locs: &[Localization],
    reference_centers: &[(f64, f64)],
    diameter: f64,
    min_n_std: f64,
    max_n_std: f64,
    min_rmsd_std: f64,
    max_rmsd_std: f64,
) -> Vec<(f64, f64)> {
    if locs.is_empty() || reference_centers.is_empty() {
        return reference_centers.to_vec();
    }
    let r = 0.5 * diameter;
    let mut ref_counts = Vec::with_capacity(reference_centers.len());
    let mut ref_rmsds = Vec::with_capacity(reference_centers.len());

    for &(cx, cy) in reference_centers {
        let mask = is_loc_at(locs, cx, cy, r);
        let picked: Vec<&Localization> = locs
            .iter()
            .zip(mask)
            .filter_map(|(l, m)| if m { Some(l) } else { None })
            .collect();
        if picked.is_empty() {
            continue;
        }
        let n = picked.len() as f64;
        let mx = picked.iter().map(|l| l.x).sum::<f64>() / n;
        let my = picked.iter().map(|l| l.y).sum::<f64>() / n;
        let rmsd = (picked
            .iter()
            .map(|l| (l.x - mx).powi(2) + (l.y - my).powi(2))
            .sum::<f64>()
            / n)
            .sqrt();
        ref_counts.push(n);
        ref_rmsds.push(rmsd);
    }

    if ref_counts.is_empty() {
        return reference_centers.to_vec();
    }

    let c_arr = Array1::from_vec(ref_counts);
    let r_arr = Array1::from_vec(ref_rmsds);
    let mean_c = np::mean(&c_arr);
    let std_c = np::std(&c_arr, 0.0).max(1.0);
    let mean_r = np::mean(&r_arr);
    let std_r = np::std(&r_arr, 0.0).max(1e-3);

    let min_c = (mean_c - min_n_std * std_c).max(1.0);
    let max_c = mean_c + max_n_std * std_c;
    let min_rmsd = (mean_r - min_rmsd_std * std_r).max(0.0);
    let max_rmsd = mean_r + max_rmsd_std * std_r;

    // Scan candidate centers on a grid of spacing r
    let ((y_min, x_min), (y_max, x_max)) = default_viewport_2d(locs);
    let step = r.max(0.5);
    let mut candidates = reference_centers.to_vec();

    let mut y = y_min + r;
    while y <= y_max - r {
        let mut x = x_min + r;
        while x <= x_max - r {
            let mask = is_loc_at(locs, x, y, r);
            let picked: Vec<&Localization> = locs
                .iter()
                .zip(mask)
                .filter_map(|(l, m)| if m { Some(l) } else { None })
                .collect();
            let n = picked.len() as f64;
            if n >= min_c && n <= max_c {
                let mx = picked.iter().map(|l| l.x).sum::<f64>() / n;
                let my = picked.iter().map(|l| l.y).sum::<f64>() / n;
                let rmsd = (picked
                    .iter()
                    .map(|l| (l.x - mx).powi(2) + (l.y - my).powi(2))
                    .sum::<f64>()
                    / n)
                    .sqrt();
                if rmsd >= min_rmsd
                    && rmsd <= max_rmsd
                    && candidates
                        .iter()
                        .all(|&(ox, oy)| (ox - mx).powi(2) + (oy - my).powi(2) > (diameter * 0.8).powi(2))
                {
                    candidates.push((mx, my));
                }
            }
            x += step;
        }
        y += step;
    }

    candidates
}

/// Links localizations across consecutive frames into single binding events (`picasso.postprocess.link`),
/// bridging up to `max_dark_time` dark frames within spatial radius `r_max`.
///
/// Populates `len`, `n`, `photon_rate`, and `dark` (if `group` is present) on the resulting
/// linked localizations.
pub fn link(
    locs: &[Localization],
    r_max: f64,
    max_dark_time: u32,
    combine_mode: LinkCombineMode,
    remove_ambiguous_lengths: bool,
) -> Vec<Localization> {
    if locs.is_empty() {
        return Vec::new();
    }

    // Sort by (group, frame)
    let mut sorted = locs.to_vec();
    sorted.sort_by(|a, b| {
        a.group
            .unwrap_or(0)
            .cmp(&b.group.unwrap_or(0))
            .then_with(|| a.frame.cmp(&b.frame))
    });

    let loc_group = trace_loc_groups(&sorted, r_max, max_dark_time);
    let mut linked = link_loc_groups(&sorted, &loc_group, combine_mode, remove_ambiguous_lengths);

    if linked.iter().any(|l| l.group.is_some()) {
        compute_dark_times(&mut linked);
    }

    linked
}

/// Mode for combining constituent localizations within a linked binding trajectory (`picasso.postprocess.link_loc_groups`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LinkCombineMode {
    /// Inverse-variance (`1 / lpx^2`, `1 / lpy^2`) weighted average of coordinates (default in Picasso).
    #[default]
    Average,
    /// Keep the coordinates of the first localization in the trajectory.
    First,
}

/// Assigns trajectory group indices `0..n_trajectories` across frames (`picasso.postprocess.get_link_groups`).
pub fn trace_loc_groups(locs: &[Localization], r_max: f64, max_dark_time: u32) -> Vec<i32> {
    let n = locs.len();
    let mut loc_group = vec![-1i32; n];
    if n == 0 {
        return loc_group;
    }
    let r2_max = r_max * r_max;
    let mut current_traj = 0i32;

    for i in 0..n {
        if loc_group[i] == -1 {
            loc_group[i] = current_traj;
            let mut last_idx = i;

            for j in (i + 1)..n {
                if locs[j].group != locs[last_idx].group {
                    break;
                }
                if loc_group[j] != -1 {
                    continue;
                }
                let dt = locs[j].frame.saturating_sub(locs[last_idx].frame);
                if dt == 0 {
                    continue;
                }
                if dt > max_dark_time + 1 {
                    break;
                }
                let dx = locs[j].x - locs[last_idx].x;
                let dy = locs[j].y - locs[last_idx].y;
                if dx * dx + dy * dy <= r2_max {
                    loc_group[j] = current_traj;
                    last_idx = j;
                }
            }
            current_traj += 1;
        }
    }

    loc_group
}

/// Aggregates localizations sharing the same trajectory index in `loc_group`
/// (`picasso.postprocess.link_loc_groups`).
pub fn link_loc_groups(
    locs: &[Localization],
    loc_group: &[i32],
    combine_mode: LinkCombineMode,
    remove_ambiguous_lengths: bool,
) -> Vec<Localization> {
    if locs.is_empty() {
        return Vec::new();
    }
    let min_frame = locs.iter().map(|l| l.frame).min().unwrap_or(0);
    let max_frame = locs.iter().map(|l| l.frame).max().unwrap_or(0);

    let mut groups: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for (idx, &g) in loc_group.iter().enumerate() {
        if g >= 0 {
            groups.entry(g).or_default().push(idx);
        }
    }

    let mut out = Vec::with_capacity(groups.len());
    for (_, indices) in groups {
        if indices.is_empty() {
            continue;
        }
        let f_first = locs[indices[0]].frame;
        let f_last = locs[*indices.last().unwrap()].frame;
        if remove_ambiguous_lengths && (f_first == min_frame || f_last == max_frame) {
            continue;
        }

        let n_count = indices.len() as u32;
        let len_frames = f_last - f_first + 1;
        let photons_sum: f64 = indices.iter().map(|&k| locs[k].photons).sum();
        let bg_sum: f64 = indices.iter().map(|&k| locs[k].bg).sum();
        let ng_sum: f64 = indices.iter().map(|&k| locs[k].net_gradient).sum();
        let sx_mean = indices.iter().map(|&k| locs[k].sx).sum::<f64>() / (n_count as f64);
        let sy_mean = indices.iter().map(|&k| locs[k].sy).sum::<f64>() / (n_count as f64);

        let mut wx_sum = 0.0;
        let mut wy_sum = 0.0;
        let mut x_wsum = 0.0;
        let mut y_wsum = 0.0;
        for &k in &indices {
            let wx = 1.0 / locs[k].lpx.max(1e-6).powi(2);
            let wy = 1.0 / locs[k].lpy.max(1e-6).powi(2);
            wx_sum += wx;
            wy_sum += wy;
            x_wsum += wx * locs[k].x;
            y_wsum += wy * locs[k].y;
        }

        let (x, y) = match combine_mode {
            LinkCombineMode::Average => (x_wsum / wx_sum.max(1e-12), y_wsum / wy_sum.max(1e-12)),
            LinkCombineMode::First => (locs[indices[0]].x, locs[indices[0]].y),
        };
        let lpx = (1.0 / wx_sum.max(1e-12)).sqrt();
        let lpy = (1.0 / wy_sum.max(1e-12)).sqrt();

        let z = if indices.iter().all(|&k| locs[k].z.is_some()) {
            let mut wz_sum = 0.0;
            let mut z_wsum = 0.0;
            for &k in &indices {
                let wz = 1.0 / locs[k].lpz.unwrap_or(1.0).max(1e-6).powi(2);
                wz_sum += wz;
                z_wsum += wz * locs[k].z.unwrap_or(0.0);
            }
            Some(z_wsum / wz_sum.max(1e-12))
        } else {
            None
        };

        let a = sx_mean.max(sy_mean).max(1e-12);
        let b = sx_mean.min(sy_mean);
        let ellipticity = (a - b) / a;

        out.push(Localization {
            frame: f_first,
            x,
            y,
            z,
            photons: photons_sum,
            sx: sx_mean,
            sy: sy_mean,
            bg: bg_sum,
            lpx,
            lpy,
            ellipticity,
            net_gradient: ng_sum,
            group: locs[indices[0]].group,
            len: Some(len_frames),
            n: Some(n_count),
            photon_rate: Some(photons_sum / (n_count as f64)),
            ..Default::default()
        });
    }

    out
}

/// Computes dark times (in frames) between consecutive linked binding events within each `group`
/// (`picasso.postprocess.compute_dark_times`). Sets `dark = Some(-1)` for the last event in a group.
pub fn compute_dark_times(locs: &mut [Localization]) {
    let n = locs.len();
    for i in 0..n {
        let g = locs[i].group;
        let len_i = locs[i].len.unwrap_or(1);
        let end_frame = locs[i].frame + len_i - 1;
        let mut found_next = None;
        for j in (i + 1)..n {
            if locs[j].group == g && locs[j].frame >= end_frame {
                found_next = Some((locs[j].frame as i32) - (end_frame as i32) - 1);
                break;
            }
        }
        locs[i].dark = Some(found_next.unwrap_or(-1));
    }
}

/// Result of Nearest-Neighbor Analysis (NeNA) experimental localization precision estimation
/// (`picasso.postprocess.nena`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NenaResult {
    /// Experimental 1D localization precision `s` in camera pixels (`nena_px * pixelsize = nm`).
    pub nena_px: f64,
    /// Fitted parameters `[s, a_short, a_corr, w, d_corr, b_bg]`.
    pub best_values: [f64; 6],
    /// Distance bin centers `d` in camera pixels.
    pub distances: Vec<f64>,
    /// Normalized empirical probability density of adjacent-frame nearest-neighbor distances.
    pub probability_density: Vec<f64>,
    /// Fitted NeNA model probability density curve evaluated at `distances`.
    pub fitted_curve: Vec<f64>,
}

/// Evaluates the NeNA adjacent-frame nearest-neighbor distance distribution model
/// (Endesfelder et al., 2014; `picasso.postprocess._nena_model`):
/// `f(d) = (d / (2 * s^2)) * exp(-d^2 / (4 * s^2)) * A_short + A_corr * Gaussian(d, d_corr, w) + b * d`.
#[inline]
pub fn nena_model(d: f64, params: &[f64; 6]) -> f64 {
    let s = params[0].max(1e-5);
    let a_short = params[1].max(0.0);
    let a_corr = params[2].max(0.0);
    let w = params[3].max(1e-4);
    let d_corr = params[4].max(0.0);
    let b = params[5].max(0.0);

    let rayleigh = (d / (2.0 * s * s)) * (-d * d / (4.0 * s * s)).exp() * a_short;
    let corr = (a_corr / (w * (2.0 * std::f64::consts::PI).sqrt()))
        * (-0.5 * ((d - d_corr) / w).powi(2)).exp();
    let linear_bg = b * d;
    rayleigh + corr + linear_bg
}

/// Estimates experimental localization precision using Nearest-Neighbor Analysis (NeNA,
/// Endesfelder et al. 2014; `picasso.postprocess.nena`) via `marray::sp::optimize::minimize`.
pub fn nena(locs: &[Localization]) -> Result<NenaResult> {
    if locs.len() < 10 {
        return Err("At least 10 localizations are required for NeNA".into());
    }
    let mut med_lpx: Vec<f64> = locs
        .iter()
        .map(|l| 0.5 * (l.lpx + l.lpy))
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect();
    med_lpx.sort_by(|a, b| a.total_cmp(b));
    let s_init = if med_lpx.is_empty() {
        0.1
    } else {
        med_lpx[med_lpx.len() / 2].clamp(0.01, 2.0)
    };

    let d_max = (10.0 * s_init).max(0.5);
    let r2_max = d_max * d_max;

    // Group localizations by frame to find nearest neighbors in frame t + 1
    let mut by_frame: BTreeMap<u32, Vec<(f64, f64)>> = BTreeMap::new();
    for l in locs {
        by_frame.entry(l.frame).or_default().push((l.x, l.y));
    }

    let mut nnds = Vec::new();
    for (&f, pts_t) in &by_frame {
        if let Some(pts_next) = by_frame.get(&(f + 1)) {
            for &(x0, y0) in pts_t {
                let mut best_d2 = f64::INFINITY;
                for &(x1, y1) in pts_next {
                    let d2 = (x1 - x0).powi(2) + (y1 - y0).powi(2);
                    if d2 < best_d2 {
                        best_d2 = d2;
                    }
                }
                if best_d2 > 0.0 && best_d2 <= r2_max {
                    nnds.push(best_d2.sqrt());
                }
            }
        }
    }

    if nnds.len() < 5 {
        // Fallback if movie has sparse frames: use consecutive localization pairs
        let mut sorted = locs.to_vec();
        sorted.sort_by_key(|l| l.frame);
        for w in sorted.windows(2) {
            let d = ((w[1].x - w[0].x).powi(2) + (w[1].y - w[0].y).powi(2)).sqrt();
            if d > 0.0 && d <= d_max {
                nnds.push(d);
            }
        }
    }

    if nnds.is_empty() {
        return Ok(NenaResult {
            nena_px: s_init,
            best_values: [s_init, 1.0, 0.0, s_init, 2.0 * s_init, 0.0],
            distances: vec![0.0, d_max],
            probability_density: vec![0.0, 0.0],
            fitted_curve: vec![0.0, 0.0],
        });
    }

    let n_bins = 100usize;
    let bin_w = d_max / (n_bins as f64);
    let mut counts = vec![0.0f64; n_bins];
    for &d in &nnds {
        let b = ((d / bin_w).floor() as usize).min(n_bins - 1);
        counts[b] += 1.0;
    }
    let area = (nnds.len() as f64) * bin_w;
    for c in &mut counts {
        *c /= area.max(1e-12);
    }
    let distances: Vec<f64> = (0..n_bins).map(|i| ((i as f64) + 0.5) * bin_w).collect();

    let x0 = Array1::from_vec(vec![s_init, 0.7, 0.2, s_init, 2.0 * s_init, 0.01]);
    let d_ref = distances.clone();
    let y_ref = counts.clone();

    let opt = minimize(
        move |p: &Array1<f64>| {
            let params = [
                p[0].abs().max(1e-4),
                p[1].max(0.0),
                p[2].max(0.0),
                p[3].abs().max(1e-4),
                p[4].abs(),
                p[5].max(0.0),
            ];
            let mut sse = 0.0;
            for (d, &y) in d_ref.iter().zip(y_ref.iter()) {
                let diff = nena_model(*d, &params) - y;
                sse += diff * diff;
            }
            sse
        },
        &x0,
        "nelder-mead",
        Some(1e-6),
        Some(1000),
    )?;

    let best = [
        opt.x[0].abs().max(1e-4),
        opt.x[1].max(0.0),
        opt.x[2].max(0.0),
        opt.x[3].abs().max(1e-4),
        opt.x[4].abs(),
        opt.x[5].max(0.0),
    ];
    let fitted_curve: Vec<f64> = distances.iter().map(|&d| nena_model(d, &best)).collect();

    Ok(NenaResult {
        nena_px: best[0],
        best_values: best,
        distances,
        probability_density: counts,
        fitted_curve,
    })
}

/// Computes the 2D radial pair-correlation function `g(r)` (`picasso.postprocess.pair_correlation`).
///
/// Returns `(bin_lower_edges, pair_correlation_curve)`.
pub fn pair_correlation(
    locs: &[Localization],
    bin_size: f64,
    r_max: f64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let vp = default_viewport_2d(locs);
    let oversampling = (1.0 / bin_size.max(1e-3)).round().max(1.0);
    let (_, img) = render_hist(locs, oversampling, vp);
    let acorr = crate::imageprocess::xcorr(&img, &img)?;
    let (h, w) = acorr.dim();
    let cy = (h / 2) as f64;
    let cx = (w / 2) as f64;
    let n_bins = (r_max / bin_size).ceil().max(1.0) as usize;
    let mut ring_sum = vec![0.0f64; n_bins];
    let mut ring_cnt = vec![0usize; n_bins];

    for iy in 0..h {
        let dy = ((iy as f64) - cy) / oversampling;
        for ix in 0..w {
            let dx = ((ix as f64) - cx) / oversampling;
            let r = (dx * dx + dy * dy).sqrt();
            if r > 0.0 && r < r_max {
                let b = ((r / bin_size).floor() as usize).min(n_bins - 1);
                ring_sum[b] += acorr[[iy, ix]];
                ring_cnt[b] += 1;
            }
        }
    }

    let bins_lower: Vec<f64> = (0..n_bins).map(|b| (b as f64) * bin_size).collect();
    let curve: Vec<f64> = (0..n_bins)
        .map(|b| {
            if ring_cnt[b] > 0 {
                ring_sum[b] / (ring_cnt[b] as f64)
            } else {
                0.0
            }
        })
        .collect();
    Ok((bins_lower, curve))
}

/// Performs Redundant Cross-Correlation (RCC) drift correction on `locs` (`picasso.postprocess.undrift`).
///
/// Splits the movie into segments of `segmentation` frames, renders each segment, solves the
/// overdetermined pairwise cross-correlation shift matrix, interpolates per-frame `(drift_x, drift_y)`,
/// and subtracts the drift from all localizations.
pub fn undrift(
    locs: &[Localization],
    segmentation: usize,
    max_iter_or_shift: Option<usize>,
    oversampling: f64,
) -> Result<(Vec<Localization>, Array1<f64>, Array1<f64>)> {
    if locs.is_empty() {
        return Err("Cannot undrift empty localizations".into());
    }
    let max_frame = locs.iter().map(|l| l.frame as usize).max().unwrap_or(0);
    let n_frames = max_frame + 1;
    let seg_len = segmentation.max(1);
    let n_segs = n_frames.div_ceil(seg_len).max(2);
    let vp = default_viewport_2d(locs);

    let mut seg_locs: Vec<Vec<Localization>> = vec![Vec::new(); n_segs];
    for l in locs {
        let s = (l.frame as usize / seg_len).min(n_segs - 1);
        seg_locs[s].push(l.clone());
    }

    let mut images = Vec::with_capacity(n_segs);
    let mut knot_frames = Vec::with_capacity(n_segs);
    for (s, group) in seg_locs.iter().enumerate() {
        let (_, img) = render(group, Some(vp), oversampling, RenderBlurMethod::Gaussian, 0.0)?;
        images.push(img);
        let f_start = s * seg_len;
        let f_end = ((s + 1) * seg_len).min(n_frames);
        knot_frames.push(0.5 * ((f_start + f_end.saturating_sub(1)) as f64));
    }

    let (shifts_x_os, shifts_y_os) = image_rcc(&images, max_iter_or_shift)?;
    let knot_dx: Vec<f64> = shifts_x_os.iter().map(|&v| v / oversampling).collect();
    let knot_dy: Vec<f64> = shifts_y_os.iter().map(|&v| v / oversampling).collect();

    let drift_x = interpolate_drift_knots(&knot_frames, &knot_dx, n_frames);
    let drift_y = interpolate_drift_knots(&knot_frames, &knot_dy, n_frames);

    let mut undrifted = locs.to_vec();
    for l in &mut undrifted {
        let f = (l.frame as usize).min(n_frames - 1);
        l.x -= drift_x[f];
        l.y -= drift_y[f];
    }

    Ok((undrifted, drift_x, drift_y))
}

/// Estimates and subtracts drift from fiducial / picked localization groups
/// (`picasso.postprocess.undrift_from_picked`).
pub fn undrift_from_picked(
    locs: &[Localization],
    picked_groups: &[Vec<Localization>],
) -> Result<(Vec<Localization>, Array1<f64>, Array1<f64>)> {
    if locs.is_empty() || picked_groups.is_empty() {
        return Err("Both locs and picked_groups must be non-empty".into());
    }
    let max_frame = locs.iter().map(|l| l.frame as usize).max().unwrap_or(0);
    let n_frames = max_frame + 1;

    let mut sum_dx = vec![0.0f64; n_frames];
    let mut sum_dy = vec![0.0f64; n_frames];
    let mut count_f = vec![0usize; n_frames];

    for group in picked_groups {
        if group.is_empty() {
            continue;
        }
        let mx = group.iter().map(|l| l.x).sum::<f64>() / (group.len() as f64);
        let my = group.iter().map(|l| l.y).sum::<f64>() / (group.len() as f64);
        for l in group {
            let f = (l.frame as usize).min(n_frames - 1);
            sum_dx[f] += l.x - mx;
            sum_dy[f] += l.y - my;
            count_f[f] += 1;
        }
    }

    let mut knot_frames = Vec::new();
    let mut knot_dx = Vec::new();
    let mut knot_dy = Vec::new();
    for f in 0..n_frames {
        if count_f[f] > 0 {
            knot_frames.push(f as f64);
            knot_dx.push(sum_dx[f] / (count_f[f] as f64));
            knot_dy.push(sum_dy[f] / (count_f[f] as f64));
        }
    }

    let drift_x = interpolate_drift_knots(&knot_frames, &knot_dx, n_frames);
    let drift_y = interpolate_drift_knots(&knot_frames, &knot_dy, n_frames);

    let mut out = locs.to_vec();
    for l in &mut out {
        let f = (l.frame as usize).min(n_frames - 1);
        l.x -= drift_x[f];
        l.y -= drift_y[f];
    }

    Ok((out, drift_x, drift_y))
}

/// Aligns multi-channel localization lists to the first channel via 2D cross-correlation
/// (`picasso.postprocess.align`).
pub fn align(channels: &[Vec<Localization>], oversampling: f64) -> Result<(Vec<Vec<Localization>>, Vec<(f64, f64)>)> {
    if channels.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let all_locs: Vec<Localization> = channels.iter().flatten().cloned().collect();
    let vp = default_viewport_2d(&all_locs);
    let (_, ref_img) = render(&channels[0], Some(vp), oversampling, RenderBlurMethod::Gaussian, 0.0)?;

    let mut aligned = Vec::with_capacity(channels.len());
    let mut shifts = Vec::with_capacity(channels.len());
    aligned.push(channels[0].clone());
    shifts.push((0.0, 0.0));

    for ch in channels.iter().skip(1) {
        let (_, img) = render(ch, Some(vp), oversampling, RenderBlurMethod::Gaussian, 0.0)?;
        let (dy_os, dx_os) = get_image_shift(&ref_img, &img, 5, None, false)?;
        let dx = dx_os / oversampling;
        let dy = dy_os / oversampling;
        let mut shifted = ch.clone();
        for l in &mut shifted {
            l.x -= dx;
            l.y -= dy;
        }
        aligned.push(shifted);
        shifts.push((dx, dy));
    }

    Ok((aligned, shifts))
}

/// Group-level summary statistics (`picasso.postprocess.groupprops`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupProps {
    pub group: i32,
    pub n_locs: usize,
    pub x_mean: f64,
    pub y_mean: f64,
    pub x_std: f64,
    pub y_std: f64,
    pub photons_mean: f64,
    pub sx_mean: f64,
    pub sy_mean: f64,
    pub bg_mean: f64,
    pub lpx_mean: f64,
    pub lpy_mean: f64,
    pub n_events: usize,
}

/// Computes summary statistics per `group` (`picasso.postprocess.groupprops`).
pub fn groupprops(locs: &[Localization]) -> Vec<GroupProps> {
    let mut groups: BTreeMap<i32, Vec<&Localization>> = BTreeMap::new();
    for l in locs {
        if let Some(g) = l.group
            && g >= 0
        {
            groups.entry(g).or_default().push(l);
        }
    }

    let mut out = Vec::with_capacity(groups.len());
    for (g, list) in groups {
        let n = list.len();
        if n == 0 {
            continue;
        }
        let nf = n as f64;
        let x_mean = list.iter().map(|l| l.x).sum::<f64>() / nf;
        let y_mean = list.iter().map(|l| l.y).sum::<f64>() / nf;
        let x_std = (list.iter().map(|l| (l.x - x_mean).powi(2)).sum::<f64>() / nf).sqrt();
        let y_std = (list.iter().map(|l| (l.y - y_mean).powi(2)).sum::<f64>() / nf).sqrt();
        let photons_mean = list.iter().map(|l| l.photons).sum::<f64>() / nf;
        let sx_mean = list.iter().map(|l| l.sx).sum::<f64>() / nf;
        let sy_mean = list.iter().map(|l| l.sy).sum::<f64>() / nf;
        let bg_mean = list.iter().map(|l| l.bg).sum::<f64>() / nf;
        let lpx_mean = list.iter().map(|l| l.lpx).sum::<f64>() / nf;
        let lpy_mean = list.iter().map(|l| l.lpy).sum::<f64>() / nf;

        let mut frames: Vec<u32> = list.iter().map(|l| l.frame).collect();
        frames.sort_unstable();
        frames.dedup();
        let mut n_events = 1usize;
        for w in frames.windows(2) {
            if w[1] > w[0] + 1 {
                n_events += 1;
            }
        }

        out.push(GroupProps {
            group: g,
            n_locs: n,
            x_mean,
            y_mean,
            x_std,
            y_std,
            photons_mean,
            sx_mean,
            sy_mean,
            bg_mean,
            lpx_mean,
            lpy_mean,
            n_events,
        });
    }

    out
}
