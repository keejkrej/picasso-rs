//! Non-Deep-Learning Single-Particle Binding Assay Pipeline for `lisca` and `lisca-binding-assay`.
//!
//! Replaces deep-learning spot detectors (PyTorch + Spotiflow) with Picasso's radial net-gradient
//! candidate identification and 2D Gaussian PSF fitting (`gausslq` / `gaussmle`), while producing
//! Spotiflow-compatible spot tables (`y, x, intensity, probability, fwhm, intens_A, intens_B, snb`),
//! cell-mask filtering, cumulative binding counts, Hungarian multi-frame tracking, and
//! coalescence / merge event detection.

use crate::localize::{FitConfig, IdentifyOptions, fit, identify, identify_in_frame};
use crate::types::{
    CameraInfo, GaussModel, Identification, Localization, Result, SpotFitMethod,
};
use marray::np::{Array2, Array3, s};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Single-particle spot record combining `lisca-binding-assay` (Spotiflow) columns and Picasso PSF metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BindingSpot {
    /// Position / field-of-view name (e.g. `"Pos0"`).
    #[serde(default)]
    pub pos: String,
    /// Timepoint / frame index (0-based).
    pub time: u32,
    /// Single-cell crop / pattern ROI identifier.
    #[serde(default)]
    pub roi: String,
    /// Sub-pixel row coordinate `y` in pixels.
    pub y: f64,
    /// Sub-pixel column coordinate `x` in pixels.
    pub x: f64,
    /// Spot intensity (fitted 2D Gaussian amplitude `intens_A` above background).
    pub intensity: f64,
    /// Normalized confidence score in `[0.0, 1.0]` derived from radial net gradient and SNR.
    pub probability: f64,
    /// Full Width at Half Maximum in pixels (`2.35482 * sqrt(sx * sy)`).
    pub fwhm: f64,
    /// Fitted 2D Gaussian amplitude above background (`intens_A`).
    #[serde(rename = "intens_A")]
    pub intens_a: f64,
    /// Fitted local background level (`intens_B`).
    #[serde(rename = "intens_B")]
    pub intens_b: f64,
    /// Signal-to-noise-and-background ratio (`intens_A / max(intens_B, 1.0)`).
    pub snb: f64,
    /// Integrated spot signal in photons (`2 * pi * sx * sy * intens_A`).
    pub photons: f64,
    /// Fitted PSF sigma along x in pixels.
    pub sx: f64,
    /// Fitted PSF sigma along y in pixels.
    pub sy: f64,
    /// Localization precision along x in pixels.
    pub lpx: f64,
    /// Localization precision along y in pixels.
    pub lpy: f64,
    /// Spot ellipticity `(max(sx, sy) - min(sx, sy)) / max(sx, sy)`.
    pub ellipticity: f64,
    /// Radial net gradient from spot identification.
    pub net_gradient: f64,
}

impl BindingSpot {
    /// Converts a Picasso [`Localization`] into a [`BindingSpot`] with `pos`, `roi`, and `min_gradient` reference.
    pub fn from_localization(
        loc: &Localization,
        pos: impl Into<String>,
        roi: impl Into<String>,
        min_gradient: f64,
    ) -> Self {
        let intens_a = loc.amplitude().max(0.0);
        let intens_b = loc.bg.max(0.0);
        let fwhm = loc.fwhm();
        let snb = loc.snb();
        // Smooth confidence score in [0, 1] combining net_gradient excess and SNB
        let ng_ratio = (loc.net_gradient / min_gradient.max(1.0)).max(0.0);
        let probability = (1.0 - (-0.5 * ng_ratio - 0.25 * snb).exp()).clamp(0.0, 1.0);

        Self {
            pos: pos.into(),
            time: loc.frame,
            roi: roi.into(),
            y: loc.y,
            x: loc.x,
            intensity: intens_a,
            probability,
            fwhm,
            intens_a,
            intens_b,
            snb,
            photons: loc.photons,
            sx: loc.sx,
            sy: loc.sy,
            lpx: loc.lpx,
            lpy: loc.lpy,
            ellipticity: loc.ellipticity,
            net_gradient: loc.net_gradient,
        }
    }

    /// Converts this [`BindingSpot`] back into a Picasso [`Localization`].
    pub fn to_localization(&self) -> Localization {
        Localization {
            frame: self.time,
            x: self.x,
            y: self.y,
            photons: self.photons,
            sx: self.sx,
            sy: self.sy,
            bg: self.intens_b,
            lpx: self.lpx,
            lpy: self.lpy,
            ellipticity: self.ellipticity,
            net_gradient: self.net_gradient,
            peak: Some(self.intens_a),
            ..Default::default()
        }
    }
}

/// Configuration for Picasso-based binding spot detection (`detect_spots_frame` / `detect_spots_stack`).
#[derive(Debug, Clone)]
pub struct BindingDetectConfig {
    /// Minimum radial net gradient threshold (`minimum_ng`).
    pub min_gradient: f64,
    /// Odd box side length in pixels (`box`, default `7`).
    pub box_size: usize,
    /// PSF fitting algorithm (`GaussLq` or `GaussMle`).
    pub fit_method: SpotFitMethod,
    /// 2D Gaussian model (`Spherical`, `Elliptic`, or `Rotated`).
    pub gauss_model: GaussModel,
    /// Optional 2D Gaussian pre-filter sigma for candidate identification.
    pub gaussian_sigma: Option<(f64, f64)>,
    /// Camera conversion configuration.
    pub camera_info: CameraInfo,
    /// Position identifier assigned to output records.
    pub pos: String,
    /// ROI identifier assigned to output records.
    pub roi: String,
    /// Whether to parallelize across frames/spots with Rayon.
    pub parallel: bool,
}

impl Default for BindingDetectConfig {
    fn default() -> Self {
        Self {
            min_gradient: 50.0,
            box_size: 7,
            fit_method: SpotFitMethod::GaussLq,
            gauss_model: GaussModel::Elliptic,
            gaussian_sigma: None,
            camera_info: CameraInfo::default(),
            pos: "Pos0".to_string(),
            roi: "crop_0".to_string(),
            parallel: true,
        }
    }
}

/// Detects and fits single-particle binding spots in a single 2D image frame `(H, W)`.
pub fn detect_spots_frame(
    frame: &Array2<f64>,
    time: u32,
    config: &BindingDetectConfig,
) -> Result<Vec<BindingSpot>> {
    let (h, w) = frame.dim();
    let mut stack = Array3::<f64>::zeros((1, h, w));
    stack.slice_mut(s![0, .., ..]).assign(frame);

    let filtered = if let Some(sig) = config.gaussian_sigma {
        crate::localize::gaussian_filter_frame(frame, sig)?
    } else {
        frame.clone()
    };

    let mut ids: Vec<Identification> =
        identify_in_frame(&filtered, 0, config.min_gradient, config.box_size, None);
    if ids.is_empty() {
        return Ok(Vec::new());
    }

    let fit_cfg = FitConfig {
        method: config.fit_method,
        model: config.gauss_model,
        box_size: config.box_size,
        eps: 1e-3,
        max_it: 1000,
        parallel: config.parallel,
    };
    let mut locs = fit(&stack, &config.camera_info, &ids, &fit_cfg)?;
    let _ = &mut ids;
    for l in &mut locs {
        l.frame = time;
    }

    Ok(locs
        .iter()
        .map(|l| BindingSpot::from_localization(l, &config.pos, &config.roi, config.min_gradient))
        .collect())
}

/// Detects and fits single-particle binding spots from a flat `u16` pixel slice of dimensions `(height, width)`.
pub fn detect_spots_u16_slice(
    pixels: &[u16],
    height: usize,
    width: usize,
    time: u32,
    config: &BindingDetectConfig,
) -> Result<Vec<BindingSpot>> {
    if pixels.len() != height * width {
        return Err(format!(
            "Pixel slice length {} does not match {}x{}",
            pixels.len(),
            height,
            width
        )
        .into());
    }
    let vec_f64: Vec<f64> = pixels.iter().map(|&v| v as f64).collect();
    let frame = Array2::from_shape_vec((height, width), vec_f64)
        .map_err(|e| crate::types::PicassoError::DimensionMismatch(e.to_string()))?;
    detect_spots_frame(&frame, time, config)
}

/// Detects and fits single-particle binding spots across a 3D time-lapse stack `(n_frames, H, W)`.
pub fn detect_spots_stack(
    movie: &Array3<f64>,
    config: &BindingDetectConfig,
) -> Result<Vec<BindingSpot>> {
    let id_opts = IdentifyOptions {
        min_gradient: config.min_gradient,
        box_size: config.box_size,
        gaussian_sigma: config.gaussian_sigma,
        temporal_median_window: None,
        roi: None,
        parallel: config.parallel,
    };
    let fit_cfg = FitConfig {
        method: config.fit_method,
        model: config.gauss_model,
        box_size: config.box_size,
        eps: 1e-3,
        max_it: 1000,
        parallel: config.parallel,
    };

    let ids = identify(movie, &id_opts)?;
    let locs = fit(movie, &config.camera_info, &ids, &fit_cfg)?;

    Ok(locs
        .iter()
        .map(|l| BindingSpot::from_localization(l, &config.pos, &config.roi, config.min_gradient))
        .collect())
}

/// Filtering criteria for [`filter_spots`] (matching `lisca-binding-assay/src/binding/filter.py`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BindingFilterConfig {
    /// Minimum fitted Gaussian amplitude (`intensity` / `intens_A`).
    pub min_intensity: Option<f64>,
    /// Maximum fitted Gaussian amplitude (`intensity` / `intens_A`).
    pub max_intensity: Option<f64>,
    /// Minimum Full Width at Half Maximum in pixels (`fwhm`).
    pub min_fwhm: Option<f64>,
    /// Maximum Full Width at Half Maximum in pixels (`fwhm`).
    pub max_fwhm: Option<f64>,
    /// Minimum spot confidence score (`probability`).
    pub min_probability: Option<f64>,
    /// Maximum spot ellipticity (`0.0 <= ellipticity < 1.0`).
    pub max_ellipticity: Option<f64>,
    /// Minimum signal-to-noise-and-background ratio (`snb`).
    pub min_snb: Option<f64>,
}

/// Filters [`BindingSpot`] records by intensity, FWHM, probability, ellipticity, SNB,
/// and an optional 2D boolean cell segmentation mask (`lisca-binding-assay/src/binding/filter.py`).
pub fn filter_spots(
    spots: &[BindingSpot],
    config: &BindingFilterConfig,
    cell_mask: Option<&Array2<bool>>,
) -> Vec<BindingSpot> {
    spots
        .iter()
        .filter(|s| {
            if !s.x.is_finite() || !s.y.is_finite() || !s.intensity.is_finite() || !s.fwhm.is_finite()
            {
                return false;
            }
            if let Some(min_i) = config.min_intensity
                && s.intensity < min_i
            {
                return false;
            }
            if let Some(max_i) = config.max_intensity
                && s.intensity > max_i
            {
                return false;
            }
            if let Some(min_w) = config.min_fwhm
                && s.fwhm < min_w
            {
                return false;
            }
            if let Some(max_w) = config.max_fwhm
                && s.fwhm > max_w
            {
                return false;
            }
            if let Some(min_p) = config.min_probability
                && s.probability < min_p
            {
                return false;
            }
            if let Some(max_e) = config.max_ellipticity
                && s.ellipticity > max_e
            {
                return false;
            }
            if let Some(min_snb) = config.min_snb
                && s.snb < min_snb
            {
                return false;
            }
            if let Some(mask) = cell_mask {
                let (h, w) = mask.dim();
                let iy = s.y.round() as isize;
                let ix = s.x.round() as isize;
                if iy < 0 || iy >= (h as isize) || ix < 0 || ix >= (w as isize) {
                    return false;
                }
                if !mask[[iy as usize, ix as usize]] {
                    return false;
                }
            }
            true
        })
        .cloned()
        .collect()
}

/// Per-frame and cumulative new-binding-event count record (`lisca-binding-assay/src/binding/spot_counts.py`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameSpotCount {
    pub pos: String,
    pub roi: String,
    pub time: u32,
    pub spot_count: usize,
    pub cumulative_count: usize,
}

/// Computes per-frame spot counts and cumulative unique binding event counts across frames
/// (`lisca-binding-assay/src/binding/spot_counts.py`).
///
/// A spot at frame `t > first_frame` increments `cumulative_count` if its Euclidean distance to
/// all previously registered spots in the same `(pos, roi)` is `>= match_distance` (default `5.0` px).
pub fn spot_counts(
    spots: &[BindingSpot],
    all_times: Option<&[u32]>,
    match_distance: f64,
) -> Vec<FrameSpotCount> {
    if spots.is_empty() {
        return Vec::new();
    }
    let mut groups: BTreeMap<(String, String), BTreeMap<u32, Vec<(f64, f64)>>> = BTreeMap::new();
    let mut time_set: BTreeMap<u32, ()> = BTreeMap::new();

    for s in spots {
        groups
            .entry((s.pos.clone(), s.roi.clone()))
            .or_default()
            .entry(s.time)
            .or_default()
            .push((s.y, s.x));
        time_set.insert(s.time, ());
    }

    let times: Vec<u32> = match all_times {
        Some(t) => t.to_vec(),
        None => time_set.into_keys().collect(),
    };

    let d2_thresh = match_distance * match_distance;
    let mut rows = Vec::new();

    for ((pos, roi), by_time) in groups {
        let mut seen_coords: Vec<(f64, f64)> = Vec::new();
        let mut cumulative = 0usize;

        for &t in &times {
            let pts = by_time.get(&t).cloned().unwrap_or_default();
            let count = pts.len();
            if seen_coords.is_empty() {
                cumulative += count;
                seen_coords.extend(pts);
            } else {
                for &(y, x) in &pts {
                    let is_new = seen_coords
                        .iter()
                        .all(|&(sy, sx)| (y - sy).powi(2) + (x - sx).powi(2) >= d2_thresh);
                    if is_new {
                        cumulative += 1;
                    }
                }
                seen_coords.extend(pts);
            }

            rows.push(FrameSpotCount {
                pos: pos.clone(),
                roi: roi.clone(),
                time: t,
                spot_count: count,
                cumulative_count: cumulative,
            });
        }
    }

    rows
}

/// Spot record augmented with multi-frame `track_id` (`lisca-binding-assay/src/binding/track_spots.py`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackedSpot {
    #[serde(flatten)]
    pub spot: BindingSpot,
    pub track_id: usize,
}

/// Coalescence / merge event between multiple active tracks (`lisca-binding-assay/src/binding/track_spots.py`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MergeEvent {
    pub pos: String,
    pub roi: String,
    pub time: u32,
    pub target_track_id: usize,
    pub source_track_ids: String,
    pub n_sources: usize,
    pub target_x: f64,
    pub target_y: f64,
    pub pre_intensity_sum: f64,
    pub post_intensity: f64,
    pub intensity_gain: f64,
    pub intensity_ratio: f64,
    pub strict: bool,
}

/// Per-trajectory summary metrics (`lisca-binding-assay/src/binding/track_spots.py`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrackSummary {
    pub pos: String,
    pub roi: String,
    pub track_id: usize,
    pub start_frame: u32,
    pub end_frame: u32,
    pub duration: u32,
    pub mean_x: f64,
    pub mean_y: f64,
    pub initial_intensity: f64,
    pub final_intensity: f64,
    pub max_intensity: f64,
    pub mean_intensity: f64,
    pub displacement: f64,
    pub merged_into: Option<usize>,
}

/// Configuration for multi-frame spot tracking and merge detection (`track_spots`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpotTrackConfig {
    /// Maximum Euclidean distance (pixels) to link a spot between consecutive frames (default `5.0`).
    pub max_distance: f64,
    /// Maximum Euclidean distance (pixels) to consider nearby ending tracks as merging into a target spot (default `8.0`).
    pub merge_distance: f64,
    /// Minimum number of converging tracks required for a merge event (default `2`).
    pub min_sources: usize,
    /// Minimum `post_intensity - max(source_intensities)` required for a `strict` merge (default `500.0`).
    pub min_intensity_gain: f64,
}

impl Default for SpotTrackConfig {
    fn default() -> Self {
        Self {
            max_distance: 5.0,
            merge_distance: 8.0,
            min_sources: 2,
            min_intensity_gain: 500.0,
        }
    }
}

/// Solves the rectangular minimum-weight bipartite assignment problem (`scipy.optimize.linear_sum_assignment`)
/// using the Jonker-Volgenant / Hungarian shortest augmenting path algorithm.
///
/// Returns `(row_idx, col_idx)` pairs for `cost` of shape `(n_rows, n_cols)`.
pub fn linear_sum_assignment(cost: &Array2<f64>) -> Vec<(usize, usize)> {
    let (n_rows, n_cols) = cost.dim();
    if n_rows == 0 || n_cols == 0 {
        return Vec::new();
    }
    let transposed = n_rows > n_cols;
    let (n, m) = if transposed {
        (n_cols, n_rows)
    } else {
        (n_rows, n_cols)
    };

    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; m + 1];
    let mut p = vec![0usize; m + 1];
    let mut way = vec![0usize; m + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![f64::INFINITY; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0usize;
            for j in 1..=m {
                if !used[j] {
                    let c = if transposed {
                        cost[[j - 1, i0 - 1]]
                    } else {
                        cost[[i0 - 1, j - 1]]
                    };
                    let cur = c - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut pairs = Vec::with_capacity(n);
    for j in 1..=m {
        if p[j] > 0 {
            let r = p[j] - 1;
            let c = j - 1;
            if transposed {
                pairs.push((c, r));
            } else {
                pairs.push((r, c));
            }
        }
    }
    pairs.sort_unstable_by_key(|&(r, _)| r);
    pairs
}

/// Tracks spots across frames within each `(pos, roi)` using Hungarian bipartite matching and
/// detects coalescence / merge events (`lisca-binding-assay/src/binding/track_spots.py`).
pub fn track_spots(
    spots: &[BindingSpot],
    config: &SpotTrackConfig,
) -> (Vec<TrackedSpot>, Vec<MergeEvent>, Vec<TrackSummary>) {
    if spots.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let mut by_roi: BTreeMap<(String, String), BTreeMap<u32, Vec<BindingSpot>>> = BTreeMap::new();
    for s in spots {
        by_roi
            .entry((s.pos.clone(), s.roi.clone()))
            .or_default()
            .entry(s.time)
            .or_default()
            .push(s.clone());
    }

    let mut all_tracked = Vec::with_capacity(spots.len());
    let mut all_merges = Vec::new();
    let mut all_summaries = Vec::new();
    let mut next_track_id = 0usize;

    for ((pos, roi), frames_map) in by_roi {
        let mut prev_tracks: Vec<(usize, BindingSpot)> = Vec::new();
        let mut roi_tracked: Vec<TrackedSpot> = Vec::new();
        let mut merged_into_map: HashMap<usize, usize> = HashMap::new();

        for (&time, curr_spots) in &frames_map {
            let n_curr = curr_spots.len();
            if prev_tracks.is_empty() {
                let mut curr_assigned = Vec::with_capacity(n_curr);
                for s in curr_spots {
                    let tid = next_track_id;
                    next_track_id += 1;
                    curr_assigned.push((tid, s.clone()));
                    roi_tracked.push(TrackedSpot {
                        spot: s.clone(),
                        track_id: tid,
                    });
                }
                prev_tracks = curr_assigned;
                continue;
            }

            let n_prev = prev_tracks.len();
            let mut dist_mat = Array2::<f64>::zeros((n_prev, n_curr));
            let mut cost_mat = Array2::<f64>::zeros((n_prev, n_curr));

            for (i, (_, ps)) in prev_tracks.iter().enumerate() {
                for (j, cs) in curr_spots.iter().enumerate() {
                    let d = ((ps.y - cs.y).powi(2) + (ps.x - cs.x).powi(2)).sqrt();
                    dist_mat[[i, j]] = d;
                    cost_mat[[i, j]] = if d <= config.max_distance { d } else { 1e6 + d };
                }
            }

            let pairs = linear_sum_assignment(&cost_mat);
            let mut matched_prev = HashSet::new();
            let mut matched_curr = HashSet::new();
            let mut curr_track_ids = vec![0usize; n_curr];

            for (pi, cj) in pairs {
                if dist_mat[[pi, cj]] <= config.max_distance {
                    matched_prev.insert(pi);
                    matched_curr.insert(cj);
                    curr_track_ids[cj] = prev_tracks[pi].0;
                }
            }

            for cj in 0..n_curr {
                if !matched_curr.contains(&cj) {
                    curr_track_ids[cj] = next_track_id;
                    next_track_id += 1;
                }
            }

            // Detect coalescence / merge events for unmatched previous tracks
            let unmatched_prev: Vec<usize> = (0..n_prev)
                .filter(|pi| !matched_prev.contains(pi))
                .collect();

            if !unmatched_prev.is_empty() {
                let mut merge_candidates: HashMap<usize, Vec<usize>> = HashMap::new();
                for &pi in &unmatched_prev {
                    let mut best_cj = None;
                    let mut best_d = f64::INFINITY;
                    for cj in 0..n_curr {
                        let d = dist_mat[[pi, cj]];
                        if d <= config.merge_distance && d < best_d {
                            best_d = d;
                            best_cj = Some(cj);
                        }
                    }
                    if let Some(cj) = best_cj {
                        merge_candidates.entry(cj).or_default().push(pi);
                    }
                }

                for (cj, mut source_pis) in merge_candidates {
                    if let Some((primary_pi, _)) = (0..n_prev)
                        .filter(|&pi| {
                            matched_prev.contains(&pi)
                                && curr_track_ids[cj] == prev_tracks[pi].0
                        })
                        .map(|pi| (pi, ()))
                        .next()
                    {
                        source_pis.insert(0, primary_pi);
                    }

                    if source_pis.len() >= config.min_sources {
                        let target_tid = curr_track_ids[cj];
                        let mut src_tids = Vec::with_capacity(source_pis.len());
                        let mut pre_sum = 0.0;
                        let mut max_src_i = f64::NEG_INFINITY;

                        for &pi in &source_pis {
                            let (stid, ref sp) = prev_tracks[pi];
                            src_tids.push(stid);
                            pre_sum += sp.intensity;
                            if sp.intensity > max_src_i {
                                max_src_i = sp.intensity;
                            }
                            if stid != target_tid {
                                merged_into_map.insert(stid, target_tid);
                            }
                        }

                        let post_i = curr_spots[cj].intensity;
                        let gain = post_i - max_src_i;
                        let ratio = if pre_sum > 0.0 { post_i / pre_sum } else { 0.0 };
                        let strict = gain >= config.min_intensity_gain;
                        let src_str = src_tids
                            .iter()
                            .map(|id| id.to_string())
                            .collect::<Vec<_>>()
                            .join(";");

                        all_merges.push(MergeEvent {
                            pos: pos.clone(),
                            roi: roi.clone(),
                            time,
                            target_track_id: target_tid,
                            source_track_ids: src_str,
                            n_sources: src_tids.len(),
                            target_x: curr_spots[cj].x,
                            target_y: curr_spots[cj].y,
                            pre_intensity_sum: pre_sum,
                            post_intensity: post_i,
                            intensity_gain: gain,
                            intensity_ratio: ratio,
                            strict,
                        });
                    }
                }
            }

            let mut next_prev = Vec::with_capacity(n_curr);
            for (cj, cs) in curr_spots.iter().enumerate() {
                let tid = curr_track_ids[cj];
                next_prev.push((tid, cs.clone()));
                roi_tracked.push(TrackedSpot {
                    spot: cs.clone(),
                    track_id: tid,
                });
            }
            prev_tracks = next_prev;
        }

        // Summarize tracks for this ROI
        let mut by_tid: BTreeMap<usize, Vec<&TrackedSpot>> = BTreeMap::new();
        for ts in &roi_tracked {
            by_tid.entry(ts.track_id).or_default().push(ts);
        }

        for (tid, list) in by_tid {
            let n = list.len() as f64;
            let first = list[0];
            let last = *list.last().unwrap();
            let mean_x = list.iter().map(|t| t.spot.x).sum::<f64>() / n;
            let mean_y = list.iter().map(|t| t.spot.y).sum::<f64>() / n;
            let max_i = list
                .iter()
                .map(|t| t.spot.intensity)
                .fold(f64::NEG_INFINITY, f64::max);
            let mean_i = list.iter().map(|t| t.spot.intensity).sum::<f64>() / n;
            let disp = ((last.spot.x - first.spot.x).powi(2)
                + (last.spot.y - first.spot.y).powi(2))
            .sqrt();

            all_summaries.push(TrackSummary {
                pos: pos.clone(),
                roi: roi.clone(),
                track_id: tid,
                start_frame: first.spot.time,
                end_frame: last.spot.time,
                duration: (list.len() as u32),
                mean_x,
                mean_y,
                initial_intensity: first.spot.intensity,
                final_intensity: last.spot.intensity,
                max_intensity: max_i,
                mean_intensity: mean_i,
                displacement: disp,
                merged_into: merged_into_map.get(&tid).copied(),
            });
        }

        all_tracked.extend(roi_tracked);
    }

    (all_tracked, all_merges, all_summaries)
}
