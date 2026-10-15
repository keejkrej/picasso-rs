//! Single-particle spot identification, patch cutting, photon conversion, and localization pipeline.
//!
//! Ported from `picasso.localize`.

use crate::avgroi;
use crate::fitting::gaussfit::{GaussFitResult, fit_spots_gauss};
use crate::fitting::gaussmle::{self, MleMethod};
use crate::fitting::precision::{
    gauss_crlb_batch, localization_precision_scalar, sigma_uncertainty_lsq_scalar,
};
use crate::lib_utils::ensure_odd;
use crate::types::{
    Baseline, CameraInfo, FitEstimator, GaussModel, Identification, Localization, PicassoError,
    Result, Sensitivity, SpotFitMethod,
};
use marray::np::{self, Array2, Array3, s};
use marray::sp::ndimage::gaussian_filter;
use rayon::prelude::*;
use std::f64::consts::PI;

/// Options for candidate spot identification (`picasso.localize.identify`).
#[derive(Debug, Clone)]
pub struct IdentifyOptions {
    /// Minimum radial net gradient threshold (`minimum_ng`).
    pub min_gradient: f64,
    /// Odd detection box side length (`box`).
    pub box_size: usize,
    /// Optional 2D Gaussian pre-filter sigma (applied during identification only).
    pub gaussian_sigma: Option<(f64, f64)>,
    /// Optional temporal rolling median window size (in frames) for background subtraction during identification.
    pub temporal_median_window: Option<usize>,
    /// Optional ROI `(y_min, x_min, y_max, x_max)` restricting candidate search.
    pub roi: Option<((usize, usize), (usize, usize))>,
    /// Whether to run frame identification in parallel with Rayon.
    pub parallel: bool,
}

impl Default for IdentifyOptions {
    fn default() -> Self {
        Self {
            min_gradient: 100.0,
            box_size: 7,
            gaussian_sigma: None,
            temporal_median_window: None,
            roi: None,
            parallel: true,
        }
    }
}

/// Finds strict/first-argmax local maxima in a 2D image within a `(box_size, box_size)` neighborhood
/// (`picasso.localize._local_maxima`).
pub fn local_maxima(frame: &Array2<f64>, box_size: usize) -> (Vec<i32>, Vec<i32>) {
    let box_size = ensure_odd(box_size);
    let (h, w) = frame.dim();
    let box_half = box_size / 2;
    if h <= 2 * box_half + 1 || w <= 2 * box_half + 1 {
        return (Vec::new(), Vec::new());
    }

    let mut y_out = Vec::new();
    let mut x_out = Vec::new();

    for i in box_half..(h - box_half - 1) {
        for j in box_half..(w - box_half - 1) {
            let center_val = frame[[i, j]];
            if !center_val.is_finite() {
                continue;
            }
            let mut max_val = f64::NEG_INFINITY;
            let mut max_dy = 0usize;
            let mut max_dx = 0usize;
            for dy in 0..box_size {
                let r = i + dy - box_half;
                for dx in 0..box_size {
                    let c = j + dx - box_half;
                    let v = frame[[r, c]];
                    if v > max_val {
                        max_val = v;
                        max_dy = dy;
                        max_dx = dx;
                    }
                }
            }
            if max_dy == box_half && max_dx == box_half {
                y_out.push(i as i32);
                x_out.push(j as i32);
            }
        }
    }

    (y_out, x_out)
}

/// Computes the centered finite-difference gradient `(gy, gx)` at pixel `(y, x)` (`picasso.localize.gradient_at`).
#[inline]
pub fn gradient_at(frame: &Array2<f64>, y: usize, x: usize) -> (f64, f64) {
    let gy = frame[[y + 1, x]] - frame[[y - 1, x]];
    let gx = frame[[y, x + 1]] - frame[[y, x - 1]];
    (gy, gx)
}

/// Precomputes normalized radial unit vectors `(ux, uy)` of shape `(box_size, box_size)`
/// directed toward the box center (`picasso.localize._net_gradient`).
pub fn radial_unit_vectors(box_size: usize) -> (Array2<f64>, Array2<f64>) {
    let box_size = ensure_odd(box_size);
    let box_half = (box_size / 2) as f64;
    let mut ux = Array2::<f64>::zeros((box_size, box_size));
    let mut uy = Array2::<f64>::zeros((box_size, box_size));

    for r in 0..box_size {
        for c in 0..box_size {
            let vx = box_half - (c as f64);
            let vy = box_half - (r as f64);
            let norm = (vx * vx + vy * vy).sqrt();
            if norm > 0.0 {
                ux[[r, c]] = vx / norm;
                uy[[r, c]] = vy / norm;
            }
        }
    }
    (ux, uy)
}

/// Computes the radial net gradient for candidate coordinates `(y, x)` (`picasso.localize.net_gradient`).
pub fn net_gradient(
    frame: &Array2<f64>,
    y: &[i32],
    x: &[i32],
    box_size: usize,
    uy: &Array2<f64>,
    ux: &Array2<f64>,
) -> Vec<f32> {
    let box_half = (box_size / 2) as i32;
    let mut ng = vec![0.0f32; y.len()];

    for (i, (&yi, &xi)) in y.iter().zip(x.iter()).enumerate() {
        let mut sum = 0.0f64;
        for (k_index, k) in ((yi - box_half)..=(yi + box_half)).enumerate() {
            for (l_index, m) in ((xi - box_half)..=(xi + box_half)).enumerate() {
                if !(k == yi && m == xi) {
                    let (gy, gx) = gradient_at(frame, k as usize, m as usize);
                    sum += gy * uy[[k_index, l_index]] + gx * ux[[k_index, l_index]];
                }
            }
        }
        ng[i] = sum as f32;
    }

    ng
}

/// Identifies candidate spots `(y, x, net_gradient)` in a 2D image (`picasso.localize.identify_in_image`).
pub fn identify_in_image(
    image: &Array2<f64>,
    minimum_ng: f64,
    box_size: usize,
) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let box_size = ensure_odd(box_size);
    let (y_cand, x_cand) = local_maxima(image, box_size);
    if y_cand.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let (ux, uy) = radial_unit_vectors(box_size);
    let ng = net_gradient(image, &y_cand, &x_cand, box_size, &uy, &ux);

    let min_ng_f32 = minimum_ng as f32;
    let mut y_out = Vec::new();
    let mut x_out = Vec::new();
    let mut ng_out = Vec::new();

    for ((yi, xi), ng_i) in y_cand.into_iter().zip(x_cand).zip(ng) {
        if ng_i > min_ng_f32 {
            y_out.push(yi);
            x_out.push(xi);
            ng_out.push(ng_i);
        }
    }

    (y_out, x_out, ng_out)
}

/// Identifies candidate spots in a single movie frame (`picasso.localize.identify_in_frame`),
/// with optional ROI cropping.
pub fn identify_in_frame(
    frame: &Array2<f64>,
    frame_number: u32,
    minimum_ng: f64,
    box_size: usize,
    roi: Option<((usize, usize), (usize, usize))>,
) -> Vec<Identification> {
    let box_size = ensure_odd(box_size);
    let (y_vec, x_vec, ng_vec) = if let Some(((y0, x0), (y1, x1))) = roi {
        let sub = frame.slice(s![y0..y1, x0..x1]).to_owned();
        let (mut y_sub, mut x_sub, ng_sub) = identify_in_image(&sub, minimum_ng, box_size);
        for yi in &mut y_sub {
            *yi += y0 as i32;
        }
        for xi in &mut x_sub {
            *xi += x0 as i32;
        }
        (y_sub, x_sub, ng_sub)
    } else {
        identify_in_image(frame, minimum_ng, box_size)
    };

    y_vec
        .into_iter()
        .zip(x_vec)
        .zip(ng_vec)
        .map(|((y, x), net_gradient)| Identification {
            frame: frame_number,
            x,
            y,
            net_gradient,
        })
        .collect()
}

/// Applies a rolling temporal median background subtraction across a 3D movie stack
/// (`picasso.localize.TemporalMedianMovie`).
pub fn temporal_median_subtract(movie: &Array3<f64>, window: usize) -> Array3<f64> {
    let shape = movie.shape();
    let n_frames = shape[0];
    let h = shape[1];
    let w = shape[2];
    if n_frames == 0 || window <= 1 {
        return movie.clone();
    }

    let half_w = window / 2;
    let mut out = Array3::<f64>::zeros((n_frames, h, w));

    for t in 0..n_frames {
        let start = t.saturating_sub(half_w);
        let end = (t + half_w + 1).min(n_frames);
        let count = end - start;
        let mut buf = vec![0.0f64; count];

        for iy in 0..h {
            for ix in 0..w {
                for (k, f_idx) in (start..end).enumerate() {
                    buf[k] = movie[[f_idx, iy, ix]];
                }
                buf.sort_by(|a, b| a.total_cmp(b));
                let med = if count % 2 == 1 {
                    buf[count / 2]
                } else {
                    0.5 * (buf[count / 2 - 1] + buf[count / 2])
                };
                out[[t, iy, ix]] = (movie[[t, iy, ix]] - med).max(0.0);
            }
        }
    }

    out
}

/// Applies a 2D Gaussian blur to a single frame using `marray::sp::ndimage::gaussian_filter`
/// (`picasso.localize.GaussianFilterMovie`).
pub fn gaussian_filter_frame(frame: &Array2<f64>, sigma: (f64, f64)) -> Result<Array2<f64>> {
    let s = 0.5 * (sigma.0 + sigma.1);
    if s <= 0.0 {
        return Ok(frame.clone());
    }
    Ok(gaussian_filter(frame, s))
}

/// Identifies candidate spots across all frames of a 3D movie stack `(n_frames, H, W)`
/// (`picasso.localize.identify`).
pub fn identify(movie: &Array3<f64>, options: &IdentifyOptions) -> Result<Vec<Identification>> {
    let box_size = ensure_odd(options.box_size);
    let n_frames = movie.shape()[0];
    if n_frames == 0 {
        return Ok(Vec::new());
    }

    let work_movie = if let Some(win) = options.temporal_median_window
        && win > 1
    {
        temporal_median_subtract(movie, win)
    } else {
        movie.clone()
    };

    let process_frame = |t: usize| -> Result<Vec<Identification>> {
        let raw_frame = work_movie.slice(s![t, .., ..]).to_owned();
        let filtered = if let Some(sigma) = options.gaussian_sigma {
            gaussian_filter_frame(&raw_frame, sigma)?
        } else {
            raw_frame
        };
        Ok(identify_in_frame(
            &filtered,
            t as u32,
            options.min_gradient,
            box_size,
            options.roi,
        ))
    };

    let frame_lists: Vec<Vec<Identification>> = if options.parallel && n_frames > 1 {
        let res: std::result::Result<Vec<_>, PicassoError> =
            (0..n_frames).into_par_iter().map(process_frame).collect();
        res?
    } else {
        let mut v = Vec::with_capacity(n_frames);
        for t in 0..n_frames {
            v.push(process_frame(t)?);
        }
        v
    };

    let mut all_ids: Vec<Identification> = frame_lists.into_iter().flatten().collect();
    all_ids.sort_by_key(|id| (id.frame, id.y, id.x));
    Ok(all_ids)
}

/// Cuts `(box_size, box_size)` patches centered on each identification from `movie`
/// and converts ADU values to photons using `camera_info` (`picasso.localize.get_spots` + `_to_photons`).
pub fn get_spots(
    movie: &Array3<f64>,
    identifications: &[Identification],
    box_size: usize,
    camera_info: &CameraInfo,
) -> Result<Array3<f64>> {
    let box_size = ensure_odd(box_size);
    let r = (box_size / 2) as isize;
    let shape = movie.shape();
    let n_frames = shape[0];
    let h = shape[1] as isize;
    let w = shape[2] as isize;
    let n_spots = identifications.len();

    let mut spots = Array3::<f64>::zeros((n_spots, box_size, box_size));
    let gain = camera_info.gain.max(1.0);

    for (idx, id) in identifications.iter().enumerate() {
        let f = id.frame as usize;
        if f >= n_frames {
            return Err(format!("Identification frame {} out of range ({n_frames})", id.frame).into());
        }
        let yc = id.y as isize;
        let xc = id.x as isize;
        if yc - r < 0 || yc + r >= h || xc - r < 0 || xc + r >= w {
            return Err(format!(
                "Spot at (x={}, y={}) with box_size={} exceeds frame bounds ({}x{})",
                id.x, id.y, box_size, w, h
            )
            .into());
        }

        for dy in 0..box_size {
            let iy = (yc - r + (dy as isize)) as usize;
            for dx in 0..box_size {
                let ix = (xc - r + (dx as isize)) as usize;
                let adu = movie[[f, iy, ix]];
                let base = match &camera_info.baseline {
                    Baseline::Scalar(b) => *b,
                    Baseline::Map(map) => map[[iy, ix]],
                };
                let sens = match &camera_info.sensitivity {
                    Sensitivity::Scalar(s) => *s,
                    Sensitivity::Map(map) => map[[iy, ix]],
                };
                spots[[idx, dy, dx]] = (adu - base) * sens / gain;
            }
        }
    }

    Ok(spots)
}

/// Cuts `(box_size, box_size)` per-pixel readout noise variance patches (in photons^2)
/// when an sCMOS variance map is present in `camera_info` (`picasso.localize._cut_scmos_vars`).
pub fn cut_scmos_vars(
    identifications: &[Identification],
    box_size: usize,
    camera_info: &CameraInfo,
) -> Result<Option<Array3<f64>>> {
    let Some(var_map) = &camera_info.variance_map else {
        return Ok(None);
    };
    let box_size = ensure_odd(box_size);
    let r = (box_size / 2) as isize;
    let (h, w) = var_map.dim();
    let n_spots = identifications.len();
    let gain = camera_info.gain.max(1.0);

    let mut vars = Array3::<f64>::zeros((n_spots, box_size, box_size));
    for (idx, id) in identifications.iter().enumerate() {
        let yc = id.y as isize;
        let xc = id.x as isize;
        if yc - r < 0 || yc + r >= (h as isize) || xc - r < 0 || xc + r >= (w as isize) {
            return Err("Spot exceeds sCMOS variance map bounds".into());
        }
        for dy in 0..box_size {
            let iy = (yc - r + (dy as isize)) as usize;
            for dx in 0..box_size {
                let ix = (xc - r + (dx as isize)) as usize;
                let sens = match &camera_info.sensitivity {
                    Sensitivity::Scalar(s) => *s,
                    Sensitivity::Map(map) => map[[iy, ix]],
                };
                let factor = sens / gain;
                vars[[idx, dy, dx]] = var_map[[iy, ix]].max(0.0) * factor * factor;
            }
        }
    }
    Ok(Some(vars))
}

/// Converts `gaussfit` result into [`Localization`] records (`picasso.localize.locs_from_fits_gauss`).
pub fn locs_from_fits_gauss(
    identifications: &[Identification],
    fit_result: &GaussFitResult,
    box_size: usize,
    model: GaussModel,
    estimator: FitEstimator,
    em: bool,
    scmos_vars: Option<&Array3<f64>>,
) -> Vec<Localization> {
    let n = identifications.len();
    let box_offset = (box_size / 2) as f64;
    let params = &fit_result.parameters;

    let mut peaks = vec![0.0f64; n];
    let mut x_box = vec![0.0f64; n];
    let mut y_box = vec![0.0f64; n];
    let mut sx_vec = vec![0.0f64; n];
    let mut sy_vec = vec![0.0f64; n];
    let mut bg_vec = vec![0.0f64; n];
    let mut angles = if model == GaussModel::Rotated {
        Some(vec![0.0f64; n])
    } else {
        None
    };

    for i in 0..n {
        peaks[i] = params[[i, 0]];
        x_box[i] = params[[i, 1]];
        y_box[i] = params[[i, 2]];
        match model {
            GaussModel::Spherical => {
                let s = params[[i, 3]].abs();
                sx_vec[i] = s;
                sy_vec[i] = s;
                bg_vec[i] = params[[i, 4]];
            }
            GaussModel::Elliptic => {
                sx_vec[i] = params[[i, 3]].abs();
                sy_vec[i] = params[[i, 4]].abs();
                bg_vec[i] = params[[i, 5]];
            }
            GaussModel::Rotated => {
                sx_vec[i] = params[[i, 3]].abs();
                sy_vec[i] = params[[i, 4]].abs();
                bg_vec[i] = params[[i, 5]];
                if let Some(ang) = &mut angles {
                    ang[i] = params[[i, 6]];
                }
            }
        }
    }

    let photons_vec: Vec<f64> = (0..n)
        .map(|i| peaks[i] * 2.0 * PI * sx_vec[i] * sy_vec[i])
        .collect();

    let crlbs = if estimator == FitEstimator::Mle {
        Some(gauss_crlb_batch(
            &x_box,
            &y_box,
            &photons_vec,
            &bg_vec,
            &sx_vec,
            &sy_vec,
            angles.as_deref(),
            box_size,
            em,
            scmos_vars,
        ))
    } else {
        None
    };

    let mut locs = Vec::with_capacity(n);
    for (i, id) in identifications.iter().enumerate() {
        let x = x_box[i] + (id.x as f64) - box_offset;
        let y = y_box[i] + (id.y as f64) - box_offset;
        let photons = photons_vec[i];
        let sx = sx_vec[i];
        let sy = sy_vec[i];
        let bg = bg_vec[i];

        let a = sx.max(sy).max(1e-12);
        let b = sx.min(sy);
        let ellipticity = (a - b) / a;

        let (lpx, lpy, sx_unc, sy_unc, photons_unc, bg_unc) = if let Some(crlb_list) = &crlbs {
            let c = crlb_list[i];
            (
                c.crlb_x.sqrt(),
                c.crlb_y.sqrt(),
                Some(c.crlb_sx.sqrt()),
                Some(c.crlb_sy.sqrt()),
                Some(c.crlb_photons.sqrt()),
                Some(c.crlb_bg.sqrt()),
            )
        } else {
            let rv = scmos_vars.map_or(0.0, |sv| np::mean(&sv.slice(s![i, .., ..]).to_owned()));
            let lx = localization_precision_scalar(photons, sx, sy, bg, em, rv);
            let ly = localization_precision_scalar(photons, sy, sx, bg, em, rv);
            let sux = sigma_uncertainty_lsq_scalar(photons, sx, sy, bg, em, rv);
            let suy = sigma_uncertainty_lsq_scalar(photons, sy, sx, bg, em, rv);
            (lx, ly, Some(sux), Some(suy), None, None)
        };

        locs.push(Localization {
            frame: id.frame,
            x,
            y,
            photons,
            sx,
            sy,
            bg,
            lpx,
            lpy,
            ellipticity,
            net_gradient: id.net_gradient as f64,
            peak: Some(peaks[i]),
            angle: angles.as_ref().map(|a_vec| a_vec[i]),
            chi2: Some(fit_result.chi_squares[i]),
            iterations: Some(fit_result.number_iterations[i]),
            fit_state: Some(fit_result.states[i]),
            sx_unc,
            sy_unc,
            photons_unc,
            bg_unc,
            ..Default::default()
        });
    }

    locs
}

/// High-level fitting configuration for [`fit`] and [`localize`].
#[derive(Debug, Clone)]
pub struct FitConfig {
    /// Spot fitting algorithm (`GaussLq`, `GaussMle`, `GaussMleIntegrated`, or `AvgRoi`).
    pub method: SpotFitMethod,
    /// Gaussian PSF model (`Spherical`, `Elliptic`, `Rotated`) when using `GaussLq` or `GaussMle`.
    pub model: GaussModel,
    /// Odd box side length in pixels (`box`).
    pub box_size: usize,
    /// Convergence epsilon for `GaussMleIntegrated` (default `1e-3`).
    pub eps: f64,
    /// Max iterations for `GaussMleIntegrated` (default `1000`).
    pub max_it: u32,
    /// Whether to parallelize across spots with Rayon.
    pub parallel: bool,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            method: SpotFitMethod::GaussLq,
            model: GaussModel::Elliptic,
            box_size: 7,
            eps: 1e-3,
            max_it: 1000,
            parallel: true,
        }
    }
}

/// Cuts spots around `identifications` in `movie` and fits them according to `config`
/// (`picasso.localize.fit`).
pub fn fit(
    movie: &Array3<f64>,
    camera_info: &CameraInfo,
    identifications: &[Identification],
    config: &FitConfig,
) -> Result<Vec<Localization>> {
    if identifications.is_empty() {
        return Ok(Vec::new());
    }
    let box_size = ensure_odd(config.box_size);
    let spots = get_spots(movie, identifications, box_size, camera_info)?;
    let scmos_vars = cut_scmos_vars(identifications, box_size, camera_info)?;
    let em = camera_info.is_em();

    let mut locs = match config.method {
        SpotFitMethod::GaussLq => {
            let res = fit_spots_gauss(
                &spots,
                config.model,
                FitEstimator::LeastSquares,
                scmos_vars.as_ref(),
                None,
                None,
                config.parallel,
            )?;
            locs_from_fits_gauss(
                identifications,
                &res,
                box_size,
                config.model,
                FitEstimator::LeastSquares,
                em,
                scmos_vars.as_ref(),
            )
        }
        SpotFitMethod::GaussMle => {
            let res = fit_spots_gauss(
                &spots,
                config.model,
                FitEstimator::Mle,
                scmos_vars.as_ref(),
                None,
                None,
                config.parallel,
            )?;
            locs_from_fits_gauss(
                identifications,
                &res,
                box_size,
                config.model,
                FitEstimator::Mle,
                em,
                scmos_vars.as_ref(),
            )
        }
        SpotFitMethod::GaussMleIntegrated => {
            let mle_method = match config.model {
                GaussModel::Spherical => MleMethod::Sigma,
                GaussModel::Elliptic | GaussModel::Rotated => MleMethod::SigmaYx,
            };
            let (thetas, crlbs, likelihoods, iterations) = gaussmle::gaussmle(
                &spots,
                config.eps,
                config.max_it,
                mle_method,
                config.parallel,
            )?;
            gaussmle::locs_from_fits(
                identifications,
                &thetas,
                &crlbs,
                &likelihoods,
                &iterations,
                box_size,
            )
        }
        SpotFitMethod::AvgRoi => {
            let theta = avgroi::fit_spots(&spots);
            avgroi::locs_from_fits(identifications, &theta, box_size, em)
        }
    };

    locs.retain(|loc| {
        loc.x.is_finite()
            && loc.y.is_finite()
            && loc.photons.is_finite()
            && loc.photons > 0.0
            && loc.sx.is_finite()
            && loc.sy.is_finite()
    });
    locs.sort_by(|a, b| a.frame.cmp(&b.frame).then_with(|| a.y.total_cmp(&b.y)));

    Ok(locs)
}

/// Runs the complete spot identification + PSF fitting pipeline on a 3D movie stack
/// (`picasso.localize`).
pub fn localize(
    movie: &Array3<f64>,
    camera_info: &CameraInfo,
    identify_options: &IdentifyOptions,
    fit_config: &FitConfig,
) -> Result<(Vec<Identification>, Vec<Localization>)> {
    let ids = identify(movie, identify_options)?;
    let locs = fit(movie, camera_info, &ids, fit_config)?;
    Ok((ids, locs))
}
