//! Initial parameter estimation (seeding) for 2D Gaussian and 3D cubic spline PSF fitting.
//!
//! Ported from `picasso.fitting.seeds`.

use crate::types::{GaussModel, Result};
use marray::np::{Array2, Array3, s};

/// Computes initial parameter guesses for 2D Gaussian spot fitting (`picasso.fitting.seeds.initial_parameters_gauss`).
///
/// # Arguments
/// * `spots` - 3D array of shape `(n_spots, size, size)` containing spot intensities in photons.
/// * `model` - [`GaussModel::Spherical`] (5 params), [`GaussModel::Elliptic`] (6 params), or [`GaussModel::Rotated`] (7 params).
///
/// # Returns
/// 2D array of shape `(n_spots, model.n_params())`:
/// - `Spherical`: `[peak, x, y, s, bg]`
/// - `Elliptic`: `[peak, x, y, sx, sy, bg]`
/// - `Rotated`: `[peak, x, y, sx, sy, bg, angle]`
pub fn initial_parameters_gauss(spots: &Array3<f64>, model: GaussModel) -> Result<Array2<f64>> {
    let shape = spots.shape();
    let n_spots = shape[0];
    let size = shape[1];
    if size < 3 || shape[2] != size {
        return Err("Spots must have shape (n_spots, size, size) with size >= 3".into());
    }

    let n_params = model.n_params();
    let mut initial = Array2::<f64>::zeros((n_spots, n_params));
    if n_spots == 0 {
        return Ok(initial);
    }

    let center = (size as f64 / 2.0) - 0.5;
    let half = size / 2;
    let fallback = ((size as f64) / 5.0).max(1.0);
    let max_width = (size as f64) / 3.0;

    // Precompute 1D squared distance from central row/column
    let d2: Vec<f64> = (0..size)
        .map(|idx| {
            let delta = idx as f64 - half as f64;
            delta * delta
        })
        .collect();

    for i in 0..n_spots {
        let spot = spots.slice(s![i, .., ..]).to_owned();
        let mut spot_max = f64::NEG_INFINITY;
        let mut spot_min = f64::INFINITY;
        for &v in spot.iter() {
            if v > spot_max {
                spot_max = v;
            }
            if v < spot_min {
                spot_min = v;
            }
        }
        let peak = (spot_max - spot_min).max(1.0);
        let bg = spot_min.max(0.0);

        match model {
            GaussModel::Spherical => {
                let (wx, wy) = initial_widths_1d(&spot, spot_min, half, &d2, fallback, max_width);
                let s = 0.5 * (wx + wy);
                initial[[i, 0]] = peak;
                initial[[i, 1]] = center;
                initial[[i, 2]] = center;
                initial[[i, 3]] = s;
                initial[[i, 4]] = bg;
            }
            GaussModel::Elliptic => {
                let (wx, wy) = initial_widths_1d(&spot, spot_min, half, &d2, fallback, max_width);
                initial[[i, 0]] = peak;
                initial[[i, 1]] = center;
                initial[[i, 2]] = center;
                initial[[i, 3]] = wx;
                initial[[i, 4]] = wy;
                initial[[i, 5]] = bg;
            }
            GaussModel::Rotated => {
                let (wx, wy, theta) =
                    initial_shape_gauss_rotated(&spot, spot_min, half, fallback, max_width);
                initial[[i, 0]] = peak;
                initial[[i, 1]] = center;
                initial[[i, 2]] = center;
                initial[[i, 3]] = wx;
                initial[[i, 4]] = wy;
                initial[[i, 5]] = bg;
                initial[[i, 6]] = theta;
            }
        }
    }

    Ok(initial)
}

fn initial_widths_1d(
    spot: &Array2<f64>,
    spot_min: f64,
    half: usize,
    d2: &[f64],
    fallback: f64,
    max_width: f64,
) -> (f64, f64) {
    let size = d2.len();
    let mut sum_y: f64 = 0.0;
    let mut mom_y: f64 = 0.0;
    let mut sum_x: f64 = 0.0;
    let mut mom_x: f64 = 0.0;

    for idx in 0..size {
        let py = (spot[[idx, half]] - spot_min).max(0.0);
        sum_y += py;
        mom_y += py * d2[idx];

        let px = (spot[[half, idx]] - spot_min).max(0.0);
        sum_x += px;
        mom_x += px * d2[idx];
    }

    let wy_raw = if sum_y > 0.0 {
        (mom_y / sum_y).sqrt()
    } else {
        fallback
    };
    let wx_raw = if sum_x > 0.0 {
        (mom_x / sum_x).sqrt()
    } else {
        fallback
    };

    let wy = wy_raw.min(fallback).clamp(0.5, max_width);
    let wx = wx_raw.min(fallback).clamp(0.5, max_width);
    (wx, wy)
}

fn initial_shape_gauss_rotated(
    spot: &Array2<f64>,
    spot_min: f64,
    half: usize,
    fallback: f64,
    max_width: f64,
) -> (f64, f64, f64) {
    let (h, w) = spot.dim();
    let half_f = half as f64;
    let mut w_sum: f64 = 0.0;
    let mut sum_x: f64 = 0.0;
    let mut sum_y: f64 = 0.0;

    for iy in 0..h {
        let y = iy as f64 - half_f;
        for ix in 0..w {
            let x = ix as f64 - half_f;
            let weight = (spot[[iy, ix]] - spot_min).max(0.0);
            w_sum += weight;
            sum_x += weight * x;
            sum_y += weight * y;
        }
    }

    if w_sum <= 0.0 || !w_sum.is_finite() {
        let mut wu = fallback.min(fallback).clamp(0.5, max_width);
        let mut wv = fallback.min(fallback).clamp(0.5, max_width);
        if wu < 1.01 * wv {
            wu *= 1.1;
            wv *= 0.9;
        }
        return (wu, wv, 0.0);
    }

    let mean_x = sum_x / w_sum;
    let mean_y = sum_y / w_sum;

    let mut m_xx: f64 = 0.0;
    let mut m_yy: f64 = 0.0;
    let mut m_xy: f64 = 0.0;

    for iy in 0..h {
        let dy = (iy as f64 - half_f) - mean_y;
        for ix in 0..w {
            let dx = (ix as f64 - half_f) - mean_x;
            let weight = (spot[[iy, ix]] - spot_min).max(0.0) / w_sum;
            m_xx += weight * dx * dx;
            m_yy += weight * dy * dy;
            m_xy += weight * dx * dy;
        }
    }

    let mean = 0.5 * (m_xx + m_yy);
    let spread = ((0.5 * (m_xx - m_yy)).powi(2) + m_xy * m_xy).sqrt();
    let mut width_u = (mean + spread).max(0.0).sqrt();
    let mut width_v = (mean - spread).max(0.0).sqrt();
    let mut angle = -0.5 * (2.0 * m_xy).atan2(m_xx - m_yy);

    if !width_u.is_finite() || !width_v.is_finite() {
        width_u = fallback;
        width_v = fallback;
        angle = 0.0;
    } else if !angle.is_finite() {
        angle = 0.0;
    }

    width_u = width_u.min(fallback).clamp(0.5, max_width);
    width_v = width_v.min(fallback).clamp(0.5, max_width);

    if width_u < 1.01 * width_v {
        width_u *= 1.1;
        width_v *= 0.9;
    }

    (width_u, width_v, angle)
}

/// Computes initial parameter guesses `[x, y, z, peak, bg]` for 3D cubic spline PSF fitting
/// (`picasso.fitting.seeds.initial_parameters_spline`).
pub fn initial_parameters_spline(spots: &Array3<f64>, n_splines_z: usize) -> Result<Array2<f64>> {
    let shape = spots.shape();
    let n_spots = shape[0];
    let size = shape[1];
    let mut initial = Array2::<f64>::zeros((n_spots, 5));
    let center_xy = (size as f64) / 2.0;
    let center_z = (n_splines_z as f64) / 2.0;

    for i in 0..n_spots {
        let spot = spots.slice(s![i, .., ..]);
        let mut spot_max = f64::NEG_INFINITY;
        let mut spot_min = f64::INFINITY;
        for &v in spot.iter() {
            if v > spot_max {
                spot_max = v;
            }
            if v < spot_min {
                spot_min = v;
            }
        }
        let bg = spot_min.max(0.0);
        let peak = (spot_max - bg).max(1.0);
        initial[[i, 0]] = center_xy;
        initial[[i, 1]] = center_xy;
        initial[[i, 2]] = center_z;
        initial[[i, 3]] = peak;
        initial[[i, 4]] = bg;
    }

    Ok(initial)
}
