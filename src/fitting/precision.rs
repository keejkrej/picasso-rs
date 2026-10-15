//! Localization precision (Mortensen et al. 2010) and Cramér-Rao lower bound (CRLB) calculations.
//!
//! Ported from `picasso.fitting.precision`.

use marray::np::{Array1, Array2, Array3, s};
use marray::sp::linalg;
use std::f64::consts::PI;

const MIN_EPS: f64 = 1e-12;
const MIN_MU: f64 = 1e-3;
const MIN_SIGMA: f64 = 1e-3;

/// Computes the theoretical 1D localization precision according to Mortensen et al. (2010)
/// (`picasso.fitting.precision.localization_precision`).
#[inline]
pub fn localization_precision_scalar(
    photons: f64,
    s: f64,
    s_orth: f64,
    bg: f64,
    em: bool,
    readout_var: f64,
) -> f64 {
    let p = photons.max(MIN_EPS);
    let b = (bg.max(0.0) + readout_var.max(0.0)).max(MIN_EPS);
    let sa2 = s * s + (1.0 / 12.0);
    let sa_orth2 = s_orth * s_orth + (1.0 / 12.0);
    let sa = sa2.sqrt();
    let sa_orth = sa_orth2.sqrt();
    let mut v = sa2 * ((16.0 / 9.0) + (8.0 * PI * sa * sa_orth * b) / p) / p;
    if em {
        v *= 2.0;
    }
    v.max(0.0).sqrt()
}

/// Vectorized Mortensen et al. (2010) localization precision across an array of localizations.
pub fn localization_precision(
    photons: &Array1<f64>,
    s: &Array1<f64>,
    s_orth: &Array1<f64>,
    bg: &Array1<f64>,
    em: bool,
    readout_var: Option<&Array1<f64>>,
) -> Array1<f64> {
    let n = photons.len();
    let mut out = Array1::<f64>::zeros(n);
    for i in 0..n {
        let rv = readout_var.map_or(0.0, |v| v[i]);
        out[i] = localization_precision_scalar(photons[i], s[i], s_orth[i], bg[i], em, rv);
    }
    out
}

/// Estimates uncertainty in Gaussian width `s` for Least Squares fitting (`picasso.fitting.precision.sigma_uncertainty_lsq`).
#[inline]
pub fn sigma_uncertainty_lsq_scalar(
    photons: f64,
    s: f64,
    s_orth: f64,
    bg: f64,
    em: bool,
    readout_var: f64,
) -> f64 {
    let p = photons.max(MIN_EPS);
    let b = (bg.max(0.0) + readout_var.max(0.0)).max(MIN_EPS);
    let sa2 = s * s + (1.0 / 12.0);
    let sa_orth2 = s_orth * s_orth + (1.0 / 12.0);
    let sa = sa2.sqrt();
    let sa_orth = sa_orth2.sqrt();
    let mut v = (sa2 / (2.0 * p)) * (1.0 + (8.0 * PI * sa * sa_orth * b) / p);
    if em {
        v *= 2.0;
    }
    v.max(0.0).sqrt()
}

/// Vectorized LSQ Gaussian width uncertainty (`picasso.fitting.precision.sigma_uncertainty_lsq`).
pub fn sigma_uncertainty_lsq(
    photons: &Array1<f64>,
    s: &Array1<f64>,
    s_orth: &Array1<f64>,
    bg: &Array1<f64>,
    em: bool,
    readout_var: Option<&Array1<f64>>,
) -> Array1<f64> {
    let n = photons.len();
    let mut out = Array1::<f64>::zeros(n);
    for i in 0..n {
        let rv = readout_var.map_or(0.0, |v| v[i]);
        out[i] = sigma_uncertainty_lsq_scalar(photons[i], s[i], s_orth[i], bg[i], em, rv);
    }
    out
}

/// Cramér-Rao lower bound (CRLB) result for a single 2D Gaussian localization.
#[derive(Debug, Clone, Copy, Default)]
pub struct GaussCrlb {
    /// CRLB variance in `x` (`lpx = crlb_x.sqrt()`).
    pub crlb_x: f64,
    /// CRLB variance in `y` (`lpy = crlb_y.sqrt()`).
    pub crlb_y: f64,
    /// CRLB variance in `photons`.
    pub crlb_photons: f64,
    /// CRLB variance in `bg`.
    pub crlb_bg: f64,
    /// CRLB variance in `sx`.
    pub crlb_sx: f64,
    /// CRLB variance in `sy`.
    pub crlb_sy: f64,
    /// Optional CRLB variance in `angle` (when `angle.is_some()`).
    pub crlb_angle: Option<f64>,
}

/// Computes the Poisson / shifted-Poisson Cramér-Rao lower bound (CRLB) for a single 2D Gaussian spot
/// by inverting the `(6x6)` or `(7x7)` Fisher Information Matrix (`picasso.fitting.precision._gauss_crlb`).
#[allow(clippy::too_many_arguments)]
pub fn gauss_crlb_single(
    x: f64,
    y: f64,
    photons: f64,
    bg: f64,
    sx: f64,
    sy: f64,
    angle: Option<f64>,
    box_size: usize,
    em: bool,
    scmos_var: Option<&Array2<f64>>,
) -> GaussCrlb {
    let sx_c = sx.max(MIN_SIGMA);
    let sy_c = sy.max(MIN_SIGMA);
    let n_p = photons.max(MIN_EPS);
    let bg_c = bg.max(MIN_EPS);
    let n_dim = if angle.is_some() { 7 } else { 6 };

    let mut fisher = Array2::<f64>::zeros((n_dim, n_dim));
    let two_pi_sxsy = 2.0 * PI * sx_c * sy_c;

    let (ct, st) = if let Some(theta) = angle {
        (theta.cos(), theta.sin())
    } else {
        (1.0, 0.0)
    };

    for iy in 0..box_size {
        let dy = iy as f64 - y;
        for ix in 0..box_size {
            let dx = ix as f64 - x;
            let var_ij = scmos_var.map_or(0.0, |v| v[[iy, ix]].max(0.0));

            let mut d = [0.0f64; 7];
            let mu = if angle.is_some() {
                let u = dx * ct + dy * st;
                let v = -dx * st + dy * ct;
                let ex = (-0.5 * (u * u / (sx_c * sx_c) + v * v / (sy_c * sy_c))).exp();
                let signal = (n_p / two_pi_sxsy) * ex;
                let mu_val = (signal + bg_c).max(MIN_MU);

                let dmu_du = -signal * u / (sx_c * sx_c);
                let dmu_dv = -signal * v / (sy_c * sy_c);

                d[0] = -(dmu_du * ct - dmu_dv * st);
                d[1] = -(dmu_du * st + dmu_dv * ct);
                d[2] = signal / n_p;
                d[3] = 1.0;
                d[4] = signal * ((u * u) / (sx_c * sx_c * sx_c) - 1.0 / sx_c);
                d[5] = signal * ((v * v) / (sy_c * sy_c * sy_c) - 1.0 / sy_c);
                d[6] = dmu_du * v - dmu_dv * u;
                mu_val
            } else {
                let ex = (-0.5 * (dx * dx / (sx_c * sx_c) + dy * dy / (sy_c * sy_c))).exp();
                let signal = (n_p / two_pi_sxsy) * ex;
                let mu_val = (signal + bg_c).max(MIN_MU);

                d[0] = signal * dx / (sx_c * sx_c);
                d[1] = signal * dy / (sy_c * sy_c);
                d[2] = signal / n_p;
                d[3] = 1.0;
                d[4] = signal * ((dx * dx) / (sx_c * sx_c * sx_c) - 1.0 / sx_c);
                d[5] = signal * ((dy * dy) / (sy_c * sy_c * sy_c) - 1.0 / sy_c);
                mu_val
            };

            let inv_denom = 1.0 / (mu + var_ij).max(MIN_MU);
            for p in 0..n_dim {
                for q in p..n_dim {
                    let term = inv_denom * d[p] * d[q];
                    fisher[[p, q]] += term;
                    if p != q {
                        fisher[[q, p]] += term;
                    }
                }
            }
        }
    }

    let scale = if em { 2.0 } else { 1.0 };
    let cov = linalg::inv(&fisher).unwrap_or_else(|_| linalg::pinv(&fisher, 1e-12));

    if cov[[0, 0]].is_finite() && cov[[1, 1]].is_finite() && cov[[0, 0]] > 0.0 && cov[[1, 1]] > 0.0
    {
        GaussCrlb {
            crlb_x: (cov[[0, 0]].max(0.0)) * scale,
            crlb_y: (cov[[1, 1]].max(0.0)) * scale,
            crlb_photons: (cov[[2, 2]].max(0.0)) * scale,
            crlb_bg: (cov[[3, 3]].max(0.0)) * scale,
            crlb_sx: (cov[[4, 4]].max(0.0)) * scale,
            crlb_sy: (cov[[5, 5]].max(0.0)) * scale,
            crlb_angle: if n_dim == 7 {
                Some((cov[[6, 6]].max(0.0)) * scale)
            } else {
                None
            },
        }
    } else {
        let rv = scmos_var.map_or(0.0, |v| v.mean().unwrap_or(0.0));
        let lpx = localization_precision_scalar(photons, sx, sy, bg, em, rv);
        let lpy = localization_precision_scalar(photons, sy, sx, bg, em, rv);
        let sux = sigma_uncertainty_lsq_scalar(photons, sx, sy, bg, em, rv);
        let suy = sigma_uncertainty_lsq_scalar(photons, sy, sx, bg, em, rv);
        GaussCrlb {
            crlb_x: lpx * lpx,
            crlb_y: lpy * lpy,
            crlb_photons: photons.max(1.0) * scale,
            crlb_bg: bg.max(1.0) * scale / ((box_size * box_size) as f64),
            crlb_sx: sux * sux,
            crlb_sy: suy * suy,
            crlb_angle: angle.map(|_| 0.0),
        }
    }
}

/// Batch calculation of Cramér-Rao lower bounds for a list of fitted spots (`picasso.fitting.precision._gauss_crlb`).
#[allow(clippy::too_many_arguments)]
pub fn gauss_crlb_batch(
    x: &[f64],
    y: &[f64],
    photons: &[f64],
    bg: &[f64],
    sx: &[f64],
    sy: &[f64],
    angles: Option<&[f64]>,
    box_size: usize,
    em: bool,
    scmos_vars: Option<&Array3<f64>>,
) -> Vec<GaussCrlb> {
    let n = x.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let ang = angles.map(|a| a[i]);
        let var_patch = scmos_vars.map(|sv| sv.slice(s![i, .., ..]).to_owned());
        out.push(gauss_crlb_single(
            x[i],
            y[i],
            photons[i],
            bg[i],
            sx[i],
            sy[i],
            ang,
            box_size,
            em,
            var_patch.as_ref(),
        ));
    }
    out
}
