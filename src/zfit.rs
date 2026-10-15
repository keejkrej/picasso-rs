//! 3D Cylindrical Lens Astigmatism Calibration and Axial (`z`) Fitting.
//!
//! Ported from `picasso.zfit`.

use crate::types::{Localization, Result};
use marray::np::{Array1, Array2};
use marray::sp::linalg;
use marray::sp::optimize::rsminimize_scalar;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// 3D astigmatism polynomial calibration (`picasso.zfit`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationZ {
    /// Polynomial coefficients for `sx(z)` (highest power first).
    pub cx: Vec<f64>,
    /// Polynomial coefficients for `sy(z)` (highest power first).
    pub cy: Vec<f64>,
    /// Axial calibration range `(z_max - z_min)` in nanometers.
    pub z_range: f64,
    /// Axial step size `d` in nanometers used during calibration.
    pub step_nm: f64,
}

impl CalibrationZ {
    /// Evaluates polynomial `p(z)` using Horner's method.
    #[inline]
    pub fn eval_poly(coeffs: &[f64], z: f64) -> f64 {
        let mut acc = 0.0;
        for &c in coeffs {
            acc = acc * z + c;
        }
        acc
    }

    /// Evaluates polynomial derivative `dp(z)/dz`.
    #[inline]
    pub fn eval_poly_deriv(coeffs: &[f64], z: f64) -> f64 {
        let deg = coeffs.len().saturating_sub(1);
        let mut acc = 0.0;
        for (idx, &c) in coeffs.iter().take(deg).enumerate() {
            let power = (deg - idx) as f64;
            acc = acc * z + c * power;
        }
        acc
    }

    /// Evaluates calibrated `(sx(z), sy(z))` at axial position `z` (nm).
    #[inline]
    pub fn eval(&self, z: f64) -> (f64, f64) {
        (Self::eval_poly(&self.cx, z), Self::eval_poly(&self.cy, z))
    }
}

/// Fits a polynomial of degree `deg` (`y ≈ c_0 * x^deg + ... + c_deg`) using `marray::sp::linalg::lstsq`
/// with internal scaling of `x` to `[-1, 1]` for numerical conditioning.
pub fn polyfit(x: &[f64], y: &[f64], weights: Option<&[f64]>, deg: usize) -> Result<Vec<f64>> {
    let n = x.len();
    if n <= deg || y.len() != n {
        return Err(format!("Need at least {} points for degree-{deg} polynomial fit", deg + 1).into());
    }
    let scale = x
        .iter()
        .map(|v| v.abs())
        .fold(0.0f64, f64::max)
        .max(1.0);

    let n_cols = deg + 1;
    let mut a = Array2::<f64>::zeros((n, n_cols));
    let mut b = Array1::<f64>::zeros(n);

    for i in 0..n {
        let w = weights.map_or(1.0, |wv| wv[i]);
        let xi = x[i] / scale;
        let mut xp = 1.0;
        for p in 0..=deg {
            let col = deg - p;
            a[[i, col]] = xp * w;
            xp *= xi;
        }
        b[i] = y[i] * w;
    }

    let (sol, _, _, _) = linalg::lstsq(&a, &b, None)?;
    let mut coeffs = Vec::with_capacity(n_cols);
    for col in 0..n_cols {
        let power = (deg - col) as i32;
        coeffs.push(sol[col] / scale.powi(power));
    }
    Ok(coeffs)
}

/// Calibrates 3D astigmatism polynomials `(cx, cy)` of degree 6 from bead localizations
/// acquired across a known `z` stack with step `d_nm` (`picasso.zfit.calibrate_z`).
pub fn calibrate_z(
    locs: &[Localization],
    n_frames: usize,
    d_nm: f64,
    magnification_factor: f64,
) -> Result<CalibrationZ> {
    if locs.is_empty() || n_frames < 7 {
        return Err("Insufficient frames/localizations for 6th-order 3D z calibration".into());
    }
    let z_range = (n_frames.saturating_sub(1) as f64) * d_nm;
    let frame_range = (n_frames.saturating_sub(1)) as f64;

    let mut z_vals = Vec::with_capacity(locs.len());
    let mut sx_vals = Vec::with_capacity(locs.len());
    let mut sy_vals = Vec::with_capacity(locs.len());

    for loc in locs {
        if loc.sx.is_finite() && loc.sy.is_finite() && loc.sx > 0.0 && loc.sy > 0.0 {
            let z = ((loc.frame as f64) - 0.5 * frame_range) * d_nm * magnification_factor;
            z_vals.push(z);
            sx_vals.push(loc.sx);
            sy_vals.push(loc.sy);
        }
    }

    let cx0 = polyfit(&z_vals, &sx_vals, None, 6)?;
    let cy0 = polyfit(&z_vals, &sy_vals, None, 6)?;

    let mut weights = vec![1.0f64; z_vals.len()];
    let mut cx = cx0;
    let mut cy = cy0;
    for _ in 0..3 {
        let mut err_sum = 0.0;
        let mut errs = Vec::with_capacity(z_vals.len());
        for i in 0..z_vals.len() {
            let wx = CalibrationZ::eval_poly(&cx, z_vals[i]).max(0.0);
            let wy = CalibrationZ::eval_poly(&cy, z_vals[i]).max(0.0);
            let dx = sx_vals[i].sqrt() - wx.sqrt();
            let dy = sy_vals[i].sqrt() - wy.sqrt();
            let r = dx * dx + dy * dy;
            errs.push(r);
            err_sum += r;
        }
        let rms = (err_sum / (z_vals.len() as f64)).max(1e-12);
        for i in 0..z_vals.len() {
            weights[i] = if errs[i] <= 2.0 * rms + 1e-9 { 1.0 } else { 0.0 };
        }
        if weights.iter().filter(|&&w| w > 0.0).count() > 6 {
            cx = polyfit(&z_vals, &sx_vals, Some(&weights), 6)?;
            cy = polyfit(&z_vals, &sy_vals, Some(&weights), 6)?;
        }
    }

    Ok(CalibrationZ {
        cx,
        cy,
        z_range,
        step_nm: d_nm,
    })
}

/// Fits the axial coordinate `z` and residual score `d_z2` for a single `(sx, sy)` pair
/// using `marray::sp::optimize::rsminimize_scalar` (`picasso.zfit.fit_z`).
pub fn fit_z_single(
    sx: f64,
    sy: f64,
    calibration: &CalibrationZ,
    magnification_factor: f64,
) -> (f64, f64, f64) {
    let half_range = 0.5 * calibration.z_range.max(100.0);
    let sqrt_sx = sx.max(0.0).sqrt();
    let sqrt_sy = sy.max(0.0).sqrt();

    let obj = |z: f64| {
        let wx = CalibrationZ::eval_poly(&calibration.cx, z).max(0.0).sqrt();
        let wy = CalibrationZ::eval_poly(&calibration.cy, z).max(0.0).sqrt();
        let dx = sqrt_sx - wx;
        let dy = sqrt_sy - wy;
        dx * dx + dy * dy
    };

    let z_raw = rsminimize_scalar(obj, (-half_range, half_range), 1e-6);
    let dz2 = obj(z_raw);

    let z_scaled = z_raw * magnification_factor;
    let dwx_dz = CalibrationZ::eval_poly_deriv(&calibration.cx, z_raw);
    let dwy_dz = CalibrationZ::eval_poly_deriv(&calibration.cy, z_raw);
    let slope = (dwx_dz * dwx_dz + dwy_dz * dwy_dz).sqrt().max(1e-6);
    let lpz = (dz2.max(0.0).sqrt() / slope) * magnification_factor;

    (z_scaled, lpz, dz2)
}

/// Fits axial positions `z` (nm) and `lpz` (nm) for all localizations in `locs`
/// (`picasso.zfit.fit_z` / `fit_z_parallel`).
pub fn fit_z(
    locs: &mut [Localization],
    calibration: &CalibrationZ,
    magnification_factor: f64,
    filter: u8,
    parallel: bool,
) {
    let results: Vec<(f64, f64, f64)> = if parallel && locs.len() > 1 {
        locs.par_iter()
            .map(|l| fit_z_single(l.sx, l.sy, calibration, magnification_factor))
            .collect()
    } else {
        locs.iter()
            .map(|l| fit_z_single(l.sx, l.sy, calibration, magnification_factor))
            .collect()
    };

    let mut dz2_vals = Vec::with_capacity(locs.len());
    for (loc, (z, lpz, dz2)) in locs.iter_mut().zip(results) {
        loc.z = Some(z);
        loc.lpz = Some(lpz);
        dz2_vals.push(dz2);
    }

    let _ = filter;
}

/// Filters out localizations at the extreme ends of the axial calibration curve or with large
/// astigmatism residual (`picasso.zfit.filter_tails`).
pub fn filter_tails(
    locs: &[Localization],
    calibration: &CalibrationZ,
    magnification_factor: f64,
    level: u8,
) -> Vec<Localization> {
    if level == 0 || locs.is_empty() {
        return locs.to_vec();
    }
    let max_z = 0.48 * calibration.z_range * magnification_factor;
    let residuals: Vec<f64> = locs
        .iter()
        .map(|l| {
            let z = l.z.unwrap_or(0.0) / magnification_factor.max(1e-12);
            let (wx, wy) = calibration.eval(z);
            let dx = l.sx.max(0.0).sqrt() - wx.max(0.0).sqrt();
            let dy = l.sy.max(0.0).sqrt() - wy.max(0.0).sqrt();
            dx * dx + dy * dy
        })
        .collect();

    let mean_res = residuals.iter().sum::<f64>() / (residuals.len() as f64);
    let max_res = match level {
        1 => 3.0 * mean_res,
        _ => 1.5 * mean_res,
    };

    locs.iter()
        .zip(residuals)
        .filter(|(l, r)| {
            let z = l.z.unwrap_or(0.0);
            z.abs() <= max_z && *r <= max_res
        })
        .map(|(l, _)| l.clone())
        .collect()
}
