//! 2D Gaussian Levenberg-Marquardt fitter (Gpufit CPU port).
//!
//! Ported from `picasso.fitting.gaussfit`. Supports [`GaussModel::Spherical`] (5 params),
//! [`GaussModel::Elliptic`] (6 params), and [`GaussModel::Rotated`] (7 params), with both
//! Least Squares (`FitEstimator::LeastSquares`) and Poisson / sCMOS shifted-Poisson MLE
//! (`FitEstimator::Mle`) objectives.

use crate::fitting::seeds::initial_parameters_gauss;
use crate::types::{FitEstimator, FitState, GaussModel, Result};
use marray::np::{Array1, Array2, Array3, s};
use rayon::prelude::*;

const LAMBDA_INITIAL: f64 = 1e-3;
const LAMBDA_DOWN: f64 = 0.1;
const LAMBDA_UP: f64 = 10.0;

/// Result of batch 2D Gaussian PSF fitting (`picasso.fitting.gaussfit.fit_spots_gauss`).
#[derive(Debug, Clone)]
pub struct GaussFitResult {
    /// Fitted parameters of shape `(n_spots, n_params)`:
    /// - `Spherical`: `[peak, x, y, s, bg]`
    /// - `Elliptic`: `[peak, x, y, sx, sy, bg]`
    /// - `Rotated`: `[peak, x, y, sx, sy, bg, angle]`
    pub parameters: Array2<f64>,
    /// Convergence state for each spot (`0 = Converged`, `1 = MaxIterationsExceeded`, `2 = SingularHessian`).
    pub states: Array1<i32>,
    /// Final chi-square / Poisson deviance for each spot.
    pub chi_squares: Array1<f64>,
    /// Number of iterations used for each spot.
    pub number_iterations: Array1<u32>,
}

/// Evaluates the 2D Gaussian model on a `(size, size)` grid given `params` and `model`.
pub fn eval_gaussian_spot(size: usize, params: &[f64], model: GaussModel) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((size, size));
    for iy in 0..size {
        for ix in 0..size {
            let (val, _, _) = eval_pixel_and_derivs(ix as f64, iy as f64, params, model);
            out[[iy, ix]] = val;
        }
    }
    out
}

/// Evaluates model intensity `value` and partial derivatives `d[0..n_params]` at pixel `(ix, iy)`.
#[inline]
fn eval_pixel_and_derivs(
    ix: f64,
    iy: f64,
    params: &[f64],
    model: GaussModel,
) -> (f64, [f64; 7], bool) {
    let mut d = [0.0f64; 7];
    match model {
        GaussModel::Spherical => {
            let peak = params[0];
            let x0 = params[1];
            let y0 = params[2];
            let s = params[3];
            let bg = params[4];
            if s == 0.0 || !s.is_finite() {
                return (f64::NAN, d, false);
            }
            let inv_s2 = 1.0 / (s * s);
            let dx = ix - x0;
            let dy = iy - y0;
            let arg = -0.5 * (dx * dx + dy * dy) * inv_s2;
            let ex = arg.exp();
            let val = peak * ex + bg;
            if !val.is_finite() {
                return (val, d, false);
            }
            d[0] = ex;
            d[1] = peak * ex * dx * inv_s2;
            d[2] = peak * ex * dy * inv_s2;
            d[3] = peak * ex * (dx * dx + dy * dy) * inv_s2 / s;
            d[4] = 1.0;
            (val, d, true)
        }
        GaussModel::Elliptic => {
            let peak = params[0];
            let x0 = params[1];
            let y0 = params[2];
            let sx = params[3];
            let sy = params[4];
            let bg = params[5];
            if sx == 0.0 || sy == 0.0 || !sx.is_finite() || !sy.is_finite() {
                return (f64::NAN, d, false);
            }
            let inv_sx2 = 1.0 / (sx * sx);
            let inv_sy2 = 1.0 / (sy * sy);
            let dx = ix - x0;
            let dy = iy - y0;
            let arg = -0.5 * (dx * dx * inv_sx2 + dy * dy * inv_sy2);
            let ex = arg.exp();
            let val = peak * ex + bg;
            if !val.is_finite() {
                return (val, d, false);
            }
            d[0] = ex;
            d[1] = peak * ex * dx * inv_sx2;
            d[2] = peak * ex * dy * inv_sy2;
            d[3] = peak * ex * dx * dx * inv_sx2 / sx;
            d[4] = peak * ex * dy * dy * inv_sy2 / sy;
            d[5] = 1.0;
            (val, d, true)
        }
        GaussModel::Rotated => {
            let peak = params[0];
            let x0 = params[1];
            let y0 = params[2];
            let sx = params[3];
            let sy = params[4];
            let bg = params[5];
            let theta = params[6];
            if sx == 0.0 || sy == 0.0 || !sx.is_finite() || !sy.is_finite() || !theta.is_finite() {
                return (f64::NAN, d, false);
            }
            let ct = theta.cos();
            let st = theta.sin();
            let dx = ix - x0;
            let dy = iy - y0;
            let u = dx * ct - dy * st;
            let v = dx * st + dy * ct;
            let inv_sx2 = 1.0 / (sx * sx);
            let inv_sy2 = 1.0 / (sy * sy);
            let arg = -0.5 * (u * u * inv_sx2 + v * v * inv_sy2);
            let ex = arg.exp();
            let val = peak * ex + bg;
            if !val.is_finite() {
                return (val, d, false);
            }
            let pe = peak * ex;
            d[0] = ex;
            d[1] = pe * (ct * u * inv_sx2 + st * v * inv_sy2);
            d[2] = pe * (-st * u * inv_sx2 + ct * v * inv_sy2);
            d[3] = pe * u * u * inv_sx2 / sx;
            d[4] = pe * v * v * inv_sy2 / sy;
            d[5] = 1.0;
            d[6] = pe * u * v * (inv_sx2 - inv_sy2);
            (val, d, true)
        }
    }
}

/// Evaluates estimator terms `(chi2_term, weight, grad_factor, ok)` for a single pixel
/// (`picasso.fitting.gaussfit._estimator_terms`).
#[inline]
fn estimator_terms(mle: bool, value: f64, data: f64, var: f64) -> (f64, f64, f64, bool) {
    if !mle {
        let deviation = value - data;
        let chi2 = deviation * deviation;
        (chi2, 1.0, -deviation, chi2.is_finite())
    } else {
        let shifted_value = value + var;
        let shifted_data = data + var;
        if !shifted_value.is_finite() || shifted_value <= 0.0 {
            return (f64::NAN, 0.0, 0.0, false);
        }
        if shifted_data > 0.0 {
            let ratio = shifted_value / shifted_data;
            let chi2 = 2.0 * ((shifted_value - shifted_data) - shifted_data * ratio.ln());
            let weight = shifted_data / (shifted_value * shifted_value);
            let factor = -(1.0 - (shifted_data / shifted_value));
            (chi2, weight, factor, chi2.is_finite())
        } else {
            let chi2 = 2.0 * shifted_value;
            (chi2, 0.0, -1.0, chi2.is_finite())
        }
    }
}

/// Accumulates total `chi2`, gradient `grad`, and approximate Hessian `hess` over a `(size, size)` spot
/// (`picasso.fitting.gaussfit._eval_gauss_spot`).
fn eval_gauss_spot_objective(
    spot: &Array2<f64>,
    var_spot: Option<&Array2<f64>>,
    params: &[f64],
    model: GaussModel,
    mle: bool,
    need_derivatives: bool,
) -> (f64, [f64; 7], [[f64; 7]; 7], bool) {
    let (size, _) = spot.dim();
    let n_params = model.n_params();
    let mut chi2 = 0.0;
    let mut grad = [0.0f64; 7];
    let mut hess = [[0.0f64; 7]; 7];

    for iy in 0..size {
        for ix in 0..size {
            let data = spot[[iy, ix]];
            let var = var_spot.map_or(0.0, |v| v[[iy, ix]]);
            let (val, d, ok_model) = eval_pixel_and_derivs(ix as f64, iy as f64, params, model);
            if !ok_model {
                return (f64::NAN, grad, hess, false);
            }
            let (chi2_term, weight, factor, ok_est) = estimator_terms(mle, val, data, var);
            if !ok_est {
                return (f64::NAN, grad, hess, false);
            }
            chi2 += chi2_term;
            if need_derivatives {
                for p in 0..n_params {
                    grad[p] += factor * d[p];
                    for q in p..n_params {
                        hess[p][q] += weight * d[p] * d[q];
                    }
                }
            }
        }
    }

    if need_derivatives {
        for p in 0..n_params {
            for q in 0..p {
                hess[p][q] = hess[q][p];
            }
        }
    }

    (chi2, grad, hess, chi2.is_finite())
}

/// Gauss-Jordan elimination with full pivoting (`picasso.fitting.gaussfit._solve_gj`).
fn solve_gj(a_in: &[[f64; 7]; 7], b_in: &[f64; 7], n: usize) -> ([f64; 7], bool) {
    let mut a = *a_in;
    let mut b = *b_in;
    let mut indxc = [0usize; 7];
    let mut indxr = [0usize; 7];
    let mut ipiv = [0i32; 7];

    for i in 0..n {
        let mut big = 0.0;
        let mut irow = 0usize;
        let mut icol = 0usize;
        for j in 0..n {
            if ipiv[j] != 1 {
                for k in 0..n {
                    if ipiv[k] == 0 {
                        let val = a[j][k].abs();
                        if val >= big {
                            big = val;
                            irow = j;
                            icol = k;
                        }
                    }
                }
            }
        }
        ipiv[icol] += 1;
        if irow != icol {
            a.swap(irow, icol);
            b.swap(irow, icol);
        }
        indxr[i] = irow;
        indxc[i] = icol;

        let piv = a[icol][icol];
        if piv == 0.0 || !piv.is_finite() {
            return ([0.0; 7], false);
        }
        let pivinv = 1.0 / piv;
        a[icol][icol] = 1.0;
        for l in 0..n {
            a[icol][l] *= pivinv;
        }
        b[icol] *= pivinv;

        for ll in 0..n {
            if ll != icol {
                let dum = a[ll][icol];
                a[ll][icol] = 0.0;
                for l in 0..n {
                    a[ll][l] -= a[icol][l] * dum;
                }
                b[ll] -= b[icol] * dum;
            }
        }
    }

    for val in b.iter().take(n) {
        if !val.is_finite() {
            return ([0.0; 7], false);
        }
    }
    let _ = (indxc, indxr);
    (b, true)
}

/// Performs one Levenberg-Marquardt damping update and solves for `delta`
/// (`picasso.fitting.splinefit._lm_solve_step`).
///
/// Note: `grad` is accumulated with `factor = data - value` (`-deviation`), so `grad` is already
/// the negative gradient of `0.5 * chi^2` (`J^T (data - value)`). Thus we solve `H_damped * delta = grad`.
fn lm_solve_step(
    hess: &[[f64; 7]; 7],
    grad: &[f64; 7],
    scaling: &mut [f64; 7],
    lam: f64,
    update_scale: bool,
    n_params: usize,
) -> ([f64; 7], bool) {
    let mut hess_damped = *hess;
    let mut rhs = [0.0f64; 7];
    for p in 0..n_params {
        if update_scale {
            scaling[p] = scaling[p].max(hess[p][p]);
        }
        hess_damped[p][p] += scaling[p] * lam;
        rhs[p] = grad[p];
    }
    solve_gj(&hess_damped, &rhs, n_params)
}

/// Fits a single 2D spot with Levenberg-Marquardt (`picasso.fitting.gaussfit._fit_gauss_spot`).
pub fn fit_single_gauss_spot(
    spot: &Array2<f64>,
    var_spot: Option<&Array2<f64>>,
    initial: &[f64],
    model: GaussModel,
    mle: bool,
    tolerance: f64,
    max_number_iterations: u32,
) -> ([f64; 7], FitState, f64, u32) {
    let n_params = model.n_params();
    let mut params = [0.0f64; 7];
    params[..n_params].copy_from_slice(&initial[..n_params]);

    let mut scaling = [0.0f64; 7];
    let mut lam = LAMBDA_INITIAL;

    let (mut chi2, mut grad, mut hess, ok) =
        eval_gauss_spot_objective(spot, var_spot, &params[..n_params], model, mle, true);
    if !ok {
        return (params, FitState::SingularHessian, chi2, 0);
    }

    let mut state = FitState::Converged;
    let mut iteration = 0u32;
    let mut update_scale = true;

    while iteration < max_number_iterations {
        let prev_chi2 = chi2;
        let (delta, solved) = lm_solve_step(
            &hess,
            &grad,
            &mut scaling,
            lam,
            update_scale,
            n_params,
        );
        if !solved {
            state = FitState::SingularHessian;
            break;
        }

        let mut cand = params;
        for p in 0..n_params {
            cand[p] += delta[p];
        }

        let (cand_chi2, cand_grad, cand_hess, ok_cand) =
            eval_gauss_spot_objective(spot, var_spot, &cand[..n_params], model, mle, true);
        iteration += 1;
        if !ok_cand {
            lam *= LAMBDA_UP;
            update_scale = false;
            if iteration == max_number_iterations {
                state = FitState::SingularHessian;
                break;
            }
            continue;
        }
        if cand_chi2 < prev_chi2 || prev_chi2 == 0.0 {
            grad = cand_grad;
            hess = cand_hess;
            update_scale = true;
        } else {
            update_scale = false;
        }
        let converged =
            (cand_chi2 - prev_chi2).abs() < tolerance.max(tolerance * cand_chi2.abs());
        if cand_chi2 < prev_chi2 {
            params = cand;
            chi2 = cand_chi2;
            lam *= LAMBDA_DOWN;
        } else {
            lam *= LAMBDA_UP;
        }
        if converged {
            break;
        }
        if iteration == max_number_iterations {
            state = FitState::MaxIterationsExceeded;
            break;
        }
    }

    (params, state, chi2, iteration)
}

/// Default CPU optimization schedules matching `picasso.fitting.gaussfit.fit_spots_gauss`.
#[inline]
pub fn default_schedule(estimator: FitEstimator) -> &'static [(f64, u32)] {
    match estimator {
        FitEstimator::LeastSquares => &[(1e-2, 200)],
        FitEstimator::Mle => &[(1e-5, 100)],
    }
}

/// Fits a batch of 2D spots `(n_spots, size, size)` using the Gpufit-ported Levenberg-Marquardt
/// 2D Gaussian fitter (`picasso.fitting.gaussfit.fit_spots_gauss`).
///
/// Runs across spots in parallel via Rayon when `parallel == true`.
pub fn fit_spots_gauss(
    spots: &Array3<f64>,
    model: GaussModel,
    estimator: FitEstimator,
    scmos_vars: Option<&Array3<f64>>,
    initial_params: Option<&Array2<f64>>,
    schedule: Option<&[(f64, u32)]>,
    parallel: bool,
) -> Result<GaussFitResult> {
    let shape = spots.shape();
    let n_spots = shape[0];
    let size = shape[1];
    if size < 3 || shape[2] != size {
        return Err("spots must have shape (n_spots, size, size) with size >= 3".into());
    }
    if let Some(vars) = scmos_vars
        && vars.shape() != shape
    {
        return Err("scmos_vars shape must match spots shape".into());
    }

    let n_params = model.n_params();
    let mle = matches!(estimator, FitEstimator::Mle);

    let spots_work = if mle && scmos_vars.is_some() {
        let mut s_clamped = spots.clone();
        for v in s_clamped.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        s_clamped
    } else {
        spots.clone()
    };

    let init = match initial_params {
        Some(p) => {
            if p.shape() != [n_spots, n_params] {
                return Err(format!(
                    "initial_params must have shape ({n_spots}, {n_params}), got {:?}",
                    p.shape()
                )
                .into());
            }
            p.clone()
        }
        None => initial_parameters_gauss(&spots_work, model)?,
    };

    let sched = schedule.unwrap_or_else(|| default_schedule(estimator));

    let fit_one = |i: usize| -> ([f64; 7], i32, f64, u32) {
        let spot_patch = spots_work.slice(s![i, .., ..]).to_owned();
        let var_patch = scmos_vars.map(|sv| sv.slice(s![i, .., ..]).to_owned());
        let mut p = [0.0f64; 7];
        for k in 0..n_params {
            p[k] = init[[i, k]];
        }
        let mut last_state = FitState::Converged;
        let mut last_chi2 = 0.0;
        let mut total_iters = 0u32;

        for &(tol, max_iter) in sched {
            let (p_out, st, chi2, n_it) = fit_single_gauss_spot(
                &spot_patch,
                var_patch.as_ref(),
                &p[..n_params],
                model,
                mle,
                tol,
                max_iter,
            );
            p = p_out;
            last_state = st;
            last_chi2 = chi2;
            total_iters += n_it;
        }
        (p, last_state as i32, last_chi2, total_iters)
    };

    let results: Vec<([f64; 7], i32, f64, u32)> = if parallel && n_spots > 1 {
        (0..n_spots).into_par_iter().map(fit_one).collect()
    } else {
        (0..n_spots).map(fit_one).collect()
    };

    let mut parameters = Array2::<f64>::zeros((n_spots, n_params));
    let mut states = Array1::<i32>::zeros(n_spots);
    let mut chi_squares = Array1::<f64>::zeros(n_spots);
    let mut number_iterations = Array1::<u32>::zeros(n_spots);

    for (i, (p, st, chi2, iters)) in results.into_iter().enumerate() {
        for k in 0..n_params {
            parameters[[i, k]] = p[k];
        }
        states[i] = st;
        chi_squares[i] = chi2;
        number_iterations[i] = iters;
    }

    Ok(GaussFitResult {
        parameters,
        states,
        chi_squares,
        number_iterations,
    })
}
