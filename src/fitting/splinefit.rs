//! 3D Tricubic Spline PSF Levenberg-Marquardt fitter.
//!
//! Ported from `picasso.fitting.splinefit`. Fits 5 parameters `[x, y, z, peak, bg]`
//! against a calibrated 3D tricubic spline coefficient tensor `coeff` of shape
//! `(n_x, n_y, n_z, 64)` under either Least Squares or Poisson/sCMOS MLE.

use crate::fitting::seeds::initial_parameters_spline;
use crate::types::{FitEstimator, FitState, Identification, Localization, Result};
use marray::np::{Array1, Array2, Array3, ArrayD, s};
use marray::sp::linalg;
use rayon::prelude::*;

const LAMBDA_INITIAL: f64 = 1e-3;
const LAMBDA_DOWN: f64 = 0.1;
const LAMBDA_UP: f64 = 10.0;

/// Result of batch 3D cubic spline PSF fitting (`picasso.fitting.splinefit.fit_spots_spline`).
#[derive(Debug, Clone)]
pub struct SplineFitResult {
    /// Fitted parameters of shape `(n_spots, 5)`: `[x, y, z, peak, bg]`.
    pub parameters: Array2<f64>,
    /// Convergence state (`0 = Converged`, `1 = MaxIterationsExceeded`, `2 = SingularHessian`).
    pub states: Array1<i32>,
    /// Final chi-square / Poisson deviance for each spot.
    pub chi_squares: Array1<f64>,
    /// Number of iterations used for each spot.
    pub number_iterations: Array1<u32>,
}

#[inline]
fn tricubic_monomials_and_derivs(
    dx: f64,
    dy: f64,
    dz: f64,
) -> ([f64; 64], [f64; 64], [f64; 64], [f64; 64]) {
    let xp = [1.0, dx, dx * dx, dx * dx * dx];
    let yp = [1.0, dy, dy * dy, dy * dy * dy];
    let zp = [1.0, dz, dz * dz, dz * dz * dz];

    let dxp = [0.0, 1.0, 2.0 * dx, 3.0 * dx * dx];
    let dyp = [0.0, 1.0, 2.0 * dy, 3.0 * dy * dy];
    let dzp = [0.0, 1.0, 2.0 * dz, 3.0 * dz * dz];

    let mut m = [0.0f64; 64];
    let mut dm_dx = [0.0f64; 64];
    let mut dm_dy = [0.0f64; 64];
    let mut dm_dz = [0.0f64; 64];

    let mut idx = 0;
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                m[idx] = xp[i] * yp[j] * zp[k];
                dm_dx[idx] = dxp[i] * yp[j] * zp[k];
                dm_dy[idx] = xp[i] * dyp[j] * zp[k];
                dm_dz[idx] = xp[i] * yp[j] * dzp[k];
                idx += 1;
            }
        }
    }
    (m, dm_dx, dm_dy, dm_dz)
}

#[inline]
pub fn eval_spline_pixel_and_derivs(
    ix: usize,
    iy: usize,
    params: &[f64; 5],
    coeffs: &ArrayD<f64>,
) -> (f64, [f64; 5], bool) {
    let shape = coeffs.shape();
    let nx = shape[0];
    let ny = shape[1];
    let nz = shape[2];

    let x0 = params[0];
    let y0 = params[1];
    let z0 = params[2];
    let peak = params[3];
    let bg = params[4];

    if !x0.is_finite() || !y0.is_finite() || !z0.is_finite() || !peak.is_finite() || !bg.is_finite()
    {
        return (f64::NAN, [0.0; 5], false);
    }

    let xc = (ix as f64) - x0 + (nx as f64) * 0.5;
    let yc = (iy as f64) - y0 + (ny as f64) * 0.5;
    let zc = z0.clamp(0.0, (nz as f64) - 1e-7);

    let vx = (xc.floor() as isize).clamp(0, (nx - 1) as isize) as usize;
    let vy = (yc.floor() as isize).clamp(0, (ny - 1) as isize) as usize;
    let vz = (zc.floor() as isize).clamp(0, (nz - 1) as isize) as usize;

    let dx = (xc - (vx as f64)).clamp(0.0, 1.0);
    let dy = (yc - (vy as f64)).clamp(0.0, 1.0);
    let dz = (zc - (vz as f64)).clamp(0.0, 1.0);

    let (m, dm_dx, dm_dy, dm_dz) = tricubic_monomials_and_derivs(dx, dy, dz);

    let mut basis = 0.0;
    let mut db_dxc = 0.0;
    let mut db_dyc = 0.0;
    let mut db_dzc = 0.0;

    for c in 0..64 {
        let a = coeffs[[vx, vy, vz, c]];
        basis += a * m[c];
        db_dxc += a * dm_dx[c];
        db_dyc += a * dm_dy[c];
        db_dzc += a * dm_dz[c];
    }

    let val = peak * basis + bg;
    if !val.is_finite() {
        return (val, [0.0; 5], false);
    }

    let d = [-peak * db_dxc, -peak * db_dyc, peak * db_dzc, basis, 1.0];
    (val, d, true)
}

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

fn eval_spline_spot_objective(
    spot: &Array2<f64>,
    var_spot: Option<&Array2<f64>>,
    params: &[f64; 5],
    coeffs: &ArrayD<f64>,
    mle: bool,
    need_derivatives: bool,
) -> (f64, [f64; 5], [[f64; 5]; 5], bool) {
    let (size, _) = spot.dim();
    let mut chi2 = 0.0;
    let mut grad = [0.0f64; 5];
    let mut hess = [[0.0f64; 5]; 5];

    for iy in 0..size {
        for ix in 0..size {
            let data = spot[[iy, ix]];
            let var = var_spot.map_or(0.0, |v| v[[iy, ix]]);
            let (val, d, ok_model) = eval_spline_pixel_and_derivs(ix, iy, params, coeffs);
            if !ok_model {
                return (f64::NAN, grad, hess, false);
            }
            let (chi2_term, weight, factor, ok_est) = estimator_terms(mle, val, data, var);
            if !ok_est {
                return (f64::NAN, grad, hess, false);
            }
            chi2 += chi2_term;
            if need_derivatives {
                for p in 0..5 {
                    grad[p] += factor * d[p];
                    for q in p..5 {
                        hess[p][q] += weight * d[p] * d[q];
                    }
                }
            }
        }
    }

    if need_derivatives {
        for p in 0..5 {
            for q in 0..p {
                hess[p][q] = hess[q][p];
            }
        }
    }

    (chi2, grad, hess, chi2.is_finite())
}

fn solve_5x5(hess: &[[f64; 5]; 5], rhs: &[f64; 5]) -> ([f64; 5], bool) {
    let mut a = Array2::<f64>::zeros((5, 5));
    let mut b = Array1::<f64>::zeros(5);
    for i in 0..5 {
        b[i] = rhs[i];
        for j in 0..5 {
            a[[i, j]] = hess[i][j];
        }
    }
    if let Ok(sol) = linalg::solve_vec(&a, &b) {
        let mut out = [0.0f64; 5];
        for i in 0..5 {
            if !sol[i].is_finite() {
                return ([0.0; 5], false);
            }
            out[i] = sol[i];
        }
        (out, true)
    } else {
        ([0.0; 5], false)
    }
}

/// Fits a single 2D spot against a 3D tricubic spline PSF model (`picasso.fitting.splinefit`).
pub fn fit_single_spline_spot(
    spot: &Array2<f64>,
    var_spot: Option<&Array2<f64>>,
    initial: &[f64; 5],
    coeffs: &ArrayD<f64>,
    mle: bool,
    tolerance: f64,
    max_iterations: u32,
) -> ([f64; 5], FitState, f64, u32) {
    let mut params = *initial;
    let mut scaling = [0.0f64; 5];
    let mut lam = LAMBDA_INITIAL;

    let (mut chi2, mut grad, mut hess, ok) =
        eval_spline_spot_objective(spot, var_spot, &params, coeffs, mle, true);
    if !ok {
        return (params, FitState::SingularHessian, chi2, 0);
    }

    let mut state = FitState::Converged;
    let mut iter = 0u32;
    let mut update_scale = true;

    while iter < max_iterations {
        let prev_chi2 = chi2;
        let mut hess_damped = hess;
        let mut rhs = [0.0f64; 5];
        for p in 0..5 {
            if update_scale {
                scaling[p] = scaling[p].max(hess[p][p].max(1e-9));
            }
            hess_damped[p][p] += scaling[p] * lam;
            rhs[p] = grad[p];
        }

        let (delta, solved) = solve_5x5(&hess_damped, &rhs);
        if !solved {
            state = FitState::SingularHessian;
            break;
        }

        let mut cand = params;
        for p in 0..5 {
            cand[p] += delta[p];
        }

        let (cand_chi2, _, _, ok_cand) =
            eval_spline_spot_objective(spot, var_spot, &cand, coeffs, mle, false);
        let step_improved = ok_cand && cand_chi2 < chi2;
        if step_improved {
            params = cand;
            let (new_chi2, new_grad, new_hess, ok_new) =
                eval_spline_spot_objective(spot, var_spot, &params, coeffs, mle, true);
            if !ok_new {
                state = FitState::SingularHessian;
                break;
            }
            chi2 = new_chi2;
            grad = new_grad;
            hess = new_hess;
            lam *= LAMBDA_DOWN;
            update_scale = true;
        } else {
            lam *= LAMBDA_UP;
            update_scale = false;
        }

        let converged =
            step_improved && (chi2 - prev_chi2).abs() < tolerance.max(tolerance * chi2.abs());
        iter += 1;
        if converged {
            break;
        }
        if iter == max_iterations {
            state = FitState::MaxIterationsExceeded;
            break;
        }
    }

    (params, state, chi2, iter)
}

/// Fits a batch of 2D spots `(n_spots, size, size)` against a 3D tricubic spline PSF (`picasso.fitting.splinefit.fit_spots_spline`).
pub fn fit_spots_spline(
    spots: &Array3<f64>,
    coeffs: &ArrayD<f64>,
    estimator: FitEstimator,
    scmos_vars: Option<&Array3<f64>>,
    initial_params: Option<&Array2<f64>>,
    parallel: bool,
) -> Result<SplineFitResult> {
    if coeffs.ndim() != 4 || coeffs.shape()[3] != 64 {
        return Err("Spline coefficients must have 4D shape (nx, ny, nz, 64)".into());
    }
    let n_spots = spots.shape()[0];
    let nz = coeffs.shape()[2];
    let mle = matches!(estimator, FitEstimator::Mle);

    let init = match initial_params {
        Some(p) => p.clone(),
        None => initial_parameters_spline(spots, nz)?,
    };

    let (tol, max_iter) = match estimator {
        FitEstimator::LeastSquares => (1e-6, 200),
        FitEstimator::Mle => (1e-6, 100),
    };

    let fit_one = |i: usize| {
        let spot_patch = spots.slice(s![i, .., ..]).to_owned();
        let var_patch = scmos_vars.map(|sv| sv.slice(s![i, .., ..]).to_owned());
        let seed = [
            init[[i, 0]],
            init[[i, 1]],
            init[[i, 2]],
            init[[i, 3]],
            init[[i, 4]],
        ];
        fit_single_spline_spot(
            &spot_patch,
            var_patch.as_ref(),
            &seed,
            coeffs,
            mle,
            tol,
            max_iter,
        )
    };

    let results: Vec<_> = if parallel && n_spots > 1 {
        (0..n_spots).into_par_iter().map(fit_one).collect()
    } else {
        (0..n_spots).map(fit_one).collect()
    };

    let mut parameters = Array2::<f64>::zeros((n_spots, 5));
    let mut states = Array1::<i32>::zeros(n_spots);
    let mut chi_squares = Array1::<f64>::zeros(n_spots);
    let mut number_iterations = Array1::<u32>::zeros(n_spots);

    for (i, (p, st, chi2, it)) in results.into_iter().enumerate() {
        for k in 0..5 {
            parameters[[i, k]] = p[k];
        }
        states[i] = st as i32;
        chi_squares[i] = chi2;
        number_iterations[i] = it;
    }

    Ok(SplineFitResult {
        parameters,
        states,
        chi_squares,
        number_iterations,
    })
}

/// Converts spline fit results into [`Localization`] records (`picasso.fitting.splinefit.locs_from_fits_spline`).
pub fn locs_from_fits_spline(
    identifications: &[Identification],
    result: &SplineFitResult,
    coeffs: &ArrayD<f64>,
    box_size: usize,
    z_step_nm: f64,
    em: bool,
) -> Vec<Localization> {
    let box_offset = (box_size / 2) as f64;
    let nz = coeffs.shape()[2] as f64;
    let mut locs = Vec::with_capacity(identifications.len());

    for (i, id) in identifications.iter().enumerate() {
        let x_box = result.parameters[[i, 0]];
        let y_box = result.parameters[[i, 1]];
        let z_vox = result.parameters[[i, 2]];
        let peak = result.parameters[[i, 3]].max(0.0);
        let bg = result.parameters[[i, 4]].max(0.0);

        let params = [x_box, y_box, z_vox, 1.0, 0.0];
        let mut psf_sum = 0.0;
        let mut fisher = Array2::<f64>::zeros((5, 5));
        for iy in 0..box_size {
            for ix in 0..box_size {
                let (basis_val, d, ok) =
                    eval_spline_pixel_and_derivs(ix, iy, &params, coeffs);
                if ok {
                    psf_sum += basis_val.max(0.0);
                    let mu = (peak * basis_val + bg).max(1e-3);
                    let d_scaled = [
                        peak * d[0],
                        peak * d[1],
                        peak * d[2],
                        basis_val,
                        1.0,
                    ];
                    for p in 0..5 {
                        for q in p..5 {
                            let term = d_scaled[p] * d_scaled[q] / mu;
                            fisher[[p, q]] += term;
                            if p != q {
                                fisher[[q, p]] += term;
                            }
                        }
                    }
                }
            }
        }

        let photons = (peak * psf_sum).max(0.0);
        let scale = if em { 2.0 } else { 1.0 };
        let cov = linalg::inv(&fisher).unwrap_or_else(|_| linalg::pinv(&fisher, 1e-12));
        let (lpx, lpy, lpz) = if cov[[0, 0]].is_finite() && cov[[1, 1]].is_finite() {
            (
                (cov[[0, 0]].max(0.0) * scale).sqrt(),
                (cov[[1, 1]].max(0.0) * scale).sqrt(),
                (cov[[2, 2]].max(0.0) * scale).sqrt() * z_step_nm,
            )
        } else {
            (0.1, 0.1, z_step_nm * 0.1)
        };

        let x = x_box + (id.x as f64) - box_offset;
        let y = y_box + (id.y as f64) - box_offset;
        let z_nm = (z_vox - 0.5 * nz) * z_step_nm;

        locs.push(Localization {
            frame: id.frame,
            x,
            y,
            z: Some(z_nm),
            photons,
            sx: 1.0,
            sy: 1.0,
            bg,
            lpx,
            lpy,
            lpz: Some(lpz),
            ellipticity: 0.0,
            net_gradient: id.net_gradient as f64,
            peak: Some(peak),
            chi2: Some(result.chi_squares[i]),
            iterations: Some(result.number_iterations[i]),
            fit_state: Some(result.states[i]),
            ..Default::default()
        });
    }

    locs
}
