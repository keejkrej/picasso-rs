//! Legacy pixel-integrated 2D Gaussian Maximum Likelihood Estimation (Smith et al., 2010).
//!
//! Ported from `picasso.gaussmle`. Uses `marray::sp::special::erf` for exact finite-pixel
//! integration of the 2D Gaussian PSF and Newton-Raphson updates with analytical 1st & 2nd
//! derivatives and Cramér-Rao lower bounds (CRLBs).

use crate::types::{Identification, Localization, Result};
use marray::np::{Array1, Array2, Array3, s};
use marray::sp::linalg;
use marray::sp::special::erf;
use rayon::prelude::*;
use std::f64::consts::{PI, SQRT_2};

/// Parameterization method for `gaussmle` (`picasso.gaussmle.gaussmle`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MleMethod {
    /// Fixed isotropic Gaussian width (`4` fitted parameters: `[x, y, photons, bg]`).
    Fixed,
    /// Isotropic Gaussian width (`5` fitted parameters: `[x, y, photons, bg, s]`).
    Sigma,
    /// Anisotropic elliptical Gaussian widths (`6` fitted parameters: `[x, y, photons, bg, sx, sy]`).
    #[default]
    SigmaYx,
}

#[inline]
fn gaussian_integral(x: f64, mu: f64, sigma: f64) -> f64 {
    let sq_norm = 0.7071067811865476 / sigma;
    let d = x - mu;
    0.5 * (erf((d + 0.5) * sq_norm) - erf((d - 0.5) * sq_norm))
}

#[inline]
fn derivative_gaussian_integral(
    x: f64,
    mu: f64,
    sigma: f64,
    photons: f64,
    psf_other: f64,
) -> (f64, f64) {
    let d = x - mu;
    let a = (-0.5 * ((d + 0.5) / sigma).powi(2)).exp();
    let b = (-0.5 * ((d - 0.5) / sigma).powi(2)).exp();
    let sqrt_2pi = (2.0 * PI).sqrt();
    let dudt = photons * psf_other * (b - a) / (sqrt_2pi * sigma);
    let d2udt2 = photons * ((d - 0.5) * b - (d + 0.5) * a) * psf_other
        / (sqrt_2pi * sigma * sigma * sigma);
    (dudt, d2udt2)
}

#[inline]
fn g_helper(n: i32, m: i32, x: f64, mu: f64, sigma_x: f64) -> f64 {
    let a_minus = x - mu - 0.5;
    let a_plus = x - mu + 0.5;
    let exp_minus = (-(a_minus * a_minus) / (2.0 * sigma_x * sigma_x)).exp();
    let exp_plus = (-(a_plus * a_plus) / (2.0 * sigma_x * sigma_x)).exp();
    (a_minus.powi(m) * exp_minus - a_plus.powi(m) * exp_plus)
        / (sigma_x.powi(n) * (2.0 * PI).sqrt())
}

#[inline]
fn derivative_gaussian_integral_sigma(
    x: f64,
    mu: f64,
    sigma_x: f64,
    photons: f64,
    psf_other: f64,
) -> (f64, f64) {
    let dudt = photons * psf_other * g_helper(2, 1, x, mu, sigma_x);
    let d2udt2 = photons
        * psf_other
        * (g_helper(5, 3, x, mu, sigma_x) - 2.0 * g_helper(3, 1, x, mu, sigma_x));
    (dudt, d2udt2)
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn derivative_gaussian_integral_iso_sigma(
    x: f64,
    y: f64,
    mu: f64,
    nu: f64,
    sigma: f64,
    photons: f64,
    psf_x: f64,
    psf_y: f64,
) -> (f64, f64) {
    let s2 = SQRT_2 * sigma;
    let a_plus = (x - mu + 0.5) / s2;
    let a_minus = (x - mu - 0.5) / s2;
    let b_plus = (y - nu + 0.5) / s2;
    let b_minus = (y - nu - 0.5) / s2;

    let exp_ap = (-a_plus * a_plus).exp();
    let exp_am = (-a_minus * a_minus).exp();
    let exp_bp = (-b_plus * b_plus).exp();
    let exp_bm = (-b_minus * b_minus).exp();

    let fx = a_minus * exp_am - a_plus * exp_ap;
    let fy = b_minus * exp_bm - b_plus * exp_bp;
    let sqrt_pi = PI.sqrt();
    let dpsf_x = fx / (sqrt_pi * sigma);
    let dpsf_y = fy / (sqrt_pi * sigma);

    let dfx_dt = (a_plus * exp_ap * (1.0 - 2.0 * a_plus * a_plus)
        - a_minus * exp_am * (1.0 - 2.0 * a_minus * a_minus))
        / sigma;
    let dfy_dy = (b_plus * exp_bp * (1.0 - 2.0 * b_plus * b_plus)
        - b_minus * exp_bm * (1.0 - 2.0 * b_minus * b_minus))
        / sigma;

    let d2psf_x = ((-fx / (sigma * sigma)) + dfx_dt / sigma) / sqrt_pi;
    let d2psf_y = ((-fy / (sigma * sigma)) + dfy_dy / sigma) / sqrt_pi;

    let dudt = photons * (psf_y * dpsf_x + psf_x * dpsf_y);
    let d2udt2 = photons * psf_y * d2psf_x + 2.0 * dpsf_x * dpsf_y + psf_x * d2psf_y;
    (dudt, d2udt2)
}

fn sum_and_center_of_mass(spot: &Array2<f64>, size: usize) -> (f64, f64, f64) {
    let mut y = 0.0;
    let mut x = 0.0;
    let mut sum = 0.0;
    for i in 0..size {
        for j in 0..size {
            let v = spot[[i, j]];
            y += v * (i as f64);
            x += v * (j as f64);
            sum += v;
        }
    }
    if sum <= 0.0 {
        let c = ((size - 1) as f64) / 2.0;
        return (0.01, c, c);
    }
    (sum, y / sum, x / sum)
}

fn mean_filter_min(spot: &Array2<f64>, size: usize) -> f64 {
    let mut min_val = f64::INFINITY;
    for k in 0..size {
        for l in 0..size {
            let min_m = k.saturating_sub(1);
            let max_m = (k + 2).min(size);
            let min_n = l.saturating_sub(1);
            let max_n = (l + 2).min(size);
            let n = ((max_m - min_m) * (max_n - min_n)) as f64;
            let mut nsum = 0.0;
            for m in min_m..max_m {
                for nn in min_n..max_n {
                    nsum += spot[[m, nn]];
                }
            }
            let avg = nsum / n;
            if avg < min_val {
                min_val = avg;
            }
        }
    }
    min_val
}

fn initial_sigmas(spot: &Array2<f64>, bg: f64, size: usize) -> (f64, f64) {
    let size_half = size / 2;
    let mut sum_dev_y = 0.0;
    let mut sum_dev_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_x = 0.0;
    for i in 0..size {
        let d = (i as f64) - (size_half as f64);
        let d2 = d * d;
        let vy = spot[[i, size_half]] - bg;
        let vx = spot[[size_half, i]] - bg;
        sum_dev_y += vy * d2;
        sum_dev_x += vx * d2;
        sum_y += vy;
        sum_x += vx;
    }
    let mut sy = (sum_dev_y / sum_y).sqrt();
    let mut sx = (sum_dev_x / sum_x).sqrt();
    if !sy.is_finite() || sy == 0.0 {
        sy = 0.01;
    }
    if !sx.is_finite() || sx == 0.0 {
        sx = 0.01;
    }
    (sy, sx)
}

fn initial_parameters(spot: &Array2<f64>, size: usize) -> (f64, f64, f64, f64, f64, f64) {
    let (sum, y, x) = sum_and_center_of_mass(spot, size);
    let bg = mean_filter_min(spot, size);
    let photons = (sum - (size * size) as f64 * bg).max(1.0);
    let (sy, sx) = initial_sigmas(spot, bg, size);
    (x, y, photons, bg, sx, sy)
}

/// Fits a single spot using Smith et al. (2010) pixel-integrated Newton-Raphson MLE (`picasso.gaussmle`).
///
/// Returns `(thetas: [x, y, photons, bg, sx, sy], crlbs: [vx, vy, vphotons, vbg, vsx, vsy], likelihood, iterations)`.
pub fn fit_spot_gaussmle(
    spot: &Array2<f64>,
    eps: f64,
    max_it: u32,
    method: MleMethod,
) -> ([f64; 6], [f64; 6], f64, u32) {
    let (size, _) = spot.dim();
    let (x0, y0, photons0, bg0, sx0, sy0) = initial_parameters(spot, size);

    match method {
        MleMethod::Fixed | MleMethod::Sigma => {
            let s0 = 0.5 * (sx0 + sy0);
            let mut theta = [x0, y0, photons0, bg0, s0];
            let max_step = [s0, s0, 0.1 * photons0, 0.1 * bg0, 0.2 * s0];
            let n_params = if method == MleMethod::Fixed { 4 } else { 5 };
            let mut old_x = theta[0];
            let mut old_y = theta[1];
            let mut kk = 0u32;

            while kk < max_it {
                kk += 1;
                let mut numerator = [0.0f64; 5];
                let mut denominator = [0.0f64; 5];

                for ii in 0..size {
                    let ii_f = ii as f64;
                    let psf_x = gaussian_integral(ii_f, theta[0], theta[4]);
                    for jj in 0..size {
                        let jj_f = jj as f64;
                        let psf_y = gaussian_integral(jj_f, theta[1], theta[4]);

                        let (du0, d2u0) = derivative_gaussian_integral(
                            ii_f, theta[0], theta[4], theta[2], psf_y,
                        );
                        let (du1, d2u1) = derivative_gaussian_integral(
                            jj_f, theta[1], theta[4], theta[2], psf_x,
                        );
                        let du2 = psf_x * psf_y;
                        let du3 = 1.0;
                        let (du4, d2u4) = if n_params == 5 {
                            derivative_gaussian_integral_iso_sigma(
                                ii_f, jj_f, theta[0], theta[1], theta[4], theta[2], psf_x, psf_y,
                            )
                        } else {
                            (0.0, 0.0)
                        };

                        let dudt = [du0, du1, du2, du3, du4];
                        let d2udt2 = [d2u0, d2u1, 0.0, 0.0, d2u4];

                        let model = theta[2] * psf_x * psf_y + theta[3];
                        let data = spot[[jj, ii]];
                        let (mut cf, mut df) = (0.0, 0.0);
                        if model > 10e-3 {
                            cf = data / model - 1.0;
                            df = data / (model * model);
                        }
                        cf = cf.min(10e4);
                        df = df.min(10e4);

                        for ll in 0..n_params {
                            numerator[ll] += cf * dudt[ll];
                            denominator[ll] += cf * d2udt2[ll] - df * dudt[ll] * dudt[ll];
                        }
                    }
                }

                for ll in 0..n_params {
                    let update = if denominator[ll] == 0.0 {
                        (numerator[ll] * max_step[ll]).signum()
                    } else {
                        (numerator[ll] / denominator[ll]).clamp(-max_step[ll], max_step[ll])
                    };
                    theta[ll] -= update;
                }
                theta[2] = theta[2].max(1.0);
                theta[3] = theta[3].max(0.01);
                if n_params == 5 {
                    theta[4] = theta[4].clamp(0.01, size as f64);
                }

                if (old_x - theta[0]).abs() < eps && (old_y - theta[1]).abs() < eps {
                    break;
                } else {
                    old_x = theta[0];
                    old_y = theta[1];
                }
            }

            let mut m_mat = Array2::<f64>::zeros((5, 5));
            let mut log_likelihood = 0.0;
            for ii in 0..size {
                let ii_f = ii as f64;
                let psf_x = gaussian_integral(ii_f, theta[0], theta[4]);
                for jj in 0..size {
                    let jj_f = jj as f64;
                    let psf_y = gaussian_integral(jj_f, theta[1], theta[4]);
                    let (du0, _) =
                        derivative_gaussian_integral(ii_f, theta[0], theta[4], theta[2], psf_y);
                    let (du1, _) =
                        derivative_gaussian_integral(jj_f, theta[1], theta[4], theta[2], psf_x);
                    let (du4, _) = derivative_gaussian_integral_iso_sigma(
                        ii_f, jj_f, theta[0], theta[1], theta[4], theta[2], psf_x, psf_y,
                    );
                    let dudt = [du0, du1, psf_x * psf_y, 1.0, du4];
                    let model = theta[2] * psf_x * psf_y + theta[3];
                    if model > 0.0 {
                        for k in 0..5 {
                            for l in k..5 {
                                let v = dudt[l] * dudt[k] / model;
                                m_mat[[k, l]] += v;
                                m_mat[[l, k]] = m_mat[[k, l]];
                            }
                        }
                        let data = spot[[jj, ii]];
                        if data > 0.0 {
                            log_likelihood += data * model.ln() - model - data * data.ln() + data;
                        } else {
                            log_likelihood += -model;
                        }
                    }
                }
            }
            let minv = linalg::pinv(&m_mat, 1e-12);
            let mut crlbs_out = [0.0f64; 6];
            for k in 0..5 {
                crlbs_out[k] = minv[[k, k]].max(0.0);
            }
            crlbs_out[5] = crlbs_out[4];
            let thetas_out = [theta[0], theta[1], theta[2], theta[3], theta[4], theta[4]];
            (thetas_out, crlbs_out, log_likelihood, kk)
        }
        MleMethod::SigmaYx => {
            let mut theta = [x0, y0, photons0, bg0, sx0, sy0];
            let max_step = [
                sx0,
                sx0,
                0.1 * photons0,
                0.1 * bg0,
                0.2 * sx0,
                0.2 * sy0,
            ];
            let mut old_x = theta[0];
            let mut old_y = theta[1];
            let mut old_sx = theta[4];
            let mut old_sy = theta[5];
            let mut kk = 0u32;

            while kk < max_it {
                kk += 1;
                let mut numerator = [0.0f64; 6];
                let mut denominator = [0.0f64; 6];

                for ii in 0..size {
                    let ii_f = ii as f64;
                    let psf_x = gaussian_integral(ii_f, theta[0], theta[4]);
                    for jj in 0..size {
                        let jj_f = jj as f64;
                        let psf_y = gaussian_integral(jj_f, theta[1], theta[5]);

                        let (du0, d2u0) = derivative_gaussian_integral(
                            ii_f, theta[0], theta[4], theta[2], psf_y,
                        );
                        let (du1, d2u1) = derivative_gaussian_integral(
                            jj_f, theta[1], theta[5], theta[2], psf_x,
                        );
                        let du2 = psf_x * psf_y;
                        let du3 = 1.0;
                        let (du4, d2u4) = derivative_gaussian_integral_sigma(
                            ii_f, theta[0], theta[4], theta[2], psf_y,
                        );
                        let (du5, d2u5) = derivative_gaussian_integral_sigma(
                            jj_f, theta[1], theta[5], theta[2], psf_x,
                        );

                        let dudt = [du0, du1, du2, du3, du4, du5];
                        let d2udt2 = [d2u0, d2u1, 0.0, 0.0, d2u4, d2u5];

                        let model = theta[2] * psf_x * psf_y + theta[3];
                        let data = spot[[jj, ii]];
                        let (mut cf, mut df) = (0.0, 0.0);
                        if model > 10e-3 {
                            cf = data / model - 1.0;
                            df = data / (model * model);
                        }
                        cf = cf.min(10e4);
                        df = df.min(10e4);

                        for ll in 0..6 {
                            numerator[ll] += cf * dudt[ll];
                            denominator[ll] += cf * d2udt2[ll] - df * dudt[ll] * dudt[ll];
                        }
                    }
                }

                for ll in 0..6 {
                    if denominator[ll] == 0.0 {
                        theta[ll] -= numerator[ll].signum() * max_step[ll];
                    } else {
                        theta[ll] -=
                            (numerator[ll] / denominator[ll]).clamp(-max_step[ll], max_step[ll]);
                    }
                }
                theta[2] = theta[2].max(1.0);
                theta[3] = theta[3].max(0.01);
                theta[4] = theta[4].max(0.01);
                theta[5] = theta[5].max(0.01);

                if (old_x - theta[0]).abs() < eps
                    && (old_y - theta[1]).abs() < eps
                    && (old_sx - theta[4]).abs() < eps
                    && (old_sy - theta[5]).abs() < eps
                {
                    break;
                }
                old_x = theta[0];
                old_y = theta[1];
                old_sx = theta[4];
                old_sy = theta[5];
            }

            let mut m_mat = Array2::<f64>::zeros((6, 6));
            let mut log_likelihood = 0.0;
            for ii in 0..size {
                let ii_f = ii as f64;
                let psf_x = gaussian_integral(ii_f, theta[0], theta[4]);
                for jj in 0..size {
                    let jj_f = jj as f64;
                    let psf_y = gaussian_integral(jj_f, theta[1], theta[5]);
                    let (du0, _) =
                        derivative_gaussian_integral(ii_f, theta[0], theta[4], theta[2], psf_y);
                    let (du1, _) =
                        derivative_gaussian_integral(jj_f, theta[1], theta[5], theta[2], psf_x);
                    let (du4, _) = derivative_gaussian_integral_sigma(
                        ii_f, theta[0], theta[4], theta[2], psf_y,
                    );
                    let (du5, _) = derivative_gaussian_integral_sigma(
                        jj_f, theta[1], theta[5], theta[2], psf_x,
                    );
                    let dudt = [du0, du1, psf_x * psf_y, 1.0, du4, du5];
                    let model = theta[2] * psf_x * psf_y + theta[3];
                    if model > 0.0 {
                        for k in 0..6 {
                            for l in k..6 {
                                let v = dudt[l] * dudt[k] / model;
                                m_mat[[k, l]] += v;
                                m_mat[[l, k]] = m_mat[[k, l]];
                            }
                        }
                        let data = spot[[jj, ii]];
                        if data > 0.0 {
                            log_likelihood += data * model.ln() - model - data * data.ln() + data;
                        } else {
                            log_likelihood += -model;
                        }
                    }
                }
            }
            let minv = linalg::pinv(&m_mat, 1e-12);
            let mut crlbs_out = [0.0f64; 6];
            for k in 0..6 {
                crlbs_out[k] = minv[[k, k]].max(0.0);
            }
            (theta, crlbs_out, log_likelihood, kk)
        }
    }
}

/// Batch pixel-integrated MLE fitting (`picasso.gaussmle.gaussmle`).
pub fn gaussmle(
    spots: &Array3<f64>,
    eps: f64,
    max_it: u32,
    method: MleMethod,
    parallel: bool,
) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>, Array1<u32>)> {
    let n_spots = spots.shape()[0];
    let fit_one = |i: usize| {
        let spot_patch = spots.slice(s![i, .., ..]).to_owned();
        fit_spot_gaussmle(&spot_patch, eps, max_it, method)
    };

    let res: Vec<_> = if parallel && n_spots > 1 {
        (0..n_spots).into_par_iter().map(fit_one).collect()
    } else {
        (0..n_spots).map(fit_one).collect()
    };

    let mut thetas = Array2::<f64>::zeros((n_spots, 6));
    let mut crlbs = Array2::<f64>::zeros((n_spots, 6));
    let mut likelihoods = Array1::<f64>::zeros(n_spots);
    let mut iterations = Array1::<u32>::zeros(n_spots);

    for (i, (th, cr, ll, it)) in res.into_iter().enumerate() {
        for k in 0..6 {
            thetas[[i, k]] = th[k];
            crlbs[[i, k]] = cr[k];
        }
        likelihoods[i] = ll;
        iterations[i] = it;
    }

    Ok((thetas, crlbs, likelihoods, iterations))
}

/// Converts `gaussmle` fit results into [`Localization`] records (`picasso.gaussmle.locs_from_fits`).
pub fn locs_from_fits(
    identifications: &[Identification],
    thetas: &Array2<f64>,
    crlbs: &Array2<f64>,
    likelihoods: &Array1<f64>,
    iterations: &Array1<u32>,
    box_size: usize,
) -> Vec<Localization> {
    let box_offset = (box_size / 2) as f64;
    let n = identifications.len();
    let mut locs = Vec::with_capacity(n);

    for (i, id) in identifications.iter().enumerate() {
        let x = thetas[[i, 0]] + (id.x as f64) - box_offset;
        let y = thetas[[i, 1]] + (id.y as f64) - box_offset;
        let photons = thetas[[i, 2]];
        let bg = thetas[[i, 3]];
        let sx = thetas[[i, 4]];
        let sy = thetas[[i, 5]];

        let lpx = crlbs[[i, 0]].max(0.0).sqrt();
        let lpy = crlbs[[i, 1]].max(0.0).sqrt();
        let photons_unc = crlbs[[i, 2]].max(0.0).sqrt();
        let bg_unc = crlbs[[i, 3]].max(0.0).sqrt();
        let sx_unc = crlbs[[i, 4]].max(0.0).sqrt();
        let sy_unc = crlbs[[i, 5]].max(0.0).sqrt();

        let a = sx.max(sy).max(1e-12);
        let b = sx.min(sy);
        let ellipticity = (a - b) / a;
        let peak = photons / (2.0 * PI * sx.max(1e-6) * sy.max(1e-6));

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
            peak: Some(peak),
            chi2: Some(likelihoods[i]),
            iterations: Some(iterations[i]),
            sx_unc: Some(sx_unc),
            sy_unc: Some(sy_unc),
            photons_unc: Some(photons_unc),
            bg_unc: Some(bg_unc),
            ..Default::default()
        });
    }

    locs
}
