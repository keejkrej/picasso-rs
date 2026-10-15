//! G5M: Gaussian Mixture Modeling for Molecular Mapping.
//!
//! Ported from `picasso.g5m` (Kowalewski et al., 2024). Fits weighted Gaussian Mixture Models
//! to localization groups using AIC/BIC model order selection and `marray::sklearn::cluster::KMeans`
//! initialization.

use crate::types::{Localization, Result};
use marray::np::Array2;
use marray::sklearn::cluster::KMeans;
use std::collections::BTreeMap;
use std::f64::consts::PI;

/// Configuration for [`g5m`].
#[derive(Debug, Clone)]
pub struct G5mConfig {
    /// Minimum localizations required in a group/cluster to fit GMM.
    pub min_locs: usize,
    /// Maximum number of Gaussian components tested per group.
    pub max_k: usize,
    /// EM convergence tolerance on log-likelihood.
    pub tol: f64,
    /// Maximum EM iterations.
    pub max_iter: usize,
}

impl Default for G5mConfig {
    fn default() -> Self {
        Self {
            min_locs: 10,
            max_k: 5,
            tol: 1e-4,
            max_iter: 100,
        }
    }
}

/// Fits a `k`-component isotropic/diagonal Gaussian Mixture Model to 2D points `(x, y)`
/// weighted by inverse localization variance `1 / (lpx^2 + lpy^2)`.
///
/// Returns `(means, sigmas, weights, log_likelihood, aic)`.
fn fit_gmm_2d(
    pts: &Array2<f64>,
    weights_in: &[f64],
    k: usize,
    tol: f64,
    max_iter: usize,
) -> Result<(Vec<(f64, f64)>, Vec<(f64, f64)>, Vec<f64>, f64, f64)> {
    let n = pts.shape()[0];
    if n < k || k == 0 {
        return Err("Insufficient points for k-component GMM".into());
    }

    // Initialize component centers via `marray::sklearn::cluster::KMeans`
    let mut means = Vec::with_capacity(k);
    if k == 1 {
        let mx = pts.column(0).sum() / (n as f64);
        let my = pts.column(1).sum() / (n as f64);
        means.push((mx, my));
    } else {
        let mut km = KMeans::new(k, 50);
        km.fit(pts);
        if let Some(centers) = &km.cluster_centers {
            for c in 0..k {
                means.push((centers[[c, 0]], centers[[c, 1]]));
            }
        } else {
            for c in 0..k {
                let idx = (c * n) / k;
                means.push((pts[[idx, 0]], pts[[idx, 1]]));
            }
        }
    }

    let mut sigmas = vec![(0.2f64, 0.2f64); k];
    let mut mix_w = vec![1.0 / (k as f64); k];
    let mut resp = vec![vec![0.0f64; k]; n];

    let w_total: f64 = weights_in.iter().sum::<f64>().max(1e-12);
    let norm_w: Vec<f64> = weights_in
        .iter()
        .map(|&w| w * (n as f64) / w_total)
        .collect();

    let mut prev_ll = f64::NEG_INFINITY;

    for _ in 0..max_iter {
        let mut ll = 0.0;
        // E-step
        for i in 0..n {
            let x = pts[[i, 0]];
            let y = pts[[i, 1]];
            let mut sum_p = 0.0;
            for c in 0..k {
                let (mx, my) = means[c];
                let (sx, sy) = sigmas[c];
                let sx_c = sx.max(1e-3);
                let sy_c = sy.max(1e-3);
                let pdf = (1.0 / (2.0 * PI * sx_c * sy_c))
                    * (-0.5 * (((x - mx) / sx_c).powi(2) + ((y - my) / sy_c).powi(2))).exp();
                let p = (mix_w[c] * pdf).max(1e-30);
                resp[i][c] = p;
                sum_p += p;
            }
            for c in 0..k {
                resp[i][c] /= sum_p.max(1e-30);
            }
            ll += norm_w[i] * sum_p.max(1e-30).ln();
        }

        if (ll - prev_ll).abs() < tol {
            prev_ll = ll;
            break;
        }
        prev_ll = ll;

        // M-step
        for c in 0..k {
            let mut nk = 0.0;
            let mut mx = 0.0;
            let mut my = 0.0;
            for i in 0..n {
                let r = norm_w[i] * resp[i][c];
                nk += r;
                mx += r * pts[[i, 0]];
                my += r * pts[[i, 1]];
            }
            if nk > 1e-9 {
                mx /= nk;
                my /= nk;
                means[c] = (mx, my);
                mix_w[c] = (nk / (n as f64)).clamp(1e-4, 1.0);

                let mut vx = 0.0;
                let mut vy = 0.0;
                for i in 0..n {
                    let r = norm_w[i] * resp[i][c];
                    vx += r * (pts[[i, 0]] - mx).powi(2);
                    vy += r * (pts[[i, 1]] - my).powi(2);
                }
                sigmas[c] = (
                    (vx / nk).max(1e-4).sqrt(),
                    (vy / nk).max(1e-4).sqrt(),
                );
            }
        }
    }

    // Number of free parameters in 2D diagonal GMM: k * 2 (means) + k * 2 (sigmas) + (k - 1) (weights)
    let n_params = (5 * k - 1) as f64;
    let aic = 2.0 * n_params - 2.0 * prev_ll;

    Ok((means, sigmas, mix_w, prev_ll, aic))
}

/// Runs G5M molecular mapping across clustered localization groups (`picasso.g5m.g5m`),
/// selecting the optimal number of molecular coordinates `1..=config.max_k` per cluster via AIC.
pub fn g5m(locs: &[Localization], config: &G5mConfig) -> Result<Vec<Localization>> {
    let mut groups: BTreeMap<i32, Vec<&Localization>> = BTreeMap::new();
    for l in locs {
        let g = l.group.or(l.cluster).unwrap_or(0);
        if g >= 0 {
            groups.entry(g).or_default().push(l);
        }
    }

    let mut out = Vec::new();
    for (gid, list) in groups {
        let n = list.len();
        if n < config.min_locs {
            continue;
        }
        let mut pts = Array2::<f64>::zeros((n, 2));
        let mut w_in = Vec::with_capacity(n);
        for (i, l) in list.iter().enumerate() {
            pts[[i, 0]] = l.x;
            pts[[i, 1]] = l.y;
            let var = (l.lpx * l.lpx + l.lpy * l.lpy).max(1e-6);
            w_in.push(1.0 / var);
        }

        let max_k = config.max_k.min(n / 3).max(1);
        let mut best_aic = f64::INFINITY;
        let mut best_fit = None;

        for k in 1..=max_k {
            if let Ok((means, sigmas, mix_w, ll, aic)) =
                fit_gmm_2d(&pts, &w_in, k, config.tol, config.max_iter)
                && aic < best_aic
            {
                best_aic = aic;
                best_fit = Some((means, sigmas, mix_w, ll));
            }
        }

        if let Some((means, sigmas, mix_w, _)) = best_fit {
            let nf = n as f64;
            let mean_frame = (list.iter().map(|l| l.frame as f64).sum::<f64>() / nf).round() as u32;
            let mean_photons = list.iter().map(|l| l.photons).sum::<f64>() / nf;
            let mean_bg = list.iter().map(|l| l.bg).sum::<f64>() / nf;
            let mean_ng = list.iter().map(|l| l.net_gradient).sum::<f64>() / nf;

            for ((mx, my), ((sx, sy), mw)) in means.into_iter().zip(sigmas.into_iter().zip(mix_w)) {
                let n_eff = (mw * nf).max(1.0);
                let lpx = sx / n_eff.sqrt();
                let lpy = sy / n_eff.sqrt();
                let a = sx.max(sy).max(1e-12);
                let b = sx.min(sy);
                out.push(Localization {
                    frame: mean_frame,
                    x: mx,
                    y: my,
                    photons: mean_photons * mw,
                    sx,
                    sy,
                    bg: mean_bg,
                    lpx,
                    lpy,
                    ellipticity: (a - b) / a,
                    net_gradient: mean_ng,
                    group: Some(gid),
                    n: Some(n_eff.round() as u32),
                    ..Default::default()
                });
            }
        }
    }

    Ok(out)
}
