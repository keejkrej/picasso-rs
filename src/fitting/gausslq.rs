//! Least-squares 2D Gaussian PSF fitting convenience module.
//!
//! Ported from `picasso.gausslq`.

use crate::fitting::gaussfit::{fit_single_gauss_spot, fit_spots_gauss};
use crate::fitting::precision::{localization_precision_scalar, sigma_uncertainty_lsq_scalar};
use crate::fitting::seeds::initial_parameters_gauss;
use crate::types::{FitEstimator, GaussModel, Identification, Localization, Result};
use marray::np::{Array2, Array3};
use std::f64::consts::PI;

/// Fits a single 2D spot `(size, size)` using unweighted 2D elliptical Gaussian least squares
/// (`picasso.gausslq.fit_spot`).
///
/// Returns `[x, y, photons, bg, sx, sy]`.
pub fn fit_spot(spot: &Array2<f64>) -> Result<[f64; 6]> {
    let (h, w) = spot.dim();
    let mut spots = Array3::<f64>::zeros((1, h, w));
    spots.slice_mut(marray::np::s![0, .., ..]).assign(spot);
    let init = initial_parameters_gauss(&spots, GaussModel::Elliptic)?;
    let init_slice = [
        init[[0, 0]],
        init[[0, 1]],
        init[[0, 2]],
        init[[0, 3]],
        init[[0, 4]],
        init[[0, 5]],
    ];
    let (p, _, _, _) = fit_single_gauss_spot(
        spot,
        None,
        &init_slice,
        GaussModel::Elliptic,
        false,
        1e-6,
        200,
    );
    let peak = p[0];
    let x = p[1];
    let y = p[2];
    let sx = p[3].abs();
    let sy = p[4].abs();
    let bg = p[5];
    let photons = peak * 2.0 * PI * sx * sy;
    Ok([x, y, photons, bg, sx, sy])
}

/// Fits a 3D stack of spots `(n_spots, size, size)` with 2D elliptical Gaussian least squares
/// (`picasso.gausslq.fit_spots` / `fit_spots_parallel`).
///
/// Returns an `Array2<f64>` of shape `(n_spots, 6)` with columns `[x, y, photons, bg, sx, sy]`.
pub fn fit_spots(spots: &Array3<f64>, parallel: bool) -> Result<Array2<f64>> {
    let res = fit_spots_gauss(
        spots,
        GaussModel::Elliptic,
        FitEstimator::LeastSquares,
        None,
        None,
        None,
        parallel,
    )?;
    let n = res.parameters.shape()[0];
    let mut theta = Array2::<f64>::zeros((n, 6));
    for i in 0..n {
        let peak = res.parameters[[i, 0]];
        let x = res.parameters[[i, 1]];
        let y = res.parameters[[i, 2]];
        let sx = res.parameters[[i, 3]].abs();
        let sy = res.parameters[[i, 4]].abs();
        let bg = res.parameters[[i, 5]];
        let photons = peak * 2.0 * PI * sx * sy;
        theta[[i, 0]] = x;
        theta[[i, 1]] = y;
        theta[[i, 2]] = photons;
        theta[[i, 3]] = bg;
        theta[[i, 4]] = sx;
        theta[[i, 5]] = sy;
    }
    Ok(theta)
}

/// Converts `gausslq::fit_spots` parameter array `[x, y, photons, bg, sx, sy]` into [`Localization`] records
/// (`picasso.gausslq.locs_from_fits`).
pub fn locs_from_fits(
    identifications: &[Identification],
    theta: &Array2<f64>,
    box_size: usize,
    em: bool,
) -> Vec<Localization> {
    let box_offset = (box_size / 2) as f64;
    let mut locs = Vec::with_capacity(identifications.len());

    for (i, id) in identifications.iter().enumerate() {
        let x = theta[[i, 0]] + (id.x as f64) - box_offset;
        let y = theta[[i, 1]] + (id.y as f64) - box_offset;
        let photons = theta[[i, 2]];
        let bg = theta[[i, 3]];
        let sx = theta[[i, 4]].abs();
        let sy = theta[[i, 5]].abs();

        let lpx = localization_precision_scalar(photons, sx, sy, bg, em, 0.0);
        let lpy = localization_precision_scalar(photons, sy, sx, bg, em, 0.0);
        let sx_unc = sigma_uncertainty_lsq_scalar(photons, sx, sy, bg, em, 0.0);
        let sy_unc = sigma_uncertainty_lsq_scalar(photons, sy, sx, bg, em, 0.0);

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
            sx_unc: Some(sx_unc),
            sy_unc: Some(sy_unc),
            ..Default::default()
        });
    }

    locs
}
