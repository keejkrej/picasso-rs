//! Fast ROI-averaging spot quantification (`picasso.avgroi`).

use crate::fitting::precision::localization_precision_scalar;
use crate::types::{Identification, Localization};
use marray::np::{Array2, Array3, s};

/// Quantifies a single spot by summing photon counts over the ROI (`picasso.avgroi.fit_spot`).
///
/// Returns `[x, y, photons, bg, sx, sy]`.
pub fn fit_spot(spot: &Array2<f64>) -> [f64; 6] {
    let avg_roi: f64 = spot.iter().sum();
    [0.0, 0.0, avg_roi, avg_roi, 1.0, 1.0]
}

/// Quantifies a batch of spots by summing ROI photons (`picasso.avgroi.fit_spots`).
pub fn fit_spots(spots: &Array3<f64>) -> Array2<f64> {
    let n = spots.shape()[0];
    let mut theta = Array2::<f64>::zeros((n, 6));
    for i in 0..n {
        let patch = spots.slice(s![i, .., ..]).to_owned();
        let row = fit_spot(&patch);
        for k in 0..6 {
            theta[[i, k]] = row[k];
        }
    }
    theta
}

/// Converts `avgroi::fit_spots` parameters into [`Localization`] records (`picasso.avgroi.locs_from_fits`).
pub fn locs_from_fits(
    identifications: &[Identification],
    theta: &Array2<f64>,
    box_size: usize,
    em: bool,
) -> Vec<Localization> {
    let mut locs = Vec::with_capacity(identifications.len());
    let _ = box_size;

    for (i, id) in identifications.iter().enumerate() {
        let photons = theta[[i, 2]];
        let bg = theta[[i, 3]];
        let sx = theta[[i, 4]];
        let sy = theta[[i, 5]];
        let x = id.x as f64;
        let y = id.y as f64;

        let lpx = localization_precision_scalar(photons, sx, sy, bg, em, 0.0);
        let lpy = localization_precision_scalar(photons, sy, sx, bg, em, 0.0);
        let a = sx.max(sy).max(1e-12);
        let b = sx.min(sy);
        let ellipticity = (a - b) / a;

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
            ..Default::default()
        });
    }

    locs
}
