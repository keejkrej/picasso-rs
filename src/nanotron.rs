//! Nanotron: Spot-shape feature extraction and classification for multi-structure DNA-PAINT.
//!
//! Ported from `picasso.nanotron`.

use crate::render::{RenderBlurMethod, render};
use crate::types::{Localization, Result};
use marray::np::Array1;

/// Renders a normalized flattened patch vector of length `roi_pixels * roi_pixels` for a single
/// picked localization group (`picasso.nanotron.roi_to_img`).
pub fn roi_to_feature_vector(
    group_locs: &[Localization],
    pick_radius_px: f64,
    oversampling: f64,
) -> Result<Array1<f64>> {
    if group_locs.is_empty() {
        let n = ((2.0 * pick_radius_px * oversampling).ceil().max(1.0) as usize).pow(2);
        return Ok(Array1::<f64>::zeros(n));
    }
    let mx = group_locs.iter().map(|l| l.x).sum::<f64>() / (group_locs.len() as f64);
    let my = group_locs.iter().map(|l| l.y).sum::<f64>() / (group_locs.len() as f64);
    let vp = (
        (my - pick_radius_px, mx - pick_radius_px),
        (my + pick_radius_px, mx + pick_radius_px),
    );
    let (_, img) = render(
        group_locs,
        Some(vp),
        oversampling,
        RenderBlurMethod::GaussianIso,
        0.0,
    )?;
    let max_v = img.iter().copied().fold(0.0f64, f64::max).max(1e-12);
    let flat: Vec<f64> = img.iter().map(|&v| (v / max_v).clamp(0.0, 1.0)).collect();
    Ok(Array1::from_vec(flat))
}
