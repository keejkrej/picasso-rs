//! Single-particle 2D rotational and translational alignment and averaging.
//!
//! Ported from `picasso.average`.

use crate::imageprocess::{get_image_shift, xcorr};
use crate::render::{RenderBlurMethod, render};
use crate::types::{Localization, Result};
use std::collections::BTreeMap;

/// Rotates a group of localizations around `(cx, cy)` by `angle_rad` and translates by `(dx, dy)`.
pub fn transform_locs(
    locs: &[Localization],
    cx: f64,
    cy: f64,
    angle_rad: f64,
    dx: f64,
    dy: f64,
) -> Vec<Localization> {
    let ct = angle_rad.cos();
    let st = angle_rad.sin();
    let mut out = locs.to_vec();
    for l in &mut out {
        let rx = l.x - cx;
        let ry = l.y - cy;
        l.x = cx + (rx * ct - ry * st) - dx;
        l.y = cy + (rx * st + ry * ct) - dy;
    }
    out
}

/// Aligns a single particle group against a reference template image over a grid of candidate
/// angles (`picasso.average.align_group`).
pub fn align_group(
    group_locs: &[Localization],
    template: &marray::np::Array2<f64>,
    viewport: ((f64, f64), (f64, f64)),
    oversampling: f64,
    angles_rad: &[f64],
) -> Result<Vec<Localization>> {
    if group_locs.is_empty() {
        return Ok(Vec::new());
    }
    let ((y_min, x_min), (y_max, x_max)) = viewport;
    let cx = 0.5 * (x_min + x_max);
    let cy = 0.5 * (y_min + y_max);

    // Center group on viewport center first
    let mx = group_locs.iter().map(|l| l.x).sum::<f64>() / (group_locs.len() as f64);
    let my = group_locs.iter().map(|l| l.y).sum::<f64>() / (group_locs.len() as f64);
    let centered = transform_locs(group_locs, mx, my, 0.0, mx - cx, my - cy);

    let mut best_score = f64::NEG_INFINITY;
    let mut best_locs = centered.clone();

    for &ang in angles_rad {
        let rotated = transform_locs(&centered, cx, cy, ang, 0.0, 0.0);
        let (_, img) = render(
            &rotated,
            Some(viewport),
            oversampling,
            RenderBlurMethod::Gaussian,
            0.0,
        )?;
        let corr = xcorr(template, &img)?;
        let max_corr = corr.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if max_corr > best_score {
            best_score = max_corr;
            let (dy_os, dx_os) = get_image_shift(template, &img, 5, None, false)?;
            best_locs = transform_locs(
                &rotated,
                cx,
                cy,
                0.0,
                dx_os / oversampling,
                dy_os / oversampling,
            );
        }
    }

    Ok(best_locs)
}

/// Iteratively aligns and averages grouped particles (`picasso.average`).
pub fn average_particles(
    locs: &[Localization],
    r_half_box: f64,
    oversampling: f64,
    n_angles: usize,
    iterations: usize,
) -> Result<Vec<Localization>> {
    let mut groups: BTreeMap<i32, Vec<Localization>> = BTreeMap::new();
    for l in locs {
        let g = l.group.unwrap_or(0);
        if g >= 0 {
            let mut c = l.clone();
            c.group = Some(g);
            groups.entry(g).or_default().push(c);
        }
    }
    if groups.is_empty() {
        return Ok(Vec::new());
    }

    let viewport = ((-r_half_box, -r_half_box), (r_half_box, r_half_box));
    let mut party: Vec<Vec<Localization>> = groups
        .into_values()
        .map(|g| {
            let mx = g.iter().map(|l| l.x).sum::<f64>() / (g.len() as f64);
            let my = g.iter().map(|l| l.y).sum::<f64>() / (g.len() as f64);
            transform_locs(&g, mx, my, 0.0, mx, my)
        })
        .collect();

    let n_ang = n_angles.max(1);
    let angles: Vec<f64> = (0..n_ang)
        .map(|k| (k as f64) * 2.0 * std::f64::consts::PI / (n_ang as f64))
        .collect();

    for _ in 0..iterations.max(1) {
        let all_flat: Vec<Localization> = party.iter().flatten().cloned().collect();
        let (_, template) = render(
            &all_flat,
            Some(viewport),
            oversampling,
            RenderBlurMethod::Gaussian,
            0.0,
        )?;
        let mut next_party = Vec::with_capacity(party.len());
        for grp in &party {
            next_party.push(align_group(grp, &template, viewport, oversampling, &angles)?);
        }
        party = next_party;
    }

    Ok(party.into_iter().flatten().collect())
}
