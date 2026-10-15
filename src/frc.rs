//! Fourier Ring Correlation (FRC) super-resolution estimation.
//!
//! Ported from `picasso.frc`. Uses `marray::sp::fft::{Complex, fft2}`.

use crate::imageprocess::fftshift2d;
use crate::render::{RenderBlurMethod, default_viewport_2d, render};
use crate::types::{Localization, Result};
use marray::np::Array2;
use marray::sp::fft::{Complex, fft2};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Result of Fourier Ring Correlation analysis (`picasso.frc`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrcResult {
    /// Estimated FRC resolution in nanometers (at `1/7` threshold).
    pub resolution_nm: f64,
    /// Spatial frequencies `q` in `nm^-1`.
    pub spatial_frequencies: Vec<f64>,
    /// Raw radial FRC values per ring.
    pub frc_curve: Vec<f64>,
    /// Smoothed radial FRC curve.
    pub frc_smooth: Vec<f64>,
    /// Fixed threshold (`1.0 / 7.0 ≈ 0.142857`).
    pub threshold: f64,
}

/// Computes the radial Fourier Ring Correlation curve between two equal-sized 2D images
/// (`picasso.frc`).
pub fn frc_from_images(
    image1: &Array2<f64>,
    image2: &Array2<f64>,
    pixel_size_nm: f64,
) -> Result<FrcResult> {
    if image1.dim() != image2.dim() {
        return Err("image1 and image2 must have identical dimensions for FRC".into());
    }
    let (h, w) = image1.dim();
    let n = h.min(w);
    if n < 8 {
        return Err("Images must be at least 8x8 for FRC calculation".into());
    }

    // Apply Hann window to suppress edge ringing
    let mut win1 = Array2::<Complex<f64>>::zeros((h, w));
    let mut win2 = Array2::<Complex<f64>>::zeros((h, w));
    for iy in 0..h {
        let wy = 0.5 * (1.0 - (2.0 * PI * (iy as f64) / ((h - 1).max(1) as f64)).cos());
        for ix in 0..w {
            let wx = 0.5 * (1.0 - (2.0 * PI * (ix as f64) / ((w - 1).max(1) as f64)).cos());
            let weight = wy * wx;
            win1[[iy, ix]] = Complex::new(image1[[iy, ix]] * weight, 0.0);
            win2[[iy, ix]] = Complex::new(image2[[iy, ix]] * weight, 0.0);
        }
    }

    let f1 = fftshift2d(&fft2(&win1));
    let f2 = fftshift2d(&fft2(&win2));

    let max_r = n / 2;
    let cy = (h / 2) as f64;
    let cx = (w / 2) as f64;

    let mut num = vec![0.0f64; max_r];
    let mut den1 = vec![0.0f64; max_r];
    let mut den2 = vec![0.0f64; max_r];

    for iy in 0..h {
        let dy = (iy as f64) - cy;
        for ix in 0..w {
            let dx = (ix as f64) - cx;
            let r = (dx * dx + dy * dy).sqrt().round() as usize;
            if r > 0 && r < max_r {
                let c1 = f1[[iy, ix]];
                let c2 = f2[[iy, ix]];
                num[r] += (c1 * c2.conj()).re;
                den1[r] += c1.norm_sqr();
                den2[r] += c2.norm_sqr();
            }
        }
    }

    let mut freqs = Vec::with_capacity(max_r - 1);
    let mut frc_raw = Vec::with_capacity(max_r - 1);
    let fov_nm = (n as f64) * pixel_size_nm.max(1e-6);

    for r in 1..max_r {
        let q = (r as f64) / fov_nm;
        let denom = (den1[r] * den2[r]).sqrt();
        let val = if denom > 1e-12 {
            (num[r] / denom).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        freqs.push(q);
        frc_raw.push(val);
    }

    // Moving-average smoothing (window = 7)
    let m = frc_raw.len();
    let mut frc_smooth = vec![0.0f64; m];
    let half_w = 3usize;
    for i in 0..m {
        let start = i.saturating_sub(half_w);
        let end = (i + half_w + 1).min(m);
        let sum: f64 = frc_raw[start..end].iter().sum();
        frc_smooth[i] = sum / ((end - start) as f64);
    }

    let threshold = 1.0 / 7.0;
    let mut crossing_q = *freqs.last().unwrap_or(&1e-3);
    for i in 0..(m.saturating_sub(1)) {
        if frc_smooth[i] >= threshold && frc_smooth[i + 1] < threshold {
            let dy = (frc_smooth[i] - frc_smooth[i + 1]).max(1e-12);
            let frac = (frc_smooth[i] - threshold) / dy;
            crossing_q = freqs[i] + frac * (freqs[i + 1] - freqs[i]);
            break;
        }
    }
    let resolution_nm = if crossing_q > 1e-12 {
        1.0 / crossing_q
    } else {
        f64::INFINITY
    };

    Ok(FrcResult {
        resolution_nm,
        spatial_frequencies: freqs,
        frc_curve: frc_raw,
        frc_smooth,
        threshold,
    })
}

/// Splits localizations into two independent half-sets (by alternating frame blocks or even/odd frames),
/// renders both at `oversampling`, and computes the FRC resolution (`picasso.frc`).
pub fn frc(
    locs: &[Localization],
    pixelsize_nm: f64,
    oversampling: f64,
    block_size_frames: usize,
) -> Result<FrcResult> {
    if locs.len() < 4 {
        return Err("At least 4 localizations are required for FRC".into());
    }
    let vp = default_viewport_2d(locs);
    let blk = block_size_frames.max(1) as u32;
    let mut set1 = Vec::with_capacity(locs.len() / 2 + 1);
    let mut set2 = Vec::with_capacity(locs.len() / 2 + 1);

    for l in locs {
        if (l.frame / blk).is_multiple_of(2) {
            set1.push(l.clone());
        } else {
            set2.push(l.clone());
        }
    }
    if set1.is_empty() || set2.is_empty() {
        set1.clear();
        set2.clear();
        for (idx, l) in locs.iter().enumerate() {
            if idx % 2 == 0 {
                set1.push(l.clone());
            } else {
                set2.push(l.clone());
            }
        }
    }

    let (_, img1) = render(&set1, Some(vp), oversampling, RenderBlurMethod::Gaussian, 0.0)?;
    let (_, img2) = render(&set2, Some(vp), oversampling, RenderBlurMethod::Gaussian, 0.0)?;
    let eff_px_nm = pixelsize_nm / oversampling.max(1e-6);

    frc_from_images(&img1, &img2, eff_px_nm)
}
