//! Super-resolution 2D and 3D localization rendering.
//!
//! Ported from `picasso.render`.

use crate::types::{Localization, Result};
use marray::np::{Array2, Array3};
use marray::sp::ndimage::gaussian_filter;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

const MAGNIFICATION_FACTOR: f64 = 0.79;

/// Blur method for super-resolution rendering (`picasso.render.render`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RenderBlurMethod {
    /// Unblurred 2D sub-pixel histogram (`render_hist`).
    None,
    /// Individual anisotropic Gaussian blur using `(lpx, lpy)` (`render_gaussian`).
    #[default]
    Gaussian,
    /// Individual isotropic Gaussian blur using `max(lpx, lpy)` (`render_gaussian_iso`).
    GaussianIso,
    /// 2D histogram followed by 1-pixel Gaussian smoothing (`render_smooth`).
    Smooth,
    /// 2D histogram convolved with median localization precision (`render_convolve`).
    Convolve,
}

/// 2D rendering viewport `((y_min, x_min), (y_max, x_max))` in camera pixels.
pub type Viewport2D = ((f64, f64), (f64, f64));

/// 3D rendering viewport `((y_min, x_min, z_min), (y_max, x_max, z_max))` (`x, y` in pixels, `z` in nm).
pub type Viewport3D = ((f64, f64, f64), (f64, f64, f64));

/// Infers a default 2D viewport covering all localizations in `locs`.
pub fn default_viewport_2d(locs: &[Localization]) -> Viewport2D {
    if locs.is_empty() {
        return ((0.0, 0.0), (64.0, 64.0));
    }
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for l in locs {
        if l.x < min_x {
            min_x = l.x;
        }
        if l.x > max_x {
            max_x = l.x;
        }
        if l.y < min_y {
            min_y = l.y;
        }
        if l.y > max_y {
            max_y = l.y;
        }
    }
    (
        (min_y.floor().max(0.0), min_x.floor().max(0.0)),
        (max_y.ceil() + 1.0, max_x.ceil() + 1.0),
    )
}

/// Renders localizations into a 2D super-resolution image (`picasso.render.render`).
///
/// Returns `(n_rendered_locs, image)` of shape `(H_oversampled, W_oversampled)`.
pub fn render(
    locs: &[Localization],
    viewport: Option<Viewport2D>,
    oversampling: f64,
    blur_method: RenderBlurMethod,
    min_blur_width: f64,
) -> Result<(usize, Array2<f64>)> {
    let vp = viewport.unwrap_or_else(|| default_viewport_2d(locs));
    match blur_method {
        RenderBlurMethod::None => Ok(render_hist(locs, oversampling, vp)),
        RenderBlurMethod::Smooth => render_smooth(locs, oversampling, vp),
        RenderBlurMethod::Convolve => render_convolve(locs, oversampling, vp, min_blur_width),
        RenderBlurMethod::Gaussian => Ok(render_gaussian(locs, oversampling, vp, min_blur_width)),
        RenderBlurMethod::GaussianIso => {
            Ok(render_gaussian_iso(locs, oversampling, vp, min_blur_width))
        }
    }
}

/// Renders an unblurred 2D histogram of localizations (`picasso.render.render_hist`).
pub fn render_hist(
    locs: &[Localization],
    oversampling: f64,
    viewport: Viewport2D,
) -> (usize, Array2<f64>) {
    let ((y_min, x_min), (y_max, x_max)) = viewport;
    let n_pixel_y = ((y_max - y_min) * oversampling).ceil().max(1.0) as usize;
    let n_pixel_x = ((x_max - x_min) * oversampling).ceil().max(1.0) as usize;
    let mut image = Array2::<f64>::zeros((n_pixel_y, n_pixel_x));
    let mut count = 0usize;

    for l in locs {
        if l.x > x_min && l.x < x_max && l.y > y_min && l.y < y_max {
            let ix = ((l.x - x_min) * oversampling) as usize;
            let iy = ((l.y - y_min) * oversampling) as usize;
            if iy < n_pixel_y && ix < n_pixel_x {
                image[[iy, ix]] += 1.0;
                count += 1;
            }
        }
    }

    (count, image)
}

/// Renders a 2D histogram smoothed by a 1-subpixel Gaussian kernel (`picasso.render.render_smooth`).
pub fn render_smooth(
    locs: &[Localization],
    oversampling: f64,
    viewport: Viewport2D,
) -> Result<(usize, Array2<f64>)> {
    let (n, img) = render_hist(locs, oversampling, viewport);
    let smoothed = gaussian_filter(&img, 1.0);
    Ok((n, smoothed))
}

/// Renders a 2D histogram convolved with the global median localization precision (`picasso.render.render_convolve`).
pub fn render_convolve(
    locs: &[Localization],
    oversampling: f64,
    viewport: Viewport2D,
    min_blur_width: f64,
) -> Result<(usize, Array2<f64>)> {
    let (n, img) = render_hist(locs, oversampling, viewport);
    if n == 0 {
        return Ok((0, img));
    }
    let ((y_min, x_min), (y_max, x_max)) = viewport;
    let mut lps: Vec<f64> = Vec::with_capacity(n * 2);
    for l in locs {
        if l.x > x_min && l.x < x_max && l.y > y_min && l.y < y_max {
            lps.push(l.lpx);
            lps.push(l.lpy);
        }
    }
    lps.sort_by(|a, b| a.total_cmp(b));
    let med_lp = if lps.is_empty() {
        min_blur_width
    } else {
        lps[lps.len() / 2]
    };
    let blur_width = oversampling * med_lp.max(min_blur_width);
    if blur_width <= 0.0 {
        return Ok((n, img));
    }
    let blurred = gaussian_filter(&img, blur_width);
    Ok((n, blurred))
}

/// Renders each localization as an individual anisotropic 2D Gaussian with widths `(lpx, lpy)`
/// (`picasso.render.render_gaussian`).
pub fn render_gaussian(
    locs: &[Localization],
    oversampling: f64,
    viewport: Viewport2D,
    min_blur_width: f64,
) -> (usize, Array2<f64>) {
    let ((y_min, x_min), (y_max, x_max)) = viewport;
    let n_pixel_y = ((y_max - y_min) * oversampling).ceil().max(1.0) as usize;
    let n_pixel_x = ((x_max - x_min) * oversampling).ceil().max(1.0) as usize;
    let mut image = Array2::<f64>::zeros((n_pixel_y, n_pixel_x));
    let mut count = 0usize;

    for l in locs {
        if l.x > x_min && l.x < x_max && l.y > y_min && l.y < y_max {
            let x = (l.x - x_min) * oversampling;
            let y = (l.y - y_min) * oversampling;
            let blur_x = oversampling * l.lpx.max(min_blur_width).max(1e-3);
            let blur_y = oversampling * l.lpy.max(min_blur_width).max(1e-3);
            let sy = (3.0 * blur_y).ceil() as isize;
            let sx = (3.0 * blur_x).ceil() as isize;
            let yc = y.round() as isize;
            let xc = x.round() as isize;

            let i_min = (yc - sy).max(0) as usize;
            let i_max = ((yc + sy + 1) as usize).min(n_pixel_y);
            let j_min = (xc - sx).max(0) as usize;
            let j_max = ((xc + sx + 1) as usize).min(n_pixel_x);

            let norm = 1.0 / (2.0 * PI * blur_x * blur_y);
            for iy in i_min..i_max {
                let dy = (iy as f64) - y;
                let ey = -0.5 * dy * dy / (blur_y * blur_y);
                for ix in j_min..j_max {
                    let dx = (ix as f64) - x;
                    let ex = -0.5 * dx * dx / (blur_x * blur_x);
                    image[[iy, ix]] += (ex + ey).exp() * norm;
                }
            }
            count += 1;
        }
    }

    (count, image)
}

/// Renders each localization as an individual isotropic 2D Gaussian with width `max(lpx, lpy)`
/// (`picasso.render.render_gaussian_iso`).
pub fn render_gaussian_iso(
    locs: &[Localization],
    oversampling: f64,
    viewport: Viewport2D,
    min_blur_width: f64,
) -> (usize, Array2<f64>) {
    let iso_locs: Vec<Localization> = locs
        .iter()
        .map(|l| {
            let s = l.lpx.max(l.lpy);
            let mut copy = l.clone();
            copy.lpx = s;
            copy.lpy = s;
            copy
        })
        .collect();
    render_gaussian(&iso_locs, oversampling, viewport, min_blur_width)
}

/// Renders an unblurred 3D super-resolution histogram (`picasso.render.render_hist3d`).
pub fn render_hist3d(
    locs: &[Localization],
    oversampling: f64,
    viewport: Viewport3D,
    pixelsize_nm: f64,
) -> (usize, Array3<f64>) {
    let ((y_min, x_min, mut z_min), (y_max, x_max, mut z_max)) = viewport;
    let n_pixel_y = ((y_max - y_min) * oversampling).ceil().max(1.0) as usize;
    let n_pixel_x = ((x_max - x_min) * oversampling).ceil().max(1.0) as usize;
    z_min /= pixelsize_nm.max(1e-6);
    z_max /= pixelsize_nm.max(1e-6);
    let n_pixel_z = ((z_max - z_min) * oversampling).ceil().max(1.0) as usize;

    let mut volume = Array3::<f64>::zeros((n_pixel_y, n_pixel_x, n_pixel_z));
    let mut count = 0usize;

    for l in locs {
        let Some(z_nm) = l.z else { continue };
        let z_px = z_nm / pixelsize_nm.max(1e-6);
        if l.x > x_min
            && l.x < x_max
            && l.y > y_min
            && l.y < y_max
            && z_px > z_min
            && z_px < z_max
        {
            let iy = ((l.y - y_min) * oversampling) as usize;
            let ix = ((l.x - x_min) * oversampling) as usize;
            let iz = ((z_px - z_min) * oversampling) as usize;
            if iy < n_pixel_y && ix < n_pixel_x && iz < n_pixel_z {
                volume[[iy, ix, iz]] += 1.0;
                count += 1;
            }
        }
    }

    (count, volume)
}

/// Renders a 3D Gaussian-blurred super-resolution volume (`picasso.render.render_gaussian3d`).
pub fn render_gaussian3d(
    locs: &[Localization],
    oversampling: f64,
    viewport: Viewport3D,
    min_blur_width: f64,
    pixelsize_nm: f64,
) -> (usize, Array3<f64>) {
    let ((y_min, x_min, mut z_min), (y_max, x_max, mut z_max)) = viewport;
    let n_pixel_y = ((y_max - y_min) * oversampling).ceil().max(1.0) as usize;
    let n_pixel_x = ((x_max - x_min) * oversampling).ceil().max(1.0) as usize;
    z_min /= pixelsize_nm.max(1e-6);
    z_max /= pixelsize_nm.max(1e-6);
    let n_pixel_z = ((z_max - z_min) * oversampling).ceil().max(1.0) as usize;

    let mut volume = Array3::<f64>::zeros((n_pixel_y, n_pixel_x, n_pixel_z));
    let mut count = 0usize;
    let two_pi_32 = (2.0 * PI).powf(1.5);

    for l in locs {
        let Some(z_nm) = l.z else { continue };
        let z_px = z_nm / pixelsize_nm.max(1e-6);
        if l.x > x_min
            && l.x < x_max
            && l.y > y_min
            && l.y < y_max
            && z_px > z_min
            && z_px < z_max
        {
            let x = (l.x - x_min) * oversampling;
            let y = (l.y - y_min) * oversampling;
            let z = (z_px - z_min) * oversampling;

            let blur_x = oversampling * l.lpx.max(min_blur_width).max(1e-3);
            let blur_y = oversampling * l.lpy.max(min_blur_width).max(1e-3);
            let lpz_px = l
                .lpz
                .map_or(2.0 * (l.lpx + l.lpy) * MAGNIFICATION_FACTOR, |v| {
                    v / pixelsize_nm.max(1e-6)
                });
            let blur_z = oversampling * lpz_px.max(min_blur_width).max(1e-3);

            let sy = (3.0 * blur_y).ceil() as isize;
            let sx = (3.0 * blur_x).ceil() as isize;
            let sz = (3.0 * blur_z).ceil() as isize;

            let yc = y.round() as isize;
            let xc = x.round() as isize;
            let zc = z.round() as isize;

            let i_min = (yc - sy).max(0) as usize;
            let i_max = ((yc + sy + 1) as usize).min(n_pixel_y);
            let j_min = (xc - sx).max(0) as usize;
            let j_max = ((xc + sx + 1) as usize).min(n_pixel_x);
            let k_min = (zc - sz).max(0) as usize;
            let k_max = ((zc + sz + 1) as usize).min(n_pixel_z);

            let norm = 1.0 / (two_pi_32 * blur_x * blur_y * blur_z);
            for iy in i_min..i_max {
                let dy = (iy as f64) - y;
                let ey = -0.5 * dy * dy / (blur_y * blur_y);
                for ix in j_min..j_max {
                    let dx = (ix as f64) - x;
                    let ex = -0.5 * dx * dx / (blur_x * blur_x);
                    for iz in k_min..k_max {
                        let dz = (iz as f64) - z;
                        let ez = -0.5 * dz * dz / (blur_z * blur_z);
                        volume[[iy, ix, iz]] += (ex + ey + ez).exp() * norm;
                    }
                }
            }
            count += 1;
        }
    }

    (count, volume)
}
