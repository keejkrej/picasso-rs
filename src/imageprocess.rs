//! 2D FFT Cross-Correlation and Redundant Cross-Correlation (RCC) Drift Estimation.
//!
//! Ported from `picasso.imageprocess`. Uses `marray::sp::fft::{Complex, fft2, ifft2}` and
//! `marray::sp::linalg::lstsq`.

use crate::fitting::gausslq;
use crate::lib_utils::minimize_shifts;
use crate::types::Result;
use marray::np::{Array1, Array2, s};
use marray::sp::fft::{Complex, fft2, ifft2};

/// Shifts the zero-frequency component to the center of a 2D array (`numpy.fft.fftshift`).
pub fn fftshift2d<T: Clone>(arr: &Array2<T>) -> Array2<T> {
    let (h, w) = arr.dim();
    let sy = h / 2;
    let sx = w / 2;
    let mut out = arr.clone();
    for iy in 0..h {
        let dst_y = (iy + sy) % h;
        for ix in 0..w {
            let dst_x = (ix + sx) % w;
            out[[dst_y, dst_x]] = arr[[iy, ix]].clone();
        }
    }
    out
}

/// Computes the 2D FFT cross-correlation between `image_a` and `image_b`
/// (`picasso.imageprocess.xcorr`).
pub fn xcorr(image_a: &Array2<f64>, image_b: &Array2<f64>) -> Result<Array2<f64>> {
    if image_a.dim() != image_b.dim() {
        return Err("image_a and image_b must have identical 2D dimensions".into());
    }
    let (h, w) = image_a.dim();
    let ca = image_a.mapv(|x| Complex::new(x, 0.0));
    let cb = image_b.mapv(|x| Complex::new(x, 0.0));
    let fft_a = fft2(&ca);
    let fft_b = fft2(&cb);

    // Multiply conj(fft_a) * fft_b
    let mut prod = fft_a;
    for (pa, pb) in prod.iter_mut().zip(fft_b.iter()) {
        *pa = pa.conj() * (*pb);
    }

    let corr_complex = ifft2(&prod);
    let mut corr_real = Array2::<f64>::zeros((h, w));
    for iy in 0..h {
        for ix in 0..w {
            corr_real[[iy, ix]] = corr_complex[[iy, ix]].re;
        }
    }

    Ok(fftshift2d(&corr_real))
}

/// Estimates the sub-pixel `(shift_y, shift_x)` between `image_a` and `image_b` by fitting a 2D
/// Gaussian to the peak of their 2D cross-correlation (`picasso.imageprocess.get_image_shift`).
pub fn get_image_shift(
    image_a: &Array2<f64>,
    image_b: &Array2<f64>,
    box_size: usize,
    roi: Option<usize>,
    display: bool,
) -> Result<(f64, f64)> {
    let _ = display;
    let sum_a: f64 = image_a.iter().sum();
    let sum_b: f64 = image_b.iter().sum();
    if sum_a <= 0.0 || sum_b <= 0.0 {
        return Ok((0.0, 0.0));
    }

    let mut corr = xcorr(image_a, image_b)?;
    let (mut h, mut w) = corr.dim();
    let mut y_off = 0isize;
    let mut x_off = 0isize;

    if let Some(r_size) = roi
        && r_size < h
        && r_size < w
    {
        let y0 = (h - r_size) / 2;
        let x0 = (w - r_size) / 2;
        corr = corr.slice(s![y0..(y0 + r_size), x0..(x0 + r_size)]).to_owned();
        y_off = y0 as isize;
        x_off = x0 as isize;
        let dim = corr.dim();
        h = dim.0;
        w = dim.1;
    }

    // Find global maximum in `corr`
    let mut max_val = f64::NEG_INFINITY;
    let mut yc = h / 2;
    let mut xc = w / 2;
    for iy in 0..h {
        for ix in 0..w {
            let v = corr[[iy, ix]];
            if v > max_val {
                max_val = v;
                yc = iy;
                xc = ix;
            }
        }
    }

    let box_half = box_size / 2;
    let orig_dim = image_a.dim();
    let center_y = (orig_dim.0 / 2) as f64;
    let center_x = (orig_dim.1 / 2) as f64;

    if yc >= box_half && yc + box_half < h && xc >= box_half && xc + box_half < w {
        let patch = corr
            .slice(s![
                (yc - box_half)..=(yc + box_half),
                (xc - box_half)..=(xc + box_half)
            ])
            .to_owned();
        if let Ok(theta) = gausslq::fit_spot(&patch) {
            let x_fit = theta[0] + (xc as f64) - (box_half as f64) + (x_off as f64);
            let y_fit = theta[1] + (yc as f64) - (box_half as f64) + (y_off as f64);
            if x_fit.is_finite() && y_fit.is_finite() {
                return Ok((y_fit - center_y, x_fit - center_x));
            }
        }
    }

    let y_global = (yc as isize + y_off) as f64;
    let x_global = (xc as isize + x_off) as f64;
    Ok((y_global - center_y, x_global - center_x))
}

/// Computes Redundant Cross-Correlation (RCC) shifts across a list of rendered segment images
/// (`picasso.imageprocess.rcc`).
///
/// Returns `(shifts_x, shifts_y)` of length `segments.len()`.
pub fn rcc(
    segments: &[Array2<f64>],
    max_shift: Option<usize>,
) -> Result<(Array1<f64>, Array1<f64>)> {
    let n_segments = segments.len();
    if n_segments == 0 {
        return Err("No segments provided to rcc".into());
    }
    let mut shifts_x = Array2::<f64>::zeros((n_segments, n_segments));
    let mut shifts_y = Array2::<f64>::zeros((n_segments, n_segments));
    let roi = max_shift.map(|ms| ms * 2 + 1);

    for i in 0..(n_segments - 1) {
        for j in (i + 1)..n_segments {
            let (dy, dx) = get_image_shift(&segments[i], &segments[j], 5, roi, false)?;
            shifts_y[[i, j]] = dy;
            shifts_x[[i, j]] = dx;
            shifts_y[[j, i]] = -dy;
            shifts_x[[j, i]] = -dx;
        }
    }

    minimize_shifts(&shifts_x, &shifts_y, None)
}
