//! Utility and mathematical helper functions ported from `picasso.lib`.

use crate::types::{Localization, Result};
use marray::np::{self, Array1, Array2};
use marray::sp::linalg;

/// Ensures an integer box size is odd (rounds up even integers to `n + 1`, and clamps to `>= 3`).
#[inline]
pub fn ensure_odd(size: usize) -> usize {
    let s = size.max(3);
    if s.is_multiple_of(2) { s + 1 } else { s }
}

/// Evaluates the cumulative exponential function used in dark-time fitting (`picasso.lib.cumulative_exponential`):
/// `f(x; a, t, c) = a * (1 - exp(-x / t)) + c`.
#[inline]
pub fn cumulative_exponential(x: f64, a: f64, t: f64, c: f64) -> f64 {
    if t.abs() < 1e-12 {
        a + c
    } else {
        a * (1.0 - (-x / t).exp()) + c
    }
}

/// Checks which localizations fall inside a circular region of radius `r` centered at `(x, y)` (`picasso.lib.is_loc_at`).
pub fn is_loc_at(locs: &[Localization], x: f64, y: f64, r: f64) -> Vec<bool> {
    let r2 = r * r;
    locs.iter()
        .map(|loc| {
            let dx = loc.x - x;
            let dy = loc.y - y;
            dx.abs() <= r && dy.abs() <= r && (dx * dx + dy * dy) <= r2
        })
        .collect()
}

/// Checks if a 2D point `(px, py)` lies inside a closed polygon defined by `vertices` using ray-casting.
pub fn point_in_polygon(px: f64, py: f64, vertices: &[(f64, f64)]) -> bool {
    let n = vertices.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = vertices[i];
        let (xj, yj) = vertices[j];
        let intersect = ((yi > py) != (yj > py))
            && (px < (xj - xi) * (py - yi) / ((yj - yi) + 1e-30) + xi);
        if intersect {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Checks if a 2D point `(px, py)` lies inside a rotated rectangle defined by centerline `(p1, p2)` and `width`.
pub fn point_in_rectangle(
    px: f64,
    py: f64,
    p1: (f64, f64),
    p2: (f64, f64),
    width: f64,
) -> bool {
    let vx = p2.0 - p1.0;
    let vy = p2.1 - p1.1;
    let len2 = vx * vx + vy * vy;
    if len2 <= 1e-24 {
        return false;
    }
    let len = len2.sqrt();
    let ux = vx / len;
    let uy = vy / len;
    let dx = px - p1.0;
    let dy = py - p1.1;
    let proj_along = dx * ux + dy * uy;
    let proj_perp = (-dx * uy + dy * ux).abs();
    proj_along >= 0.0 && proj_along <= len && proj_perp <= 0.5 * width
}

/// Computes optimal histogram bin edges using the Freedman-Diaconis rule (`picasso.lib.calculate_optimal_bins`).
pub fn calculate_optimal_bins(data: &[f64], max_n_bins: Option<usize>) -> Result<Array1<f64>> {
    if data.is_empty() {
        return Err("Cannot compute optimal bins for empty data".into());
    }
    let arr = Array1::from_vec(data.to_vec());
    let iqr = np::percentile(&arr, 75.0) - np::percentile(&arr, 25.0);
    let data_max = np::max(&arr);
    let data_min = np::min(&arr);
    if data_max <= data_min {
        return Ok(np::linspace(data_min, data_min + 1.0, 2));
    }
    let bin_size = 2.0 * iqr * (data.len() as f64).powf(-1.0 / 3.0);
    let mut n_bins = if bin_size <= 1e-12 || !bin_size.is_finite() {
        ((data.len() as f64).sqrt().ceil() as usize).max(2)
    } else {
        (((data_max - data_min) / bin_size).ceil() as usize).max(2)
    };
    if let Some(max_b) = max_n_bins {
        n_bins = n_bins.min(max_b.max(2));
    }
    Ok(np::linspace(data_min, data_max, n_bins + 1))
}

/// Solves overdetermined pairwise frame-segment shift matrices (`picasso.lib.minimize_shifts`)
/// using `marray::sp::linalg::lstsq`.
///
/// Given pairwise shifts `shifts_x[i, j]` and `shifts_y[i, j]` between segment `i` and segment `j` (`0 <= i < j < n_channels`),
/// returns the optimal relative drift trajectory `(shift_x, shift_y)` of length `n_channels` with `shift[0] = 0`.
pub fn minimize_shifts(
    shifts_x: &Array2<f64>,
    shifts_y: &Array2<f64>,
    weights: Option<&Array2<f64>>,
) -> Result<(Array1<f64>, Array1<f64>)> {
    let shape = shifts_x.shape();
    let n_channels = shape[0];
    if shape[1] != n_channels || shifts_y.shape() != shape {
        return Err("shifts_x and shifts_y must be square (N, N) matrices of equal shape".into());
    }
    if n_channels <= 1 {
        return Ok((
            Array1::<f64>::zeros(n_channels),
            Array1::<f64>::zeros(n_channels),
        ));
    }

    let n_equations = n_channels * (n_channels - 1) / 2;
    let n_unknowns = n_channels - 1;
    let mut a = Array2::<f64>::zeros((n_equations, n_unknowns));
    let mut bx = Array1::<f64>::zeros(n_equations);
    let mut by = Array1::<f64>::zeros(n_equations);

    let mut eq = 0;
    for i in 0..(n_channels - 1) {
        for j in (i + 1)..n_channels {
            let w = weights.map_or(1.0, |w_mat| w_mat[[i, j]]);
            for k in i..j {
                a[[eq, k]] = w;
            }
            bx[eq] = shifts_x[[i, j]] * w;
            by[eq] = shifts_y[[i, j]] * w;
            eq += 1;
        }
    }

    let (sol_x, _, _, _) = linalg::lstsq(&a, &bx, None)?;
    let (sol_y, _, _, _) = linalg::lstsq(&a, &by, None)?;

    let mut traj_x = Array1::<f64>::zeros(n_channels);
    let mut traj_y = Array1::<f64>::zeros(n_channels);
    let mut acc_x = 0.0;
    let mut acc_y = 0.0;
    for k in 0..n_unknowns {
        acc_x -= sol_x[k];
        acc_y -= sol_y[k];
        traj_x[k + 1] = acc_x;
        traj_y[k + 1] = acc_y;
    }

    Ok((traj_x, traj_y))
}
