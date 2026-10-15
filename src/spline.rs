//! 2D Bicubic and 3D Tricubic Catmull-Rom PSF Spline Calibration.
//!
//! Ported from `picasso.spline`.

use crate::types::Result;
use marray::np::{Array2, Array3, ArrayD, IxDyn};

/// Returns the 1D Catmull-Rom cubic polynomial coefficients `[a0, a1, a2, a3]` for 4 grid samples.
pub fn catmull_rom_1d_coeffs(f_m1: f64, f_0: f64, f_1: f64, f_2: f64) -> [f64; 4] {
    let df_0 = 0.5 * (f_1 - f_m1);
    let df_1 = 0.5 * (f_2 - f_0);
    let a0 = f_0;
    let a1 = df_0;
    let a2 = 3.0 * (f_1 - f_0) - 2.0 * df_0 - df_1;
    let a3 = 2.0 * (f_0 - f_1) + df_0 + df_1;
    [a0, a1, a2, a3]
}

/// Fits 2D bicubic spline coefficients of shape `(H - 1, W - 1, 16)` from a 2D image
/// (`picasso.spline.fit_spline_coefficients_2d`).
pub fn fit_spline_coefficients_2d(image: &Array2<f64>) -> Result<Array3<f64>> {
    let (h, w) = image.dim();
    if h < 4 || w < 4 {
        return Err("Image must be at least 4x4 for 2D bicubic spline fitting".into());
    }
    let nx = h - 1;
    let ny = w - 1;
    let mut coeffs = Array3::<f64>::zeros((nx, ny, 16));

    let get_px = |ix: isize, iy: isize| -> f64 {
        let cx = ix.clamp(0, (h - 1) as isize) as usize;
        let cy = iy.clamp(0, (w - 1) as isize) as usize;
        image[[cx, cy]]
    };

    for i in 0..nx {
        for j in 0..ny {
            let mut row_coeffs = [[0.0f64; 4]; 4];
            for (r_idx, di) in (-1isize..=2).enumerate() {
                let c = catmull_rom_1d_coeffs(
                    get_px(i as isize + di, j as isize - 1),
                    get_px(i as isize + di, j as isize),
                    get_px(i as isize + di, j as isize + 1),
                    get_px(i as isize + di, j as isize + 2),
                );
                row_coeffs[r_idx] = c;
            }
            for q in 0..4 {
                let col_c = catmull_rom_1d_coeffs(
                    row_coeffs[0][q],
                    row_coeffs[1][q],
                    row_coeffs[2][q],
                    row_coeffs[3][q],
                );
                for p in 0..4 {
                    coeffs[[i, j, p * 4 + q]] = col_c[p];
                }
            }
        }
    }

    Ok(coeffs)
}

/// Evaluates a 2D bicubic spline at continuous coordinates `(x, y)` (`picasso.spline.eval_spline_2d`).
pub fn eval_spline_2d(coeffs: &Array3<f64>, x: f64, y: f64) -> f64 {
    let shape = coeffs.shape();
    let nx = shape[0];
    let ny = shape[1];
    let ix = (x.floor() as isize).clamp(0, (nx - 1) as isize) as usize;
    let iy = (y.floor() as isize).clamp(0, (ny - 1) as isize) as usize;
    let dx = (x - (ix as f64)).clamp(0.0, 1.0);
    let dy = (y - (iy as f64)).clamp(0.0, 1.0);

    let xp = [1.0, dx, dx * dx, dx * dx * dx];
    let yp = [1.0, dy, dy * dy, dy * dy * dy];

    let mut sum = 0.0;
    for p in 0..4 {
        for q in 0..4 {
            sum += coeffs[[ix, iy, p * 4 + q]] * xp[p] * yp[q];
        }
    }
    sum
}

/// Fits 3D tricubic spline coefficients of shape `(nx - 1, ny - 1, nz - 1, 64)` from a 3D PSF volume
/// (`picasso.spline.fit_spline_coefficients_3d`).
pub fn fit_spline_coefficients_3d(volume: &Array3<f64>) -> Result<ArrayD<f64>> {
    let shape = volume.shape();
    let nx = shape[0];
    let ny = shape[1];
    let nz = shape[2];
    if nx < 4 || ny < 4 || nz < 4 {
        return Err("3D PSF volume must have dimensions >= 4 along all axes".into());
    }

    let sx = nx - 1;
    let sy = ny - 1;
    let sz = nz - 1;
    let mut coeffs = ArrayD::<f64>::zeros(IxDyn(&[sx, sy, sz, 64]));

    let get_vox = |ix: isize, iy: isize, iz: isize| -> f64 {
        let cx = ix.clamp(0, (nx - 1) as isize) as usize;
        let cy = iy.clamp(0, (ny - 1) as isize) as usize;
        let cz = iz.clamp(0, (nz - 1) as isize) as usize;
        volume[[cx, cy, cz]]
    };

    for i in 0..sx {
        for j in 0..sy {
            for k in 0..sz {
                let mut cz = [[[0.0f64; 4]; 4]; 4];
                for (xi, di) in (-1isize..=2).enumerate() {
                    for (yj, dj) in (-1isize..=2).enumerate() {
                        cz[xi][yj] = catmull_rom_1d_coeffs(
                            get_vox(i as isize + di, j as isize + dj, k as isize - 1),
                            get_vox(i as isize + di, j as isize + dj, k as isize),
                            get_vox(i as isize + di, j as isize + dj, k as isize + 1),
                            get_vox(i as isize + di, j as isize + dj, k as isize + 2),
                        );
                    }
                }

                let mut cyz = [[[0.0f64; 4]; 4]; 4];
                for xi in 0..4 {
                    for r in 0..4 {
                        let c_y = catmull_rom_1d_coeffs(
                            cz[xi][0][r],
                            cz[xi][1][r],
                            cz[xi][2][r],
                            cz[xi][3][r],
                        );
                        for q in 0..4 {
                            cyz[xi][q][r] = c_y[q];
                        }
                    }
                }

                for q in 0..4 {
                    for r in 0..4 {
                        let c_x = catmull_rom_1d_coeffs(
                            cyz[0][q][r],
                            cyz[1][q][r],
                            cyz[2][q][r],
                            cyz[3][q][r],
                        );
                        for p in 0..4 {
                            let idx = p * 16 + q * 4 + r;
                            coeffs[[i, j, k, idx]] = c_x[p];
                        }
                    }
                }
            }
        }
    }

    Ok(coeffs)
}

/// Evaluates a 3D tricubic spline at continuous voxel coordinates `(x, y, z)` (`picasso.spline.eval_spline_3d`).
pub fn eval_spline_3d(coeffs: &ArrayD<f64>, x: f64, y: f64, z: f64) -> f64 {
    let shape = coeffs.shape();
    let nx = shape[0];
    let ny = shape[1];
    let nz = shape[2];

    let ix = (x.floor() as isize).clamp(0, (nx - 1) as isize) as usize;
    let iy = (y.floor() as isize).clamp(0, (ny - 1) as isize) as usize;
    let iz = (z.floor() as isize).clamp(0, (nz - 1) as isize) as usize;

    let dx = (x - (ix as f64)).clamp(0.0, 1.0);
    let dy = (y - (iy as f64)).clamp(0.0, 1.0);
    let dz = (z - (iz as f64)).clamp(0.0, 1.0);

    let xp = [1.0, dx, dx * dx, dx * dx * dx];
    let yp = [1.0, dy, dy * dy, dy * dy * dy];
    let zp = [1.0, dz, dz * dz, dz * dz * dz];

    let mut sum = 0.0;
    let mut idx = 0;
    for p in 0..4 {
        for q in 0..4 {
            for r in 0..4 {
                sum += coeffs[[ix, iy, iz, idx]] * xp[p] * yp[q] * zp[r];
                idx += 1;
            }
        }
    }
    sum
}

/// Aligns and averages multiple 3D bead z-stacks `(n_beads, nx, ny, nz)` into a normalized 3D PSF volume
/// (`picasso.spline`).
pub fn average_bead_stacks(stacks: &[Array3<f64>]) -> Result<Array3<f64>> {
    if stacks.is_empty() {
        return Err("No bead stacks provided".into());
    }
    let shape = stacks[0].shape();
    let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
    let mut avg = Array3::<f64>::zeros((nx, ny, nz));

    for st in stacks {
        if st.shape() != shape {
            return Err("All bead stacks must have identical 3D shape".into());
        }
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for &v in st.iter() {
            if v < min_v {
                min_v = v;
            }
            if v > max_v {
                max_v = v;
            }
        }
        let denom = (max_v - min_v).max(1e-12);
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    avg[[i, j, k]] += (st[[i, j, k]] - min_v) / denom;
                }
            }
        }
    }

    let n = stacks.len() as f64;
    for v in avg.iter_mut() {
        *v /= n;
    }
    Ok(avg)
}
