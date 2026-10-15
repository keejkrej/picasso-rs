//! sCMOS camera per-pixel offset, variance, and gain calibration.
//!
//! Ported from `picasso.scmos`.

use crate::types::{Baseline, CameraInfo, Result, Sensitivity};
use marray::np::{Array2, Array3};

/// Per-pixel sCMOS calibration maps (`picasso.scmos`).
#[derive(Debug, Clone)]
pub struct ScmosCalibration {
    /// Per-pixel dark offset (baseline) in ADU, shape `(H, W)`.
    pub offset: Array2<f64>,
    /// Per-pixel dark readout variance in ADU^2, shape `(H, W)`.
    pub variance: Array2<f64>,
    /// Per-pixel conversion gain in `ADU / e-`, shape `(H, W)`.
    pub gain: Array2<f64>,
    /// Per-pixel sensitivity (`1.0 / gain`, in `e- / ADU`), shape `(H, W)`.
    pub sensitivity: Array2<f64>,
}

impl ScmosCalibration {
    /// Converts this calibration into a [`CameraInfo`] ready for [`crate::localize::fit`].
    pub fn to_camera_info(&self) -> CameraInfo {
        CameraInfo {
            baseline: Baseline::Map(self.offset.clone()),
            sensitivity: Sensitivity::Map(self.sensitivity.clone()),
            gain: 1.0,
            qe: 1.0,
            variance_map: Some(self.variance.clone()),
        }
    }
}

/// Computes temporal mean and variance across axis 0 of a 3D frame stack `(n_frames, H, W)`.
pub fn stack_mean_and_var(stack: &Array3<f64>) -> Result<(Array2<f64>, Array2<f64>)> {
    let shape = stack.shape();
    let n_frames = shape[0];
    let h = shape[1];
    let w = shape[2];
    if n_frames == 0 {
        return Err("Cannot compute mean/variance of empty stack".into());
    }

    let mut mean = Array2::<f64>::zeros((h, w));
    let mut var = Array2::<f64>::zeros((h, w));
    let nf = n_frames as f64;

    for iy in 0..h {
        for ix in 0..w {
            let mut sum = 0.0;
            for t in 0..n_frames {
                sum += stack[[t, iy, ix]];
            }
            let mu = sum / nf;
            mean[[iy, ix]] = mu;

            if n_frames > 1 {
                let mut sq = 0.0;
                for t in 0..n_frames {
                    let d = stack[[t, iy, ix]] - mu;
                    sq += d * d;
                }
                var[[iy, ix]] = sq / nf;
            }
        }
    }

    Ok((mean, var))
}

/// Calibrates per-pixel sCMOS `offset`, `variance`, and `gain` from a dark stack and a sequence
/// of illuminated stacks at varying light levels (`picasso.scmos.calibrate_scmos`).
///
/// For each pixel `(y, x)`:
/// - `offset = mean(dark_stack)`
/// - `variance = var(dark_stack)`
/// - `gain = sum((V_k - variance) * (M_k - offset)) / sum((M_k - offset)^2)` (least-squares slope through origin)
pub fn calibrate_scmos(
    dark_stack: &Array3<f64>,
    illuminated_stacks: &[Array3<f64>],
) -> Result<ScmosCalibration> {
    if illuminated_stacks.is_empty() {
        return Err("At least one illuminated stack is required for sCMOS gain calibration".into());
    }
    let (offset, variance) = stack_mean_and_var(dark_stack)?;
    let (h, w) = offset.dim();

    let mut num = Array2::<f64>::zeros((h, w));
    let mut den = Array2::<f64>::zeros((h, w));

    for illum in illuminated_stacks {
        if illum.shape()[1] != h || illum.shape()[2] != w {
            return Err("Illuminated stack spatial dimensions do not match dark stack".into());
        }
        let (m_k, v_k) = stack_mean_and_var(illum)?;
        for iy in 0..h {
            for ix in 0..w {
                let dm = (m_k[[iy, ix]] - offset[[iy, ix]]).max(0.0);
                let dv = (v_k[[iy, ix]] - variance[[iy, ix]]).max(0.0);
                num[[iy, ix]] += dv * dm;
                den[[iy, ix]] += dm * dm;
            }
        }
    }

    let mut gain = Array2::<f64>::zeros((h, w));
    let mut sensitivity = Array2::<f64>::zeros((h, w));

    for iy in 0..h {
        for ix in 0..w {
            let g = if den[[iy, ix]] > 1e-12 {
                (num[[iy, ix]] / den[[iy, ix]]).max(1e-4)
            } else {
                1.0
            };
            gain[[iy, ix]] = g;
            sensitivity[[iy, ix]] = 1.0 / g;
        }
    }

    Ok(ScmosCalibration {
        offset,
        variance,
        gain,
        sensitivity,
    })
}

/// Validates and sanitizes an [`ScmosCalibration`], replacing any non-finite or non-positive
/// entries with spatial median values (`picasso.scmos`).
pub fn validate_calibration(cal: &mut ScmosCalibration) {
    let sanitize = |arr: &mut Array2<f64>, min_val: f64, fallback: f64| {
        let mut valid: Vec<f64> = arr
            .iter()
            .copied()
            .filter(|v| v.is_finite() && *v >= min_val)
            .collect();
        let replacement = if valid.is_empty() {
            fallback
        } else {
            valid.sort_by(|a, b| a.total_cmp(b));
            valid[valid.len() / 2]
        };
        for v in arr.iter_mut() {
            if !v.is_finite() || *v < min_val {
                *v = replacement;
            }
        }
    };

    sanitize(&mut cal.offset, 0.0, 100.0);
    sanitize(&mut cal.variance, 1e-6, 1.0);
    sanitize(&mut cal.gain, 1e-4, 1.0);
    let (h, w) = cal.gain.dim();
    for iy in 0..h {
        for ix in 0..w {
            cal.sensitivity[[iy, ix]] = 1.0 / cal.gain[[iy, ix]];
        }
    }
}
