//! 3D Interferometric / Astigmatic Phase Unwrapping (`picasso.toroidal_unwrap`).

use crate::types::Localization;

/// Unwraps periodic axial `z` coordinates across a modulo period `period_nm` (`picasso.toroidal_unwrap`).
pub fn unwrap_z_periodic(locs: &mut [Localization], period_nm: f64) {
    if period_nm <= 0.0 {
        return;
    }
    for l in locs {
        if let Some(z) = l.z {
            let wrapped = ((z + 0.5 * period_nm).rem_euclid(period_nm)) - 0.5 * period_nm;
            l.z = Some(wrapped);
        }
    }
}
