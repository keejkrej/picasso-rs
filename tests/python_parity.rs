//! Direct numerical parity tests comparing Python `picasso` (`jungmannlab/picasso`)
//! against Rust `picasso` (`picasso-rs`) on the exact same inputs.

use approx::assert_abs_diff_eq;
use marray::np::{Array2, Array3, s};
use picasso::avgroi;
use picasso::fitting::gaussfit::fit_spots_gauss;
use picasso::fitting::gausslq;
use picasso::fitting::gaussmle::{self, MleMethod};
use picasso::fitting::seeds::initial_parameters_gauss;
use picasso::localize::{FitConfig, IdentifyOptions, get_spots, identify_in_image, localize};
use picasso::types::{CameraInfo, FitEstimator, GaussModel, Identification, SpotFitMethod};
use serde::Deserialize;
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug, Deserialize)]
struct PythonPicassoReference {
    ident_y: Vec<i32>,
    ident_x: Vec<i32>,
    ident_ng: Vec<f64>,
    init_sph: Vec<Vec<f64>>,
    init_ell: Vec<Vec<f64>>,
    init_rot: Vec<Vec<f64>>,
    gaussfit_sph_lsq_thetas: Vec<Vec<f64>>,
    gaussfit_sph_lsq_chi2: Vec<f64>,
    gaussfit_sph_mle_thetas: Vec<Vec<f64>>,
    gaussfit_sph_mle_chi2: Vec<f64>,
    gaussfit_ell_lsq_thetas: Vec<Vec<f64>>,
    gaussfit_ell_lsq_chi2: Vec<f64>,
    gaussfit_rot_lsq_thetas: Vec<Vec<f64>>,
    gaussmle_sigmaxy_thetas: Vec<Vec<f64>>,
    gaussmle_sigmaxy_crlbs: Vec<Vec<f64>>,
    avgroi_thetas: Vec<Vec<f64>>,
}

fn find_python() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let venv_py = manifest_dir.join(".venv").join("bin").join("python");
    if venv_py.is_file() {
        return venv_py;
    }
    PathBuf::from("python3")
}

#[test]
fn test_python_picasso_vs_rust_picasso_parity() {
    let py = find_python();
    let script = r#"
import json
import numpy as np
import picasso.localize as localize
import picasso.fitting.gaussfit as gaussfit
import picasso.fitting.seeds as seeds
import picasso.gaussmle as gaussmle
import picasso.avgroi as avgroi

h, w = 32, 32
img = np.full((h, w), 100.0, dtype=np.float32)
spots_truth = [
    (15.3, 16.7, 2500.0, 1.2, 1.35),
    (7.8, 9.2, 1800.0, 1.15, 1.15),
    (23.1, 22.4, 3200.0, 1.3, 1.25),
]
for (tx, ty, tphot, tsx, tsy) in spots_truth:
    norm = 1.0 / (2.0 * np.pi * tsx * tsy)
    for r in range(h):
        for c in range(w):
            dx = c - tx
            dy = r - ty
            g = np.exp(-0.5 * ((dx/tsx)**2 + (dy/tsy)**2))
            img[r, c] += np.float32(tphot * norm * g)

y, x, ng = localize.identify_in_image(img, 500.0, 7)
frames = np.zeros_like(x)
spots = localize._cut_spots_numba(img[np.newaxis, :, :], frames, x, y, 7)

init_sph = seeds.initial_parameters_gauss(spots, 7, spherical=True)
init_ell = seeds.initial_parameters_gauss(spots, 7, spherical=False, rotated=False)
init_rot = seeds.initial_parameters_gauss(spots, 7, rotated=True)

p_sph_lsq, c_sph_lsq, _, _ = gaussfit.fit_spots(gaussfit.SPHERICAL, spots.copy(), init_sph, mle=False)
p_sph_mle, c_sph_mle, _, _ = gaussfit.fit_spots(gaussfit.SPHERICAL, spots.copy(), init_sph, mle=True)
p_ell_lsq, c_ell_lsq, _, _ = gaussfit.fit_spots(gaussfit.ELLIPTIC, spots.copy(), init_ell, mle=False)
p_rot_lsq, c_rot_lsq, _, _ = gaussfit.fit_spots(gaussfit.ROTATED, spots.copy(), init_rot, mle=False)

thetas_mle, crlbs_mle, _, _ = gaussmle.gaussmle(spots.copy(), 1e-4, 1000, method="sigmaxy")
thetas_avg = avgroi.fit_spots(spots.copy())

out = {
    "ident_y": y.tolist(),
    "ident_x": x.tolist(),
    "ident_ng": [float(v) for v in ng],
    "init_sph": init_sph.astype(float).tolist(),
    "init_ell": init_ell.astype(float).tolist(),
    "init_rot": init_rot.astype(float).tolist(),
    "gaussfit_sph_lsq_thetas": p_sph_lsq.astype(float).tolist(),
    "gaussfit_sph_lsq_chi2": c_sph_lsq.astype(float).tolist(),
    "gaussfit_sph_mle_thetas": p_sph_mle.astype(float).tolist(),
    "gaussfit_sph_mle_chi2": c_sph_mle.astype(float).tolist(),
    "gaussfit_ell_lsq_thetas": p_ell_lsq.astype(float).tolist(),
    "gaussfit_ell_lsq_chi2": c_ell_lsq.astype(float).tolist(),
    "gaussfit_rot_lsq_thetas": p_rot_lsq.astype(float).tolist(),
    "gaussmle_sigmaxy_thetas": thetas_mle.astype(float).tolist(),
    "gaussmle_sigmaxy_crlbs": crlbs_mle.astype(float).tolist(),
    "avgroi_thetas": thetas_avg.astype(float).tolist(),
}
print(json.dumps(out))
"#;

    let output = Command::new(&py)
        .arg("-c")
        .arg(script)
        .output()
        .expect("Failed to run Python picasso reference script");
    assert!(
        output.status.success(),
        "Python picasso script failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let py_ref: PythonPicassoReference =
        serde_json::from_slice(&output.stdout).expect("Failed to parse Python picasso JSON");

    // Build identical image in Rust (matching float32 accumulation as in Python)
    let (h, w) = (32usize, 32usize);
    let mut img_f32 = Array2::<f32>::from_elem((h, w), 100.0f32);
    let spots_truth = [
        (15.3_f64, 16.7_f64, 2500.0_f64, 1.2_f64, 1.35_f64),
        (7.8_f64, 9.2_f64, 1800.0_f64, 1.15_f64, 1.15_f64),
        (23.1_f64, 22.4_f64, 3200.0_f64, 1.3_f64, 1.25_f64),
    ];
    for &(tx, ty, tphot, tsx, tsy) in &spots_truth {
        let norm = 1.0 / (2.0 * std::f64::consts::PI * tsx * tsy);
        for r in 0..h {
            for c in 0..w {
                let dx = (c as f64) - tx;
                let dy = (r as f64) - ty;
                let g = (-0.5 * ((dx / tsx).powi(2) + (dy / tsy).powi(2))).exp();
                img_f32[[r, c]] += (tphot * norm * g) as f32;
            }
        }
    }
    let img = img_f32.mapv(|v| v as f64);

    // 1. Compare `identify_in_image`
    let (ry, rx, rng) = identify_in_image(&img, 500.0, 7);
    assert_eq!(ry, py_ref.ident_y, "identify y mismatch");
    assert_eq!(rx, py_ref.ident_x, "identify x mismatch");
    for (&r_ng, &p_ng) in rng.iter().zip(py_ref.ident_ng.iter()) {
        assert_abs_diff_eq!(r_ng as f64, p_ng, epsilon = 1e-2);
    }

    // 2. Cut spots & compare `initial_parameters_gauss`
    let mut movie = Array3::<f64>::zeros((1, h, w));
    movie.slice_mut(s![0, .., ..]).assign(&img);
    let idents: Vec<Identification> = rx
        .iter()
        .zip(ry.iter())
        .zip(rng.iter())
        .map(|((&x, &y), &ng)| Identification {
            frame: 0,
            x,
            y,
            net_gradient: ng,
        })
        .collect();
    let raw_cam = CameraInfo::new(0.0, 1.0, 1.0);
    let spots: Array3<f64> = get_spots(&movie, &idents, 7, &raw_cam).unwrap();

    let r_init_sph = initial_parameters_gauss(&spots, GaussModel::Spherical).unwrap();
    let r_init_ell = initial_parameters_gauss(&spots, GaussModel::Elliptic).unwrap();
    let r_init_rot = initial_parameters_gauss(&spots, GaussModel::Rotated).unwrap();

    for i in 0..spots.shape()[0] {
        for p in 0..5 {
            assert_abs_diff_eq!(r_init_sph[[i, p]], py_ref.init_sph[i][p], epsilon = 1e-4);
        }
        for p in 0..6 {
            assert_abs_diff_eq!(r_init_ell[[i, p]], py_ref.init_ell[i][p], epsilon = 1e-4);
        }
        for p in 0..7 {
            assert_abs_diff_eq!(r_init_rot[[i, p]], py_ref.init_rot[i][p], epsilon = 1e-4);
        }
    }

    // 3. Compare `gaussfit` (Spherical LSQ, Spherical MLE, Elliptic LSQ, Rotated LSQ)
    let r_sph_lsq = fit_spots_gauss(
        &spots,
        GaussModel::Spherical,
        FitEstimator::LeastSquares,
        None,
        None,
        None,
        false,
    )
    .unwrap();
    let r_sph_mle = fit_spots_gauss(
        &spots,
        GaussModel::Spherical,
        FitEstimator::Mle,
        None,
        None,
        None,
        false,
    )
    .unwrap();
    let r_ell_lsq = fit_spots_gauss(
        &spots,
        GaussModel::Elliptic,
        FitEstimator::LeastSquares,
        None,
        None,
        None,
        false,
    )
    .unwrap();
    let r_rot_lsq = fit_spots_gauss(
        &spots,
        GaussModel::Rotated,
        FitEstimator::LeastSquares,
        None,
        None,
        None,
        false,
    )
    .unwrap();

    for i in 0..spots.shape()[0] {
        for p in 0..5 {
            let tol_lsq = (py_ref.gaussfit_sph_lsq_thetas[i][p].abs() * 1e-4).max(1e-3);
            assert_abs_diff_eq!(
                r_sph_lsq.parameters[[i, p]],
                py_ref.gaussfit_sph_lsq_thetas[i][p],
                epsilon = tol_lsq
            );
            let tol_mle = (py_ref.gaussfit_sph_mle_thetas[i][p].abs() * 1e-4).max(1e-3);
            assert_abs_diff_eq!(
                r_sph_mle.parameters[[i, p]],
                py_ref.gaussfit_sph_mle_thetas[i][p],
                epsilon = tol_mle
            );
        }
        assert_abs_diff_eq!(
            r_sph_lsq.chi_squares[i],
            py_ref.gaussfit_sph_lsq_chi2[i],
            epsilon = 1e-2
        );
        assert_abs_diff_eq!(
            r_sph_mle.chi_squares[i],
            py_ref.gaussfit_sph_mle_chi2[i],
            epsilon = 1e-3
        );

        for p in 0..6 {
            let tol_ell = (py_ref.gaussfit_ell_lsq_thetas[i][p].abs() * 1e-4).max(1e-3);
            assert_abs_diff_eq!(
                r_ell_lsq.parameters[[i, p]],
                py_ref.gaussfit_ell_lsq_thetas[i][p],
                epsilon = tol_ell
            );
        }
        assert_abs_diff_eq!(
            r_ell_lsq.chi_squares[i],
            py_ref.gaussfit_ell_lsq_chi2[i],
            epsilon = 1e-4
        );

        for p in 0..7 {
            let tol_rot = (py_ref.gaussfit_rot_lsq_thetas[i][p].abs() * 1e-4).max(1e-3);
            assert_abs_diff_eq!(
                r_rot_lsq.parameters[[i, p]],
                py_ref.gaussfit_rot_lsq_thetas[i][p],
                epsilon = tol_rot
            );
        }
    }

    // 4. Compare `gaussmle` (integrated Gaussian MLE with CRLB)
    let (r_mle_thetas, r_mle_crlbs, _, _) =
        gaussmle::gaussmle(&spots, 1e-4, 1000, MleMethod::SigmaYx, false).unwrap();
    for i in 0..spots.shape()[0] {
        for p in 0..6 {
            let rel_tol = (py_ref.gaussmle_sigmaxy_thetas[i][p].abs() * 1e-3).max(1e-3);
            assert_abs_diff_eq!(
                r_mle_thetas[[i, p]],
                py_ref.gaussmle_sigmaxy_thetas[i][p],
                epsilon = rel_tol
            );
            let crlb_tol = (py_ref.gaussmle_sigmaxy_crlbs[i][p].abs() * 1e-2).max(1e-4);
            assert_abs_diff_eq!(
                r_mle_crlbs[[i, p]],
                py_ref.gaussmle_sigmaxy_crlbs[i][p],
                epsilon = crlb_tol
            );
        }
    }

    // 5. Compare `gausslq` sub-pixel center recovery
    let r_lq = gausslq::fit_spots(&spots, false).unwrap();
    for i in 0..spots.shape()[0] {
        assert_abs_diff_eq!(
            r_lq[[i, 0]],
            py_ref.gaussfit_ell_lsq_thetas[i][1],
            epsilon = 0.02
        );
        assert_abs_diff_eq!(
            r_lq[[i, 1]],
            py_ref.gaussfit_ell_lsq_thetas[i][2],
            epsilon = 0.02
        );
    }

    // 6. Compare `avgroi`
    let r_avg = avgroi::fit_spots(&spots);
    for i in 0..spots.shape()[0] {
        for p in 0..6 {
            let tol = (py_ref.avgroi_thetas[i][p].abs() * 1e-4).max(1e-4);
            assert_abs_diff_eq!(r_avg[[i, p]], py_ref.avgroi_thetas[i][p], epsilon = tol);
        }
    }

    // 7. Verify end-to-end `localize`
    let cam = CameraInfo::new(100.0, 1.0, 1.0);
    let id_opts = IdentifyOptions {
        min_gradient: 500.0,
        box_size: 7,
        ..Default::default()
    };
    let fit_cfg = FitConfig {
        method: SpotFitMethod::GaussLq,
        model: GaussModel::Elliptic,
        box_size: 7,
        ..Default::default()
    };
    let (_, locs) = localize(&movie, &cam, &id_opts, &fit_cfg).unwrap();
    assert_eq!(locs.len(), 3);
}
