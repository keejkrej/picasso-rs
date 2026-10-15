//! Comprehensive unit and integration test suite for `picasso-rs`.

use approx::assert_abs_diff_eq;
use marray::np::{Array2, Array3, s};
use picasso::aim::aim;
use picasso::average::average_particles;
use picasso::binding::{
    BindingDetectConfig, BindingFilterConfig, SpotTrackConfig, detect_spots_frame,
    detect_spots_stack, detect_spots_u16_slice, filter_spots, linear_sum_assignment, spot_counts,
    track_spots,
};
use picasso::clusterer::{cluster, dbscan, find_cluster_centers};
use picasso::design::generate_lattice_sites;
use picasso::fitting::gaussfit::{eval_gaussian_spot, fit_spots_gauss};
use picasso::fitting::gausslq;
use picasso::fitting::gaussmle::{self, MleMethod};
use picasso::fitting::precision::{gauss_crlb_single, localization_precision_scalar};
use picasso::fitting::seeds::initial_parameters_gauss;
use picasso::fitting::splinefit::{fit_spots_spline, locs_from_fits_spline};
use picasso::frc::frc;
use picasso::g5m::{G5mConfig, g5m};
use picasso::io::{
    load_binding_spots_csv, load_info, load_locs_csv, load_tiff_stack, save_binding_spots_csv,
    save_info, save_locs_csv, save_tiff_stack_u16,
};
use picasso::localize::{FitConfig, IdentifyOptions, identify_in_image, localize};
use picasso::nanotron::roi_to_feature_vector;
use picasso::plot::{plot_drift, plot_frc, plot_lnp_binding, plot_nena, plot_rendered_image};
use picasso::postprocess::{
    LinkCombineMode, compute_local_density, groupprops, link, nena, pair_correlation, picked_locs,
    undrift,
};
use picasso::render::{RenderBlurMethod, render, render_gaussian3d, render_hist3d};
use picasso::scmos::{calibrate_scmos, validate_calibration};
use picasso::simulate::{SimulatedEmitter, SimulationConfig, simulate_movie};
use picasso::spinna::{compare_nnd_distributions, nearest_neighbor_distances};
use picasso::spline::{
    eval_spline_2d, eval_spline_3d, fit_spline_coefficients_2d, fit_spline_coefficients_3d,
};
use picasso::toroidal_unwrap::unwrap_z_periodic;
use picasso::types::{
    CameraInfo, FitEstimator, GaussModel, Identification, Localization, PickShape, SpotFitMethod,
};
use picasso::zfit::{calibrate_z, filter_tails, fit_z};

#[test]
fn test_identify_and_gaussfit_models_lsq_and_mle() {
    let size = 9usize;
    let true_params_ell = [500.0, 4.25, 3.80, 1.15, 1.35, 15.0];
    let spot = eval_gaussian_spot(size, &true_params_ell, GaussModel::Elliptic);

    let mut spots = Array3::<f64>::zeros((1, size, size));
    spots.slice_mut(s![0, .., ..]).assign(&spot);

    // 1. Test seeds
    let seeds = initial_parameters_gauss(&spots, GaussModel::Elliptic).unwrap();
    assert_eq!(seeds.shape(), &[1, 6]);
    assert!(seeds[[0, 0]] > 400.0);

    // 2. Test Elliptic LSQ
    let res_lsq = fit_spots_gauss(
        &spots,
        GaussModel::Elliptic,
        FitEstimator::LeastSquares,
        None,
        None,
        None,
        false,
    )
    .unwrap();
    assert_eq!(res_lsq.states[0], 0);
    assert_abs_diff_eq!(res_lsq.parameters[[0, 0]], 500.0, epsilon = 1e-2);
    assert_abs_diff_eq!(res_lsq.parameters[[0, 1]], 4.25, epsilon = 1e-3);
    assert_abs_diff_eq!(res_lsq.parameters[[0, 2]], 3.80, epsilon = 1e-3);
    assert_abs_diff_eq!(res_lsq.parameters[[0, 3]].abs(), 1.15, epsilon = 1e-3);
    assert_abs_diff_eq!(res_lsq.parameters[[0, 4]].abs(), 1.35, epsilon = 1e-3);
    assert_abs_diff_eq!(res_lsq.parameters[[0, 5]], 15.0, epsilon = 1e-2);

    // 3. Test Elliptic MLE
    let res_mle = fit_spots_gauss(
        &spots,
        GaussModel::Elliptic,
        FitEstimator::Mle,
        None,
        None,
        None,
        false,
    )
    .unwrap();
    assert_abs_diff_eq!(res_mle.parameters[[0, 1]], 4.25, epsilon = 1e-3);
    assert_abs_diff_eq!(res_mle.parameters[[0, 2]], 3.80, epsilon = 1e-3);

    // 4. Test Spherical model
    let true_params_sph = [400.0, 3.9, 4.1, 1.2, 10.0];
    let spot_sph = eval_gaussian_spot(size, &true_params_sph, GaussModel::Spherical);
    let mut spots_sph = Array3::<f64>::zeros((1, size, size));
    spots_sph.slice_mut(s![0, .., ..]).assign(&spot_sph);
    let res_sph = fit_spots_gauss(
        &spots_sph,
        GaussModel::Spherical,
        FitEstimator::LeastSquares,
        None,
        None,
        None,
        false,
    )
    .unwrap();
    assert_abs_diff_eq!(res_sph.parameters[[0, 1]], 3.9, epsilon = 1e-3);
    assert_abs_diff_eq!(res_sph.parameters[[0, 2]], 4.1, epsilon = 1e-3);
    assert_abs_diff_eq!(res_sph.parameters[[0, 3]].abs(), 1.2, epsilon = 1e-3);

    // 5. Test Rotated model
    let true_params_rot = [600.0, 4.1, 3.9, 1.6, 0.95, 12.0, 0.45];
    let spot_rot = eval_gaussian_spot(size, &true_params_rot, GaussModel::Rotated);
    let mut spots_rot = Array3::<f64>::zeros((1, size, size));
    spots_rot.slice_mut(s![0, .., ..]).assign(&spot_rot);
    let res_rot = fit_spots_gauss(
        &spots_rot,
        GaussModel::Rotated,
        FitEstimator::LeastSquares,
        None,
        None,
        None,
        false,
    )
    .unwrap();
    assert_abs_diff_eq!(res_rot.parameters[[0, 1]], 4.1, epsilon = 1e-2);
    assert_abs_diff_eq!(res_rot.parameters[[0, 2]], 3.9, epsilon = 1e-2);
}

#[test]
fn test_gaussmle_integrated_and_gausslq() {
    let emitters = vec![SimulatedEmitter {
        frame: 0,
        x: 3.3,
        y: 3.7,
        photons: 2500.0,
        sx: 1.1,
        sy: 1.2,
    }];
    let frame = picasso::simulate::render_noiseless_frame(7, 7, 10.0, &emitters);
    let mut spots = Array3::<f64>::zeros((1, 7, 7));
    spots.slice_mut(s![0, .., ..]).assign(&frame);

    let (thetas, crlbs, _, _) =
        gaussmle::gaussmle(&spots, 1e-4, 500, MleMethod::SigmaYx, false).unwrap();
    assert_abs_diff_eq!(thetas[[0, 0]], 3.3, epsilon = 0.02);
    assert_abs_diff_eq!(thetas[[0, 1]], 3.7, epsilon = 0.02);
    assert_abs_diff_eq!(thetas[[0, 2]], 2500.0, epsilon = 50.0);
    assert!(crlbs[[0, 0]] > 0.0 && crlbs[[0, 0]] < 0.01);

    let lq_theta = gausslq::fit_spot(&frame).unwrap();
    assert_abs_diff_eq!(lq_theta[0], 3.3, epsilon = 0.05);
    assert_abs_diff_eq!(lq_theta[1], 3.7, epsilon = 0.05);
}

#[test]
fn test_precision_mortensen_and_crlb() {
    let lp_no_em = localization_precision_scalar(1000.0, 1.2, 1.2, 10.0, false, 0.0);
    let lp_em = localization_precision_scalar(1000.0, 1.2, 1.2, 10.0, true, 0.0);
    assert_abs_diff_eq!(lp_em / lp_no_em, std::f64::consts::SQRT_2, epsilon = 1e-6);

    let crlb = gauss_crlb_single(3.0, 3.0, 1000.0, 10.0, 1.2, 1.2, None, 7, false, None);
    assert!(crlb.crlb_x.sqrt() > 0.02 && crlb.crlb_x.sqrt() < 0.15);
    assert_abs_diff_eq!(crlb.crlb_x, crlb.crlb_y, epsilon = 1e-9);
}

#[test]
fn test_full_localize_pipeline_on_simulated_movie() {
    let emitters = vec![
        SimulatedEmitter {
            frame: 0,
            x: 16.25,
            y: 20.75,
            photons: 3000.0,
            sx: 1.2,
            sy: 1.2,
        },
        SimulatedEmitter {
            frame: 0,
            x: 42.60,
            y: 35.10,
            photons: 2800.0,
            sx: 1.15,
            sy: 1.25,
        },
        SimulatedEmitter {
            frame: 1,
            x: 16.30,
            y: 20.70,
            photons: 3100.0,
            sx: 1.2,
            sy: 1.2,
        },
    ];
    let sim_cfg = SimulationConfig {
        height: 64,
        width: 64,
        n_frames: 2,
        bg_photons: 10.0,
        baseline_adu: 100.0,
        sensitivity: 1.0,
        gain: 1.0,
        readout_noise_adu: 0.0,
        shot_noise: false,
        random_seed: Some(123),
    };
    let movie = simulate_movie(&emitters, &sim_cfg).unwrap();

    // Test identify_in_image directly on frame 0
    let frame0 = movie.slice(s![0, .., ..]).to_owned();
    let (y_ids, x_ids, _) = identify_in_image(&frame0, 500.0, 7);
    assert_eq!(y_ids.len(), 2);
    assert_eq!(x_ids.len(), 2);

    let cam = CameraInfo::new(100.0, 1.0, 1.0);
    let id_opts = IdentifyOptions {
        min_gradient: 500.0,
        box_size: 7,
        ..Default::default()
    };
    let fit_cfg = FitConfig {
        method: SpotFitMethod::GaussMle,
        model: GaussModel::Elliptic,
        box_size: 7,
        ..Default::default()
    };

    let (ids, locs) = localize(&movie, &cam, &id_opts, &fit_cfg).unwrap();
    assert_eq!(ids.len(), 3);
    assert_eq!(locs.len(), 3);

    assert_abs_diff_eq!(locs[0].x, 16.25, epsilon = 0.08);
    assert_abs_diff_eq!(locs[0].y, 20.75, epsilon = 0.08);
    assert_abs_diff_eq!(locs[1].x, 42.60, epsilon = 0.08);
    assert_abs_diff_eq!(locs[1].y, 35.10, epsilon = 0.08);
}

#[test]
fn test_3d_zfit_and_cubic_splines() {
    // 1. Test 3D astigmatism calibration & fitting
    let n_frames = 21usize;
    let d_nm = 40.0;
    let mut cal_locs = Vec::new();
    for f in 0..n_frames {
        let z = ((f as f64) - 10.0) * d_nm;
        let sx = 1.1 + 0.0012 * z + 2.5e-6 * z * z;
        let sy = 1.1 - 0.0012 * z + 2.5e-6 * z * z;
        cal_locs.push(Localization {
            frame: f as u32,
            x: 20.0,
            y: 20.0,
            photons: 2000.0,
            sx,
            sy,
            bg: 10.0,
            lpx: 0.05,
            lpy: 0.05,
            ..Default::default()
        });
    }
    let cal = calibrate_z(&cal_locs, n_frames, d_nm, 1.0).unwrap();
    let mut test_locs = vec![cal_locs[15].clone()]; // z = +200 nm
    fit_z(&mut test_locs, &cal, 1.0, 0, false);
    let fitted_z = test_locs[0].z.unwrap();
    assert_abs_diff_eq!(fitted_z, 200.0, epsilon = 15.0);
    let filtered = filter_tails(&test_locs, &cal, 1.0, 1);
    assert_eq!(filtered.len(), 1);

    // 2. Test 2D & 3D Catmull-Rom cubic splines
    let mut img2d = Array2::<f64>::zeros((9, 9));
    for iy in 0..9 {
        for ix in 0..9 {
            let dx = (ix as f64) - 4.0;
            let dy = (iy as f64) - 4.0;
            img2d[[iy, ix]] = (-0.5 * (dx * dx + dy * dy) / 1.5).exp();
        }
    }
    let c2d = fit_spline_coefficients_2d(&img2d).unwrap();
    let v_center = eval_spline_2d(&c2d, 4.0, 4.0);
    assert_abs_diff_eq!(v_center, 1.0, epsilon = 1e-6);

    let mut vol3d = Array3::<f64>::zeros((7, 7, 7));
    for ix in 0..7 {
        for iy in 0..7 {
            for iz in 0..7 {
                let dx = (ix as f64) - 3.0;
                let dy = (iy as f64) - 3.0;
                let dz = (iz as f64) - 3.0;
                vol3d[[ix, iy, iz]] = (-0.5 * (dx * dx + dy * dy + dz * dz) / 1.5).exp();
            }
        }
    }
    let c3d = fit_spline_coefficients_3d(&vol3d).unwrap();
    let v3_center = eval_spline_3d(&c3d, 3.0, 3.0, 3.0);
    assert_abs_diff_eq!(v3_center, 1.0, epsilon = 1e-6);

    // Fit a spot with 3D spline fitter
    let mut spot_stack = Array3::<f64>::zeros((1, 7, 7));
    for iy in 0..7 {
        for ix in 0..7 {
            spot_stack[[0, iy, ix]] = 500.0 * vol3d[[ix, iy, 3]] + 10.0;
        }
    }
    let sp_res =
        fit_spots_spline(&spot_stack, &c3d, FitEstimator::LeastSquares, None, None, false).unwrap();
    let ids = vec![Identification {
        frame: 0,
        x: 10,
        y: 10,
        net_gradient: 1000.0,
    }];
    let sp_locs = locs_from_fits_spline(&ids, &sp_res, &c3d, 7, 50.0, false);
    assert_eq!(sp_locs.len(), 1);
    assert!(sp_locs[0].photons > 100.0);
}

#[test]
fn test_scmos_calibration() {
    let mut dark = Array3::<f64>::zeros((4, 4, 4));
    let mut illum1 = Array3::<f64>::zeros((4, 4, 4));
    let mut illum2 = Array3::<f64>::zeros((4, 4, 4));

    let deltas = [-1.0, 1.0, -1.0, 1.0];
    for t in 0..4 {
        for iy in 0..4 {
            for ix in 0..4 {
                dark[[t, iy, ix]] = 100.0 + deltas[t]; // mean 100, var 1
                illum1[[t, iy, ix]] = 200.0 + deltas[t] * 14.177446878757825; // mean 200 (+100 ADU), var 201 (+200 ADU^2 => gain = 2.0)
                illum2[[t, iy, ix]] = 300.0 + deltas[t] * 20.024984394500787; // mean 300 (+200 ADU), var 401 (+400 ADU^2 => gain = 2.0)
            }
        }
    }

    let mut cal = calibrate_scmos(&dark, &[illum1, illum2]).unwrap();
    validate_calibration(&mut cal);
    assert_abs_diff_eq!(cal.offset[[0, 0]], 100.0, epsilon = 1e-6);
    assert_abs_diff_eq!(cal.variance[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(cal.gain[[0, 0]], 2.0, epsilon = 1e-4);
    assert_abs_diff_eq!(cal.sensitivity[[0, 0]], 0.5, epsilon = 1e-4);
}

#[test]
fn test_postprocessing_clustering_g5m_render_and_frc() {
    // Generate two distinct clusters of localizations across frames
    // (leave z = None so 2D local density and 2D clustering operate in xy plane)
    let mut locs = Vec::new();
    for f in 0..40u32 {
        let jx = ((f as f64) * 0.7).sin() * 0.08;
        let jy = ((f as f64) * 1.1).cos() * 0.08;
        locs.push(Localization {
            frame: f,
            x: 15.0 + jx,
            y: 15.0 + jy,
            photons: 1500.0,
            sx: 1.1,
            sy: 1.1,
            bg: 10.0,
            lpx: 0.06,
            lpy: 0.06,
            ellipticity: 0.0,
            net_gradient: 800.0,
            ..Default::default()
        });
        locs.push(Localization {
            frame: f,
            x: 35.0 - jx,
            y: 35.0 - jy,
            photons: 1800.0,
            sx: 1.1,
            sy: 1.1,
            bg: 10.0,
            lpx: 0.06,
            lpy: 0.06,
            ellipticity: 0.0,
            net_gradient: 900.0,
            ..Default::default()
        });
    }

    // 1. Local density (2D)
    compute_local_density(&mut locs, 1.0).unwrap();
    assert_eq!(locs[0].density, Some(40));

    // 2. Picking (Circles)
    let picks = PickShape::Circles {
        centers: vec![(15.0, 15.0), (35.0, 35.0)],
        diameter: 3.0,
    };
    let grouped = picked_locs(&locs, &picks, true);
    assert_eq!(grouped[0].len(), 80);

    // 3. Trajectory linking & dark times
    let linked = link(&grouped[0], 0.5, 1, LinkCombineMode::Average, false);
    assert_eq!(linked.len(), 2);
    assert_eq!(linked[0].n, Some(40));
    assert_eq!(linked[0].len, Some(40));

    // 4. NeNA precision
    let nena_res = nena(&locs).unwrap();
    assert!(nena_res.nena_px > 0.01 && nena_res.nena_px < 0.5);

    // 5. Pair correlation
    let (bins, pcorr) = pair_correlation(&locs, 0.5, 5.0).unwrap();
    assert_eq!(bins.len(), pcorr.len());

    // 6. Undrift (RCC) & AIM
    let (undrifted_rcc, dx_rcc, dy_rcc) = undrift(&locs, 10, Some(5), 2.0).unwrap();
    assert_eq!(undrifted_rcc.len(), locs.len());
    assert_eq!(dx_rcc.len(), 40);
    assert_eq!(dy_rcc.len(), 40);

    let aim_res = aim(&locs, 10, 0.2, 1.0).unwrap();
    assert_eq!(aim_res.locs.len(), locs.len());

    // 7. Clustering (SMLM density clusterer + DBSCAN + find_cluster_centers)
    let clustered = cluster(&locs, 1.0, 5, true, None, None).unwrap();
    assert_eq!(clustered.len(), 80);
    let centers = find_cluster_centers(&clustered, Some(100.0));
    assert_eq!(centers.len(), 2);
    assert!(centers[0].area >= 0.0);

    let db_clustered = dbscan(&locs, 1.0, 5, 5, None).unwrap();
    assert_eq!(db_clustered.len(), 80);

    let props = groupprops(&clustered);
    assert_eq!(props.len(), 2);

    // 8. G5M molecular mapping
    let g5m_locs = g5m(
        &clustered,
        &G5mConfig {
            min_locs: 5,
            max_k: 2,
            tol: 1e-3,
            max_iter: 30,
        },
    )
    .unwrap();
    assert!(!g5m_locs.is_empty());

    // 9. Particle averaging
    let avg_locs = average_particles(&clustered, 3.0, 2.0, 4, 1).unwrap();
    assert_eq!(avg_locs.len(), 80);

    // 10. 2D & 3D Rendering + FRC
    for method in [
        RenderBlurMethod::None,
        RenderBlurMethod::Smooth,
        RenderBlurMethod::Convolve,
        RenderBlurMethod::Gaussian,
        RenderBlurMethod::GaussianIso,
    ] {
        let (cnt, img) =
            render(&locs, Some(((10.0, 10.0), (40.0, 40.0))), 2.0, method, 0.05).unwrap();
        assert_eq!(cnt, 80);
        assert_eq!(img.dim(), (60, 60));
    }

    let mut locs_3d = locs.clone();
    for (idx, l) in locs_3d.iter_mut().enumerate() {
        l.z = Some(if idx % 2 == 0 { 50.0 } else { -50.0 });
        l.lpz = Some(15.0);
    }

    let (cnt3d, vol3d) = render_hist3d(
        &locs_3d,
        1.0,
        ((10.0, 10.0, -200.0), (40.0, 40.0, 200.0)),
        100.0,
    );
    assert_eq!(cnt3d, 80);
    assert_eq!(vol3d.shape(), &[30, 30, 4]);

    let (cnt3d_g, _) = render_gaussian3d(
        &locs_3d,
        1.0,
        ((10.0, 10.0, -200.0), (40.0, 40.0, 200.0)),
        0.05,
        100.0,
    );
    assert_eq!(cnt3d_g, 80);

    let frc_res = frc(&locs, 130.0, 2.0, 2).unwrap();
    assert!(frc_res.resolution_nm.is_finite());

    // 11. SPINNA, Design, Nanotron, and Toroidal Unwrap
    let sites = generate_lattice_sites(3, 4, 20.0, 20.0, true);
    assert_eq!(sites.len(), 12);
    let mut site_coords = Array2::<f64>::zeros((12, 2));
    for (i, &(x, y)) in sites.iter().enumerate() {
        site_coords[[i, 0]] = x;
        site_coords[[i, 1]] = y;
    }
    let nnds = nearest_neighbor_distances(&site_coords, 2).unwrap();
    assert_eq!(nnds.len(), 2);
    let (ks_stat, ks_p) = compare_nnd_distributions(&nnds[0], &nnds[0]).unwrap();
    assert_abs_diff_eq!(ks_stat, 0.0, epsilon = 1e-9);
    assert!(ks_p >= 0.99);

    let feat = roi_to_feature_vector(&clustered[0..40], 2.0, 2.0).unwrap();
    assert_eq!(feat.len(), 64);

    let mut z_locs = locs_3d.clone();
    z_locs[0].z = Some(350.0);
    unwrap_z_periodic(&mut z_locs, 300.0);
    assert_abs_diff_eq!(z_locs[0].z.unwrap(), 50.0, epsilon = 1e-6);
}

#[test]
fn test_lisca_binding_assay_pipeline_and_mplot_io() {
    // Simulate a 3-frame LNP binding time-lapse stack where:
    // - Frame 0 has 2 spots: Spot A at (20, 20) and Spot B at (26.0, 20) (>= 5 px apart for 7x7 local maxima, within merge_distance = 8 px)
    // - Frame 1 has Spot A and Spot B coalesced into a brighter Spot A' at (20.2, 20.1), plus a new Spot C at (45, 45)
    // - Frame 2 has Spot A' at (20.3, 20.1) and Spot C at (45.1, 45.0)
    let emitters = vec![
        SimulatedEmitter {
            frame: 0,
            x: 20.0,
            y: 20.0,
            photons: 4000.0,
            sx: 1.2,
            sy: 1.2,
        },
        SimulatedEmitter {
            frame: 0,
            x: 26.0,
            y: 20.0,
            photons: 3500.0,
            sx: 1.2,
            sy: 1.2,
        },
        SimulatedEmitter {
            frame: 1,
            x: 20.2,
            y: 20.1,
            photons: 12000.0,
            sx: 1.2,
            sy: 1.2,
        },
        SimulatedEmitter {
            frame: 1,
            x: 45.0,
            y: 45.0,
            photons: 5000.0,
            sx: 1.2,
            sy: 1.2,
        },
        SimulatedEmitter {
            frame: 2,
            x: 20.3,
            y: 20.1,
            photons: 11800.0,
            sx: 1.2,
            sy: 1.2,
        },
        SimulatedEmitter {
            frame: 2,
            x: 45.1,
            y: 45.0,
            photons: 5100.0,
            sx: 1.2,
            sy: 1.2,
        },
    ];
    let sim_cfg = SimulationConfig {
        height: 64,
        width: 64,
        n_frames: 3,
        bg_photons: 15.0,
        baseline_adu: 100.0,
        sensitivity: 1.0,
        gain: 1.0,
        readout_noise_adu: 0.0,
        shot_noise: false,
        random_seed: Some(7),
    };
    let movie = simulate_movie(&emitters, &sim_cfg).unwrap();

    let det_cfg = BindingDetectConfig {
        min_gradient: 300.0,
        box_size: 7,
        fit_method: SpotFitMethod::GaussLq,
        gauss_model: GaussModel::Elliptic,
        gaussian_sigma: None,
        camera_info: CameraInfo::new(100.0, 1.0, 1.0),
        pos: "Pos0".to_string(),
        roi: "cell_0".to_string(),
        parallel: true,
    };

    // 1. Test single-frame detection & u16 slice detection
    let f0 = movie.slice(s![0, .., ..]).to_owned();
    let spots_f0 = detect_spots_frame(&f0, 0, &det_cfg).unwrap();
    assert_eq!(spots_f0.len(), 2);
    let u16_slice: Vec<u16> = f0.iter().map(|&v| v.round() as u16).collect();
    let spots_u16 = detect_spots_u16_slice(&u16_slice, 64, 64, 0, &det_cfg).unwrap();
    assert_eq!(spots_u16.len(), 2);

    // 2. Test full stack detection
    let all_spots = detect_spots_stack(&movie, &det_cfg).unwrap();
    assert_eq!(all_spots.len(), 6);
    for sp in &all_spots {
        assert!(sp.fwhm > 1.5 && sp.fwhm < 4.5);
        assert!(sp.intensity > 100.0);
        assert!(sp.probability > 0.5 && sp.probability <= 1.0);
        assert!(sp.snb > 1.0);
    }

    // 3. Test cell mask & parameter filtering
    let mut mask = Array2::<bool>::from_elem((64, 64), true);
    mask[[0, 0]] = false;
    let flt_cfg = BindingFilterConfig {
        min_intensity: Some(50.0),
        min_fwhm: Some(1.0),
        max_fwhm: Some(5.0),
        min_probability: Some(0.1),
        max_ellipticity: Some(0.5),
        min_snb: Some(1.0),
        ..Default::default()
    };
    let filtered_spots = filter_spots(&all_spots, &flt_cfg, Some(&mask));
    assert_eq!(filtered_spots.len(), 6);

    // 4. Test per-frame and cumulative spot counts
    let counts = spot_counts(&filtered_spots, Some(&[0, 1, 2]), 3.0);
    assert_eq!(counts.len(), 3);
    assert_eq!(counts[0].spot_count, 2);
    assert_eq!(counts[0].cumulative_count, 2);
    assert_eq!(counts[1].spot_count, 2);
    assert_eq!(counts[1].cumulative_count, 3); // Spot C at (45, 45) is a new binding event!
    assert_eq!(counts[2].spot_count, 2);
    assert_eq!(counts[2].cumulative_count, 3);

    // 5. Test Hungarian linear_sum_assignment & multi-frame tracking + merge detection
    let mut cost = Array2::<f64>::zeros((2, 2));
    cost[[0, 0]] = 1.0;
    cost[[0, 1]] = 10.0;
    cost[[1, 0]] = 10.0;
    cost[[1, 1]] = 2.0;
    assert_eq!(linear_sum_assignment(&cost), vec![(0, 0), (1, 1)]);

    let trk_cfg = SpotTrackConfig {
        max_distance: 3.0,
        merge_distance: 8.0,
        min_sources: 2,
        min_intensity_gain: 100.0,
    };
    let (tracked, merges, summaries) = track_spots(&filtered_spots, &trk_cfg);
    assert_eq!(tracked.len(), 6);
    assert_eq!(merges.len(), 1);
    assert!(merges[0].strict);
    assert_eq!(merges[0].n_sources, 2);
    assert_eq!(summaries.len(), 3);

    // 6. Test I/O (TIFF stack, CSV tables, YAML info) & mplot figure saving
    let dir = tempfile::tempdir().unwrap();
    let tif_path = dir.path().join("test_movie.tif");
    save_tiff_stack_u16(&tif_path, &movie).unwrap();
    let (loaded_movie, meta) = load_tiff_stack(&tif_path).unwrap();
    assert_eq!(loaded_movie.shape(), &[3, 64, 64]);
    assert_eq!(meta.frames, Some(3));

    let spots_csv = dir.path().join("spots.csv");
    save_binding_spots_csv(&spots_csv, &filtered_spots).unwrap();
    let loaded_spots = load_binding_spots_csv(&spots_csv).unwrap();
    assert_eq!(loaded_spots.len(), 6);

    let locs_vec: Vec<Localization> = filtered_spots.iter().map(|s| s.to_localization()).collect();
    let locs_csv = dir.path().join("locs.csv");
    save_locs_csv(&locs_csv, &locs_vec).unwrap();
    let loaded_locs = load_locs_csv(&locs_csv).unwrap();
    assert_eq!(loaded_locs.len(), 6);

    let yaml_path = dir.path().join("info.yaml");
    save_info(&yaml_path, &[meta]).unwrap();
    let loaded_info = load_info(&yaml_path).unwrap();
    assert_eq!(loaded_info.len(), 1);

    // Test mplot figure generation
    let plot_lnp_path = dir.path().join("lnp_binding.png");
    plot_lnp_binding(&counts, &filtered_spots, &plot_lnp_path).unwrap();
    assert!(plot_lnp_path.exists());

    let plot_img_path = dir.path().join("rendered.png");
    plot_rendered_image(&f0, "Frame 0", &plot_img_path).unwrap();
    assert!(plot_img_path.exists());

    let dx = marray::np::Array1::from_vec(vec![0.0, 0.1, 0.2]);
    let dy = marray::np::Array1::from_vec(vec![0.0, -0.05, -0.1]);
    let plot_drift_path = dir.path().join("drift.png");
    plot_drift(&dx, &dy, &plot_drift_path).unwrap();
    assert!(plot_drift_path.exists());

    let dummy_nena = picasso::postprocess::NenaResult {
        nena_px: 0.08,
        best_values: [0.08, 0.8, 0.1, 0.08, 0.16, 0.01],
        distances: vec![0.02, 0.06, 0.10, 0.14, 0.18],
        probability_density: vec![2.0, 5.0, 4.0, 2.0, 1.0],
        fitted_curve: vec![2.1, 4.9, 4.1, 1.9, 1.0],
    };
    let plot_nena_path = dir.path().join("nena.png");
    plot_nena(&dummy_nena, 130.0, &plot_nena_path).unwrap();
    assert!(plot_nena_path.exists());

    let dummy_frc = picasso::frc::FrcResult {
        resolution_nm: 25.0,
        spatial_frequencies: vec![0.01, 0.02, 0.03, 0.04, 0.05],
        frc_curve: vec![0.9, 0.6, 0.3, 0.1, 0.05],
        frc_smooth: vec![0.88, 0.58, 0.31, 0.11, 0.04],
        threshold: 1.0 / 7.0,
    };
    let plot_frc_path = dir.path().join("frc.png");
    plot_frc(&dummy_frc, &plot_frc_path).unwrap();
    assert!(plot_frc_path.exists());
}
