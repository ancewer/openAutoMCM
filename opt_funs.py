import numpy as np
from scipy.optimize import minimize, curve_fit
from pyswarm import pso
from scipy.stats import norm
import matplotlib.pyplot as plt
from mc_utils import *
from bdl_utils import read_bdl_file, write_to_bdl, create_bdl_file
import warnings
import time  # For time statistics
import logging  # For logging functionality
import os
from base_funs import *
from scipy import ndimage
from math import atan, cos, sin, radians, degrees


@timeit
def calc_absolute_dose(gantry, tps, mc_config, engine):
    """
    Adjust the number of protons per MU.
    """
    logger.info("Starting calc_absolute_dose...")
    WorkDir = mc_config.work_dir
    bdl_file = mc_config.bdl_file
    IC_diameter = mc_config.IC_diameter_ad

    if tps.name.lower() == 'raystation':
        meas_data = tps.absolute_dose_data
        spot_spacing = meas_data['spot_spacing']
        field_size = meas_data['field_size']
        meas_energies = meas_data['meas_energies']
        meas_depths = meas_data['meas_depths']
        isoc_depth = meas_data['isoc_depth']
        Dose_meas = meas_data['Dose_meas']
    elif tps.name.lower() == 'eclipse':
        spot_spacing = tps.Spots_spacing[0]
        field_size = tps.Field_size
        meas_energies = tps.all_energies
        meas_depths = tps.Calibration_depth * np.ones(len(meas_energies))
        isoc_depth = tps.Isocenter_depth * np.ones(len(meas_energies))
        Dose_meas = [x['Dose_meas'] for x in tps.idd_data]

    # Calculate the number of spots
    if len(field_size) == 2:
        nbspots1 = round(field_size[0] / spot_spacing + 1)
        nbspots2 = round(field_size[1] / spot_spacing + 1)
        nb_spots = nbspots1 * nbspots2
    else:
        nbspots1 = round(field_size[0] / spot_spacing + 1)
        nb_spots = nbspots1 ** 2

    if engine.lower() == 'mcsquare':
        new_spacing = mc_config.simu_spacing_ad
        MC2Data = {}
        ct_info = process_ct_to_mhd(mc_config.ct_template_path, WorkDir, new_spacing)
        true_center, center, Z1, Y1, v1, v3, coord = calculate_grids_ad(ct_info, new_spacing, ct_info['Size'], nbspots1,
                                                                        nbspots2, spot_spacing)
        MC2Data['Z1'] = Z1
        MC2Data['Y1'] = Y1
        MC2Data['center'] = center
        MC2Data['true_center'] = true_center
        MC2Data['v1'] = v1
        MC2Data['v3'] = v3

    bdl_parameters = read_bdl_file(bdl_file)
    energybdl = bdl_parameters[:, 0]

    protons_per_mu = np.zeros(len(energybdl))
    for j in range(len(energybdl)):
        current_energy = energybdl[j]
        k = np.where(meas_energies == current_energy)[0]
        iso_depth = isoc_depth[k[0]]
        ms_depth = meas_depths[k[0]]
        logger.info(f'[{current_energy}] Start calculate absolute dose (protons/MU)')
        if engine.lower() == 'mcsquare':
            mc_config.generate_mc2_config(WorkDir, mc_config.simu_ad_protons, 'CT.mhd', 'plan_AD.txt',
                                          os.path.join(mc_config.program_path, 'Scanners/Water_only'), bdl_file)
            center[1] = (ct_info['Size'][1] - 1) * new_spacing[1] - iso_depth + new_spacing[1] / 2
            # if mc_config.simu_ad_speed == False:
            #     mc_config.create_ad_plan_file(WorkDir, False, nb_spots, center, current_energy, coord=coord)
            # else:
            #     mc_config.create_ad_plan_file(WorkDir, True, nb_spots, center, current_energy, coord=coord)
            mc_config.create_ad_plan_file(WorkDir, tps.one_spot, nb_spots, center, current_energy, coord=coord)
            mc_config.run_mc2_simulation(WorkDir, mc_config.program_path, tps)

            _, Dose_data = mc_config.import_mhd_data(os.path.join(WorkDir, 'Outputs', 'Dose.mhd'))

            # if mc_config.simu_ad_speed == True:
            #     Dose_data = rotate_beamlet(Dose_data, nb_spots, ms_depth, new_spacing, coord, gantry.vsad_xy[0], gantry.vsad_xy[1], ct_info['ImagePositionPatient'],
            #                    true_center, iso_depth)

            Dose_data = np.transpose(Dose_data, (2, 0, 1))
            Dose_data = Dose_data * 1.602176e-19 * 1000  # eV/g per proton to Gy/proton

            x = np.arange(new_spacing[1] / 2, meas_depths[k] + 1, new_spacing[1])
            diff = np.abs(x - meas_depths[k])
            min_diff = np.min(diff)
            ind = np.where(diff == min_diff)[0]
            d = np.mean(Dose_data[:, :, ind], axis=2)
            mask = np.sqrt((Y1 - true_center[0]) ** 2 + (Z1 - true_center[2]) ** 2) <= IC_diameter[0] / 2
            mask = np.fliplr(mask)
            d = np.flipud(np.fliplr(d))
            if tps.one_spot:
                d[mask == 0] = 0
                dose_value = np.trapezoid(np.trapezoid(d, v3, axis=1), v1)
                protons_per_mu[j] = tps.RBE * Dose_meas[k[0]] / dose_value
            else:
                protons_per_mu[j] = tps.RBE * Dose_meas[k[0]] / np.mean(d[mask])
        elif engine.lower() == 'topas':
            mc_config.generate_topas_ad_config(gantry, mc_config, IC_diameter, iso_depth, ms_depth, current_energy,
                                               spot_spacing, field_size)
            mc_config.run_topas_simulation(WorkDir, 'pbs_ad.txt')
            Dose_data = mc_config.import_topas_ad_csv(os.path.join(WorkDir, 'pbs_ad_result.csv'))
            Dose_data = Dose_data / mc_config.simu_ad_protons / tps.RBE
            protons_per_mu[j] = Dose_meas[k[0]] / Dose_data
        logger.info(f'[{current_energy}] Finish calculate absolute dose [protons/MU: {protons_per_mu[j]:e}]')
        write_to_bdl(bdl_file, current_energy, 4, protons_per_mu[j])
    logger.info("Completed calc_absolute_dose.")
    return protons_per_mu


@timeit
def calc_phase_space(gantry, tps, mc_config):
    """
    Adjust phase space parameters and write them to the BDL file.
    """
    logger.info("Starting calc_phase_space...")
    bdl_file = mc_config.bdl_file
    nozzle2iso = gantry.nozzle2iso

    bdl_parameters = read_bdl_file(bdl_file)
    bdl_energies = bdl_parameters[:, 0]

    data = tps.spot_profile_data
    depths = data['depths']
    all_depths = data['all_depths']
    all_energies = data['all_energies']
    all_curves = data['all_curves']
    data_blocks = data['data_blocks']
    if tps.name.lower() == 'raystation':
        xycurve = ['X', 'Y']
    else:
        xycurve = ['MeasuredSpotFluenceX', 'MeasuredSpotFluenceY']

    coeff = np.zeros((len(depths), 3))
    coeff_cs = np.zeros((len(bdl_energies), 3, 2))
    s_nozzle = np.zeros((len(bdl_energies), 2))
    corr_nozzle = np.zeros((len(bdl_energies), 2))

    for j in range(len(bdl_energies)):
        ind1 = np.where(all_energies == bdl_energies[j])[0]
        logger.info(f"[{bdl_energies[j]}] Calculate of phase space parameters")
        for i in range(2):
            ind2 = np.where(np.array(all_curves) == xycurve[i])[0]
            for k in range(len(depths)):
                ind3 = np.where(all_depths == depths[k])[0]
                ind = np.intersect1d(np.intersect1d(ind1, ind2), ind3)
                if len(ind) == 0:
                    warnings.warn(f'Warning: not found for profile for depth {depths[k]}, energy {bdl_energies[j]}...')
                    continue

                block_idx = ind[0]
                data = data_blocks[block_idx]
                if tps.name.lower() == 'raystation':
                    X = data[:, i]
                    Z = data[:, 2]
                else:
                    X = data[:, 0]
                    Z = data[:, 3]
                Z = Z / np.sum(Z)

                def gauss(xdata, x1, x2, x3):
                    return x3 * np.exp(-((xdata - x1) ** 2) / (2 * x2 ** 2))

                sigma_guess2 = np.sqrt(np.sum(Z * (X - np.sum(X * Z) / np.sum(Z)) ** 2) / np.sum(Z))
                sigma_guess3 = abs(X[np.where(Z >= max(Z) / 2)[0][-1]] - X[np.where(Z >= max(Z) / 2)[0][0]]) / 2.355
                p0 = [X[np.argmax(Z)], sigma_guess3, max(Z)]
                popt, _ = curve_fit(gauss, X, Z, p0=p0)
                coeff[k, :] = popt

            div = np.polyfit(depths, coeff[:, 1], 1)

            if div[0] > 0:
                X0 = [coeff[0, 1] if 0 not in depths else coeff[depths == 0, 1][0], -abs(div[0]), -0.6]
                lb = [0, 0, -0.99]
                ub = [20, 0.5, 0]
            else:
                X0 = [coeff[0, 1] if 0 not in depths else coeff[depths == 0, 1][0], abs(div[0]), 0.6]
                lb = [0, 0, 0]
                ub = [20, 0.5, 0.99]

            def cs_func(xdata, x1, x2, x3):
                return np.sqrt(x1 ** 2 - 2 * x3 * x1 * x2 * xdata + x2 ** 2 * xdata ** 2)

            popt_cs, _ = curve_fit(cs_func, depths, coeff[:, 1], p0=X0, bounds=(lb, ub))
            coeff_cs[j, :, i] = popt_cs

            s_nozzle[j, i] = np.sqrt(
                coeff_cs[j, 0, i] ** 2 -
                2 * coeff_cs[j, 2, i] * coeff_cs[j, 0, i] * coeff_cs[j, 1, i] * nozzle2iso +
                coeff_cs[j, 1, i] ** 2 * nozzle2iso ** 2
            )
            corr_nozzle[j, i] = (
                    (coeff_cs[j, 2, i] * coeff_cs[j, 0, i] - coeff_cs[j, 1, i] * nozzle2iso) /
                    s_nozzle[j, i]
            )

    data_to_write = np.vstack([
        s_nozzle[:, 0], coeff_cs[:, 1, 0], corr_nozzle[:, 0],
        s_nozzle[:, 1], coeff_cs[:, 1, 1], corr_nozzle[:, 1]
    ]).T
    write_to_bdl(bdl_file, bdl_energies, [6, 7, 8, 9, 10, 11], data_to_write)
    logger.info("Completed calc_phase_space.")

def process_ct_to_mhd(ct_path, work_dir, new_spacing=(1.0, 1.0, 1.0)):
    """
    Process a CT file and save as an MHD file with new spacing.
    """
    # Ensure work_dir exists
    os.makedirs(work_dir, exist_ok=True)
    output_mhd_path = os.path.join(work_dir, 'CT.mhd')

    # Determine file extension
    ext = os.path.splitext(ct_path)[1].lower()

    # Read the image
    if ext == '.dcm':
        # Read DICOM file (assuming single file; for series, use ImageSeriesReader)
        reader = sitk.ImageFileReader()
        reader.SetFileName(ct_path)
        image = reader.Execute()
    elif ext == '.mhd':
        # Read MHD file
        reader = sitk.ImageFileReader()
        reader.SetFileName(ct_path)
        image = reader.Execute()
    else:
        raise ValueError("Error: please enter a valid CT name (dcm or mhd)")

    # Extract CT info
    ct_info = {
        'Size': list(image.GetSize()),  # (x, y, z)
        'Spacing': list(image.GetSpacing()),  # (x, y, z)
        'ImagePositionPatient': list(image.GetOrigin())  # (x, y, z)
    }

    # Calculate resolution factor and new size
    resolution_fact = np.array(ct_info['Spacing']) / np.array(new_spacing)
    new_size = [int(round(s * r)) for s, r in zip(ct_info['Size'], resolution_fact)]

    # Resample the image to new spacing
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)  # Use original image as reference
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)  # Linear interpolation
    resampled_image = resampler.Execute(image)

    # Write the resampled image as MHD
    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_mhd_path)
    writer.Execute(resampled_image)

    # Update ct_info with new values
    ct_info['Size'] = np.array(new_size)
    ct_info['Spacing'] = np.array(list(new_spacing))
    return ct_info

def calculate_grids(ct_info, new_spacing, isoc_depth):
    """
    Calculate true center and 3D coordinate grids.
    """
    # Extract origin and ensure it's a NumPy array
    origin = np.array(ct_info['ImagePositionPatient'])
    new_size = np.array(ct_info['Size'])
    new_spacing = np.array(new_spacing)

    # Calculate true center
    true_center = origin + (new_size - 1) * new_spacing / 2 + new_spacing / 2

    # Calculate center relative to origin
    center = true_center - origin
    # Adjust y-coordinate (center[1]) with isoc_depth
    center[1] = (new_size[1] - 1) * new_spacing[1] - isoc_depth + new_spacing[1] / 2

    # Generate vectors with spacing, shifted by half voxel
    v1 = np.arange(origin[0], origin[0] + new_size[0] * new_spacing[0], new_spacing[0]) + new_spacing[0] / 2
    v2 = np.arange(origin[1], origin[1] + new_size[1] * new_spacing[1], new_spacing[1]) + new_spacing[1] / 2
    v3 = np.arange(origin[2], origin[2] + new_size[2] * new_spacing[2], new_spacing[2]) + new_spacing[2] / 2

    # Create 3D coordinate grids
    X, Y, Z = np.meshgrid(v1, v2, v3, indexing='ij')
    logger.info("Completed calculate_grids.")
    return true_center, center, X, Y, Z

def calculate_grids_ad(ct_info, new_spacing, new_size, nbspots1, nbspots2, spot_spacing):
    """
    Calculate the CT image grids and spot positions.
    """
    origin = np.array(ct_info['ImagePositionPatient'])
    new_size = np.array(new_size)
    new_spacing = np.array(new_spacing)

    # calculate true center
    true_center = origin + new_spacing / 2 + (new_size - 1) * new_spacing / 2

    # correct center (relative to origin)
    center = true_center - origin

    v1 = np.arange(origin[0], origin[0] + (new_size[0] - 1) * new_spacing[0] + new_spacing[0] / 2, new_spacing[0])
    v2 = np.arange(origin[1], origin[1] + (new_size[1] - 1) * new_spacing[1] + new_spacing[1] / 2, new_spacing[1])
    v3 = np.arange(origin[2], origin[2] + (new_size[2] - 1) * new_spacing[2] + new_spacing[2] / 2, new_spacing[2])
    v1 = v1 + new_spacing[0] / 2
    v2 = v2 + new_spacing[1] / 2
    v3 = v3 + new_spacing[2] / 2

    # Create 2D coordinate grids (Z1 along Z and Y1 along Y)
    Z1, Y1 = np.meshgrid(v3, v1)  # attention: v3 is Z,v1 is Y

    # Generate spot position coordinates
    c1 = np.arange(-(nbspots1 - 1) * spot_spacing / 2, (nbspots1 - 1) * spot_spacing / 2 + spot_spacing, spot_spacing)
    c2 = np.arange(-(nbspots2 - 1) * spot_spacing / 2, (nbspots2 - 1) * spot_spacing / 2 + spot_spacing, spot_spacing)
    C1, C2 = np.meshgrid(c1, c2)

    # reshape
    C = np.stack((C1.T, C2.T), axis=-1)  # shape: (nbspots2, nbspots1, 2)
    coord = C.reshape(-1, 2)  # Reshape to (nbspots1 * nbspots2, 2)
    z_column = np.ones((coord.shape[0], 1))  # Add a z column, all 1
    coord = np.hstack((coord, z_column))  # Merge to (nbspots1 * nbspots2, 3)
    return true_center, center, Z1, Y1, v1, v3, coord


@timeit
def opt_energy_spectrum(gantry, tps, mc_config, engine):
    """
    Optimize the energy spectrum parameters and write them to the BDL file.
    """
    logger.info("Starting opt_energy_spectrum...")
    target_tol = gantry.es_goal
    nozzle2iso = gantry.nozzle2iso
    WorkDir = mc_config.work_dir
    bdl_file = mc_config.bdl_file
    new_spacing = mc_config.simu_spacing_es

    bdl_parameters = read_bdl_file(bdl_file)
    if tps.name.lower() == 'raystation':
        idd_data = tps.idd_data
        isoc_depth = tps.isocenter_depth
        IC_diameter = tps.IC_diameter_es
    elif tps.name.lower() == 'eclipse':
        idd_data = tps.idd_data
        isoc_depth = tps.isocenter_depth
        IC_diameter = tps.IC_diameter_es
    else:
        raise ValueError(f"Not supported TPS data [{tps.name}]")

    if engine.lower() == 'mcsquare':
        MC2Data = {}
        ct_info = process_ct_to_mhd(mc_config.ct_template_path, WorkDir, new_spacing)
        true_center, center, X, Y, Z = calculate_grids(ct_info, new_spacing, isoc_depth)
        MC2Data['X'] = X
        MC2Data['Y'] = Y
        MC2Data['Z'] = Z
        MC2Data['center'] = center
        MC2Data['true_center'] = true_center

    BP_MC = np.load('BP_library.npy', allow_pickle=True)
    BPMC_x = BP_MC[:, 0]
    BPMC_y = BP_MC[:, 1:]
    BPMC_energies = np.arange(60, 250.2, 0.2)

    bdl_energies = np.zeros(bdl_parameters.shape[0])
    sol_e = np.zeros((bdl_parameters.shape[0], 2))
    fval_e = np.zeros(bdl_parameters.shape[0])
    errange = np.zeros(bdl_parameters.shape[0])
    errel20 = np.zeros(bdl_parameters.shape[0])
    err_FWHM = np.zeros(bdl_parameters.shape[0])
    dtpd = np.zeros(bdl_parameters.shape[0])
    mptpd = np.zeros(bdl_parameters.shape[0])
    goal = np.zeros(bdl_parameters.shape[0])

    for i in range(bdl_parameters.shape[0]):
        current_energy = bdl_parameters[i, 0]
        logger.info(f"[{current_energy}] Optimization of energy spectrum parameters")
        IDD = [x['idd'] for x in idd_data if x['energy'] == current_energy][0]
        IDD = IDD[np.argsort(IDD[:, 0])]
        delta = IDD[-1, 0] - IDD[-2, 0]
        supp = np.arange(IDD[-1, 0], IDD[-1, 0] + 30 + delta, delta)
        IDD = np.vstack([IDD, np.column_stack([supp, np.zeros_like(supp)])])
        IDD = IDD[IDD[:, 0] >= new_spacing[1] / 2]
        IDD_sum = np.column_stack([IDD[:, 0], IDD[:, 1] / trapz_with_sort(IDD[:, 0], IDD[:, 1])])

        ind_before = np.where(IDD_sum[:, 1] >= max(IDD_sum[:, 1]) * 0.8)[0][-1]
        ind_after = ind_before + 1
        interp_r80 = interp1d([IDD_sum[ind_after, 1], IDD_sum[ind_before, 1]],
                              [IDD_sum[ind_after, 0], IDD_sum[ind_before, 0]])
        r80_meas = interp_r80(max(IDD_sum[:, 1]) * 0.8)
        log_r = np.log(r80_meas / 10)
        E_0 = np.exp(3.464048 + 0.561372013 * log_r - 0.004900892 * log_r ** 2 + 0.001684756748 * log_r ** 3)

        mask = (BPMC_energies >= current_energy - 7) & (BPMC_energies <= current_energy + 7)
        BPMC_preopt = BPMC_y[:, mask]

        x0 = np.array([E_0, 0.5])
        ub = np.array([E_0 + gantry.es_e_dif, gantry.es_es_range[1]])
        lb = np.array([E_0 - gantry.es_e_dif, gantry.es_es_range[0]])

        def funct(x):
            return fun_preopt(x, BPMC_x, BPMC_preopt, BPMC_energies[mask], IDD_sum)

        sol, fval = pattern_search(funct, x0, bounds=(lb, ub), steptol=1e-4, meshtol=1e-4, tolfun=1e-6, tolx=5e-5)
        x0 = np.array([sol[0], sol[1] / current_energy * 100])

        global funct_called_counter, line, y_values
        funct_called_counter = 0
        global line, goal_values, fig, ax
        goal_values = []

        fig, ax = plt.subplots(figsize=(10, 6))
        line, = ax.plot([], [], 'bo-', label='goal value')
        ax.set_xlabel('evaluation times')
        ax.set_ylabel('target goal')
        ax.set_title('goal curve')
        ax.grid(True)
        ax.legend()

        def funct(x):
            if engine == 'topas':
                return fun_es(gantry, tps, mc_config, engine, x, IDD_sum, current_energy, IC_diameter, new_spacing)
            else:
                return fun_es(gantry, tps, mc_config, engine, x, IDD_sum, current_energy, IC_diameter, new_spacing,
                              ct_info['Size'], isoc_depth, MC2Data)

        global best_x, best_fval,iteration_counter
        best_x = None
        best_fval = float('inf')
        iteration_counter=0
        def callback(xk):
            global best_x, best_fval, iteration_counter
            iteration_counter+=1
            fval = funct(xk)
            logger.info(f"iteration_counter: {iteration_counter}, x: {xk}, goal: {fval}")
            if fval < best_fval:
                best_fval = fval
                best_x = xk.copy()
            if fval < target_tol:
                logger.info(f"Stopping optimization: fval < {target_tol}")
                raise StopIteration

        logger.info(f'guess x0: {x0}')
        # options = {
        #     'xatol': 1e-3,
        #     'fatol': 5e-2,
        #     'maxfev': 120,
        #     'disp': True
        # }

        try:
            ub = np.array([E_0 + gantry.es_e_dif, gantry.es_es_range[1]])
            lb = np.array([E_0 - gantry.es_e_dif, gantry.es_es_range[0]])
            options = {'maxiter': 200, 'xatol': 1e-3, 'fatol': 5e-2}
            # options = {'maxiter': 120, 'xtol': 1e-3, 'ftol': 5e-2}
            result = minimize(
                fun=funct,
                x0=x0,
                bounds=list(zip(lb, ub)),
                method='nelder-mead',
                # method='powell',
                options=options,
                callback=callback
            )
            sol_e[i] = result.x
            fval_e[i] = result.fun
        except StopIteration:
            sol_e[i] = best_x
            fval_e[i] = best_fval
            logger.info("Optimization terminated by callback")

        # if fval_e[i] > target_tol:
        #     x0 = np.array([sol_e[0], sol_e[1] / current_energy * 100])
        #     sol_ps, fval_ps = pattern_search(funct, x0, bounds=(lb, ub),
        #                                      steptol=1e-4, meshtol=1e-4, tolfun=1e-6, tolx=5e-5)
        #     if fval_ps < fval_e[i]:
        #         sol_e[i] = sol_ps
        #         fval_e[i] = fval_ps

        logger.info(f"Optimized solution: {sol_e[i]}, Final value: {fval_e[i]}")
        write_to_bdl(bdl_file, current_energy, [2, 3], sol_e[i])

        logger.info(f"[{current_energy}] Start Monte Carlo simulation for energy spread, engine: {engine}")
        if engine.lower().__contains__('mcsquare') and not tps.skip_simu:
            mc_config.create_es_plan_file(WorkDir, ct_info['Size'], new_spacing, isoc_depth, current_energy)
            mc_config.generate_mc2_config(WorkDir, mc_config.simu_es_protons_final, 'CT.mhd', 'OneSpot.txt',
                                          os.path.join(mc_config.program_path, 'Scanners/Water_only'), bdl_file)
            MCdose = mc_config.run_mc2_simulation_and_compute_idd(WorkDir, mc_config.program_path, IC_diameter,
                                                                  new_spacing, IDD, MC2Data, tps)
        elif engine.lower().__contains__('topas') and not tps.skip_simu:
            mc_config.generate_topas_idd_config(gantry, mc_config, tps, current_energy, new_spacing)
            mc_config.run_topas_simulation(WorkDir, 'pbs_idd.txt')
            Dose_data = mc_config.import_topas_idd_csv(os.path.join(WorkDir, 'pbs_idd_result.csv'), dataLines=None)
            absci = Dose_data[:, 0] + new_spacing[1] / 2
            MCdose = interp1d(absci, Dose_data[:, 1], bounds_error=False)(IDD[:, 0])
            MCdose = MCdose / trapz_with_sort(IDD[:, 0], MCdose)
        elif engine.lower().__contains__('moqui') and not tps.skip_simu:
            pass
        logger.info(f"[{current_energy}] Finish Monte Carlo simulation for energy spread, engine: {engine}")

        # Evaluate optimization results (r80, r20, FWHM, etc.)
        ind_before = np.where(MCdose >= max(MCdose) * 0.8)[0][-1]
        ind_after = ind_before + 1
        r80_simu = interp1d([MCdose[ind_after], MCdose[ind_before]],
                            [IDD_sum[ind_after, 0], IDD_sum[ind_before, 0]])(max(MCdose) * 0.8)
        r80_meas = interp_r80(max(IDD_sum[:, 1]) * 0.8)
        errange[i] = r80_meas - r80_simu

        ind_before = np.where(MCdose >= max(MCdose) * 0.2)[0][-1]
        ind_after = ind_before + 1
        r20_simu = interp1d([MCdose[ind_after], MCdose[ind_before]],
                            [IDD_sum[ind_after, 0], IDD_sum[ind_before, 0]])(max(MCdose) * 0.2)
        ind_before = np.where(IDD_sum[:, 1] >= max(IDD_sum[:, 1]) * 0.2)[0][-1]
        ind_after = ind_before + 1
        r20_meas = interp1d([IDD_sum[ind_after, 1], IDD_sum[ind_before, 1]],
                            [IDD_sum[ind_after, 0], IDD_sum[ind_before, 0]])(max(IDD_sum[:, 1]) * 0.2)
        errel20[i] = r20_meas - r20_simu

        t = 60
        ind_before_r = np.where(IDD_sum[:, 1] >= max(IDD_sum[:, 1]) * t / 100)[0][-1]
        ind_after_r = ind_before_r + 1
        ind_after_l = np.where(IDD_sum[:, 1] >= max(IDD_sum[:, 1]) * t / 100)[0][0]
        ind_before_l = ind_after_l - 1
        r_m = interp1d([IDD_sum[ind_after_r, 1], IDD_sum[ind_before_r, 1]],
                       [IDD_sum[ind_after_r, 0], IDD_sum[ind_before_r, 0]])(max(IDD_sum[:, 1]) * t / 100)
        l_m = interp1d([IDD_sum[ind_before_l, 1], IDD_sum[ind_after_l, 1]],
                       [IDD_sum[ind_before_l, 0], IDD_sum[ind_after_l, 0]])(max(IDD_sum[:, 1]) * t / 100)
        FWHM_meas = r_m - l_m
        ind_before_r = np.where(MCdose >= max(MCdose) * t / 100)[0][-1]
        ind_after_r = ind_before_r + 1
        ind_after_l = np.where(MCdose >= max(MCdose) * t / 100)[0][0]
        ind_before_l = ind_after_l - 1
        r_s = interp1d([MCdose[ind_after_r], MCdose[ind_before_r]],
                       [IDD_sum[ind_after_r, 0], IDD_sum[ind_before_r, 0]])(max(MCdose) * t / 100)
        l_s = interp1d([MCdose[ind_before_l], MCdose[ind_after_l]],
                       [IDD_sum[ind_before_l, 0], IDD_sum[ind_after_l, 0]])(max(MCdose) * t / 100)
        FWHM_MC = r_s - l_s
        err_FWHM[i] = FWHM_meas - FWHM_MC

        dtpd[i] = (max(IDD_sum[:, 1]) - max(MCdose)) * 100 / max(IDD_sum[:, 1])
        L = r80_meas - IDD_sum[0, 0]
        mptpd[i] = 0
        for j in range(len(IDD_sum[:, 1]) - 1):
            if IDD_sum[j, 1] > max(IDD_sum[:, 1]) / 40:
                mptpd[i] += (
                        (abs(IDD_sum[j, 1] - MCdose[j]) / IDD_sum[j, 1]) *
                        ((IDD_sum[j + 1, 0] - IDD_sum[j, 0]) / L) * 100
                )

        goal[i] = 10 * errange[i] ** 2 + errel20[i] ** 2 + (1.2 * err_FWHM[i] - dtpd[i]) ** 2

        header = (
            f"# Energy (MeV): {current_energy:.6f}\n"
            f"# Range Error: {errange[i]:.6f}\n"
            f"# Relative Error 20%: {errel20[i]:.6f}\n"
            f"# FWHM Error: {err_FWHM[i]:.6f}\n"
            f"# DTPD: {dtpd[i]:.6f}\n"
            f"# goal: {goal[i]:.6f}\n"
        )
        logger.info("Header Metrics:\n" + header.strip())

        depths = IDD_sum[:, 0]
        meas_dose = IDD_sum[:, 1]
        sim_dose = MCdose

        tmp_dir = os.path.join(WorkDir, 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)

        csv_filename = os.path.join(tmp_dir, f'idd_E{current_energy:.1f}.csv')
        df = pd.DataFrame({
            'Depth (mm)': depths,
            'Measured Dose (IDD)': meas_dose,
            'Simulated Dose (MC)': sim_dose
        })

        with open(csv_filename, 'w', newline='\n', encoding='utf-8') as f:
            f.write(header)
            df.to_csv(f, index=False, lineterminator='\n')
        logger.info(f"Dose and metrics data saved to {csv_filename}")

        plt.figure(figsize=(10, 6))
        plt.plot(depths, meas_dose, 'b-', label='Measured IDD')
        plt.plot(depths, sim_dose, 'r--', label='Simulated MC Dose')
        plt.xlabel('Depth (mm)')
        plt.ylabel('Dose (Normalized)')
        plt.title(f'IDD Comparison at Energy {current_energy:.1f} MeV')
        plt.grid(True)
        plt.legend()
        plot_filename = os.path.join(WorkDir, 'tmp', f'idd_E{current_energy:.1f}.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Dose plot saved to {plot_filename}")

    logger.info("Completed opt_energy_spectrum.")

@timeit
def fun_es(gantry, tps, mc_config, engine, x, IDD, current_energy, IC_diameter, new_spacing, new_size=None, isocenter_depth=None, MC2Data=None):
    """
    Energy spectrum optimization objective function.

    Parameters:
        gantry: Gantry parameters object.
        tps: Treatment planning system data.
        mc_config: Monte Carlo configuration.
        engine (str): Simulation engine.
        x: Optimization parameters [mean energy, energy spread].
        IDD: Measured depth dose data [depth, dose].
        current_energy: Current beam energy (MeV).
        IC_diameter: Ionization chamber diameter.
        new_spacing: New spatial resolution [radius, depth] (mm).
        new_size: Optional grid size.
        isocenter_depth: Optional isocenter depth.
        MC2Data: Optional MC2 simulation data.

    Returns:
        float: Objective function value.
    """
    WorkDir = mc_config.work_dir
    nozzle2iso = gantry.nozzle2iso
    bdl_file = mc_config.bdl_file
    
    global  funct_called_counter
    funct_called_counter += 1  # Increment iteration counter

    # Update average energy and expansion in bdl file
    write_to_bdl(bdl_file, current_energy, [2, 3], x)

    # MC 模拟
    print(f"[{current_energy}] Start Monte Carlo simulation for energy spread, engine:{engine}")
    if engine.lower().__contains__('mcsquare'):
        # Create the plan file
        mc_config.create_es_plan_file(WorkDir, new_size, new_spacing, isocenter_depth, current_energy)
        mc_config.generate_mc2_config(WorkDir, mc_config.simu_es_protons, 'CT.mhd', 'OneSpot.txt',
                                            os.path.join(mc_config.program_path, 'Scanners/Water_only'), bdl_file)
        MCdose = mc_config.run_mc2_simulation_and_compute_idd(WorkDir, mc_config.program_path,  IC_diameter,
                                                                                new_spacing, IDD, MC2Data, tps)
    elif engine.lower().__contains__('topas'):
        mc_config.generate_topas_idd_config(gantry, mc_config, tps, current_energy, new_spacing)
        mc_config.run_topas_simulation(WorkDir, 'pbs_idd.txt')
        Dose_data = mc_config.import_topas_idd_csv(os.path.join(WorkDir, 'pbs_idd_result.csv'), dataLines=None)
        # Create tmp folder if it doesn't exist
        # 计算 MC 深度剂量
        absci = Dose_data[:, 0] + new_spacing[1] / 2
        MCdose = interp1d(absci, Dose_data[:, 1], bounds_error=False)(IDD[:, 0])
        MCdose = MCdose / trapz_with_sort(IDD[:, 0], MCdose)
    elif engine.lower().__contains__('moqui'):
        pass
    logger.info(f"[{current_energy}] Finish Monte Carlo simulation for energy spread, engine: {engine}")

    # r80 relative error
    ind_before = np.where(MCdose >= max(MCdose) * 0.8)[0][-1]
    ind_after = ind_before + 1
    r80_simu = interp1d([MCdose[ind_after], MCdose[ind_before]],
                        [IDD[ind_after, 0], IDD[ind_before, 0]])(max(MCdose) * 0.8)
    ind_before = np.where(IDD[:, 1] >= max(IDD[:, 1]) * 0.8)[0][-1]
    ind_after = ind_before + 1
    r80_meas = interp1d([IDD[ind_after, 1], IDD[ind_before, 1]],
                        [IDD[ind_after, 0], IDD[ind_before, 0]])(max(IDD[:, 1]) * 0.8)
    errel = r80_meas - r80_simu

    # r20 relative error
    ind_before = np.where(MCdose >= max(MCdose) * 0.2)[0][-1]
    ind_after = ind_before + 1
    r20_simu = interp1d([MCdose[ind_after], MCdose[ind_before]],
                        [IDD[ind_after, 0], IDD[ind_before, 0]])(max(MCdose) * 0.2)
    ind_before = np.where(IDD[:, 1] >= max(IDD[:, 1]) * 0.2)[0][-1]
    ind_after = ind_before + 1
    r20_meas = interp1d([IDD[ind_after, 1], IDD[ind_before, 1]],
                        [IDD[ind_after, 0], IDD[ind_before, 0]])(max(IDD[:, 1]) * 0.2)
    errel20 = r20_meas - r20_simu

    # Peak dose difference (%)
    dtpd = (max(IDD[:, 1]) - max(MCdose)) * 100 / max(IDD[:, 1])

    # Average point-to-point difference (%)
    L = r80_meas - IDD[0, 0]
    mptpd = 0
    for j in range(len(IDD[:, 1]) - 1):
        if IDD[j, 1] > max(IDD[:, 1]) / 40:
            mptpd += (abs(IDD[j, 1] - MCdose[j]) / IDD[j, 1]) * ((IDD[j + 1, 0] - IDD[j, 0]) / L) * 100

    # FWHM at t%
    t = 60
    ind_before_r = np.where(IDD[:, 1] >= max(IDD[:, 1]) * t / 100)[0][-1]
    ind_after_r = ind_before_r + 1
    ind_after_l = np.where(IDD[:, 1] >= max(IDD[:, 1]) * t / 100)[0][0]
    ind_before_l = ind_after_l - 1
    r_m = interp1d([IDD[ind_after_r, 1], IDD[ind_before_r, 1]],
                   [IDD[ind_after_r, 0], IDD[ind_before_r, 0]])(max(IDD[:, 1]) * t / 100)
    l_m = interp1d([IDD[ind_before_l, 1], IDD[ind_after_l, 1]],
                   [IDD[ind_before_l, 0], IDD[ind_after_l, 0]])(max(IDD[:, 1]) * t / 100)
    FWHM_meas = r_m - l_m
    ind_before_r = np.where(MCdose >= max(MCdose) * t / 100)[0][-1]
    ind_after_r = ind_before_r + 1
    ind_after_l = np.where(MCdose >= max(MCdose) * t / 100)[0][0]
    ind_before_l = ind_after_l - 1
    r_s = interp1d([MCdose[ind_after_r], MCdose[ind_before_r]],
                   [IDD[ind_after_r, 0], IDD[ind_before_r, 0]])(max(MCdose) * t / 100)
    l_s = interp1d([MCdose[ind_before_l], MCdose[ind_after_l]],
                   [IDD[ind_before_l, 0], IDD[ind_after_l, 0]])(max(MCdose) * t / 100)
    FWHM_MC = r_s - l_s
    err_FWHM = FWHM_meas - FWHM_MC

    # 目标函数
    goal = float(10 * errel**2 + errel20**2 + (1.2 * err_FWHM - dtpd)**2)

    # print("mptpd:", mptpd)
    # print("errel:", errel)
    # print("dtpd:", dtpd)
    # print("err_FWHM:", err_FWHM)
    logger.info(f'funct_called_counter: {funct_called_counter}, x: {x}, goal: {goal}')

    # Create tmp folder if it doesn't exist
    tmp_dir = os.path.join(WorkDir, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    global line, goal_values, fig, ax
    goal_values.append(goal)
    line.set_xdata(range(1, len(goal_values) + 1))
    line.set_ydata(goal_values)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

    tmp_dir = os.path.join(WorkDir, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    convergence_filename = f'convergence_E{current_energy}.png'
    fig.savefig(os.path.join(tmp_dir, convergence_filename), dpi=300, bbox_inches='tight')

    # if engine.lower()=='topas':
    #     # Plot IDD match
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(IDD[:, 0], IDD[:, 1], label='Measured IDD', color='blue')
    #     plt.plot(IDD[:, 0], MCdose, label='Simulated MCdose', color='red', linestyle='--')
    #     plt.xlabel('Depth (mm)')
    #     plt.ylabel('Dose')
    #     plt.title(f'IDD Match - Iter {funct_called_counter}, goal={goal:.4f}, E={x[0]:2f}MeV, Spread={x[1]:.4f}')
    #     plt.legend()
    #     plt.grid(True)
    #     # Save the plot
    #     plot_filename = f'idd_match_iter{funct_called_counter}_y{goal:.4f}_E{x[0]:2f}_spread{x[1]:.4f}.png'
    #     plt.savefig(os.path.join(tmp_dir, plot_filename), dpi=300, bbox_inches='tight')
    #     plt.close()  # Close the figure to free memory
    return goal

@timeit
def pattern_search(funct, x0, bounds, steptol=1e-4, meshtol=1e-4, tolfun=1e-6, tolx=5e-5):
    """
    Pattern search optimization combined with particle swarm optimization.

    Parameters:
        funct: Objective function to minimize.
        x0: Initial guess (array-like).
        bounds: Tuple (lb, ub) of lower and upper bounds.
        steptol: Tolerance for step size convergence.
        meshtol: Tolerance for mesh size (unused in this version).
        tolfun: Tolerance for function value convergence.
        tolx: Tolerance for variable convergence.

    Returns:
        tuple: (sol, fval) Optimized solution and function value.

        Pattern search optimization combined with particle swarm optimization.
    """
    logger.info("Starting pattern_search...")
    lb, ub = bounds
    x0 = np.clip(x0, lb, ub)  # Ensure x0 is within bounds

    # PSO optimization with reduced parameters for speed
    sol, fval = pso(
        func=funct,
        lb=lb,
        ub=ub,
        swarmsize=20,  # Reduced number of particles
        maxiter=50,    # Reduced maximum iterations
        minstep=steptol,
        minfunc=tolfun,
        debug=False    # Disable debug output for speed
    )

    # Local refinement using Nelder-Mead (faster than L-BFGS-B, no gradient needed)
    result = minimize(
        fun=funct,
        x0=sol,
        bounds=list(zip(lb, ub)),
        method='Nelder-Mead',
        tol=tolfun,
        options={'xatol': tolx, 'fatol': tolfun, 'maxiter': 200}  # Limit iterations
    )
    sol = result.x
    fval = result.fun
    sol = np.clip(sol, lb, ub)
    logger.info("Completed pattern_search.")
    return sol, fval

def fun_preopt(x, BPMC_x, BPMC_preopt, BPMC_energies, IDD):
    """
    Pre-optimization objective function for energy spectrum fitting.

    Parameters:
        x: Optimization parameters [mean energy, energy spread].
        BPMC_x: Monte Carlo depth coordinates (array).
        BPMC_preopt: Monte Carlo pre-optimized dose matrix (2D array).
        BPMC_energies: Monte Carlo energy array (array).
        IDD: Measured data [depth, dose] (2D array).

    Returns:
        float: Objective function value (scalar).
    """
    # Precompute constants
    idd_depths = IDD[:, 0]
    idd_dose = IDD[:, 1]
    idd_max = np.max(idd_dose)

    # Gaussian weights
    gw = norm.pdf(BPMC_energies, loc=x[0], scale=x[1])

    # Vectorized weighted MC dose
    SimDose = np.dot(BPMC_preopt, gw)
    MCdose = interp1d(BPMC_x, SimDose, bounds_error=False, fill_value=0)(idd_depths)
    try:
        SimDose = MCdose / np.trapezoid(MCdose, idd_depths)
    except:
        SimDose = MCdose / np.trapz(MCdose, idd_depths)

    sim_max = np.max(SimDose)
    # Vectorized R80 and R20 calculations
    def get_range(dose, thresh, depths):
        ind = np.where(dose >= thresh)[0]
        if len(ind) < 2:
            return depths[-1]
        ind_before, ind_after = ind[-2], ind[-1]
        return interp1d([dose[ind_after], dose[ind_before]],
                        [depths[ind_after], depths[ind_before]],
                        bounds_error=False, fill_value=depths[-1])(thresh)

    r80_simu = get_range(SimDose, sim_max * 0.8, idd_depths)
    r80_meas = get_range(idd_dose, idd_max * 0.8, idd_depths)
    errel = r80_meas - r80_simu

    r20_simu = get_range(SimDose, sim_max * 0.2, idd_depths)
    r20_meas = get_range(idd_dose, idd_max * 0.2, idd_depths)
    errel20 = r20_meas - r20_simu

    # Peak dose difference (%)
    dtpd = (idd_max - sim_max) * 100 / idd_max

    # Average point-to-point difference (%)
    mask = idd_dose > idd_max * 0.005
    if np.any(mask):
        L = r80_meas - idd_depths[0]
        diff = np.abs(idd_dose[mask] - SimDose[mask]) / idd_dose[mask]
        weights = np.diff(idd_depths[mask]) / L
        mptpd = np.sum(diff[:-1] * weights) * 100
    else:
        mptpd = 0

    # FWHM at 60%
    t = 0.6
    def get_fwhm(dose, thresh, depths):
        inds = np.where(dose >= thresh)[0]
        if len(inds) < 2:
            return 0
        l_idx, r_idx = inds[0], inds[-1]
        l_val = interp1d(dose[l_idx-1:l_idx+1], depths[l_idx-1:l_idx+1],
                         bounds_error=False, fill_value=depths[0])(thresh)
        r_val = interp1d(dose[r_idx-1:r_idx+1], depths[r_idx-1:r_idx+1],
                         bounds_error=False, fill_value=depths[-1])(thresh)
        return r_val - l_val

    FWHM_meas = get_fwhm(idd_dose, idd_max * t, idd_depths)
    FWHM_MC = get_fwhm(SimDose, sim_max * t, idd_depths)
    err_FWHM = FWHM_meas - FWHM_MC

    # Objective function, 10, 1.2
    goal = 9.9 * errel**2 + errel20**2 + (1.21 * err_FWHM - dtpd)**2 + mptpd**2
    return float(goal)

@timeit
def rotate_beamlet(Dose_data, nb_spots, ms_depth, new_spacing, coord, sadx, sady, origin, true_centre, iso_depth):
    """
    Optimized version of rotate_beamlet for speed.
    """
    # Precompute constants
    Dose_data = Dose_data / nb_spots  # Scale dose once
    depth_limit = ms_depth / new_spacing[1] + 15 / new_spacing[1]
    if Dose_data.shape[0] > depth_limit:
        Dose_data = Dose_data[:int(np.ceil(depth_limit)), :, :]

    sizeIn = Dose_data.shape
    D = np.zeros(sizeIn, dtype=Dose_data.dtype)  # Preallocate total dose array

    # Precompute shifts (assuming iso_depth is scalar or same length as spots)
    shift_z = origin[0] - true_centre[0]  # For z-axis rotation
    shift_y = sady - iso_depth  # For z-axis rotation
    shift_x = sadx - iso_depth  # For x-axis rotation
    shift_z2 = origin[2] - true_centre[2]  # For x-axis rotation

    # Precompute angles for all spots
    alpha = np.degrees(-np.atan(coord[:, 0] / sadx))  # Shape: (nb_spots,)
    beta = np.degrees(np.atan(coord[:, 1] / sady))  # Shape: (nb_spots,)

    # Loop over spots with optimized transformations
    for i in range(nb_spots):
        # Rotation angles in radians
        alpha_rad = radians(alpha[i])
        beta_rad = radians(beta[i])

        # Simplified rotation matrices (around z and x axes)
        cos_a, sin_a = cos(alpha_rad), sin(alpha_rad)
        cos_b, sin_b = cos(beta_rad), sin(beta_rad)

        # Rotation around z-axis (beta)
        Rz = np.array([
            [cos_b, -sin_b, 0],
            [sin_b, cos_b, 0],
            [0, 0, 1]
        ])

        # Rotation around x-axis (alpha)
        Rx = np.array([
            [1, 0, 0],
            [0, cos_a, -sin_a],
            [0, sin_a, cos_a]
        ])

        # Total rotation matrix (Rz @ Rx)
        R = Rz @ Rx

        # Translation components
        offset_z = np.array([shift_z, shift_y, 0])
        offset_x = np.array([0, shift_x, shift_z2])

        # Total transformation matrix (simplified)
        MM = np.eye(4)
        MM[:3, :3] = R
        offset = R @ offset_z + offset_x  # Combine translations

        # Apply affine transformation
        tform = np.linalg.inv(R)  # Inverse for ndimage
        rotated = ndimage.affine_transform(
            Dose_data,
            matrix=tform,
            offset=offset,
            output_shape=sizeIn,
            mode='constant',
            cval=0.0
        )

        # Accumulate dose
        D += rotated

    return D


# 示例调用
if __name__ == "__main__":
    # fname1 = r'C:\Users\m313763\Desktop\Dr.Liang\TOPAS_Auto_Modeling\AutoMCM\Sample_data_RayStation\Measured_GTR5_PristineBraggPeaks.csv'
    fname2 = r'C:\Users\m313763\Desktop\Dr.Liang\TOPAS_Auto_Modeling\AutoMCM\Sample_data_Eclipse\Measured_Depth_Dose_Parallel_Plate.txt'
