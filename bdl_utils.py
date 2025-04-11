import os
import numpy as np
from scipy.io import loadmat

def read_bdl_file(bdl_file):
    # Read bdl file
    A = open(bdl_file, 'r', encoding='utf-8').read().splitlines()
    start_index = [i for i, line in enumerate(A) if 'NominalEnergy' in line][0] + 1
    bdl_parameters = np.loadtxt(bdl_file, skiprows=start_index)
    # Ensure bdl_parameters is a 2D array
    bdl_parameters = np.atleast_2d(bdl_parameters)
    return bdl_parameters

def write_to_bdl(bdl_file, energies, columns, data):
    """
    Write data to specified columns in the bdl file.
    Parameters:
        bdlname: Path to the bdl file
        energies: Array of energies to match rows
        columns: Indices of columns to update (1-based, consistent with MATLAB)
        data: Data to write, shape (len(energies), len(columns))
    """
    # Read bdl file
    all_lines = open(bdl_file, encoding='utf-8').read().splitlines()
    start_index = next(i for i, line in enumerate(all_lines) if 'NominalEnergy' in line) + 1
    # Load bdl data part and ensure 2D array
    bdl_parameters = np.atleast_2d(np.loadtxt(bdl_file, skiprows=start_index))
    # Ensure input is 1D or 2D array
    energies, data = np.atleast_1d(energies), np.atleast_2d(data)
    # Update specified rows and columns
    for i, current_energy in enumerate(energies):
        row_idx = np.flatnonzero(bdl_parameters[:, 0] == current_energy)
        if not row_idx.size:
            raise ValueError(f"energy {current_energy} was not found in bdl file！")
        bdl_parameters[row_idx[0], np.array(columns) - 1] = data[i]
    # Convert to string and merge header and data
    all_lines_updated = all_lines[:start_index] + ['\t'.join(map(str, row)) for row in bdl_parameters]
    # Write to file, ensuring correct line endings
    with open(bdl_file, 'w', newline='\n') as f:
        for i, line in enumerate(all_lines_updated):
            f.write(line + ('' if i == len(all_lines_updated) - 1 else '\n'))

def create_bdl_file(gantry, tps, mc_config):
    """self.gantry, self.tps, self.mc_config
    Create bdl file.
    Parameters:
        bdl_file: Path to the bdl file
        nominal_energies: List of nominal energies or 0 (extract all energies from MeasFile)
        nozzle2iso: Distance from nozzle to isocenter (mm)
        vsad_x: Distance from SMX to isocenter (mm)
        vsad_y: Distance from SMY to isocenter (mm)
        MeasFile: Path to the measurement file
    """
    bdl_file= mc_config.bdl_file
    nominal_energies = gantry.energies_to_tune
    nozzle2iso = gantry.nozzle2iso
    vsad_x = gantry.vsad_xy[0]
    vsad_y = gantry.vsad_xy[1]
    # Create directory
    os.makedirs(os.path.dirname(bdl_file), exist_ok=True)
    # Ensure nominal_energies is a 1D array
    if len(nominal_energies)==0:
        nominal_energies = np.atleast_1d(tps.all_energies)
    else:
        nominal_energies = np.atleast_1d(nominal_energies)
    l = len(nominal_energies)
    # Initialize data array (18 columns)
    data = np.zeros((l, 18))
    data[:, 0:2] = nominal_energies[:, None]  # NominalEnergy 和 MeanEnergy
    data[:, 2] = 0.5  # EnergySpread
    data[:, 3] = 100000000  # ProtonsMU
    data[:, [4, 5, 8, 12, 15]] = 1, 5, 5, 5, 5  # Weight1, SpotSize1x, SpotSize1y, SpotSize2x, SpotSize2y
    data[:, [6, 9, 13, 16]] = 0.003  # Divergence1x, Divergence1y, Divergence2x, Divergence2y
    data[:, [7, 10, 14, 17]] = 0.2  # Correlation1x, Correlation1y, Correlation2x, Correlation2y
    # Sort by the first column (NominalEnergy)
    data = data[data[:, 0].argsort()]
    with open(bdl_file, 'w', newline='\n') as fid:
        line_change = '\n'
        fid.write(
            f'--UPenn beam model (double gaussian)--{line_change}'
            f'# Beam data library for P1{line_change}{line_change}'
            f'Nozzle exit to Isocenter distance{line_change}{nozzle2iso:.2f}{line_change}{line_change}'
            f'SMX to Isocenter distance{line_change}{vsad_x:.2f}{line_change}{line_change}'
            f'SMY to Isocenter distance{line_change}{vsad_y:.2f}{line_change}{line_change}'
            f'Beam parameters{line_change}{l} energies{line_change}{line_change}'
            'NominalEnergy\tMeanEnergy\tEnergySpread\tProtonsMU\tWeight1\tSpotSize1x\tDivergence1x\tCorrelation1x\t'
            'SpotSize1y\tDivergence1y\tCorrelation1y\tWeight2\tSpotSize2x\tDivergence2x\tCorrelation2x\t'
            f'SpotSize2y\tDivergence2y\tCorrelation2y{line_change}'
        )
        np.savetxt(
            fid, data,
            fmt=['%.3f', '%.6f', '%.6f', '%.1f'] + ['%.6f'] * 14,
            delimiter='\t'
        )
    print(f'energies to tune: {nominal_energies}')

