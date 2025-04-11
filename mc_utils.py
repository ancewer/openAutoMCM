import os,shutil
import platform
import subprocess
import numpy as np
from scipy.interpolate import interp1d
import SimpleITK as sitk
import pandas as pd
from bdl_utils import read_bdl_file
from base_funs import timeit, trapz_with_sort

class TopasConfigObject:
    work_dir = ''
    program_path =''
    config_pbs_beam_file = ''
    config_pbs_idd_file = ''
    config_pbs_ad_file = ''
    bdl_file = ''

    """Class to export MC2 configuration to a file based on a template."""
    def __init__(self,work_dir,program_path,bdl_file):
        self.work_dir = work_dir
        self.program_path = program_path
        self.bdl_file = bdl_file

    @staticmethod
    def import_topas_idd_csv(filename, dataLines=None):
        """
        Import TOPAS data from a CSV file.

        Parameters:
            filename: Path to the file
            dataLines: Data line range [start, end], defaults to [9, Inf] (from line 9 to end)
        Returns:
            PBSIdd070: Imported data (numpy array, containing only columns 3 and 4)
        """
        if dataLines is None:
            dataLines = [9, float('inf')]  # Default from line 9 to end

        skiprows = dataLines[0] - 1 if dataLines[0] > 1 else 0
        nrows = int(dataLines[1] - dataLines[0] + 1) if dataLines[1] != float('inf') else None

        try:
            df = pd.read_csv(
                filename,
                delimiter=',',
                skiprows=skiprows,
                nrows=nrows,
                header=None,
                usecols=[2, 3],
                dtype=float,
                skip_blank_lines=False,
                keep_default_na=False,
                engine='python'
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"{filename} not existed！")

        return df.values

    @timeit
    @staticmethod
    def run_topas_simulation(WorkDir, fname):
        cur_dir = os.getcwd()
        os.chdir(WorkDir)
        with open(os.path.join(WorkDir, 'run_topas.sh'),'w', encoding='utf-8', newline='\n') as f:
            f.write(f"#!/bin/bash\nset -x\nsource ~/.bashrc\nexport TOPAS_G4_DATA_DIR=~/topas/G4Data\n"
                    f"export PATH=$PATH:~/topas/bin\ntopas {fname}")
        os.system("wsl bash -c 'bash \\run_topas.sh'")
        os.chdir(cur_dir)

    @staticmethod
    def generate_topas_idd_config(gantry, mc_config, tps, energy, new_spacing):
        """
        Generate TOPAS configuration files, updating PBS_Beam.txt and PBS_Idd.txt.

        Parameters:
            SimulatedProtons: Number of simulated protons
            nozzle2iso: Distance from nozzle to isocenter (mm)
            energy: Target energy (MeV)
            BDL: Path to BDL file
            new_spacing: New spacing parameters [RMax, ZBins] (mm)
        """
        SimulatedProtons = mc_config.simu_es_protons
        nozzle2iso=gantry.nozzle2iso
        bdl_parameters = read_bdl_file(mc_config.bdl_file)
        work_dir = mc_config.work_dir

        row_idx = np.where(bdl_parameters[:, 0] == energy)[0][0]
        mywords = bdl_parameters[row_idx, :]

        lines = open(mc_config.config_pbs_beam_file, encoding='utf-8').read().splitlines()
        for i in range(len(lines)):
            if 'd:So/PrimBeam/BeamEnergy' in lines[i]:
                lines[i] = f'd:So/PrimBeam/BeamEnergy = {mywords[1]} MeV'
            elif 'u:So/PrimBeam/BeamEnergySpread' in lines[i]:
                lines[i] = f'u:So/PrimBeam/BeamEnergySpread = {mywords[2]}'
            elif 'd:So/PrimBeam/SigmaX' in lines[i]:
                lines[i] = f'd:So/PrimBeam/SigmaX = {mywords[5]} mm'
            elif 'd:So/PrimBeam/SigmaY' in lines[i]:
                lines[i] = f'd:So/PrimBeam/SigmaY = {mywords[8]} mm'
            elif 'u:So/PrimBeam/CorrelationX' in lines[i]:
                lines[i] = f'u:So/PrimBeam/CorrelationX = {mywords[7]}'
            elif 'u:So/PrimBeam/CorrelationY' in lines[i]:
                lines[i] = f'u:So/PrimBeam/CorrelationY = {mywords[10]}'
            elif 'u:So/PrimBeam/SigmaXPrime' in lines[i]:
                lines[i] = f'u:So/PrimBeam/SigmaXPrime = {mywords[6]}'
            elif 'u:So/PrimBeam/SigmaYPrime' in lines[i]:
                lines[i] = f'u:So/PrimBeam/SigmaYPrime = {mywords[9]}'
            elif 'i:So/PrimBeam/NumberOfHistoriesInRun' in lines[i]:
                lines[i] = f'i:So/PrimBeam/NumberOfHistoriesInRun = {int(SimulatedProtons)}'
            elif 'd:Ge/BeamPosition/TransZ' in lines[i]:
                lines[i] = f'd:Ge/BeamPosition/TransZ = {-nozzle2iso} mm'

        with open(os.path.join(work_dir, 'pbs_beam.txt'), 'w', encoding='utf-8') as fid:
            for i in range(len(lines)):
                fid.write(lines[i] + '\n')

        newlines = open(mc_config.config_pbs_idd_file, encoding='utf-8').read().splitlines()
        for i in range(len(newlines)):
            if 'd:Ge/BPCregion/RMax' in newlines[i]:
                newlines[i] = f'd:Ge/BPCregion/RMax = {tps.IC_diameter_es/2} mm'
            elif 'i:Ge/BPCregion/ZBins' in newlines[i]:
                newlines[i] = f'i:Ge/BPCregion/ZBins = {int(400 / new_spacing[1])}'

        with open(os.path.join(work_dir, 'pbs_idd.txt'), 'w', encoding='utf-8') as fid:
            for i in range(len(newlines)):
                fid.write(newlines[i] + '\n')

    @staticmethod
    def generate_topas_ad_config(gantry, mc_config, IC_diameter, iso_depth, ms_depth, energy,
                          spot_spacing, field_size):
        """
        Generate TOPAS AD file.

        Parameters:
            SimulatedProtons: Total number of simulated protons
            nozzle2iso: Distance from nozzle to isocenter (mm)
            IC_diameter: Ion chamber sensitive volume [diameter, depth] (mm)
            iso_depth: Isocenter depth (mm)
            ms_depth: Measurement depth (mm)
            energy: Energy (MeV)
            spot_spacing: Spot spacing (mm)
            field_size: Field size [width, height] or [size] (mm)
            vsad_x: SADX (mm)
            vsad_y: SADY (mm)
            bdl: Path to bdl file
        """
        fname = 'pbs_ad.txt'
        vsad_x = gantry.vsad_xy[0]
        vsad_y = gantry.vsad_xy[1]
        SimulatedProtons = mc_config.simu_ad_protons
        nozzle2iso = gantry.nozzle2iso
        work_dir = mc_config.work_dir

        bdl_parameters = read_bdl_file(mc_config.bdl_file)

        row_idx = np.where(bdl_parameters[:, 0] == energy)[0][0]
        mywords = bdl_parameters[row_idx, :]

        if len(field_size) == 1:
            field_size = [field_size[0], field_size[0]]
        # Generate x and y coordinates
        x = np.arange(-field_size[0] / 2, field_size[0] / 2 + spot_spacing, spot_spacing)
        y = np.arange(-field_size[1] / 2, field_size[1] / 2 + spot_spacing, spot_spacing)
        # Calculate spot positions adjusted by focal distances
        x_pos = x * (vsad_x - nozzle2iso) / vsad_x
        y_pos = y * (vsad_y - nozzle2iso) / vsad_y
        # Calculate angles in degrees
        x_angle = np.array([np.degrees(np.atan(xi / vsad_x)) for xi in x])
        y_angle = np.array([np.degrees(np.atan(yi / vsad_y)) for yi in y])
        # Compute total number of spots
        total_len = len(x) * len(y)
        # Initialize output strings
        x_pos_line = ''
        y_pos_line = ''
        x_angle_line = ''
        y_angle_line = ''
        particles_line = ''
        times_line = ''
        # Loop over all spots to generate output strings
        for i in range(len(x)):
            for j in range(len(y)):
                # Append x position to the string
                x_pos_line += ' ' + str(x_pos[i])
                # Append y position to the string
                y_pos_line += ' ' + str(y_pos[j])
                # Append x angle (using y_angle[j] as per the original code)
                x_angle_line += ' ' + str(y_angle[j])
                # Append y angle (using -x_angle[i] as per the original code)
                y_angle_line += ' ' + str(-x_angle[i])
                # Append number of particles per spot
                particles_per_spot = int(SimulatedProtons / total_len)  # Convert to integer
                particles_line += ' ' + str(particles_per_spot)
                # Append time index for the spot
                times_line += ' ' + str(i * len(y) + j + 1)

        # Remove leading/trailing whitespace from the strings
        x_pos_line = x_pos_line.strip()
        y_pos_line = y_pos_line.strip()
        x_angle_line = x_angle_line.strip()
        y_angle_line = y_angle_line.strip()
        particles_line = particles_line.strip()
        times_line = times_line.strip()

        lines = open(mc_config.config_pbs_ad_file, encoding='utf-8').read().splitlines()
        for i in range(len(lines)):
            if 'd:So/PrimBeam/BeamEnergy' in lines[i]:
                lines[i] = f'd:So/PrimBeam/BeamEnergy = {mywords[1]} MeV'
            elif 'u:So/PrimBeam/BeamEnergySpread' in lines[i]:
                lines[i] = f'u:So/PrimBeam/BeamEnergySpread = {mywords[2]}'
            elif 'd:So/PrimBeam/SigmaX' in lines[i]:
                lines[i] = f'd:So/PrimBeam/SigmaX = {mywords[5]} mm'
            elif 'd:So/PrimBeam/SigmaY' in lines[i]:
                lines[i] = f'd:So/PrimBeam/SigmaY = {mywords[8]} mm'
            elif 'u:So/PrimBeam/CorrelationX' in lines[i]:
                lines[i] = f'u:So/PrimBeam/CorrelationX = {mywords[7]}'
            elif 'u:So/PrimBeam/CorrelationY' in lines[i]:
                lines[i] = f'u:So/PrimBeam/CorrelationY = {mywords[10]}'
            elif 'u:So/PrimBeam/SigmaXPrime' in lines[i]:
                lines[i] = f'u:So/PrimBeam/SigmaXPrime = {mywords[6]}'
            elif 'u:So/PrimBeam/SigmaYPrime' in lines[i]:
                lines[i] = f'u:So/PrimBeam/SigmaYPrime = {mywords[9]}'
            elif 'i:Tf/NumberOfSequentialTimes' in lines[i]:
                lines[i] = f'i:Tf/NumberOfSequentialTimes = {total_len}'
            elif 'd:Tf/TimelineEnd' in lines[i]:
                lines[i] = f'd:Tf/TimelineEnd = {total_len} s'
            elif 'dv:Tf/Particles/Times' in lines[i]:
                lines[i] = f'dv:Tf/Particles/Times = {total_len} {times_line} s'
            elif 'iv:Tf/Particles/Values' in lines[i]:
                lines[i] = f'iv:Tf/Particles/Values = {total_len} {particles_line}'
            elif 'dv:Tf/xtrans/Values' in lines[i]:
                lines[i] = f'dv:Tf/xtrans/Values = {total_len} {x_pos_line} mm'
            elif 'dv:Tf/ytrans/Values' in lines[i]:
                lines[i] = f'dv:Tf/ytrans/Values = {total_len} {y_pos_line} mm'
            elif 'dv:Tf/xrot/Values' in lines[i]:
                lines[i] = f'dv:Tf/xrot/Values = {total_len} {x_angle_line} deg'
            elif 'dv:Tf/yrot/Values' in lines[i]:
                lines[i] = f'dv:Tf/yrot/Values = {total_len} {y_angle_line} deg'
            elif 'd:Ge/WaterBox/TransZ' in lines[i]:
                lines[i] = f'd:Ge/WaterBox/TransZ = {200 - iso_depth} mm'
            elif 'd:Ge/BPCregion/TransZ' in lines[i]:
                lines[i] = f'd:Ge/BPCregion/TransZ = {ms_depth - 200} mm'
            elif 'd:Ge/BPCregion/RMax' in lines[i]:
                lines[i] = f'd:Ge/BPCregion/RMax = {IC_diameter[0] / 2} mm'
            elif 'd:Ge/BPCregion/HL' in lines[i]:
                lines[i] = f'd:Ge/BPCregion/HL = {IC_diameter[1] / 2} mm'
            elif 'd:Ge/BeamPosition/TransZ' in lines[i]:
                lines[i] = f'd:Ge/BeamPosition/TransZ = {-nozzle2iso} mm'

        with open(os.path.join(work_dir, 'pbs_ad.txt'), 'w', newline='\n') as fid:
            for line in lines:
                fid.write(f'{line}\n')

    @staticmethod
    def import_topas_ad_csv(filename):
        """
        Import TOPAS AD data from a text file.

        Parameters:
            filename: Path to the file
            dataLines: Data line range [start, end], defaults to [6, Inf] (from line 6 to end)
        Returns:
            ADResult: Numeric data from the first column (numpy array)
        """
        ADResult = np.loadtxt(filename, skiprows=5)
        return ADResult

class MC2ConfigObject:
    work_dir = ''
    program_path =''
    config_file = ''
    ct_template_path = ''
    bdl_file = ''

    """Class to export MC2 configuration to a file based on a template."""
    def __init__(self,work_dir,program_path,config_file,ct_template_path, bdl_file):
        self.work_dir = work_dir
        self.program_path = program_path
        self.config_file = config_file
        self.ct_template_path = ct_template_path
        self.bdl_file = bdl_file

        with open(config_file, 'r', encoding='utf-8') as f:
            self.template = f.read()

    def _replace_bool(self, tag, value):
        """Replace a boolean tag in the template with 'True' or 'False'."""
        if value == 1 or str(value).lower() == 'true':
            self.template = self.template.replace(tag, 'True')
        else:
            self.template = self.template.replace(tag, 'False')

    def _replace_vector(self, tag, values):
        """Replace a vector tag in the template with space-separated values."""
        tmp = f"{values[0]} {values[1]} {values[2]}"
        self.template = self.template.replace(tag, tmp)

    def export_mc2_config(self, mc2_config):
        """
        Export the MC2 configuration to a file.

        Parameters:
        - mc2_config (dict): MC2 configuration dictionary
        """
        # Simulation parameters
        self.template = self.template.replace('{NUMBER_OF_THREADS}', str(mc2_config['NumberOfThreads']))
        self.template = self.template.replace('{RNG_SEED}', str(mc2_config['RNG_seed']))
        self.template = self.template.replace('{NUMBER_OF_PRIMARIES}', str(mc2_config['NumberOfPrimaries']))
        self.template = self.template.replace('{E_CUT_PRO}', str(mc2_config['E_Cut_Pro']))
        self.template = self.template.replace('{D_MAX}', str(mc2_config['D_Max']))
        self.template = self.template.replace('{EPSILON_MAX}', str(mc2_config['Epsilon_Max']))
        self.template = self.template.replace('{TE_MIN}', str(mc2_config['Te_Min']))

        # Input files (convert backslashes to forward slashes)
        self.template = self.template.replace('{CT_FILE_NAME}', mc2_config['CT'].replace('\\', '/'))
        self.template = self.template.replace('{SCANNER_HU_DENSITY_CONVERSION}',
                                              os.path.join(mc2_config['ScannerDirectory'], 'HU_Density_Conversion.txt').replace('\\', '/'))
        self.template = self.template.replace('{SCANNER_HU_MATERIAL_CONVERSION}',
                                              os.path.join(mc2_config['ScannerDirectory'], 'HU_Material_Conversion.txt').replace('\\', '/'))
        self.template = self.template.replace('{BDL_FILE_NAME}', mc2_config['BDL_File'].replace('\\', '/'))
        self.template = self.template.replace('{PBS_PLAN_FILE_NAME}', mc2_config['Plan'].replace('\\', '/'))

        # Physical parameters
        self._replace_bool('{ENABLE_NUCLEAR_INTER}', mc2_config['Simulate_Nuclear_Interactions'])
        self._replace_bool('{ENABLE_SECONDARY_PROTONS}', mc2_config['Simulate_Secondary_Protons'])
        self._replace_bool('{ENABLE_SECONDARY_DEUTERONS}', mc2_config['Simulate_Secondary_Deuterons'])
        self._replace_bool('{ENABLE_SECONDARY_ALPHAS}', mc2_config['Simulate_Secondary_Alphas'])

        # 4D simulation
        self._replace_bool('{ENABLE_4D_MODE}', mc2_config['Simu_4D_Mode'])
        self._replace_bool('{ENABLE_DOSE_ACCUMULATION}', mc2_config['Dose_4D_Accumulation'])
        self.template = self.template.replace('{FIELD_TYPE}', mc2_config['Field_type'])
        self._replace_bool('{ENABLE_REF_FROM_4DCT}', mc2_config['Create_Ref_from_4DCT'])
        self._replace_bool('{ENABLE_4DCT_FROM_REF}', mc2_config['Create_4DCT_from_Ref'])
        self._replace_bool('{ENABLE_DYNAMIC_DELIVERY}', mc2_config['Dynamic_delivery'])
        self.template = self.template.replace('{BREATHING_PERIOD}', str(mc2_config['Breathing_period']))

        # Robustness simulation
        self._replace_bool('{ENABLE_ROBUSTNESS_MODE}', mc2_config['Robustness_Mode'])
        self.template = self.template.replace('{ROBUSTNESS_SCENARIO_SELECTION}', mc2_config['ScenarioSelection'])
        self._replace_bool('{ROBUSTNESS_COMPUTE_NOMINAL}', mc2_config['Robust_Compute_Nominal'])
        self._replace_vector('{ROBUSTNESS_SYSTEMATIC_SETUP}', mc2_config['Robust_Systematic_Setup'])
        self._replace_vector('{ROBUSTNESS_RANDOM_SETUP}', mc2_config['Robust_Random_Setup'])
        self.template = self.template.replace('{ROBUSTNESS_RANGE_ERROR}', str(mc2_config['Robust_Range_Error']))
        self.template = self.template.replace('{ROBUSTNESS_SYSTEMATIC_AMPLI}', str(mc2_config['Robust_Systematic_Amplitude']))
        self.template = self.template.replace('{ROBUSTNESS_RANDOM_AMPLI}', str(mc2_config['Robust_Random_Amplitude']))
        self.template = self.template.replace('{ROBUSTNESS_SYSTEMATIC_PERIOD}', str(mc2_config['Robust_Systematic_Period']))
        self.template = self.template.replace('{ROBUSTNESS_RANDOM_PERIOD}', str(mc2_config['Robust_Random_Period']))

        # Beamlet simulation
        self._replace_bool('{ENABLE_BEAMLET_MODE}', mc2_config['Beamlet_Mode'])
        self._replace_bool('{ENABLE_BEAMLET_PARALLEL}', mc2_config['Beamlet_Parallelization'])

        # Output parameters
        self.template = self.template.replace('{OUTPUT_DIR}', mc2_config['Output_Directory'])
        self._replace_bool('{ENABLE_ENERGY_ASCII_OUT}', mc2_config['Out_Energy_ASCII'])
        self._replace_bool('{ENABLE_ENERGY_MHD_OUT}', mc2_config['Out_Energy_MHD'])
        self._replace_bool('{ENABLE_ENERGY_SPARSE_OUT}', mc2_config['Out_Energy_Sparse'])
        self._replace_bool('{ENABLE_DOSE_ASCII_OUT}', mc2_config['Out_Dose_ASCII'])
        self._replace_bool('{ENABLE_DOSE_MHD_OUT}', mc2_config['Out_Dose_MHD'])
        self._replace_bool('{ENABLE_DOSE_SPARSE_OUT}', mc2_config['Out_Dose_Sparse'])
        self._replace_bool('{ENABLE_LET_ASCII_OUT}', mc2_config['Out_LET_ASCII'])
        self._replace_bool('{ENABLE_LET_MHD_OUT}', mc2_config['Out_LET_MHD'])
        self._replace_bool('{ENABLE_LET_SPARSE_OUT}', mc2_config['Out_LET_Sparse'])
        self._replace_bool('{ENABLE_DENSITIES_OUT}', mc2_config['Out_Densities'])
        self._replace_bool('{ENABLE_MATERIALS_OUT}', mc2_config['Out_Materials'])
        self._replace_bool('{ENABLE_COMPUTE_DVH}', mc2_config['Compute_DVH'])
        self.template = self.template.replace('{DOSE_SPARSE_THRESHOLD}', str(mc2_config['Dose_Sparse_Threshold']))
        self.template = self.template.replace('{ENERGY_SPARSE_THRESHOLD}', str(mc2_config['Energy_Sparse_Threshold']))
        self.template = self.template.replace('{LET_SPARSE_THRESHOLD}', str(mc2_config['LET_Sparse_Threshold']))
        self._replace_bool('{ENABLE_PG_SCORING}', mc2_config['PG_scoring'])
        self.template = self.template.replace('{PG_LOW_CUT}', str(mc2_config['PG_LowEnergyCut']))
        self.template = self.template.replace('{PG_HI_CUT}', str(mc2_config['PG_HighEnergyCut']))
        self.template = self.template.replace('{PG_SPECTRUM_NUMBIN}', str(mc2_config['PG_Spectrum_NumBin']))
        self.template = self.template.replace('{PG_SPECTRUM_BINNING}', str(mc2_config['PG_Spectrum_Binning']))
        self.template = self.template.replace('{LET_METHOD}', mc2_config['LET_Method'])
        self._replace_bool('{ENABLE_BEAM_DOSE}', mc2_config['Export_beam_dose'])
        self.template = self.template.replace('{DOSE_TO_WATER}', mc2_config['DoseToWater'])
        self._replace_bool('{ENABLE_SEGMENTATION}', mc2_config['Dose_Segmentation'])
        self.template = self.template.replace('{SEGMENTATION_THRESHOLD}', str(mc2_config['Density_Threshold_for_Segmentation']))

        # Write the configuration file
        output_path = os.path.join(mc2_config['WorkDir'], 'config.txt')
        os.makedirs(mc2_config['WorkDir'], exist_ok=True)
        # print(f"Write configuration file: {output_path}")
        with open(output_path, 'w', newline='\n', encoding='utf-8') as fid:
            fid.write(self.template)

    def generate_mc2_config(self, work_dir, number_of_primaries, ct, plan, scanner_directory=None, bdl_file=None):
        """
        Generate an MC2 configuration dictionary for MCsquare simulation.

        Parameters:
        - work_dir (str): Working directory
        - number_of_primaries (int): Number of primaries for simulation
        - ct (str): Path to CT file
        - plan (str): Path to plan file
        - scanner_directory (str, optional): Path to scanner directory (default: UCL_Toshiba)
        - bdl_file (str, optional): Path to BDL file (default: TRENTO_GTR2.txt)

        Returns:
        - mc2_config (dict): Configuration dictionary
        """
        # Current directory
        cur_dir = os.getcwd()

        # Resolve absolute paths
        os.chdir(scanner_directory)
        scanner_directory = os.getcwd()

        file_path, file_name = os.path.split(bdl_file)
        # file_ext = os.path.splitext(bdl_file)[1]
        os.chdir(file_path)
        bdl_file = os.path.join(os.getcwd(), file_name)

        os.chdir(cur_dir)  # Return to original directory

        # Initialize configuration dictionary
        mc2_config = {}

        # Set working directory
        mc2_config['WorkDir'] = work_dir

        # Simulation parameters
        mc2_config['NumberOfThreads'] = 0
        mc2_config['RNG_seed'] = 0
        mc2_config['NumberOfPrimaries'] = number_of_primaries
        mc2_config['E_Cut_Pro'] = 0.5
        mc2_config['D_Max'] = 0.2
        mc2_config['Epsilon_Max'] = 0.25
        mc2_config['Te_Min'] = 0.05

        # Input files
        mc2_config['CT'] = ct
        mc2_config['ScannerDirectory'] = scanner_directory
        mc2_config['BDL_File'] = bdl_file
        mc2_config['Plan'] = plan

        # Physical parameters
        mc2_config['Simulate_Nuclear_Interactions'] = 1
        mc2_config['Simulate_Secondary_Protons'] = 1
        mc2_config['Simulate_Secondary_Deuterons'] = 1
        mc2_config['Simulate_Secondary_Alphas'] = 1

        # 4D simulation
        mc2_config['Simu_4D_Mode'] = 0
        mc2_config['Dose_4D_Accumulation'] = 0
        mc2_config['Field_type'] = 'Velocity'
        mc2_config['Create_Ref_from_4DCT'] = 0
        mc2_config['Create_4DCT_from_Ref'] = 0
        mc2_config['Dynamic_delivery'] = 0
        mc2_config['Breathing_period'] = 7.0

        # Robustness simulation
        mc2_config['Robustness_Mode'] = 0
        mc2_config['ScenarioSelection'] = 'All'
        mc2_config['Robust_Compute_Nominal'] = 1
        mc2_config['Robust_Systematic_Setup'] = [0.25, 0.25, 0.25]
        mc2_config['Robust_Random_Setup'] = [0.1, 0.1, 0.1]
        mc2_config['Robust_Range_Error'] = 3.0
        mc2_config['Robust_Systematic_Amplitude'] = 5.0
        mc2_config['Robust_Random_Amplitude'] = 5.0
        mc2_config['Robust_Systematic_Period'] = 5.0
        mc2_config['Robust_Random_Period'] = 5.0

        # Beamlet simulation
        mc2_config['Beamlet_Mode'] = 0
        mc2_config['Beamlet_Parallelization'] = 0

        # Output parameters
        mc2_config['Output_Directory'] = 'Outputs'
        mc2_config['Out_Energy_ASCII'] = 0
        mc2_config['Out_Energy_MHD'] = 0
        mc2_config['Out_Energy_Sparse'] = 0
        mc2_config['Out_Dose_ASCII'] = 0
        mc2_config['Out_Dose_MHD'] = 1
        mc2_config['Out_Dose_Sparse'] = 0
        mc2_config['Out_LET_ASCII'] = 0
        mc2_config['Out_LET_MHD'] = 0
        mc2_config['Out_LET_Sparse'] = 0
        mc2_config['Out_Densities'] = 0
        mc2_config['Out_Materials'] = 0
        mc2_config['Compute_DVH'] = 0
        mc2_config['Dose_Sparse_Threshold'] = 0.0
        mc2_config['Energy_Sparse_Threshold'] = 0.0
        mc2_config['LET_Sparse_Threshold'] = 0.0
        mc2_config['PG_scoring'] = 0
        mc2_config['PG_LowEnergyCut'] = 0.0
        mc2_config['PG_HighEnergyCut'] = 50.0
        mc2_config['PG_Spectrum_NumBin'] = 150
        mc2_config['PG_Spectrum_Binning'] = 0.1
        mc2_config['LET_Method'] = 'StopPow'
        mc2_config['Export_beam_dose'] = 0
        mc2_config['DoseToWater'] = 'Disabled'
        mc2_config['Dose_Segmentation'] = 0
        mc2_config['Density_Threshold_for_Segmentation'] = 0.01

        # Export the configuration (placeholder for openreggui_Export_MC2_Config)
        self.export_mc2_config(mc2_config)

    @staticmethod
    def create_es_plan_file(work_dir, new_size, new_spacing, isoc_depth, energy):
        """
        Create a treatment plan file (OneSpot.txt) with specified parameters.

        Parameters:
        - work_dir (str): Working directory to save the file
        - new_size (list/tuple): Size of the image (x, y, z)
        - new_spacing (list/tuple): Spacing of the image (x, y, z)
        - isoc_depth (list): List of isocenter depth values
        - energybdl (list): List of energy values (MeV)
        - num_energ (int): Index of the energy to use (0-based)
        - ind1 (int): Index for isoc_depth (default 0)
        """
        # Ensure work_dir exists
        os.makedirs(work_dir, exist_ok=True)
        file_path = os.path.join(work_dir, 'OneSpot.txt')

        # Convert inputs to lists/arrays if they aren't already
        new_size = list(new_size)
        new_spacing = list(new_spacing)

        # Calculate centre (assuming origin is at [0, 0, 0] or adjusted elsewhere)
        centre = [0, 0, 0]  # Placeholder; adjust if origin is provided
        centre[0] = (new_size[0] - 1) * new_spacing[0] / 2  # x (example assumption)
        centre[1] = (new_size[1] - 1) * new_spacing[1] - isoc_depth + new_spacing[1] / 2  # y
        centre[2] = (new_size[2] - 1) * new_spacing[2] / 2  # z (example assumption)

        # Define the plan content
        plan_content = (
            "#TREATMENT-PLAN-DESCRIPTION\n"
            "#PlanName\n"
            "OneSpot\n"
            "#NumberOfFractions\n"
            "1\n"
            "##FractionID\n"
            "1\n"
            "##NumberOfFields\n"
            "1\n"
            "###FieldsID\n"
            "1\n"
            "#TotalMetersetWeightOfAllFields\n"
            "1\n\n"
            "#FIELD-DESCRIPTION\n"
            "###FieldID\n"
            "1\n"
            "###FinalCumulativeMeterSetWeight\n"
            "1\n"
            "###GantryAngle\n"
            "0\n"
            "###PatientSupportAngle\n"
            "0\n"
            "###IsocenterPosition\n"
            f"{centre[0]:.6f}\t{centre[1]:.6f}\t{centre[2]:.6f}\n"
            "###NumberOfControlPoints\n"
            "1\n\n"
            "#SPOTS-DESCRIPTION\n"
            "####ControlPointIndex\n"
            "1\n"
            "####SpotTunnedID\n"
            "1\n"
            "####CumulativeMetersetWeight\n"
            "1\n"
            "####Energy (MeV)\n"
            f"{energy:.2f}\n"
            "####NbOfScannedSpots\n"
            "1\n"
            "####X Y Weight\n"
            "0 0 1"
        )

        # Write to file
        with open(file_path, 'w', newline='\n', encoding='utf-8') as fid:
            fid.write(plan_content)

    @staticmethod
    def run_mc2_simulation_and_compute_idd(work_dir, mc_square_dir, ic_diameter, new_spacing,
                                           idd, MC2Data=None, tps=None):
        """
        Run MCsquare simulation, import dose data, and compute MC depth dose.

        Parameters:
        - mc_square_dir (str): Directory containing MCsquare executable
        - work_dir (str): Working directory
        - X (np.ndarray): X-coordinate grid from previous computation
        - true_center (list): True center coordinates [x, y, z]
        - ic_diameter (list): List of IC diameters
        - ind1 (int): Index for ic_diameter (default 0)
        - new_spacing (list): Spacing [x, y, z]
        - idd (np.ndarray): 2D array with depth-dose data [depth, dose]

        Returns:
        - dose_data (np.ndarray): 3D dose array
        - mc_dose (np.ndarray): Computed MC depth dose
        """
        MC2ConfigObject.run_mc2_simulation(work_dir,mc_square_dir,tps)
        mc_dose = MC2ConfigObject.analyse_mc2_idd(work_dir, ic_diameter, new_spacing, idd, MC2Data)

        return mc_dose

    @staticmethod
    @timeit
    def run_mc2_simulation(work_dir,mc_square_dir, tps=None):
        """
        Runs an MCsquare simulation.

        Parameters:
        mc_square_dir (str): Directory containing the MCsquare executable
        work_dir (str): Working directory
        tps (object, optional): Object containing a skip_simu attribute to control whether to skip the simulation

        Returns:
        None (runs the simulation and generates output files)
        """
        skip_simu = tps.skip_simu

        if platform.system() == 'Windows' and not skip_simu:
            src_materials = os.path.join(mc_square_dir, 'Materials')
            dest_materials = os.path.join(work_dir, 'Materials')
            if os.path.exists(src_materials) and not os.path.exists(dest_materials):
                subprocess.run(['xcopy', src_materials, dest_materials, '/E', '/I', '/Y'], check=True)

            os.chdir(work_dir)
            # if os.path.exists("Outputs"):
            #     shutil.rmtree("Outputs")
            os.makedirs("Outputs", exist_ok=True)
            mc_square_exe = os.path.join(mc_square_dir, 'MCsquare.bat')
            with open('log.txt', 'a', encoding='utf-8') as log_file:
                # print('Start Monte Carlo simulation for energy spread, engine: MCsquare')
                subprocess.run(mc_square_exe, stdout=log_file, stderr=subprocess.STDOUT, shell=True)

        elif platform.system() == 'Linux' and not skip_simu:
            mc_square_exe = os.path.join(mc_square_dir, 'MCsquare_double')
            subprocess.run(['dos2unix', mc_square_exe], check=True)
            cmd = f"cd {work_dir} && {mc_square_exe}"
            subprocess.run(cmd, shell=True, check=True)

    @staticmethod
    def analyse_mc2_idd(work_dir, ic_diameter, new_spacing, idd, MC2Data=None):
        """
        Process MCsquare simulation results and calculate MC depth dose.

        Parameters:
        work_dir (str): working directory
        ic_diameter (float): IC diameter
        new_spacing (list): spatial resolution [dx, dy, dz]
        idd (np.ndarray): 2D array containing depth-dose data [depth, dose]
        MC2Data (dict, optional): dictionary containing X, Z and true_center for mask calculation

        Returns:
        mc_dose (np.ndarray): calculated MC depth dose
        """
        dose_file = os.path.join(work_dir, 'Outputs', 'Dose.mhd')
        reader = sitk.ImageFileReader()
        reader.SetFileName(dose_file)
        dose_image = reader.Execute()
        dose_data = sitk.GetArrayFromImage(dose_image)  # 形状: (z, y, x)
        dose_data = np.transpose(dose_data, (2, 1, 0))  # 转换为 [x, y, z]

        # Based on IC diameter mask data
        if MC2Data is not None:
            radius = ic_diameter / 2
            mask = np.sqrt((MC2Data['X'] - MC2Data['true_center'][0]) ** 2 +
                           (MC2Data['Z'] - MC2Data['true_center'][2]) ** 2) > radius
            dose_data[mask] = 0

        # Sum along the x and z axes to get the dose in the depth direction (y axis)
        # Reverse the y-axis order
        dose_sum = np.sum(np.sum(dose_data, axis=2), axis=0)[::-1]

        # Generate depth axis (absci)
        absci = np.arange(len(dose_sum)) * new_spacing[1] + new_spacing[1] / 2

        # Interpolate to match the depth of IDD
        interp_func = interp1d(absci, dose_sum, kind='linear', fill_value='extrapolate')
        mc_dose = interp_func(idd[:, 0])
        mc_dose[mc_dose < 0] = 0  # Ensure dose is non-negative

        # Normalize using trapezoidal integration
        norm_factor = trapz_with_sort(idd[:, 0], mc_dose)
        if norm_factor != 0:
            mc_dose = mc_dose / norm_factor

        return mc_dose

    @staticmethod
    def import_mhd_data(filename):
        """
        Import MHD data and metadata from a file.

        Parameters:
        - filename (str): Path to the .mhd file

        Returns:
        - data_info (dict): Dictionary with metadata (size, spacing, origin, etc.)
        - data_data (np.ndarray): 3D array of image data
        """
        # Check if file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found!")

        # Read MHD file
        print(f"Read MHD file: {filename}")

        # Use SimpleITK to read the file
        reader = sitk.ImageFileReader()
        reader.SetFileName(filename)
        image = reader.Execute()

        # Extract metadata into data_info
        data_info = {
            'Size': list(image.GetSize()),  # [x, y, z]
            'ElementSpacing': list(image.GetSpacing()),  # [x, y, z]
            'Offset': list(image.GetOrigin()),  # [x, y, z]
            'Direction': list(image.GetDirection()),  # 3x3 matrix as flat list
            'PixelType': sitk.GetPixelIDValueAsString(image.GetPixelID()),
            'MetaData': {key: image.GetMetaData(key) for key in image.GetMetaDataKeys()}
        }

        # Get image data as NumPy array
        data_data = sitk.GetArrayFromImage(image)  # Shape: [z, y, x]

        data_data = np.flip(data_data, axis=0)
        data_data = np.flip(data_data, axis=1)
        # data_data = np.transpose(data_data, (2, 1, 0))
        return data_info, data_data

    @staticmethod
    def create_ad_plan_file(work_dir, is_one_spot, nb_spots, centre, energy, coord=None):
        """
        Generate treatment plan file plan_AD.txt.

        Parameters:
        WorkDir (str): working directory path
        speed_mode (int): mode selection (0 for multi-spot mode, 1 for single spot mode)
        nb_spots (int): total number of spots
        centre (array-like): isocenter position [x, y, z]
        energybdl (array-like): energy array
        num_energ (int): current energy index
        coord (array-like, optional): spot coordinate array, shape (nb_spots, 3), only used when speed_mode=0

        Return:
        None (generate file)
        """
        file_path = os.path.join(work_dir, 'plan_AD.txt')

        with open(file_path, 'w', newline='\n', encoding='utf-8') as fid:
            if is_one_spot == False:
                # Multi-spot mode
                fid.write('#TREATMENT-PLAN-DESCRIPTION\n')
                fid.write('#PlanName\nplan_AD\n')
                fid.write('#NumberOfFractions\n1\n')
                fid.write('##FractionID\n1\n')
                fid.write('##NumberOfFields\n1\n')
                fid.write('###FieldsID\n1\n')
                fid.write('#TotalMetersetWeightOfAllFields\n%d\n\n' % nb_spots)
                fid.write('#FIELD-DESCRIPTION\n')
                fid.write('###FieldID\n1\n')
                fid.write('###FinalCumulativeMeterSetWeight\n%d\n' % nb_spots)
                fid.write('###GantryAngle\n0\n')
                fid.write('###PatientSupportAngle\n0\n')
                fid.write('###IsocenterPosition\n%f\t%f\t%f\n' % (centre[0], centre[1], centre[2]))
                fid.write('###NumberOfControlPoints\n1\n\n')
                fid.write('#SPOTS-DESCRIPTION\n')
                fid.write('####ControlPointIndex\n1\n')
                fid.write('####SpotTunnedID\n1\n')
                fid.write('####CumulativeMetersetWeight\n%d\n' % nb_spots)
                fid.write('####Energy (MeV)\n%f\n' % energy)
                fid.write('####NbOfScannedSpots\n%d\n' % nb_spots)
                fid.write('####X Y Weight\n')

                if coord is not None:
                    for spot in coord:
                        fid.write('%f %f %f\n' % (spot[0], spot[1], spot[2]))
            else:
                # Single spot mode
                fid.write('#TREATMENT-PLAN-DESCRIPTION\n')
                fid.write('#PlanName\nOneSpot\n')
                fid.write('#NumberOfFractions\n1\n')
                fid.write('##FractionID\n1\n')
                fid.write('##NumberOfFields\n1\n')
                fid.write('###FieldsID\n1\n')
                fid.write('#TotalMetersetWeightOfAllFields\n1\n\n')
                fid.write('#FIELD-DESCRIPTION\n')
                fid.write('###FieldID\n1\n')
                fid.write('###FinalCumulativeMeterSetWeight\n1\n')
                fid.write('###GantryAngle\n0\n')
                fid.write('###PatientSupportAngle\n0\n')
                fid.write('###IsocenterPosition\n%f\t%f\t%f\n' % (centre[0], centre[1], centre[2]))
                fid.write('###NumberOfControlPoints\n1\n\n')
                fid.write('#SPOTS-DESCRIPTION\n')
                fid.write('####ControlPointIndex\n1\n')
                fid.write('####SpotTunnedID\n1\n')
                fid.write('####CumulativeMetersetWeight\n1\n')
                fid.write('####Energy (MeV)\n%f\n' % energy)
                fid.write('####NbOfScannedSpots\n1\n')
                fid.write('####X Y Weight\n0 0 1\n')

