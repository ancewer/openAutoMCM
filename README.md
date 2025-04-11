# AutoMCM Usage Instructions

## Overview

AutoMCM is an open-source, Python-based tool designed to automate beam commissioning for pencil beam scanning (PBS) proton therapy across multiple Monte Carlo (MC) engines, such as MCsquare and TOPAS. This document provides instructions for using the script to configure, tune, and validate beam models based on treatment planning system (TPS) data.
(This code is partially adapted from the matlab implementation in https://gitlab.com/openmcsquare/commissioning/.)
## Prerequisites

### Software Requirements
- **MC Engines**:
  - **MCsquare**: Download and place the executable in `MCsquare/MCsquare`.
  - **TOPAS**: Install and configure in your pc (ensure compatibility with your system).
- **Operating System**: Compatible with Windows and Linux.

### Directory Structure
Organize your project directory as follows:
```
project_root/
├── MachineData/
│   ├── Sample_data_Eclipse/
│   └── Sample_data_RayStation/
├── MCsquare/
│   ├── Work_Dir_MC2/
│   ├── MCsquare (executable)
│   └── template/
├── Topas/
│   ├── Work_Dir_Topas/
│   └── template/
├── BDL/
└── script.py
```

## Script Overview

The script defines a `BeamTuner` class to handle beam commissioning steps:
- **BDL File Creation**: Generates beam definition files.
- **Phase Space Optimization**: Tunes spot size and divergence.
- **Energy Spectrum Optimization**: Adjusts energy spread.
- **Absolute Dose Calibration**: Calibrates protons per monitor unit (MU).

The `main()` function configures the environment and executes the tuning process.

## Usage Instructions

### 1. Setup Configuration
Modify the `main()` function to match your setup:

#### Gantry Settings
```python
gantry = MyClass()
gantry.name = 'GTR5'  # Gantry identifier
gantry.nozzle2iso = 450  # Nozzle-to-isocenter distance (mm)
gantry.vsad_xy = [2950, 9100]  # Virtual source-to-axis distances (mm)
gantry.energies_to_tune = [70]  # Energies to commission (MeV)
gantry.es_goal = 0.01  # Energy spectrum optimization goal
gantry.es_e_dif = 2  # Energy difference tolerance
gantry.es_es_range = [0.01, 1.0]  # Energy spread range
```

#### MC Engine Selection
```python
mc_engine = 'mcsquare'  # Options: 'mcsquare' or 'topas'
```
- **MCsquare Config**:
  ```python
  work_dir = os.path.join(cwd_path, r'MCsquare\Work_Dir_MC2')
  program_path = os.path.join(cwd_path, r'MCsquare\MCsquare')
  bdl_file = os.path.join(bdl_dir, f'BDL_{gantry.name}_{mc_engine}_{tps_name}.txt')
  config_file = os.path.join(cwd_path, r'MCsquare\template\ConfigTemplate.txt')
  ct_template_path = os.path.join(cwd_path, r'MCsquare\template\ct_mhd\CT\CT.mhd')
  mc_config = MC2ConfigObject(work_dir, program_path, config_file, ct_template_path, bdl_file)
  mc_config.simu_spacing_es = [2, 0.5, 2]  # Voxel spacing for energy spectrum (mm)
  mc_config.simu_spacing_ad = [1, 1, 1]  # Voxel spacing for absolute dose (mm)
  mc_config.IC_diameter_ad = [9.9, 0.6]  # Ion chamber diameter (mm)
  mc_config.simu_es_protons = 5e5  # Protons for initial energy spectrum
  mc_config.simu_es_protons_final = 1e6  # Protons for final energy spectrum
  mc_config.simu_ad_protons = 1e6  # Protons for absolute dose
  ```
- **TOPAS Config**:
  ```python
  work_dir = os.path.join(cwd_path, r'Topas\Work_Dir_Topas')
  program_path = os.path.join(cwd_path, r'*')
  bdl_file = os.path.join(bdl_dir, f'BDL_{gantry.name}_{mc_engine}_{tps_name}.txt')
  mc_config = TopasConfigObject(work_dir, program_path, bdl_file)
  mc_config.config_pbs_beam_file = os.path.join(cwd_path, r'Topas\template\pbs_beam.txt')
  mc_config.config_pbs_idd_file = os.path.join(cwd_path, r'Topas\template\pbs_idd.txt')
  mc_config.config_pbs_ad_file = os.path.join(cwd_path, r'Topas\template\pbs_ad.txt')
  mc_config.simu_spacing_es = [None, 1, None]  # Voxel spacing
  mc_config.IC_diameter_ad = [9.9, 0.6]  # Ion chamber diameter
  scaling = 10  # Adjust proton scaling if needed
  mc_config.simu_es_protons = 5e5 / scaling
  mc_config.simu_es_protons_final = 1e6 / scaling
  mc_config.simu_ad_protons = 1e7 / scaling
  ```

#### TPS Data
```python
tps_name = 'raystation'  # Options: 'raystation' or 'eclipse'
```
- **RayStation**:
  ```python
  data_dir = r'MachineData\Sample_data_RayStation'
  idd_file = os.path.join(data_dir, 'Measured_*_PristineBraggPeaks.csv')
  spot_profile_file = os.path.join(data_dir, 'Measured_*_SpotProfiles.csv')
  absolute_dose_file = os.path.join(data_dir, 'Measured_*_AbsoluteDosimetry.csv')
  tps = RayStationMeas(idd_file, spot_profile_file, absolute_dose_file, skip_simu=False)
  ```
- **Eclipse**:
  ```python
  data_dir = r'MachineData\Sample_data_Eclipse\data\*'
  idd_file = os.path.join(data_dir, '*_Measured Depth Dose (raw).txt')
  spot_profile_file = [os.path.join(data_dir, '*_Measured Spot Fluence Profile X.txt'),
                       os.path.join(data_dir, '*_Measured Spot Fluence Profile Y.txt')]
  tps = EclipseMeas(idd_file, spot_profile_file)
  ```

### 2. Running the Script
1. Save the script as `beam_tuner.py`.
2. Ensure all paths and files exist as configured.
3. Execute:
   ```bash
   python beam_tuner.py
   ```

### 3. Tuning Steps
Uncomment desired steps in `run_all()`:
```python
def run_all(self):
    self.create_bdl_file()  # Generate BDL file
    self.calc_phase_space()  # Optimize phase space
    self.opt_energy_spectrum()  # Tune energy spectrum
    self.calc_absolute_dose()  # Calibrate absolute dose
```
- Current script runs only `calc_absolute_dose()`. Adjust as needed.

### 4. Output
- **Logs**: Check console or log file (configured in `opt_funs.py`) for progress and timing.
- **Files**: Results saved in `work_dir` (e.g., `MCsquare/Work_Dir_MC2`).

## Troubleshooting
- **Missing Files**: Verify paths to MC executables, templates, and TPS data.
- **Errors**: Ensure `opt_funs.py` and `tps_utils.py` are in the same directory and properly configured.
- **Performance**: Adjust proton counts (e.g., `simu_es_protons`) for faster runs, balancing accuracy.

## Notes
- **Customization**: Modify `gantry.energies_to_tune` for multiple energies.
- **Dependencies**: Ensure `os`, `time`, and `logging` are available (standard libraries).
- **Support**: For issues, consult the AutoMCM documentation or community resources.
