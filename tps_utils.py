import numpy as np
import os, re

class EclipseMeas:
    """Class for storing Eclipse proton therapy measurement parameters"""
    idd_data = []
    spot_profile_data={}
    all_energies = []
    skip_simu = False
    def __init__(self, idd_file, spot_profile_file,calibration_depth=20.0, isocenter_depth=0.0, field_size=(0.0, 0.0),
                 spots_spacing=(1.0, 1.0), sad_factor=1.0, ic_correction_factor=1.0,skip_simu=False):
        """
        Initialize Eclipse proton therapy measurement parameters
        :param calibration_depth: Depth used for absolute dose calibration (mm)
        :param isocenter_depth: Isocenter depth during measurement (mm)
        :param field_size: Observation field size (mm), [0,0] represents single point measurement
        :param spots_spacing: Spot spacing (mm)
        :param sad_factor: SAD calculation factor, default 1.0
        :param ic_correction_factor: Ionization chamber correction factor, default 1.0
        """
        self.calibration_depth = calibration_depth
        self.isocenter_depth = isocenter_depth
        self.field_size = tuple(field_size)
        self.spots_spacing = tuple(spots_spacing)
        self.sad_factor = sad_factor
        self.ic_correction_factor = ic_correction_factor
        self.idd_file = idd_file
        self.spot_profile_file = spot_profile_file
        self.name = 'Eclipse'
        self.Isocenter_depth_ES = 0 # only for Eclipse measurements: depth of isocenter during measurements
        self.integration_diameter = 120 # only for Eclipse measurements: diameter of IC used for measurements (or integration diameter if simulations instead of measurements)
        self.skip_simu = skip_simu

    def parse_spot_profile_file(self):
        """
        Read and parse spot profile data from an Eclipse-format file.

        Parameters:
            file_path (str): Path to the measurement data file
            app: optional, GUI application object with UIFigure attribute for alerts (default None)

        Returns:
            dict: Dictionary containing:
                - depths (np.ndarray): Unique depth values [mm]
                - all_depths (np.ndarray): All depth values [mm]
                - all_energies (np.ndarray): All energy values [MeV]
                - all_curves (list): All curve types [X/Y]
                - data_blocks (list): Arrays for each data block, shape (n_points, 3)
                - block_indices (list): Row indices corresponding to each data block
        """
        print(f'parse_spot_profile_file: {self.spot_profile_file}')
        # Read file content
        with open(self.spot_profile_file[0], 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        with open(self.spot_profile_file[1], 'r', encoding='utf-8') as f:
            lines += f.read().splitlines()
        # Find key lines (Eclipse format)
        mask_begin = np.array([i for i, line in enumerate(lines) if '<' in line])
        mask_end = np.array([i for i, line in enumerate(lines) if '>' in line])
        mask_energy = np.array([i for i, line in enumerate(lines) if 'ENERGY' in line])
        mask_depth = np.array([i for i, line in enumerate(lines) if 'ZPOS' in line])
        mask_xy = np.array([i for i, line in enumerate(lines) if 'TYPE' in line])

        # Adjust mask_begin and mask_end to ensure proper block delineation
        if len(mask_begin) > 1:
            dmb = np.diff(mask_begin)
            ind = np.where(dmb > 1)[0]
            mask_begin = np.concatenate(([mask_begin[0]], mask_begin[ind+1]))
        if len(mask_end) > 1:
            dmb = np.diff(mask_end)
            mask_end = np.concatenate((mask_end[np.where(dmb > 1)[0]], [mask_end[-1]]))

        # Validation checks
        if len(mask_depth) == 0:
            print("No spot profile data found. Abort spot profile import.")
            return None
        elif len(mask_depth) != len(mask_xy) or len(mask_depth) != len(mask_energy):
            print("Incorrect amount of data. Abort spot profile import.")
            return None

        # Extract depths, energies, and curve types
        all_depths = np.array([float(re.sub(r'[^\-0-9.]', '', lines[i])) for i in mask_depth])
        all_energies = np.array([float(re.sub(r'[^0-9.]', '', lines[i])) for i in mask_energy])
        all_curves = [lines[i].split()[-1] for i in mask_xy]  # Last word is curve type
        depths = np.unique(all_depths)

        # Pre-read all data blocks
        data_blocks = []
        block_indices = []
        for begin, end in zip(mask_begin, mask_end):
            # Extract lines between begin and end, removing < and >, and strip whitespace
            z = [re.sub(r'[<>]', '', line).strip() for line in lines[begin:end + 1] if '<' in line and '>' in line]
            # Convert each line to a list of floats
            data = np.array([list(map(float, line.split())) for line in z if line], dtype=float)
            if data.size > 0:  # Only append non-empty blocks
                data_blocks.append(data)
                block_indices.append(begin)  # Record original row index

        # Construct output dictionary
        self.spot_profile_data = {
            'depths': depths,
            'all_depths': all_depths,
            'all_energies': all_energies,
            'all_curves': all_curves,
            'data_blocks': data_blocks,
            'block_indices': block_indices
        }

    def parse_idd_file(self, gantry, tps):
        """
        Parse a depth-dose data file and store IDD data in a dictionary by energy.

        Parameters:
        - file_path (str): Path to the data file

        Returns:
        - Dict[str, List[Tuple[float, float]]]: Dictionary with energy as keys and list of (depth, dose) tuples as values
        """
        print(f'parse_eclipse_idd_file: {self.idd_file}')
        energy_data = {}
        current_energy = None
        idd_data = []

        with open(self.idd_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        for line in lines:
            line = line.strip()

            # Match %ENERGY line to start a new block
            energy_match = re.match(r'%ENERGY\s+(\d+\.\d+)', line)
            if energy_match:
                # If we were already collecting data, save the previous block
                if current_energy and idd_data:
                    energy_data[current_energy] = idd_data
                # Start new block
                current_energy = energy_match.group(1)
                idd_data = []
                continue

            # End of block
            if line == '$ENOM' and current_energy:
                if idd_data:
                    energy_data[current_energy] = idd_data
                current_energy = None
                idd_data = []
                continue

            # Parse data lines within a block (e.g., <+008.2 +000.0 +000.0 +000.374>)
            data_match = re.match(r'<([-+]\d+\.\d+)\s+([-+]\d+\.\d+)\s+([-+]\d+\.\d+)\s+([-+]\d+\.\d+)>', line)
            if data_match and current_energy:
                depth = float(data_match.group(1))  # First column: depth (mm)
                dose = float(data_match.group(4))  # Fourth column: dose (relative value)
                idd_data.append([depth, dose])

        # Handle the case where file ends without $ENOM (unlikely here, but for robustness)
        if current_energy and idd_data:
            energy_data[current_energy] = idd_data

        dict_list = [{"energy": float(k), "idd": np.array(v)} for k, v in energy_data.items()]

        meas_depths = np.ones(len(dict_list)) * tps.Calibration_depth
        isoc_depth = np.ones(len(dict_list)) * tps.Isocenter_depth  # 假设 Iso_depth 是一个标量
        # Calculate the number of spots based on field_size and spot_spacing
        if len(tps.Field_size) == 2:  # 检查 field_size 是否为 nxn 形式
            nbspots1 = round(tps.Field_size[0] / tps.Spots_spacing[0] + 1)
            nbspots2 = round(tps.Field_size[1] / tps.Spots_spacing[1] + 1)
        else:
            nbspots1 = round(tps.Field_size[0] / tps.Spots_spacing[0] + 1)
            nbspots2 = nbspots1
        Dose_meas = np.zeros(len(dict_list))
        for i in range(len(dict_list)):
            # Calculate the spot spacing (spacing at the isocenter)
            spacingx = (gantry.vsad_xy[0] + meas_depths[i] - isoc_depth[i]) / gantry.vsad_xy[0] * tps.Spots_spacing[0]
            spacingy = (gantry.vsad_xy[1] + meas_depths[i] - isoc_depth[i]) / gantry.vsad_xy[1] * tps.Spots_spacing[-1]
            MUfactor = 1 / (spacingx * spacingy)
            scaling =  MUfactor / (nbspots1 * nbspots2) / tps.SAD_factor / tps.IC_correction_factor
            Dose_meas[i] = np.interp(meas_depths[i], dict_list[i]['idd'][:, 0], dict_list[i]['idd'][:, 1]) * scaling
            # print(scaling)
            dict_list[i]['Dose_meas'] = Dose_meas[i]


        self.idd_data = dict_list
        self.all_energies = [x['energy'] for x in dict_list]

    def __repr__(self):
        return (f"EclipseMeasurement(Calibration Depth={self.calibration_depth} mm, "
                f"Isocenter Depth={self.isocenter_depth} mm, Field Size={self.field_size} mm, "
                f"Spots Spacing={self.spots_spacing} mm, SAD Factor={self.sad_factor}, "
                f"IC Correction Factor={self.ic_correction_factor})")

class RayStationMeas:
    """Class for storing RayStation proton therapy measurement parameters"""
    idd_data=[]
    absolute_dose_data={}
    spot_profile_data={}
    all_energies = []
    skip_simu = False
    def __init__(self, idd_file, spot_profile_file, absolute_dose_file, skip_simu=False):
        """
        Initialize Eclipse proton therapy measurement parameters
        :param calibration_depth: Depth used for absolute dose calibration (mm)
        :param isocenter_depth: Isocenter depth during measurement (mm)
        :param field_size: Observation field size (mm), [0,0] represents single point measurement
        :param spots_spacing: Spot spacing (mm)
        :param sad_factor: SAD calculation factor, default 1.0
        :param ic_correction_factor: Ionization chamber correction factor, default 1.0
        """
        self.idd_file = idd_file
        self.spot_profile_file = spot_profile_file
        self.absolute_dose_file = absolute_dose_file
        self.name = 'RayStation'
        self.skip_simu = skip_simu

    def parse_spot_profile_file(self):
        """
        Read and parse the fluence data in the measurement file.

        Parameters:
        file_path (str): measurement data file path

        Returns:
        dict: dictionary containing the following key values:
        - depths (np.ndarray): unique depth value [mm]
        - all_depths (np.ndarray): all depth values ​​[mm]
        - all_energies (np.ndarray): all energy values ​​[MeV]
        - all_curves (list): all curve types [X/Y]
        - data_blocks (list): array of each data block, shape (n_points, 3)
        - block_indices (list): row index corresponding to each data block
        """
        # Read file content
        print(f'parse_spot_profile_file: {self.spot_profile_file}')
        with open(self.spot_profile_file, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()

        mask_begin = [i for i, line in enumerate(lines) if 'Measured fluence curve' in line]
        mask_end = [i for i, line in enumerate(lines) if 'End:' in line]
        mask_energy = [i for i, line in enumerate(lines) if 'Nominal beam energy' in line]
        mask_depth = [i for i, line in enumerate(lines) if 'Air profile depth' in line]
        mask_xy = [i for i, line in enumerate(lines) if 'Curve type' in line]

        all_depths = np.array([float(re.sub(r'[^-0-9.]', '', lines[i])) for i in mask_depth])
        all_energies = np.array([float(re.sub(r'[^0-9.]', '', lines[i])) for i in mask_energy])
        all_curves = [lines[i].split()[-1] for i in mask_xy]  # 最后一词为曲线类型
        depths = np.unique(all_depths)

        data_blocks = []
        block_indices = []
        for begin, end in zip(mask_begin, mask_end):
            z = [re.sub(r'[;]', '', line).strip() for line in lines[begin + 1:end]]
            data = np.array([list(map(float, line.split())) for line in z])
            data_blocks.append(data)
            block_indices.append(begin)

        self.spot_profile_data = {
            'depths': depths,
            'all_depths': all_depths,
            'all_energies': all_energies,
            'all_curves': all_curves,
            'data_blocks': data_blocks,
            'block_indices': block_indices
        }

    def parse_absolute_dose_file(self):
        """
        Extract parameters from RayStation measurement data file.

        Parameters:
        file_path (str): measurement data file path

        Returns:
        dict: dictionary with the following keys:
        - meas_energies (np.ndarray): measurement energy [MeV]
        - meas_depths (np.ndarray): measurement depth [mm]
        - isoc_depth (np.ndarray): distance from isocenter to phantom surface [mm]
        - dose_meas (np.ndarray): measurement dose [Gy/MU]
        - spot_spacing (float): spot spacing [mm]
        - field_size (list): scan field size [x, y] [mm]
        """
        print(f'parse_ad_file: {self.absolute_dose_file}')
        with open(self.absolute_dose_file, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()

        result = {}

        spot_spacing_line = next((line for line in lines if 'Spot spacing [mm]' in line), None)
        if spot_spacing_line:
            result['spot_spacing'] = float(re.search(r'[\d.]+', spot_spacing_line).group())

        field_size_line = next((line for line in lines if 'Scanned field size (x y) [mm]' in line), None)
        if field_size_line:
            result['field_size'] = [float(x) for x in re.findall(r'[\d.]+', field_size_line)]

        start_idx = next(i for i, line in enumerate(lines) if 'Measurement (Nominal' in line) + 1
        end_idx = next(i for i, line in enumerate(lines) if 'End:' in line)
        data_lines = lines[start_idx:end_idx]

        Data = np.array([list(map(float, re.findall(r'[\d.]+', line))) for line in data_lines])

        result['meas_energies'] = Data[:, 0]
        result['meas_depths'] = Data[:, 1]

        # Determine whether the isocenter depth is global or row by row
        isoc_depth_label_idx = next(
            (i for i, line in enumerate(lines) if 'Isocenter to phantom surface distance' in line), None)
        if isoc_depth_label_idx < start_idx - 1:  # Global isocenter depth
            isoc_depth_value = float(re.search(r'[\d.]+', lines[isoc_depth_label_idx]).group())
            result['isoc_depth'] = isoc_depth_value * np.ones(len(result['meas_energies']))
            result['Dose_meas'] = Data[:, 2] / 100  # Dose conversion from cGy/MU to Gy/MU
        else:  # Separate isocenter depth for each row
            result['isoc_depth'] = Data[:, 2]
            result['Dose_meas'] = Data[:, 3] / 100

        self.absolute_dose_data = result

    def parse_idd_file(self):
        print(f'parse_idd_file: {self.idd_file}')
        file_content = open(self.idd_file, encoding='utf-8').read().split('\n')
        data = {}
        current_energy = None
        capture_data = False

        isoc_depth = -1
        IC_diameter = -1
        for line in file_content:
            stripped_line = line.strip()
            if stripped_line.startswith('Isocenter to phantom surface distance'):
                isoc_depth = float(stripped_line.split(';')[-1].strip())
            if stripped_line.startswith('Detector lateral side/diameter'):
                IC_diameter = float(stripped_line.split(';')[-1].strip())
            # Detection energy identifier (e.g. "Nominal beam energy [MeV]:; 70")
            if stripped_line.startswith('Nominal beam energy [MeV]:;'):
                current_energy = stripped_line.split(';')[-1].strip()
                data[current_energy] = []

            # Detect the start mark of the data segment
            elif stripped_line.startswith('Measured dose curve'):
                capture_data = True
            # Data segment end marker
            elif stripped_line == 'End:;':
                capture_data = False
            # Data row parsing
            elif capture_data and current_energy is not None:
                if ';' in stripped_line:
                    parts = stripped_line.split(';')
                    if len(parts) >= 2:
                        try:
                            depth = float(parts[0].strip())
                            dose = float(parts[1].strip())
                            data[current_energy].append((depth, dose))
                        except ValueError:
                            pass  # Ignore formatting errors
        dict_list = [{"energy": float(k), "idd": np.array(v)} for k, v in data.items()]
        self.idd_data = dict_list
        self.IC_diameter_es = IC_diameter
        self.isocenter_depth = isoc_depth
        self.all_energies = [x['energy'] for x in dict_list]