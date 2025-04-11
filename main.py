from opt_funs import *
from tps_utils import *
import time
import logging

# Logger is already set up in opt_funs.py, so we can use it here
logger = logging.getLogger()

class BeamTuner:
    """A class to handle beam tuning processes for MC simulations."""

    def __init__(self, gantry, engine, tps, mc_config):
        """Initialize the BeamTuner with default paths and settings."""
        self.gantry = gantry
        self.engine = engine.lower()
        self.tps = tps
        self.tps.name = self.tps.name.lower()
        self.mc_config = mc_config
        logger.info("Initialized BeamTuner.")

    @timeit
    def create_bdl_file(self):
        """Create the BDL template."""
        logger.info(f"Creating BDL template: {self.mc_config.bdl_file}")
        create_bdl_file(self.gantry, self.tps, self.mc_config)
        logger.info("BDL template created.")

    @timeit
    def calc_phase_space(self):
        """Optimize phase space parameters."""
        logger.info("Starting phase space optimization...")
        calc_phase_space(self.gantry, self.tps, self.mc_config)
        logger.info("Phase space optimization completed.")

    @timeit
    def opt_energy_spectrum(self):
        """Optimize energy spectrum parameters."""
        logger.info("Starting energy spectrum optimization...")
        opt_energy_spectrum(self.gantry, self.tps, self.mc_config, self.engine)
        logger.info("Energy spectrum optimization completed.")

    @timeit
    def calc_absolute_dose(self):
        """Optimize protons per MU."""
        logger.info("Starting absolute dose calculation...")
        calc_absolute_dose(self.gantry, self.tps, self.mc_config, self.engine)
        logger.info("Absolute dose calculation completed.")

    def run_all(self):
        """Execute all tuning steps sequentially."""
        start_time = time.time()
        logger.info("Starting run_all process...")

        # self.create_bdl_file()
        # self.calc_phase_space()
        # self.opt_energy_spectrum()
        self.calc_absolute_dose()

        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Total execution time for run_all: {total_time:.2f} seconds.")

def main():
    start_time = time.time()
    logger.info("Starting main program...")

    cwd_path = os.getcwd()

    class MyClass:
        pass

    gantry = MyClass()
    gantry.name = 'GTR5'
    gantry.nozzle2iso = 450
    gantry.vsad_xy = [2950, 9100]
    gantry.energies_to_tune = [70]
    gantry.es_goal = 0.01
    gantry.es_e_dif = 2
    gantry.es_es_range = [0.01, 1.0]

    skip_simu = False
    tps_name = 'raystation'
    mc_engine = 'mcsquare'

    if mc_engine.lower() == 'mcsquare':
        work_dir = os.path.join(cwd_path, r'MCsquare\Work_Dir_MC2')
        os.makedirs(os.path.dirname(work_dir), exist_ok=True)
        program_path = os.path.join(cwd_path, r'MCsquare\MCsquare')
        bdl_dir = os.path.join(cwd_path, 'BDL')
        config_file = os.path.join(cwd_path, r'MCsquare\template\ConfigTemplate.txt')
        ct_template_path = os.path.join(cwd_path, r'MCsquare\template\ct_mhd\CT\CT.mhd')
        bdl_file = os.path.join(bdl_dir, f'BDL_{gantry.name}_{mc_engine}_{tps_name}.txt')
        mc_config = MC2ConfigObject(work_dir, program_path, config_file, ct_template_path, bdl_file)
        mc_config.simu_spacing_es = [2, 0.5, 2]
        mc_config.simu_spacing_ad = [1, 1, 1]
        mc_config.IC_diameter_ad = [9.9, 0.6]
        mc_config.simu_es_protons = 5e5
        mc_config.simu_es_protons_final = 1e6
        # mc_config.simu_ad_speed = True
        mc_config.simu_ad_protons = 1e6
    elif mc_engine.lower() == 'topas':
        work_dir = os.path.join(cwd_path, r'Topas\Work_Dir_Topas')
        os.makedirs(os.path.dirname(work_dir), exist_ok=True)
        program_path = os.path.join(cwd_path, r'Topas\*')
        bdl_dir = os.path.join(cwd_path, 'BDL')
        bdl_file = os.path.join(bdl_dir, f'BDL_{gantry.name}_{mc_engine}_{tps_name}.txt')
        mc_config = TopasConfigObject(work_dir, program_path, bdl_file)
        mc_config.config_pbs_beam_file = os.path.join(cwd_path, r'Topas\template\pbs_beam.txt')
        mc_config.config_pbs_idd_file = os.path.join(cwd_path, r'Topas\template\pbs_idd.txt')
        mc_config.config_pbs_ad_file = os.path.join(cwd_path, r'Topas\template\pbs_ad.txt')
        mc_config.simu_spacing_es = [None, 1, None]
        mc_config.IC_diameter_ad = [9.9, 0.6]
        scaling = 10
        mc_config.simu_es_protons = 5e5 / scaling
        mc_config.simu_es_protons_final = 1e6 / scaling
        mc_config.simu_ad_protons = 1e7 / scaling

    if tps_name.lower().__contains__('eclipse'):
        data_dir = r'MachineData\Sample_data_Eclipse\data\AZ_VAC'
        idd_file = os.path.join(data_dir, 'VAC_Option - OPTION01_Measured Depth Dose (raw).txt')
        spot_profile_file = [os.path.join(data_dir, 'VAC_Beam Spot - BEAMSPOT01_Measured Spot Fluence Profile X.txt'),
                             os.path.join(data_dir, 'VAC_Beam Spot - BEAMSPOT01_Measured Spot Fluence Profile Y.txt')]
        tps = EclipseMeas(idd_file, spot_profile_file)
        tps.parse_spot_profile_file()
        tps.RBE = 1
        tps.Calibration_depth = 20
        tps.Isocenter_depth = 0
        tps.one_spot = False
        tps.Field_size = [100, 100] if not tps.one_spot else [0, 0]
        tps.Spots_spacing = [2.5, 2.5] if not tps.one_spot else [1, 1]
        tps.SAD_factor = 1
        tps.IC_correction_factor = 1
        tps.parse_idd_file(gantry, tps)
        tps.IC_diameter_es = 120
        gantry.energies_to_tune = tps.all_energies if len(gantry.energies_to_tune) == 0 else [gantry.energies_to_tune]
        tuner = BeamTuner(gantry=gantry, engine=mc_engine, tps=tps, mc_config=mc_config)
    elif tps_name.lower().__contains__('raystation'):
        data_dir = r'MachineData\Sample_data_RayStation'
        idd_file = os.path.join(data_dir, 'Measured_GTR5_PristineBraggPeaks.csv')
        spot_profile_file = os.path.join(data_dir, 'Measured_GTR5_SpotProfiles.csv')
        absolute_dose_file = os.path.join(data_dir, 'Measured_GTR5_AbsoluteDosimetry.csv')
        tps = RayStationMeas(idd_file, spot_profile_file, absolute_dose_file, skip_simu=skip_simu)
        tps.one_spot = False
        tps.RBE = 1
        tps.parse_spot_profile_file()
        tps.parse_idd_file()
        tps.parse_absolute_dose_file()
        gantry.energies_to_tune = tps.all_energies if len(gantry.energies_to_tune) == 0 else [gantry.energies_to_tune]
        tuner = BeamTuner(gantry=gantry, engine=mc_engine, tps=tps, mc_config=mc_config)

    tuner.run_all()

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total execution time for main program: {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()