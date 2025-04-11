import logging  # For logging functionality
import time
import numpy as np
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set up logging
def setup_logging(log_file='beam_tuning.log'):
    """
    Set up logging to output to both console and a log file.

    Parameters:
        log_file (str): Path to the log file.
    """
    # Ensure the log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # Log to file
            logging.StreamHandler()  # Log to console
        ]
    )
    logger = logging.getLogger()
    return logger

# Initialize logger
logger = setup_logging(log_file='logs/beam_tuning.log')

# Decorator to measure execution time of functions
def timeit(func):
    """
    Decorator to measure the execution time of a function.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"Function '{func.__name__}' took {execution_time:.2f} seconds to execute.")
        return result

    return wrapper

def mat_to_npy(mat_file_path, npy_file_path=None, variable_name=None):
    """
    Convert MATLAB .mat file to Python .npy file.

    Parameters:
        mat_file_path (str): Path to input .mat file.
        npy_file_path (str, optional): Path to save .npy file.
        variable_name (str, optional): Name of variable to convert.

    Raises:
        ValueError: If no valid variables in MAT file.
        KeyError: If specified variable not found in MAT file.
    """
    mat_data = loadmat(mat_file_path)

    mat_data = {k: v for k, v in mat_data.items() if not k.startswith('__')}

    if not mat_data:
        raise ValueError("no valid variables in MAT file")

    if variable_name is None:
        variable_name = list(mat_data.keys())[0]
        print(f"default {variable_name}")

    if variable_name not in mat_data:
        raise KeyError(f"{variable_name} was not found, only has {list(mat_data.keys())}")

    data = mat_data[variable_name]
    data = np.asarray(data)

    if npy_file_path is None:
        npy_file_path = mat_file_path.replace('.mat', '.npy')

    np.save(npy_file_path, data)
    print(f"{variable_name} was saved as {npy_file_path}")

def trapz_with_sort(x, y):
    """
    Encapsulates np.trapezoid, ensuring x is increasing
    Parameters:
    x: independent variable value
    y: integrated function value
    Returns:
    Integration result
    """
    x = np.asarray(x)
    y = np.asarray(y)
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    try:
        result = np.trapezoid(y_sorted, x_sorted)
    except ValueError:
        result = np.trapz(y_sorted, x_sorted)
    return result


