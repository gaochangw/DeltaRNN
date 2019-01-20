import csv
import matplotlib.pyplot as plt
import numpy as np


def write_log(logfile, log_dict):
    """
    Writes a '.csv' logfile
    :param logfile (str): output filename
    :param log_dict (ordered dictionary): dictionary that contains parameters to log
    :return: None
    """
    with open(logfile, 'a') as f:
        c = csv.writer(f)
        if log_dict['epoch'] == 0:  # write header for first epoch (dubbed as 0th epoch)
            c.writerow(log_dict.keys())

        c.writerow(log_dict.values())
