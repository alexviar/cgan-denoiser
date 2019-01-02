# -*- coding: utf-8 -*-
"""
oasis.py

Methods to prepare data from OASIS Brains (https://www.oasis-brains.org/)

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function
from config import Config as config
import data_processing
import artefacts
import utils
import nibabel as nib
import numpy as np
import os


def get_scan_paths(data_path='.', file_type='nii', scan_type='', selected_runs=(1,)):
    """
    Find the path for all scan files matching certain criteria.

    Arguments:
        data_path: string. Top-level path to search for files.
        file_type: string. File extension for the desired scans.
        scan_type: string. Type of scan, e.g. T1w, T2w, bold, FLAIR.
        selected_runs: tuple of int. If a scan has multiple runs 
            (indicated with a run number in the filename), select only 
            these run numbers.
    Returns:
        Valid scan file paths as a list of strings.
    """
    # Find MR data files
    data_files = []
    for root, dirs, files in os.walk(data_path):
        for cfile in files:
            if ".{}".format(file_type) in cfile:
                data_file = os.path.join(root, cfile)
                data_files.append(data_file)

    # Get specific data files for experiment
    exp_data_files = []
    for data_file in data_files:
        # Check for correct scan type e.g. T1-weighted, FLAIR
        if scan_type not in data_file:
            continue
        # Check if there are multiple runs and if this file is the desired run
        run_pos = data_file.find('run-')
        if run_pos != -1:
            try:
                run_num = int(data_file[run_pos + 4: run_pos + 6])
                if run_num not in selected_runs:
                    continue
            except TypeError:
                print('Expected a run number: {}'.format(
                    data_file[run_pos + 4: run_pos + 6]))
        # All clear, this is one of the files we want
        exp_data_files.append(data_file)

    return exp_data_files


def prepare_oasis3_dataset():
    print("Preparing OASIS3 data...")
    
    data_path = '/home/ben/projects/honours/datasets/oasis3/exp2'
    data_files = get_scan_paths(data_path, 'nii', 'T1w', (1, 2))
    num_scans = 30
    slice_min = 130
    slice_max = 170
    num_slice = abs(slice_max - slice_min)
    n = 256

    # Check for consistent dimensions
    count = 0
    accepted_files = []
    for i, data_file in enumerate(data_files):
        # print(nib.load(data_file).get_fdata().shape)
        if nib.load(data_file).get_fdata().shape == (176, 256, 256):
            if count >= num_scans:
                break
            count += 1
            accepted_files.append(data_file)
    print("Found {} volumes matching criteria".format(count))

    data = np.zeros((len(accepted_files) * num_slice, n, n), 
        dtype=np.float32)

    for i, data_file in enumerate(accepted_files):
        nimg = nib.load(data_file)
        # Get the 3D image data (series of grayscale slices)
        scan = nimg.get_data().astype(np.float32)
        scan_transposed = np.transpose(scan, (2, 0, 1))
        scan_sliced = scan_transposed[slice_min:slice_max, :, :]
        scan_bounded = data_processing.imbound(scan_sliced, bounds=(n, n), center=True)
        data[num_slice*i: num_slice*(i+1)] = scan_bounded

    data_artefact = artefacts.add_turbulence(data)

    data_combined = np.zeros((data.shape[0], data.shape[1], 2*data.shape[2]))
    save_path = os.path.join(config.data_path, 'train')
    utils.safe_makedirs(save_path)
    save_path = os.path.join(save_path, '{}.jpg')  # to be formatted
    for i, (raw_label, raw_input) in enumerate(zip(data, data_artefact)):
        raw_label_formatted = data_processing.normalise(raw_label, new_range=(0, 255)).astype(np.int)
        raw_input_formatted = data_processing.normalise(raw_input, new_range=(0, 255)).astype(np.int)
        data_combined[i, :, :config.raw_size] = raw_label_formatted
        data_combined[i, :, config.raw_size:2*config.raw_size] = raw_input_formatted
        utils.imsave(data_combined[i], save_path.format(i))
    print('Finished loading')


if __name__ == '__main__':
    prepare_oasis3_dataset()