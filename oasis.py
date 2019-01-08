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
import filenames
import utils
import nibabel as nib
import numpy as np
import os
import time


def preprocess(data, pad_size):
    """
    Process the data into an appropriate format for CNN-based models.
    """
    # Reverse the flipping about the x-axis caused by data reading
    data = np.flipud(data)

    # Ensure it is padded to square
    data_rows, data_cols = data.shape
    # print("Original shape", (data_rows, data_cols))
    preprocessed_data = np.zeros( (pad_size, pad_size), data.dtype)
    preprocessed_data[:data_rows, :data_cols] = data
    # print("Padded shape", preprocessed_data.shape)
    # Normalise to suitable value range
    preprocessed_data = data_processing.normalise(preprocessed_data)
    return preprocessed_data


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


def prepare_oasis3_dataset(data_path, save_artefacts=False, category='train',
        max_scans=24, skip=0, shape=(176, 256, 256), 
        slice_min=130, slice_max=170, n=256):

    print("Preparing OASIS3 data...")
    data_files = get_scan_paths(data_path, 'nii', 'T1w', (1, 2))
    num_slice = abs(slice_max - slice_min)

    # Check for consistent dimensions
    count = 0
    accepted_files = []
    for i, data_file in enumerate(data_files):
        if i < skip:
            continue
        # print(nib.load(data_file).get_fdata().shape)
        if nib.load(data_file).get_fdata().shape == (176, 256, 256):
            if count >= max_scans:
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

    if save_artefacts:
        data_combined = np.zeros((data.shape[0], data.shape[1], 2*data.shape[2]))
        start = time.time()
        data_artefact = artefacts.add_turbulence(data)
        print("Applying turbulence to {} images took {}s".format(len(data), time.time() - start))
    else:
        data_combined = np.zeros_like(data)

    save_path = os.path.join(config.data_path, category)
    utils.safe_makedirs(save_path)
    save_path = os.path.join(save_path, '{}.nii.gz')  # to be formatted
    for i, raw_label in enumerate(data):
        data_combined[i, :, :config.raw_size] = raw_label
    if save_artefacts:
        for i, raw_input in enumerate(data_artefact):
            data_combined[i, :, config.raw_size:2*config.raw_size] = raw_input
    for i in range(len(data_combined)):
        nib.save(nib.Nifti1Image(data_combined[i], np.eye(4)), save_path.format(i))
    print('Finished loading')


def get_oasis1_dataset(input_path, label_path, test_cases, max_training_cases, size):
    # Load input data
    # Get list of filenames and case IDs from path where 3D volumes are
    input_list, case_list = filenames.getSortedFileListAndCases(
        input_path, 0, "*.nii.gz", True)
    # print("Cases: {}".format(case_list))

    # Load label data
    # Get list of filenames and case IDs from path where 3D volumes are
    label_list, case_list = filenames.getSortedFileListAndCases(
        label_path, 0, "*.nii.gz", True)
    
    # Check number of inputs
    mu = len(input_list)
    if mu != len(label_list):
        print("Warning: inputs and labels don't match!")
    if mu < 1:
        print("Error: No input data found. Exiting.")
        quit()
        
    if test_cases:
        mu -= len(test_cases) #leave 1 outs
    if max_training_cases > 0:
        mu = max_training_cases
        
    print("Loading input ...")
    # Get initial input size, assumes all same size
    first = nib.load(input_list[0]).get_data().astype(np.float32)
    first = preprocess(first, size)
    # print("Input shape:", first.shape)
    input_rows, input_cols = first.shape
    
    # Use 3D array to store all inputs and labels
    train_inputs = np.ndarray((mu, input_rows, input_cols), dtype=np.float32)
    train_labels = np.ndarray((mu, input_rows, input_cols), dtype=np.float32)
    test_inputs = np.ndarray((len(test_cases), input_rows, input_cols), dtype=np.float32)
    test_labels = np.ndarray((len(test_cases), input_rows, input_cols), dtype=np.float32)
    
    # Process each 3D volume
    i = 0
    count = 0
    out_count = 0
    for input_name, label, _ in zip(input_list, label_list, case_list):
        # Load MR nifti file
        input_data = nib.load(input_name).get_data().astype(np.float32)
        input_data = preprocess(input_data, size) 
        # print("Slice shape:", input_data.shape)
        # print("Loaded", image)

        # Load label nifti file
        label = nib.load(label).get_data().astype(np.float32)
        label = preprocess(label, size) 
        # label[label > 0] = 1.0  # binary threshold labels
        # print("Slice shape:", label.shape)
        # print("Loaded", label)

        # Check for test case
        if count in test_cases:
            test_inputs[out_count] = input_data
            test_labels[out_count] = label
            count += 1
            out_count += 1
            continue

        train_inputs[i] = input_data
        train_labels[i] = label
        i += 1 
        count += 1
        
        if i == max_training_cases:
            break
    
    # labels, inputs = preprocess_train_batch(labels, inputs, 
    #                                         new_range=(-1, 1),
    #                                         current_range=None, 
    #                                         axis=None, cropping='topleft', 
    #                                         hflip=0, vflip=2)

    train_inputs = train_inputs[..., np.newaxis]
    train_labels = train_labels[..., np.newaxis]
    test_inputs = test_inputs[..., np.newaxis]
    test_labels = test_labels[..., np.newaxis]

    print("Used", i, "images for training")

    return (train_inputs, train_labels), (test_inputs, test_labels), case_list



def get_oasis1_dataset_test(input_path, label_path, test_cases, max_test_cases, size):
    # Load input data
    # Get list of filenames and case IDs from path where 3D volumes are
    input_list, case_list = filenames.getSortedFileListAndCases(
        input_path, 0, "*.nii.gz", True)
    # print("Cases: {}".format(case_list))

    # Load label data
    # Get list of filenames and case IDs from path where 3D volumes are
    label_list, case_list = filenames.getSortedFileListAndCases(
        label_path, 0, "*.nii.gz", True)
    
    # Check number of inputs
    if len(input_list) != len(label_list):
        print("Warning: inputs and labels don't match!")
    if len(input_list) < 1:
        print("Error: No input data found. Exiting.")
        quit()
        
    print("Loading input ...")
    # Get initial input size, assumes all same size
    first = nib.load(input_list[0]).get_data().astype(np.float32)
    first = preprocess(first, size)
    # print("Input shape:", first.shape)
    input_rows, input_cols = first.shape
    
    # Use 3D array to store all inputs and labels
    test_inputs = np.ndarray((len(test_cases), input_rows, input_cols), dtype=np.float32)
    test_labels = np.ndarray((len(test_cases), input_rows, input_cols), dtype=np.float32)
    
    # Process each 3D volume
    count = 0
    out_count = 0
    for input_name, label, _ in zip(input_list, label_list, case_list):
        # Check for test case
        if count in test_cases:
            # Load MR nifti file
            input_data = nib.load(input_name).get_data().astype(np.float32)
            input_data = preprocess(input_data, size) 
            # print("Slice shape:", input_data.shape)
            # print("Loaded", image)

            # Load label nifti file
            label = nib.load(label).get_data().astype(np.float32)
            label = preprocess(label, size) 
            # label[label > 0] = 1.0  # binary threshold labels
            # print("Slice shape:", label.shape)
            # print("Loaded", label)

            test_inputs[out_count] = input_data
            test_labels[out_count] = label
            out_count += 1
            if out_count == max_test_cases:
                break
        count += 1

    test_inputs = test_inputs[..., np.newaxis]
    test_labels = test_labels[..., np.newaxis]

    print("Used {} images for testing".format(out_count))

    return test_inputs, test_labels


if __name__ == '__main__':
    prepare_oasis3_dataset('/home/ben/projects/honours/datasets/oasis3/exp2', 
        category='train', max_scans=24, skip=0)
    prepare_oasis3_dataset('/home/ben/projects/honours/datasets/oasis3/exp2', 
        category='valid', max_scans=6, skip=24)
    prepare_oasis3_dataset('/home/ben/projects/honours/datasets/oasis3/exp2', 
        category='test', max_scans=9, skip=30)
