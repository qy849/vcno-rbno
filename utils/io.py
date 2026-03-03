import os
import yaml
import numpy
import pickle
import torch
import csv
import glob
import re

# YAML file operations
def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.full_load(f)

def save_yaml(path, data):
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
def extract_yaml_filenames(directory):
    return [f.split('.yaml')[0] for f in os.listdir(directory) if f.endswith('.yaml')]



# NumPy file operations
def load_npy(path):
    return numpy.load(path)

def save_npy(path, data):
    numpy.save(path, data)



# Pickle file operations
def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pkl(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def extract_pkl_filenames(directory):
    return [f.split('.pkl')[0] for f in os.listdir(directory) if f.endswith('.pkl')]



# PyTorch file operations
def load_pt(path):
    return torch.load(path)

def save_pt(path, data):
    torch.save(data, path)

def load_pth(path):
    return torch.load(path)

def save_pth(path, data):
    torch.save(data, path)


# CSV file operations (for dictionary)
def load_csv(path):
    my_dict = {}
    with open(path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            for key, value in row.items():
                my_dict.setdefault(key, []).append(value)

    my_dict = {key: numpy.array([float(value) for value in values]) for key, values in my_dict.items()}
    return my_dict

# def save_csv(path, my_dict):
#     with open(path, 'w', newline='') as file:
#         csv_writer = csv.writer(file)
#         csv_writer.writerow(my_dict.keys())
#         for row in zip(*my_dict.values()):
#             csv_writer.writerow(row)
def save_csv(path, my_dict):
    max_len = max(len(lst) for lst in my_dict.values())
    with open(path, 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(my_dict.keys())
        for i in range(max_len):
            row = []
            for key in my_dict.keys():
                try:
                    value = my_dict[key][i]
                    if value is None:
                        row.append('')
                    else:
                        row.append(value)
                except IndexError:
                    row.append('')
            csv_writer.writerow(row)

# Text file operations
def load_txt(path):
    try:
        with open(path, 'r') as file:
            lines = file.read().splitlines()
        
        if len(lines) == 1:
            # If there's only one line in the file, assume it's a single float
            return float(lines[0])
        elif len(lines) > 1:
            # If there are multiple lines, assume it's a list of floats
            return [float(line) for line in lines]
        else:
            # File is empty
            return None
    except (IOError, ValueError):
        # Handle file not found or parsing errors
        return None

def save_txt(path, data):
    # Ensure data is a list, even if it's a single number
    if not isinstance(data, list):
        data = [data]
    # Open the file for writing
    with open(path, 'w') as file:
        # Write each number to the file, one per line
        for num in data:
            file.write(f'{num}\n')


def load_derivative_m_labels(directory:str, basis_name=None):
    if basis_name is None:
        file_pattern = directory + "/derivative_m_labels_*.npy"
    else:
        file_pattern = directory + f"/{basis_name}_derivative_m_labels_*.npy"
    files = glob.glob(file_pattern)
    # Sort files based on the numerical value extracted from the filename
    def extract_number(filename):
        # Using regex to find the number before the last underscore
        match = re.search(r'(\d+)_\d+\.npy', filename)
        if match:
            return int(match.group(1))
        return 0  # In case no number is found, return 0

    files_sorted = sorted(files, key=extract_number)

    # Load each sorted file and store it in a list
    arrays = [numpy.load(file) for file in files_sorted]

    # Concatenate all the arrays along dimension 0
    all_data = numpy.concatenate(arrays, axis=0)

    return all_data


def load_derivative_x_labels(directory:str): 
    file_pattern = directory + "/derivative_x_labels_*.npy"
    files = glob.glob(file_pattern)
    # Sort files based on the numerical value extracted from the filename
    def extract_number(filename):
        # Using regex to find the number before the last underscore
        match = re.search(r'(\d+)_\d+\.npy', filename)
        if match:
            return int(match.group(1))
        return 0  # In case no number is found, return 0

    files_sorted = sorted(files, key=extract_number)

    # Load each sorted file and store it in a list
    arrays = [numpy.load(file) for file in files_sorted]

    # Concatenate all the arrays along dimension 0
    all_data = numpy.concatenate(arrays, axis=0)

    return all_data