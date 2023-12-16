# %%
import torch
import numpy as np
import os
from utils import *

torch.cuda.is_available()

# %% Load raw data along with its bvalues
save_folder = 'synthetic'

data_files = []
bvalues_files = []
for i in range(1,17):
    for j in range(1,3):
        path = 'raw_data/{}.{}/00039.nii.gz'.format(i,j)
        if os.path.isfile(path):
            data_files.append(path)
        path = 'raw_data/{}.{}/00039.bval'.format(i,j)
        if os.path.isfile(path):
            bvalues_files.append(path)

# %%
# Make sure all files have the same sequence of bvalues
bvalues = []
for i in bvalues_files:
    text_file = np.genfromtxt(i)
    bvalues.append(np.array2string(text_file))

assert len(set(bvalues)) == 1
GLOBAL_BVALUES = np.array(np.genfromtxt(bvalues_files[0]))

# %% Individually train IVIMNET on each sample and output the predicted feature maps
for i in range(len(data_files)):
    bvals_file = bvalues_files[i]
    data_file = data_files[i]
    datatot, bvalues, valid_id, S0, sx, sy, sz, n_b_values = parse_data(
        data_file, bvals_file)

    paramsNN, _ = NN_fit(datatot, bvalues)

    # For each parameter (D, f, Dp)
    for k in range(3):
        # remove outliers
        Q1 = np.percentile(paramsNN[k], 25, method='midpoint')
        Q3 = np.percentile(paramsNN[k], 75, method='midpoint')
        IQR = Q3 - Q1
        mask_lower = paramsNN[k] < Q1 - 1.5*IQR
        mask_upper = paramsNN[k] > Q3 + 1.5*IQR
        paramsNN[k][mask_lower + mask_upper] = 0
        
        # ensure D, f, Dp are non-negative
        minimum = np.min(paramsNN[k])
        if minimum <0:
            paramsNN[k] -= minimum

        # again, to avoid an offset in the outliers
        paramsNN[k][mask_lower + mask_upper] = 0


    synth_signal, synth_bvals = params_to_signal(paramsNN, GLOBAL_BVALUES, valid_id, sx, sy, sz, S0)
    subfolder = data_files[i].split('/')[1]
    save_params(paramsNN, valid_id, sx, sy, sz, save_folder, subfolder)
    save_signal(synth_signal, save_folder, subfolder)
    save_bvals(synth_bvals, save_folder, subfolder)
    save_source(data_file, bvals_file, save_folder, subfolder)
    save_valid_id(valid_id, save_folder, subfolder)


