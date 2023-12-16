# %%
import torch
torch.cuda.is_available()
import numpy as np
from utils import *

# %% Load sample 16.1
id_test = '16.1'
test_data = 'synthetic/{}/data.nii.gz'.format(id_test)
test_bvals = 'synthetic/{}/bvalues.bval'.format(id_test)
test_params = 'synthetic/{}/params.nii.gz'.format(id_test)

# %% LSQ fit sample
datatot, bvalues, valid_id, S0, sx, sy, sz, n_b_values = parse_data(test_data, test_bvals, SNR=15) #FIX SNR
paramslsq = lsq_fit(datatot, bvalues)

# %% save estimated parameter maps
path = prepare_path('saved_preds',id_test,'LSQfit')

params_names = ['D', 'f', 'Dp']
for k in range(len(params_names)):
    img = np.zeros([sx * sy * sz])
    img[valid_id] = paramslsq[k]
    img = np.reshape(img, [sx, sy, sz])
    np.save(path + '/' + params_names[k], img)


