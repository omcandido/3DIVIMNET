# %%
import torch
torch.cuda.is_available()
import numpy as np
from datetime import datetime
import nibabel as nib
from torch import nn
import os
from utils import *
from IVIMNET.deep import Net, checkarg
from hyperparams import hyperparams as hp

# %% load data
data_files = []
bvalues_files = []
params_files = []
for i in range(1,17):
    for j in range(1,3):
        path = 'synthetic/{}.{}/data.nii.gz'.format(i,j)
        if os.path.isfile(path):
            data_files.append(path)
        path = 'synthetic/{}.{}/bvalues.bval'.format(i,j)
        if os.path.isfile(path):
            bvalues_files.append(path)
        path = 'synthetic/{}.{}/params.nii.gz'.format(i,j)
        if os.path.isfile(path):
            params_files.append(path)

# %% train and test data. IVIMNET internally further splits train data into train and validation data.
train_data = data_files[:28]
train_params = params_files[:28]
test_data = data_files[-2:]
test_params = params_files[-2:]

# %%
# Make sure all files have the same sequence of bvalues
bvalues = []
for i in bvalues_files:
    text_file = np.genfromtxt(i)
    bvalues.append(np.array2string(text_file))

assert len(set(bvalues)) == 1
GLOBAL_BVALUES = np.array(np.genfromtxt(bvalues_files[0]))

# %% Create a big array with all train_data
full_datatot=[]
for i, data in enumerate(train_data):
    bvals_file = bvalues_files[i]
    data_file = data
    print(data_file, bvals_file)
    datatot, _, _, _, _, _, _, _ = parse_data(data_file, bvals_file, SNR=(10,30))
    full_datatot.append(datatot)
     
     
full_datatot = np.concatenate(full_datatot)

# %% Train IVIMNET
paramsNN, net = NN_fit(full_datatot, GLOBAL_BVALUES)
timestamp = datetime.now()
model_path = prepare_path('saved_models', 'IVIMNET', timestamp)
torch.save(net.state_dict(), model_path + '/model.pt')



# == MODEL EVALUATION ==
# %%
arg = hp()
arg = checkarg(arg)
net = Net(torch.FloatTensor(GLOBAL_BVALUES[:]).to(arg.train_pars.device), arg.net_pars).to('cuda')
net.load_state_dict(torch.load(model_path + '/model.pt'))
net.eval()

# %% PREDICT FOR test_data
pred_d =[]
pred_f =[]
pred_dp =[]
valid_ids = []
for i, data in enumerate(test_data):
    bvals_file = bvalues_files[i]
    data_file = data
    print(data_file, bvals_file)
    datatot, _, valid_id, _, _, _, _, _ = parse_data(data, bvals_file, SNR=15) #FIX SNR
    valid_ids.append(valid_id)
    datatot = torch.tensor(datatot).to('cuda', dtype=torch.float32)
    s_, D_, f_, Dp_, s0_ = net(datatot)

    pred_d.append(D_)
    pred_f.append(f_)
    pred_dp.append(Dp_)

# %%
sx = 144
sy = 144
sz = 18
n_b = 12
n_samples = len(test_data)

# Save predictions
# %%
path = prepare_path('saved_preds',16,'IVIMNET')

D_valid = []
for i in range(n_samples):
    vol = np.zeros((sx, sy, sz)).flatten()
    vol[valid_ids[i]] = pred_d[i].detach().cpu().flatten()
    D_valid.append(vol)
D_valid = np.concatenate(D_valid).reshape((n_samples, sx, sy, sz))
np.save(path + '/D.npy', D_valid)

f_valid = []
for i in range(n_samples):
    vol = np.zeros((sx, sy, sz)).flatten()
    vol[valid_ids[i]] = pred_f[i].detach().cpu().flatten()
    f_valid.append(vol)
f_valid = np.concatenate(f_valid).reshape((n_samples, sx, sy, sz))
np.save(path + '/f.npy', f_valid)

Dp_valid = []
for i in range(n_samples):
    vol = np.zeros((sx, sy, sz)).flatten()
    vol[valid_ids[i]] = pred_dp[i].detach().cpu().flatten()
    Dp_valid.append(vol)
Dp_valid = np.concatenate(Dp_valid).reshape((n_samples, sx, sy, sz))
np.save(path + '/Dp.npy', Dp_valid)


