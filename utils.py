import os
import time
import nibabel as nib
import numpy as np
import IVIMNET.deep as deep
import torch
from IVIMNET.fitting_algorithms import fit_dats
from hyperparams import hyperparams as hp
import matplotlib.pyplot as plt

cons_min = [0.0, 0.0, 0.005] # D, f, Dp
cons_max = [0.005, 0.7, 0.2]  # D, f, Dp

def minmax_rescale(data, min, max):
    min_ = data.min()
    max_ = data.max()
    return min + ( ( (data-min_) * (max-min) ) / (max_ - min_) )


def parse_data(data_file, bvals_file, remove_background=True, normalize=True, SNR=None):
    """
    Reads a raw image along with its bvalues and preprocesses it. 
    For more info see Example_3_volunteer.py from the IVIMNET repo: https://github.com/oliverchampion/IVIMNET.git
    - data_file: path to the signal image
    - bvals_file: path to the bvalues used to acquire the given signal
    - remove_background: removes pixels below 0.5*medianOfTheSignal
    - Normalize: normalizes signal by dividing by the average S0
    - SNR: can be None (no noise added), int (Gaussian noise added at given SNR) or 2-Tuple (Gaussian noise added between SNR[0] and SNR[1])
    """
    ### load patient data
    print('Load patient data \n')
    # load and init b-values
    text_file = np.genfromtxt(bvals_file)
    bvalues = np.array(text_file)
    selsb = np.array(bvalues) == 0
    # load nifti
    data = nib.load(data_file)
    datas = data.get_fdata()
    # reshape image for fitting
    sx, sy, sz, n_b_values = datas.shape 
    X_dw = np.reshape(datas, (sx * sy * sz, n_b_values))

    S0 = calc_s0(X_dw, selsb)
    valid_id = mask_background(S0)
    
    if remove_background:
        ### select only relevant values, delete background and noise, and normalise data
        X_dw = X_dw[valid_id, :]

    if SNR is not None:
        meansig=np.nanmean(X_dw[:,selsb])
        if type(SNR) is tuple:
            SNR = SNR[0]+np.random.rand(1)*(SNR[1]-SNR[0])
        noise=meansig/SNR
        X_dw=X_dw+np.random.randn(np.shape(X_dw)[0],np.shape(X_dw)[1])*noise
        X_dw[X_dw<0]=0

    if normalize:
        # normalise data
        S0 = np.nanmean(X_dw[:, selsb], axis=1).astype('<f')
        X_dw = X_dw / S0[:, None]

    print('Patient data loaded\n')

    return X_dw, bvalues, valid_id, S0, sx, sy, sz, n_b_values

def calc_s0(data, selsb):
    S0 = np.nanmean(data[:, selsb], axis=1)
    S0[S0 != S0] = 0
    S0 = np.squeeze(S0)
    return S0

def mask_background(S0):
    mask = (S0 > (0.5 * np.median(S0[S0 > 0]))) 
    return mask

def lsq_fit(datatot, bvalues):
    print('Conventional fitting\n')
    arg = hp()
    arg = deep.checkarg(arg)
    arg.fit.do_fit = True
    start_time = time.time()
    paramslsq = fit_dats(bvalues.copy(), datatot.copy()[:, :len(bvalues)], arg.fit)
    elapsed_time1 = time.time() - start_time
    print('\ntime elapsed for lsqfit: {}\n'.format(elapsed_time1))
    return paramslsq

def NN_fit(datatot, bvalues):
    arg = hp()
    arg = deep.checkarg(arg)
    print('NN fitting\n')
    res = [i for i, val in enumerate(datatot != datatot) if not val.any()] # Remove NaN data
    start_time = time.time()
    # train network
    net = deep.learn_IVIM(datatot[res], bvalues, arg)
    elapsed_time1net = time.time() - start_time
    print('\ntime elapsed for Net: {}\n'.format(elapsed_time1net))
    start_time = time.time()
    # predict parameters
    paramsNN = deep.predict_IVIM(datatot, bvalues, net, arg)
    elapsed_time1netinf = time.time() - start_time
    print('\ntime elapsed for Net inf: {}\n'.format(elapsed_time1netinf))
    print('\ndata length: {}\n'.format(len(datatot)))
    if arg.train_pars.use_cuda:
        torch.cuda.empty_cache()
    
    return paramsNN, net

def params_to_signal(params, bvalues, valid_id, sx, sy, sz, S0 = None):
    D = params[0].reshape(-1, 1)
    f = params[1].reshape(-1, 1)
    Dp = params[2].reshape(-1, 1)

    D[D<0]=0
    Dp[Dp<0]=0
    f[f<0]=0

    unique_bvals = list(set(bvalues))
    n_bvals = len(unique_bvals)
    unique_bvals = np.array(unique_bvals).reshape(1,-1)
    
    S = f * np.exp(-unique_bvals*Dp) + (1-f) * np.exp(-unique_bvals*D)
    if S0 is not None:
        S *= S0[:,None]
    
    img = np.zeros((sx * sy * sz, n_bvals))

    img[valid_id, :] = S
    img = img.reshape((sx, sy, sz, n_bvals))
    return img, unique_bvals
    
def prepare_path(*folders):
    folders = (str(f) for f in folders)
    path = '/'.join(folders)
    os.makedirs(path) if not os.path.exists(path) else None
    return path

def save_params(params, valid_id, sx, sy, sz, folder, subfolder):
    imgs = []
    for k in range(4):
        img = np.zeros([sx * sy * sz])
        img[valid_id] = params[k]
        img = np.reshape(img, [sx, sy, sz])
        imgs.append(img)
    imgs = np.stack(imgs, axis=-1)
    path = prepare_path(folder, subfolder)
    nib.save(nib.Nifti1Image(imgs, np.eye(4)),'{}/params.nii.gz'.format(path))

def save_signal(signal, folder, subfolder):
    path = prepare_path(folder, subfolder)
    nib.save(nib.Nifti1Image(signal, np.eye(4)), '{}/data.nii.gz'.format(path))

def save_bvals(bvals, folder, subfolder):
    path = prepare_path(folder, subfolder)
    np.savetxt('{}/bvalues.bval'.format(path), np.array(bvals).reshape((1,-1)),fmt='%d')

def save_valid_id(valid_id, folder, subfolder):
    path = prepare_path(folder, subfolder)
    np.save('{}/valid_id.npy'.format(path), valid_id)
    
def save_source(data_file, bvals_file, folder, subfolder):
    path = prepare_path(folder, subfolder)
    with open('{}/source.txt'.format(path), 'w') as f:
        f.write(data_file)
        f.write('\n')
        f.write(bvals_file)

def plot_hist(data_file, bvals_file, bins=60, b_idx=None, remove_background=True, normalize=True):
    datatot, bvalues, valid_id, S0, sx, sy, sz, n_b_values = parse_data(
        data_file, bvals_file, remove_background, normalize)
    
    if b_idx == None:
        plt.hist(datatot.flatten(), bins=bins)
    else:
        plt.hist(datatot[:,b_idx], bins=bins)   
    plt.show()