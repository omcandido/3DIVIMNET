# %%
from pytorch3dunet.unet3d.model import *
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from utils import prepare_path
from datetime import datetime
import os

# %%
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

# %%
# Make sure all files have the same sequence of bvalues
bvalues = []
for i in bvalues_files:
    text_file = np.genfromtxt(i)
    bvalues.append(np.array2string(text_file))

assert len(set(bvalues)) == 1
bvalues = np.array(np.genfromtxt(bvalues_files[0]))

# %%
class IVIMDataset(Dataset):
    def __init__(self, data_files, bvalues, params_files, snr=None) -> None:
        self.data_files = data_files
        self.bvalues = bvalues
        self.params_files = params_files
        self.snr = snr
        super().__init__()

    def __len__(self):
        return len(self.data_files)
    
    #TODO: do all this preprocessing outside and save a lot of time
    #TODO: alternative: load only D, f, Dp and create augmented data on-the-fly
    def __getitem__(self, idx):
        # load and init b-values
        selsb = np.array(self.bvalues) == 0
        # load nifti
        data = nib.load(self.data_files[idx])
        datas = data.get_fdata()
        # reshape image for fitting
        sx, sy, sz, n_b_values = datas.shape
        X_dw = np.reshape(datas, (sx * sy * sz, n_b_values))
        ### select only relevant values, delete background and noise, and normalise data
        S0 = np.nanmean(X_dw[:, selsb], axis=1)
        S0[S0 != S0] = 0
        S0 = np.squeeze(S0)
        valid_id = (S0 > (0.5 * np.median(S0[S0 > 0])))
        datatot = X_dw[valid_id, :]
        #add noise
        meansig=np.nanmean(datatot[:,selsb])
        SNR = self.snr
        if SNR is not None:
            meansig=np.nanmean(X_dw[:,selsb])
            if type(SNR) is tuple:
                SNR = SNR[0]+np.random.rand(1)*(SNR[1]-SNR[0])
            noise=meansig/SNR
            datatot=datatot+np.random.randn(np.shape(datatot)[0],np.shape(datatot)[1])*noise
            datatot[datatot<0]=0
        # normalise data
        S0 = np.nanmean(datatot[:, selsb], axis=1).astype('<f')
        S0[S0<0.1]=0.1
        datatot = datatot / S0[:, None]
        datatot = np.clip(datatot,0,2.5)
        data_norm = np.zeros(sx*sy*sz*n_b_values).reshape((-1, n_b_values))
        data_norm[valid_id,:] = datatot
        data_norm = data_norm.reshape((sx,sy,sz,n_b_values))
        data_norm = data_norm.transpose(-1,0,1,2)

        # load ground truth parameter maps
        params = nib.load(self.params_files[idx])
        params = params.get_fdata()
        params = params.transpose(-1,0,1,2)
        params = params[:3]
        
        return data_norm, valid_id, params

# %%
train_data = IVIMDataset(data_files[:25], bvalues, params_files[:25], snr=(10, 30))
valid_data = IVIMDataset(data_files[25:28], bvalues, params_files[25:28], snr=10)
test_data = IVIMDataset(data_files[-2:], bvalues, params_files[-2:], snr=15)

# %%
train_dataloader = DataLoader(train_data, batch_size=5, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=2, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=2, shuffle=False)

# %%
n_b_values = len(bvalues)
model = UNet3D(n_b_values, 3, num_groups=1, is_segmentation=False, f_maps=(128)).to('cuda', dtype=torch.float32)

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)
cons_min = [-0.002, -0.3, -0.06] # D, f, Dp
cons_max = [0.007, 1, 0.3]  # D, f, Dp


# %%
def get_snet(D, f, Dp, b):
    Snet = (f * torch.exp(-Dp*b) + (1-f) * torch.exp(-D*b))
    return torch.squeeze(Snet)

# %%
def predict_batch(model, x_batch, mask_batch):
    pred = model.forward(x_batch)
    pred = torch.sigmoid(pred)

    D_ = cons_min[0] + pred[:,0,:,:,:].flatten(1)[mask_batch] * (cons_max[0] - cons_min[0])
    f_ = cons_min[1] + pred[:,1,:,:,:].flatten(1)[mask_batch] * (cons_max[1] - cons_min[1])
    Dp_ = cons_min[2] + pred[:,2,:,:,:].flatten(1)[mask_batch] * (cons_max[2] - cons_min[2])
    # S0_ = cons_min[3] + pred[:,3,:,:,:].flatten(1)[batch_valid_id] * (cons_max[3] - cons_min[3])

    return D_, f_, Dp_

def eval_batch(x_batch, D_, f_, Dp_, y_batch, mask_batch, bvalues):
    criterion = torch.nn.MSELoss()
    
    # ugly non-vectorised version:
    physics_loss = torch.tensor(0).to('cuda', dtype=torch.float32)
    for j, b in enumerate(bvalues):
        Snet = get_snet(D_, f_, Dp_, b)
        physics_loss += criterion(Snet, x_batch[:,j,:,:,:].flatten(1)[mask_batch])
    physics_loss /= n_b_values
    
    D_loss = np.sqrt(criterion(D_, y_batch[:,0,:,:,:].flatten(1)[mask_batch]).item())
    f_loss = np.sqrt(criterion(f_, y_batch[:,1,:,:,:].flatten(1)[mask_batch]).item())
    Dp_loss = np.sqrt(criterion(Dp_, y_batch[:,2,:,:,:].flatten(1)[mask_batch]).item())
    return physics_loss, D_loss, f_loss, Dp_loss

def eval_dataloader(model, dataloader):
    epoch_loss = np.zeros(4) #losses per epoch: [total, D , f, Dp]
    for batch_data, batch_valid_id, batch_y in dataloader:
        batch_data = batch_data.to('cuda', dtype=torch.float32)
        batch_y = batch_y.to('cuda', dtype=torch.float32)
        
        D_, f_, Dp_ = predict_batch(model, batch_data, batch_valid_id)
        batch_loss, D_loss, f_loss, Dp_loss = eval_batch(batch_data, D_, f_, Dp_, batch_y, batch_valid_id, bvalues)

        epoch_loss[0] += batch_loss.item()
        epoch_loss[1] += D_loss
        epoch_loss[2] += f_loss
        epoch_loss[3] += Dp_loss
        del batch_data, batch_valid_id, batch_y, D_, f_, Dp_, D_loss, f_loss, Dp_loss

    epoch_loss /= len(dataloader)
    return epoch_loss
    

# %%
def train(model, train_dataloader, valid_dataloader, optimizer, patience=10):
    timestamp = datetime.now()
    model.train()
    epoch_n = 0
    n_bad_epochs = 0
    best_loss = np.inf
    while n_bad_epochs < patience:
        epoch_n +=1
        epoch_loss = np.zeros(4) #losses per epoch: [total, D , f, Dp]
        tqdm_epoch = tqdm(train_dataloader)
        tqdm_epoch.set_description(desc='Epoch#{}'.format(epoch_n))
        for batch_data, batch_valid_id, batch_y in tqdm_epoch:
            batch_data = batch_data.to('cuda', dtype=torch.float32)
            batch_y = batch_y.to('cuda', dtype=torch.float32)
            
            D_, f_, Dp_ = predict_batch(model, batch_data, batch_valid_id)
            batch_loss, D_loss, f_loss, Dp_loss = eval_batch(batch_data, D_, f_, Dp_, batch_y, batch_valid_id, bvalues)
            
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            epoch_loss[0] += batch_loss.item()
            epoch_loss[1] += D_loss
            epoch_loss[2] += f_loss
            epoch_loss[3] += Dp_loss

            tqdm_epoch.set_postfix({'physics': ('%.3e '% batch_loss.item()), 'D': ('%.3e '%  D_loss), 'f': ('%.3e '% f_loss), 'Dp': ('%.3e '% Dp_loss)})
            del batch_loss, batch_data, batch_valid_id, batch_y, D_, f_, Dp_

        epoch_loss /= len(train_dataloader)
        tqdm_epoch.write('Training - Physics:{:.3e}, D: {:.3e}, f: {:.3e}, Dp: {:.3e}'.format(*epoch_loss))
        torch.cuda.empty_cache()

        model.eval()
        val_loss = eval_dataloader(model, valid_dataloader)
        if val_loss[0] < best_loss:
            best_loss = val_loss[0]
            n_bad_epochs = 0
            model_path = prepare_path('saved_models', '3DIVIMNET', timestamp)
            torch.save(model.state_dict(), model_path + '/model.pt')
        else:
            n_bad_epochs += 1
        tqdm_epoch.write('Eval (#bad: {}) - Physics:{:.3e}, D: {:.3e}, f: {:.3e}, Dp: {:.3e}'.format(n_bad_epochs, *val_loss))

    return model_path + '/model.pt'


# %%
saved_model = train(model, train_dataloader, valid_dataloader, optimizer, patience=10)

# %%
model.load_state_dict(torch.load(saved_model))
model.eval()
batch_data, batch_valid_id, batch_y = next(iter(test_dataloader))
batch_data = batch_data.to('cuda', dtype=torch.float32)
batch_valid_id = batch_valid_id.to('cuda', dtype=torch.float32)
batch_y = batch_y.to('cuda', dtype=torch.float32)

pred = model.forward(batch_data)
pred = torch.sigmoid(pred)

D_ = cons_min[0] + pred[:,0] * (cons_max[0] - cons_min[0])
f_ = cons_min[1] + pred[:,1] * (cons_max[1] - cons_min[1])
Dp_ = cons_min[2] + pred[:,2] * (cons_max[2] - cons_min[2])


path = prepare_path('saved_preds',16,'3DIVIMNET')

D_valid = (D_ * batch_valid_id.reshape(D_.shape)).detach().cpu().numpy()
f_valid = (f_ * batch_valid_id.reshape(f_.shape)).detach().cpu().numpy()
Dp_valid = (Dp_ * batch_valid_id.reshape(Dp_.shape)).detach().cpu().numpy()

np.save(path + '/D.npy', D_valid)
np.save(path + '/f.npy', f_valid)
np.save(path + '/Dp.npy', Dp_valid)


