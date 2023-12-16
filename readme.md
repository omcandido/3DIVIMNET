# 3DIVIMNET
## About
This repo demonstrates how to create synthetic IVIM data (of which the grount truth parameter maps are known) to test an image-wise model derived from [IVIMNET]((https://github.com/oliverchampion/IVIMNET)) that takes and entire image as input. The proposed 3DIVIMNET model has a [3D-UNET](https://github.com/wolny/pytorch-3dunet) architecture. This means that, unlike voxel-wise models, like [IVMNET](https://github.com/oliverchampion/IVIMNET), 3DIVIMNET takes an entire 3D image as input, with the bvalues as channel inputs. This architecture benefits from spatial information, potentially improving the IVIM quantification.

## Installation
1. Install the environment: `conda env create -f environment.yml`
2. Install [IVIMNET](https://github.com/oliverchampion/IVIMNET): You can download the repo and add the IVIMNET/IVIMNET folder to your PYTHONPATH variable.
3. Install [pytorch3dunet](https://github.com/wolny/pytorch-3dunet): same as IVIMNET -or follow the INSTALLATION instructions in the readme of the pytorch3dunet repo.

## Synthetise data
To test 3DIVIMNET, we need anatomically correct data so that the model can exploit the spatial correlations in the image. To do that, we (1) estimate the parameter maps of a real image using IVIMNET, (2) feed these parameter maps to a forward model (bi-exponential IVIM model) to get normalised synthetic signal, and (3) de-normalise it by multiplying it by S0 (signal at b=0). The estimated parameter maps are used as ground truth values of the estimated non-normalised signal. To create the synthetic data:
1. Put you raw data (image + bvalues) under the `raw_data` folder. If you have multiple scans from the same patient/volunteer, create subfolders for each scan (e.g., `raw_data/7.2` is the sample #2 of the patient/volunteer #7).
2. run `synthetise.py`
The synthetised data can be found under the `synthetic folder`.

## Train the models
1. Do the LSQ fit by running: `LSQ_fit.py`
2. Train IVIMNET by running: `IVIMNET_train.py`
3. Train 3DIVIMNET by running: `3DIVIMNET_train.py`
Note that during training, Gaussian noise is added to the data at SNR between 10 and 30. During testing SNR is fixed to 15. 
The above scripts train the corresponding models and test them on the last sample of the dataset (in the code, this is sample 16.1). The predictions for the test data can be found under the `saved_preds` folder.

## Results
See the [analysis notebook](analysis.ipynb).