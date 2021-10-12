# SC-Net
Here is source code for paper "Self-Supervised Cryo-Electron Tomography Volumetric Image Restoration from Single Noisy Volume with Sparsity Constraint" accepted in ICCV 2021.<br>

# Introduction
Cryo-Electron Tomography (cryo-ET) is a powerful tool for 3D cellular visualization. Due to instrumental limitations, cryo-ET images and their volumetric reconstruction suffer from extremely low signal-to-noise ratio. In this paper, we propose a novel end-to-end self-supervised learning model, the Sparsity Constrained Network (SC-Net), to restore volumetric image from single noisy data in cryo-ET. The proposed method only requires a single noisy data as training input and no ground-truth is needed in the whole training procedure. A new target function is proposed to preserve both local smoothness and detailed structure. Additionally, a novel procedure for the simulation of electron tomographic photographing is designed to help the evaluation of methods. Experiments are done on three simulated data and four real-world data. The results show that our method could produce a strong enhancement for a single very noisy cryo-ET volumetric data, which is much better than the state-of-the-art Noise2Void, and with a competitive performance comparing with Noise2Noise.

# Operation System
Ubuntu 18.04 or CentOS7

# Requirements
Python 3.6.13 <br>
Pytorch 1.7.1 <br>
opencv-python 4.5.1 <br>
numpy 1.19.2 <br>
scikit-image 0.17.1 <br>
scikit-learn 0.24.2 <br>
mrcfile 1.3.0 <br>
topaz-em 0.2.4 <br>
numba 0.51.2 <br>

# Pretrained Model
 Pretrained models for real-world datasets : https://drive.google.com/file/d/18yaCdxlLbNU_eg1cIkgFIE9LpaxiF83E/view?usp=sharing 
 Real-World Training dataset will be available on Google Drive.
 This may take a few days, coming soon...

More information is on the way......
