<h1> ZS-DeconvNet </h1>

This is the source codes and instructions for <b>ZS-DeconvNet</b>, a self-supervised deep-learning tool for instant denoising and super-resolution in optical fluorescence microscopy. This package includes the Python implementation of training and inference, and MATLAB implementation of training data generation and simulation of PSF. You should be able to download our code and set up the environment within five minutes.

<h2> Content </h2>

<ul>
  <li><a href="#File structure">1. File structure</a></li>
  <li><a href="#Environment">2. Environment</a></li>
  <li><a href="#Data Pre-processing">3. Training dataset generation</a></li>
  <li><a href="#Implementation of Python code1">4. Train a new model</a></li>
  <li><a href="#Implementation of Python code2">5. Test a well-trained model</a></li>
  <li><a href="#PSF">6. PSF Generation</a></li>
</ul>

<hr>

<h2 id="File structure">1. File structure</h2>

- <code>./data_augment_recorrupt_matlab</code> includes the MATLAB codes for generating training datasets and simulation of PSF
  
  + `./data_augment_recorrupt_matlab/GenData4ZS-DeconvNet` includes the MATLAB codes for generating training dataset for 2D and 3D ZS-DeconvNet, as well as one demo for generating simulated 3D PSF
  
  + `./data_augment_recorrupt_matlab/GenData4ZS-DeconvNet-SIM` includes the MATLAB codes for generating training dataset for 2D and 3D ZS-DeconvNet for Structured Illumination Microscopy, as well as one demo for generating simulated SIM beads
  
  + `./data_augment_recorrupt_matlab/XxUtils` includes common tool packages

- <code>./train_inference_python</code> includes the Python codes of training and inference, and the required dependencies
  
  - <code>./train_inference_python/models</code> includes the optional models
  - <code>./train_inference_python/utils</code> is the tool package

It is recommended to download the demo test data and pre-trained models contained in `saved_models` from [our open-source datasets](https://drive.google.com/drive/folders/1XAOuLYXYFCxlElRwvik_fs7TqZlRixGv?usp=sharing), and place it under the folder with the same name so that:

+ `./saved_models` includes pre-trained models for testing, and for each modality and structure `xx`:
  
  - `./saved_models/xx/saved_model` includes corresponding the pre-trained model and inference result, which you should be able to get by processing the raw test data with the pre-trained model
  - `./saved_models/xx/test_data` includes raw test data

<hr>

<h2 id="Environment">2. Environment</h2>

Our environment is:

- Windows 10
- Python 3.9.7
- Tensorflow-gpu 2.5.0
- NVIDIA GPU (GeForce RTX 3090) + CUDA (11.4)

To use our code, you should create a virtual environment and install the required packages first.

```
$ conda create -n zs-deconvnet python=3.9.7 
$ conda activate zs-deconvnet
$ pip install -r requirements.txt
```

After that, remember to install the right version of CUDA and cuDNN, if you want to use GPU. You can get the compatible version (e.g., cudatoolkit==11.3.1, cudnn==8.2.1) by searching

```
$ conda search cudatoolkit --info
$ conda search cudnn --info
```

then installing the corresponding version

```
$ conda install cudatoolkit==11.3.1
$ conda install cudnn==8.2.1
```

<hr>

<h2 id="Data pre-processing">3. Training dataset generation</h2>

Dataset generation for ZS-DeconvNet:

+ Prepare a folder of raw data. Download [our open-source raw data with corresponding PSF](https://www.zenodo.org/record/7261163) of various modalities or use your own raw data. 

+ Open `./data_augment_recorrupt_matlab/GenData4ZS-DeconvNet/main_augm.m` and replace the parameter `data_folder` with your raw data directory. 

+ The default output path is `./your_augmented_datasets/`.

Dataset generation for ZS-DeconvNet for SIM:

+ Prepare a folder of raw SIM data.

+ Run `./data_augment_recorrupt_matlab/GenData4ZS-DeconvNet-SIM/Create_corrupt_img_2D.m` or `./data_augment_recorrupt_matlab/GenData4ZS-DeconvNet-SIM/Create_corrupt_img_3D.m` to generate re-corrupted data.

+ Perform SIM reconstruction on the re-corrupted data.

+ Use the reconstructed data for training. You can set `augment_flag` to True in corresponding Python codes to augment the reconstructed data if needed.

<hr>

<h2  id="Implementation of Python code1">4. Train a new model</h2>

Skip this part if you do not wish to train a new model. You can just test the demo test data using our provided pre-trained models. 

To train a new model, you need to:

+ Generated the training dataset following the instructions in the previous part and obtain the path to the corresponding PSF.

+ Choose a test image/volume.

+ Choose the demo file based on needs:
  
  + `./train_inference_python/train_demo_2D.sh`: train with 2D wide-field data and alike.
  
  + `./train_inference_python/train_demo_3D.sh`: train with 3D wide-field, confocal, lattice light-sheet, lattice light-sheet SIM data and alike.
  
  + `./train_inference_python/train_demo_2DSIM.sh`: train with 2D reconstructed SIM data.
  
  + `./train_inference_python/train_demo_3DSIM.sh`: train with 3D reconstructed SIM data.

+ Set `otf_or_psf_path` (or `psf_path`), `data_dir`, `folder` and `test_images_path` in said demo file. Remember to examine other parameters like patch size and dx as well because their default value may not fit your training data.

+ Run it in your terminal.

+ The result wills be saved to <code>./your_saved_models/</code>.

+ Run <code>tensorboard --logdir [save_weights_dir]/[save_weights_name]/graph</code> to monitor the training process via tensorboard if needed.

+ Other **detailed description of each input argument of the python codes** can be found in the comments of the demo file. You can set them accordingly.

<hr>

<h2  id="Implementation of Python code1">5. Test a well-trained model</h2>

To test a well-trained ZS-DeconvNet model, you should:

+ Change the weight paths in <code>./train_inference_python/infer_demo_2D.sh</code> or <code>./train_inference_python/infer_demo_3D.sh</code> accordingly, or just use the default options given by us. The inference of SIM data is the same as other type of microscopy so no additional code is provided, though remember to set the model name correctly.
+ Run it in your terminal.
+ The output will be saved to the folder where you load weights, e.g., if you load weights from <code>./train_inference_python/saved_models/.../weights_40000.h5</code>, then the output will be saved to <code>./train_inference_python/saved_models/.../Inference/</code>.

<hr>

<h2  id="PSF">6. PSF Generation</h2>

For researchers who cannot get access to the experimental captured PSF, we provide a MATLAB script to generate simulated PSFs.

<h3> 6.1 Generation of 3D Wide-field PSF </h3>

We provide script to generate 3D wide-field PSF given the NA, emission wavelength, pixel size, refrative index and PSF size. The output can be directly used for our (3D) ZS-DeconvNet.

+ Adjust the parameters in `./data_augment_recorrupt_matlab/GenData4ZS-DeconvNet/create_PSF.m`.
+ Run `./data_augment_recorrupt_matlab/GenData4ZS-DeconvNet/create_PSF.m` and the simulated PSF will be saved to your designated file path.

<h3> 6.2 Generation of 3D SIM PSF </h3>

We provide script to generate raw 3D-SIM images of a simulated bead given the excitation NA, PSF/OTF of the imaging system, excitation wavelength, and pixel size. After 3D-SIM reconstruction via either commercial or open-source software, the simulated SR-SIM image stack of beads can be used as PSF input to our ZS-DeconvNet-SIM.

+ Run `./data_augment_recorrupt_matlab/GenData4ZS-DeconvNet-SIM/main_create_simu_beads.m` and the generated raw 3D-SIM images of a simulated bead `img_sim` will be saved to your MATLAB workspace. 
+ Detailed descriptions of parameters is given in the comments of `./data_augment_recorrupt_matlab/GenData4ZS-DeconvNet-SIM/main_create_simu_beads.m`. You can change them according to your needs.