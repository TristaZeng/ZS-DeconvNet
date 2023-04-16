<h1> ZS-DeconvNet </h1>

This is the source codes and instructions for <b>ZS-DeconvNet</b>, a self-supervised deep-learning tool for instant denoising and super-resolution in optical fluorescence microscopy. This package includes the Python implementation of training and inference, MATLAB implementation of data augmentation, as well as some demo data and pre-trained network models.

<h2> Content </h2>

<ul>
  <li><a href="#File structure">1. File structure</a></li>
  <li><a href="#Environment">2. Environment</a></li>
  <li><a href="#Data Pre-processing">3. Training dataset generation</a></li>
  <li><a href="#Implementation of Python code1">4. Train a new model</a></li>
  <li><a href="#Implementation of Python code2">5. Test a well-trained model</a></li>
</ul>

<hr>

<h2 id="File structure">1. File structure</h2>

- `./saved_models` includes pre-trained models for testing, and for each modality and structure `xx`:
  - <code>./saved_models/xx/saved_model</code> includes corresponding pre-trained model and inference result
  - <code>./saved_models/xx/test_data</code> includes raw test data
- <code>./data_augment_recorrupt_matlab</code> includes the MATLAB codes for generating training datasets
- <code>./train_inference_python</code> includes the Python codes of training and inference, and the required dependencies
  - <code>./train_inference_python/models</code> includes the optional models
  - <code>./train_inference_python/utils</code> is the tool package

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

+ Prepare a folder of raw data. Download our open-source raw data of various modalities or use your own raw data. 

+ Open `./data_augment_recorrupt_matlab/demo_augm.m` and replace the parameter `data_folder` with your raw data directory. 

+ The default output path is `./your_augmented_datasets/`.

<hr>

<h2  id="Implementation of Python code1">4. Train a new model</h2>

Skip this part if you do not wish to train a new model. You can just test the demo test data using our provided pre-trained models. 

To train a new model, you need to:

+ Generated the training dataset following the instructions in the previous part.
+ Choose a test image/volume and obtain the path to the corresponding PSF.
+ Change `otf_or_psf_path` (or `psf_path` in the case of 3D), `data_dir`, `folder` and `test_images_path` in <code>./train_inference_python/train_demo_2D.sh</code> or <code>train_inference_python/train_demo_3D.sh</code>. 
+ Run it in your terminal.
+ The result wills be saved to <code>./your_saved_models/</code>.
+ Run <code>tensorboard --logdir [save_weights_dir]/[save_weights_name]/graph</code> to monitor the training process via tensorboard if needed.
+ Other detailed description of each input argument of the python codes can be found in the comments of `./train_inference_python/train_demo_2D.sh` or `train_inference_python/train_demo_3D.sh`.

<hr>

<h2  id="Implementation of Python code1">5. Test a well-trained model</h2>

To test a well-trained ZS-DeconvNet model, you should:

+ Change the weight paths in <code>./train_inference_python/infer_demo_2D.sh</code> or <code>./train_inference_python/infer_demo_3D.sh</code> accordingly, or just use the default options given by us. 
+ Run it in your terminal.
+ The output will be saved to the folder where you load weights, e.g., if you load weights from <code>./train_inference_python/saved_models/.../weights_40000.h5</code>, then the output will be saved to <code>./train_inference_python/saved_models/.../Inference/</code>.
