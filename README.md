# Improved SEGAN
Tricks to improve [SEGAN](https://github.com/santi-pdp/segan) performance. Eveything is re-implemented into Keras with Tensorflow backend.

Supporting document with evaluation results and other details can be found [here](https://arxiv.org/pdf/2002.08796.pdf).

**Deepak Baby, _iSEGAN: Improved Speech Enhancement Generative Adversarial Networks_, Arxiv preprint, 2020.**

----
### Pre-requisites
1. Install [tensorflow](https://www.tensorflow.org/) and [keras](https://keras.io/)
1. Install [tqdm](https://pypi.org/project/tqdm/) for profiling the training progress
1. The experiments are conducted on a dataset from Valentini et. al.,  and are downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/2791). The following script can be used to download the dataset. *Requires [sox](http://sox.sourceforge.net/) for converting to 16kHz*.
    ```bash
    $ ./download_dataset.sh
    ```

### Running the model
1. **Prepare data for training and testing the various models**. The folder path may be edited if you keep the database in a different folder. This script is to be executed only once and the all the models reads from the same location.
    ```python
    python prepare_data.py
    ```
2. **Running the models**. The training and evaluation of the various segan models are implemented in `run_isegan.py`. which offers several cGAN configurations. Edit the ```opts``` variable for choosing the cofiguration. The results will be automatically saved to different folders. The folder name is generated from ```files_ops.py ``` and the foldername automatically includes different configuration options.
    
The options are:    
* **Different normalizations**      
    * Instance Normalization    
    * Batch Normalization     
    * Batch Renormalization     
    * Group Normalization     
    * Spectral Normalization    
* **One Sided Label Smoothing**: Encouranging the discriminator to estimate soft probabilities (0.8, 0.9, etc.) on the real samples.    
* **Trainable Auditory filter-bank layer**: The first layer is initialized using a gammatone filterbank and use it as a trainable layer.    
* **Pre-emphasis Layer** : Incorporating the pre-emphasis operation as a trainable layer.    

3. **Evaluation on testset is also done together with training**. Set ```TEST_SEGAN = False``` for disabling testing. 

----
### Misc
* **This code loads all the data into memory for speeding up training**. But if you dont have enough memory, it is possible  to read the mini-batches from the disk using HDF5 read. In ```run_<xxx>.py``` 
  ```python
  clean_train_data = np.array(fclean['feat_data'])
  noisy_train_data = np.array(fnoisy['feat_data'])
  ```
  change the above lines to 
  ```python
  clean_train_data = fclean['feat_data']
  noisy_train_data = fnoisy['feat_data']
  ```
  **But this can lead to a slow-down of about 20 times (on the test machine)** as the mini-batches are to be read from the disk over several epochs.

---- 
### References
[1] S. Pascual, A. Bonafonte, and J. Serra, _SEGAN: speech enhancement generative adversarial network_, in INTERSPEECH., ISCA, Aug 2017, pp. 3642–3646.

----
#### Credits
The keras implementation of cGAN is based on the following repos
* [SEGAN](https://github.com/santi-pdp/segan)
* [DCGAN](https://github.com/carpedm20/DCGAN-tensorflow)
* [pix2pix](https://github.com/phillipi/pix2pix)


