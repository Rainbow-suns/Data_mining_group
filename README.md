# 1. Data preprocessing
The model in this project is for the binary task, so two sets of pictures of different categories need to be named as follows:
```code
First class: n(no. of picture)
Second class: t(no. of picture)
```
And place the image in the corresponding folder according to the different model you want to run:
### VGGnet & ResNet
The training dataset, valid dataset and the test dataset are placed in **data/train/**, **data/valid/**  and **data/test/** separately. You no longer need to put images in different folders by category.

### SVM
The training dataset and the test dataset are placed in **data/raw_data/train/** and **data/raw_data/test/** separately, and divide the images into two folders by different categories.

### Image format
In this project, the size of all images needs to be **256 x 256**. You can run the following code to resize all the images in a folder:
```bash
python transfer.py --folder_path Folder_path --size 256
```
_Folder_path_ is the path to the folder where you store your images.

# 2. Running Environment
It is recommended to use IntelliJ IDEA Community Edition 2022.2.3 to run the code of this project. After opening IDEA, use the python3.9 version, choose to let IDEA automatically download the missing packages, and then IDEA will automatically download the packages required by the program, and most packages can be installed directly.
For Pytorch packages, manual download is required.

First enter https://pytorch.org/get-started/locally/, select Stable (2.0.0) for PyTorch Build, then select the corresponding option according to the actual situation of your own computer, and finally copy the command in Run this Command to In the terminal in your IDEA, you can complete the installation of pytorch.

After the installation is complete, enter pip install opencv-python in the terminal to install the opencv package.

The packages required for this project are as follows:
1. torch~=1.12.0+cu116
2. numpy~=1.22.3
3. pillow~=9.0.1
4. torchvision~=0.13.0+cu116
5. seaborn~=0.11.0
6. pandas~=1.1.3
7. matplotlib ~=3.3.2
8. scikit-learn ~=0.23.2
9. opencv-python~=4.6.0.66
10. scipy~=1.7.3

Once installed, you can run the code.

# 3. Training & Test
### VGGnet & ResNet
Training VGG model:
```bash
Python VGGTrain.py
```
The best result of training will be saved as **/TrainedModel/VGGnet.pth** and training result of final epoch will be saved as **/TrainedModel/VGGnetLast.pth**

Training ResNet model:
```bash
Python ResnetTrain.py
```
The best result of training will be saved as **/TrainedModel/Resnet.pth** and training result of final epoch will be saved as **/TrainedModel/LastResnet.pth**

After training, a line graph of loss function and accuracy will be automatically generated, and the running time of the program will be displayed at the same time.

* Due to the size limitation of github warehouse, we saved a set of trained models in [Google drive](https://drive.google.com/drive/folders/18wqeMtLIJrMXNqCImAOTXEQkuBazCj3O?usp=share_link)
  , which can be downloaded to **/TrainedModel/** through the link.

Test model: **/TrainedModel/VGGnet.pth**
```bash
Python VGGTest.py
```
Test model: **/TrainedModel/Resnet.pth**
```bash
Python ResnetTest.py
```
After testing, a confusion matrix diagram will be generated.

### Grid search
To find the best suitable combination of learning rate and batch size, grid search can be used:
```bash
python .\grid_search.py --algorithm_type 1
```
_algorithm_type_ : 0 represent Resnet model and 1 represent VGGnet model.
Best model will be saved in /Best_model.

### SVM
Different from neural network model. For the SVM algorithm, just run SVM.py directly to complete train and test.
```bash
Python SVM.py
```

# 4. GUI
Running GUI.py will open the front-end interface we designed. For the operation using GUI, you can check the demo video.
