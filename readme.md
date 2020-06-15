# Image Classification in Tensorflow: Python vs C++ 

Simple repository to compare the speed of image classification on python vs C++, on the webcam. 

Specifically, it aims to evaluate how much bottleneck the python program would add to image data copying, compared to C++ data copying. 

## Tensorflow Libraries

Tensorflow libraries for python are required. Also, `cppflow` uses the [tensorflow C API](https://www.tensorflow.org/install/lang_c). It should be downloaded and added to the folder `libtensorflow` in your home folder, or installed system-wide.  

## Dataset

A dataset should be added to the folder `Dataset/` in the root of the repository, containing many folders, each representing a class, with photos of that class. 

## Usage 

1. Run `Classifier.ipynb` cells to define and save the model. 

2. Compile the C++ programs defined in CMakeLists.txt

```ssh
mkdir build/
cd build/
cmake ..
build
```

3. Test executables and/or python `predict-webcam.py` to test with webcam and compare speed. 