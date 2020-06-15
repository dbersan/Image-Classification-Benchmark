cmake_minimum_required(VERSION 3.10)
project(example)

find_library(TENSORFLOW_LIB tensorflow HINT $ENV{HOME}/libtensorflow/lib)

set(CMAKE_CXX_STANDARD 17)
# find_package( OpenCV REQUIRED )
find_package( OpenCV REQUIRED PATHS /usr/local NO_DEFAULT_PATH)

# Classifier test
add_executable(classifier_test classifier.cpp cppflow/src/Model.cpp cppflow/src/Tensor.cpp)
target_include_directories(classifier_test PRIVATE include $ENV{HOME}/libtensorflow/include)
target_link_libraries (classifier_test "${TENSORFLOW_LIB}" ${OpenCV_LIBS} stdc++fs)

# Webcam test
add_executable(webcam_test webcam.cpp cppflow/src/Model.cpp cppflow/src/Tensor.cpp)
target_include_directories(webcam_test PRIVATE include $ENV{HOME}/libtensorflow/include)
target_link_libraries (webcam_test "${TENSORFLOW_LIB}" ${OpenCV_LIBS} stdc++fs)