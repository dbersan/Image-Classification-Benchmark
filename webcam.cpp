//
// Author : D Bersan
//

/*
 * Load model saved using tensorflow SavedModel format.
 *  
 * To access the names of the input and output tensors of the model, you can use the `saved_model_cli` tool, inside python bin/ folder. 
 * 
 * The syntax is (for the default tag-set and signature): `saved_model_cli show --dir /path/to/saved_model_folder/ --tag_set serve --signature_def serving_default`
 * 
 * > TODO Fix code to use other SignatureDefs other than the default one (It is also possible to define the name of the tensors via code while saving the model, but the documentation isn't so clear). 
 * 
 * More info at: https://stackoverflow.com/questions/58968918/accessing-input-and-output-tensors-of-a-tensorflow-2-0-savedmodel-via-the-c-api?noredirect=1#comment109422705_58968918
*/

#include "cppflow/include/Model.h"
#include "cppflow/include/Tensor.h"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iterator>
#include <chrono>
#include <ctime> 
#include <iostream>

using namespace cv;

Mat preprocessImage(Mat image, int size);

int main() {
    const int DEVICE_ID = 0;
    const int IMG_SIZE = 150;
    const std::string labels[] = {"bird","none","bottle","lion"};

    // Used variables
    Mat frame, preprocessed_frame, flat;
    std::vector<float> predictions;
    std::vector<float> img_data(IMG_SIZE*IMG_SIZE*3);

    auto second = std::chrono::duration<double>(1.0f);
    auto last_time = std::chrono::system_clock::now();
    std::chrono::_V2::system_clock::time_point now;
    std::chrono::duration<double> elapsed_seconds;
    double fps;
    std::string fps_str;
    std::string text;

    // Display variables
    const auto puttext_point = cv::Point(10,450);
    const auto puttext_color = cv::Scalar(20,200,10);

    // Initialize neural network
    std::cout<<"Current tensorflow version: "<< TF_Version() << std::endl;
    Model m("../model");

    // Input and output Tensors
    Tensor input(m, "serving_default_input_layer");
    Tensor prediction(m, "StatefulPartitionedCall");

    // Initialize webcam
    VideoCapture cap;
    if(!cap.open(DEVICE_ID))
        return 0;

    while(true){
        cap >> frame;
        if( frame.empty() ) break; // end of video stream
        
        // Get fps
        now = std::chrono::system_clock::now();
        elapsed_seconds = now-last_time;
        fps = (second/elapsed_seconds);
        std::stringstream stream;
        stream << std::fixed << std::setprecision(2) << fps;
        fps_str = stream.str();
        last_time = now;

        // Predict frame        
        preprocessed_frame = preprocessImage(frame, IMG_SIZE);
        int rows = preprocessed_frame.rows;
        int cols = preprocessed_frame.cols;
        int channels = preprocessed_frame.channels();

        // Put data inside vector
        //img_data.assign(preprocessed_frame.begin<float>(), preprocessed_frame.end<float>());

        // Assign to vector for 3 channel image
        // Souce: https://stackoverflow.com/a/56600115/2076973
        flat = preprocessed_frame.reshape(1, preprocessed_frame.total() * channels);
        img_data = preprocessed_frame.isContinuous()? flat : flat.clone(); 

        // Feed data to input tensor
        input.set_data(img_data, {1, rows, cols, 3});
        
        // Run and show predictions
        m.run(input, prediction);
        
        // Get tensor with predictions
        predictions = prediction.Tensor::get_data<float>();
        int index = (int) std::distance(predictions.begin(),std::max_element(predictions.begin(), predictions.end()));
        
        // Put text
        text = fps_str + " " + labels[index];
        cv::putText(frame, 
            text,
            puttext_point, // Coordinates
            cv::FONT_HERSHEY_SIMPLEX, // Font
            1.0, // Scale. 2.0 = 2x bigger
            puttext_color, // BGR Color
            2, // Line Thickness (Optional)
            2); // Anti-alias (Optional)

        // Show image
        imshow( "Frame", frame );

        if( waitKey(1) == 27 ) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}

Mat preprocessImage(Mat image, int size){
    Mat result, cropped; 
    int h = image.rows;
    int w = image.cols;
    
    int start_col = (int) std::max(w/2.0-h/2.0, 0.0);
    int end_col   = (int) std::min(start_col+h, w);

    Rect roi(start_col, 0, end_col, h);
    cropped = image(roi); // crop image
    resize(cropped, result, cv::Size(size,size)); // resize to final size
    cvtColor(result, result, CV_BGR2RGB); // RGB
    result.convertTo(result, CV_32F, 1.0/255.0); // 0 - 1.0 range
    int c = result.channels();
    return result;
}