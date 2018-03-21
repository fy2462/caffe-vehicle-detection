// Brief Sample of using OpenCV dnn module in real time with device capture, video and image.
// VIDEO DEMO: https://www.youtube.com/watch?v=NHtRlndE2cg

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>

using namespace std;
using namespace cv;
using namespace cv::dnn;

static const char* about =
"This sample uses You only look once (YOLO)-Detector (https://arxiv.org/abs/1612.08242) to detect objects on camera/video/image.\n"
"Models can be downloaded here: https://pjreddie.com/darknet/yolo/\n"
"Default network is 416x416.\n"
"Class names can be downloaded here: https://github.com/pjreddie/darknet/tree/master/data\n";

static const char* params =
"{ help           | false | print usage         }"
"{ cfg            |       | model configuration }"
"{ model          |       | model weights       }"
"{ camera_device  | 0     | camera device number}"
"{ source         |       | video or image for detection}"
"{ style          | box   | box or line style draw }"
"{ min_confidence | 0.3  | min confidence      }"
"{ class_names    |       | File with class names, [PATH-TO-DARKNET]/data/coco.names }";

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, params);

    if (parser.get<bool>("help"))
    {
        cout << about << endl;
        parser.printMessage();
        return 0;
    }

    String modelConfiguration = parser.get<String>("cfg");
    String modelBinary = parser.get<String>("model");
    String image_source = parser.get<String>("source");
    String class_name_label = parser.get<String>("class_names");
    class_name_label = "/Users/fengyan04/Github/yolo-detection/yolo/yolo-voc/voc.names";
    //image_source = "/Users/fengyan04/Github/yolo-detection/video_3.mp4";
    modelConfiguration = "/Users/fengyan04/Github/yolo-detection/yolo/yolo-voc/yolo-voc.cfg";
    modelBinary = "/Users/fengyan04/Github/yolo-detection/yolo/yolo-voc/yolo-voc-544.weights";
    String output_video_path = "/Users/fengyan04/Github/yolo-detection/videoOut1.mp4";

    //! [Initialize network]
    dnn::Net net = readNetFromDarknet(modelConfiguration, modelBinary);
    //! [Initialize network]

    if (net.empty())
    {
        cerr << "Can't load network by using the following files: " << endl;
        cerr << "cfg-file:     " << modelConfiguration << endl;
        cerr << "weights-file: " << modelBinary << endl;
        cerr << "Models can be downloaded here:" << endl;
        cerr << "https://pjreddie.com/darknet/yolo/" << endl;
        exit(-1);
    }

    VideoCapture cap;

    if (image_source.empty())
    {
        int cameraDevice = parser.get<int>("camera_device");
        cap = VideoCapture(cameraDevice);
        if(!cap.isOpened())
        {
            cout << "Couldn't find camera: " << cameraDevice << endl;
            return -1;
        }
    }
    else
    {
        cap.open(image_source);
        if(!cap.isOpened())
        {
            cout << "Couldn't open image or video: " << parser.get<String>("video") << endl;
            return -1;
        }
    }

    vector<String> classNamesVec;
    ifstream classNamesFile(class_name_label.c_str());
    if (classNamesFile.is_open())
    {
        string className = "";
        while (std::getline(classNamesFile, className))
            classNamesVec.push_back(className);
    }

    String object_roi_style = parser.get<String>("style");
    /*
     * CV_FOURCC('P', 'I', 'M', '1') = MPEG-1 codec
     * CV_FOURCC('M', 'J', 'P', 'G') = motion-jpeg codec
     * CV_FOURCC('M', 'P', '4', '2') = MPEG-4.2 codec
     * CV_FOURCC('D', 'I', 'V', '3') = MPEG-4.3 codec
     * CV_FOURCC('D', 'I', 'V', 'X') = MPEG-4 codec
     * CV_FOURCC('U', '2', '6', '3') = H263 codec
     * CV_FOURCC('I', '2', '6', '3') = H263I codec
     * CV_FOURCC('F', 'L', 'V', '1') = FLV1 codec
     */
    double rate = cap.get(CV_CAP_PROP_FPS);
    int frameH = (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT);
    int frameW = (int) cap.get(CV_CAP_PROP_FRAME_WIDTH);
    int numFrames = (int) cap.get(CV_CAP_PROP_FRAME_COUNT);
    //frameH = frameW = 544;
    //VideoWriter writer(output_video_path, CV_FOURCC('D', 'I', 'V', 'X'), rate, Size(frameW, frameH));
    int index_frame = 0;
    for(;;)
    {
        Mat frame;
        cap >> frame; // get a new frame from camera/video or read image
        //resize(frame, frame, Size(frameW, frameH));
        index_frame++;
        if (frame.empty())
        {
            waitKey();
            break;
        }

        if (frame.channels() == 4)
            cvtColor(frame, frame, COLOR_BGRA2BGR);

        //! [Prepare blob]
        Mat inputBlob = blobFromImage(frame, 1 / 255.F, Size(416, 416), Scalar(), true, false); //Convert Mat to batch of images
        //! [Prepare blob]

        //! [Set input blob]
        net.setInput(inputBlob, "data");                   //set the network input
        //! [Set input blob]

        //! [Make forward pass]
        Mat detectionMat = net.forward("detection_out");   //compute output
        //! [Make forward pass]

        vector<double> layersTimings;
        double tick_freq = getTickFrequency();
        double time_ms = net.getPerfProfile(layersTimings) / tick_freq * 1000;
       /* putText(frame, format("FPS: %.2f ; time: %.2f ms; percent: %d%%", 1000.f / time_ms, time_ms, (int)(index_frame*100/numFrames)),
                Point(20, 20), 0, 0.5, Scalar(0, 0, 255));*/

        float confidenceThreshold = parser.get<float>("min_confidence");
        for (int i = 0; i < detectionMat.rows; i++)
        {
            const int probability_index = 5;
            const int probability_size = detectionMat.cols - probability_index;
            float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);

            size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
            float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);

            if (confidence > confidenceThreshold)
            {
                float x_center = detectionMat.at<float>(i, 0) * frame.cols;
                float y_center = detectionMat.at<float>(i, 1) * frame.rows;
                float width = detectionMat.at<float>(i, 2) * frame.cols;
                float height = detectionMat.at<float>(i, 3) * frame.rows;
                Point p1(cvRound(x_center - width / 2), cvRound(y_center - height / 2));
                Point p2(cvRound(x_center + width / 2), cvRound(y_center + height / 2));
                Rect object(p1, p2);

                Scalar object_roi_color(0, 255, 0);

                if (object_roi_style == "box")
                {
                    rectangle(frame, object, object_roi_color);
                }
                else
                {
                    Point p_center(cvRound(x_center), cvRound(y_center));
                    line(frame, object.tl(), p_center, object_roi_color, 1);
                }

                String className = objectClass < classNamesVec.size() ? classNamesVec[objectClass] : cv::format("unknown(%d)", objectClass);
                String label = format("%s: %.2f", className.c_str(), confidence);
                int baseLine = 0;
                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                rectangle(frame, Rect(p1, Size(labelSize.width, labelSize.height + baseLine)),
                          object_roi_color, CV_FILLED);
                putText(frame, label, p1 + Point(0, labelSize.height),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));
            }
        }
        //writer << frame;
        imshow("YOLO: Detections", frame);
        if (waitKey(33) >= 0) break;
    }
    //writer.release();
    return 0;
} // main
