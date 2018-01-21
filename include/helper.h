#ifndef HELPER
#define HELPER
#include <iostream>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string.h>
#include <string>
#include <sstream>
#include <iomanip> // setprecision
#include <fstream>
#include <stdlib.h>
#include <cmath>
#include <ctime>


#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/ximgproc/segmentation.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;

int getdir (string dir, vector<string> &files);

void shuffleRows(cv::Mat &feats, cv::Mat &labels);

float compute_Iou(cv::Rect rect1, cv::Rect rect2, int debug=0);

void sort(cv::Mat &mat);

void display_output(cv::Mat imOut, cv::Rect currRect, int i_label,  float conf);


#endif
