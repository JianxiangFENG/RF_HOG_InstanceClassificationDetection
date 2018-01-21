#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/objdetect.hpp"
#include "visual_hog.h"
//#include <string>

//using namespace std;

int main(int argc, char **argv){
	
	std::string img_path = argv[1];
	int scale_factor = atoi(argv[2]);
	cv::Mat image = cv::imread(img_path);
	std::cout<< "Image path is: "<<img_path<<". and its size:"<<image.size()<<std::endl;
	if (image.empty()){
		std::cout<< "image is empty!"<<std::endl;
		return -1;
	}

	// extract hog features
	cv::HOGDescriptor hog;
	cv::Size imgSize = image.size();
	cv::Size blockStride(8, 8);
	cv::Size blockSize(16, 16);
	cv::Size cellSize(16, 16);
	hog.blockStride = blockStride;
	hog.blockSize = blockSize;
	hog.cellSize = cellSize;
	hog.nbins = 6;

	int padding_width = (imgSize.width-blockSize.width)%blockStride.width;
	int padding_height = (imgSize.height-blockSize.height)%blockStride.height;
	// cv::Size reSize(imgSize.width+(blockStride.width-padding_width), 
	// 			imgSize.height+(blockStride.height-padding_height));
	cv::Size reSize(128, 112);
	cv::resize(image, image, reSize);
	imgSize = image.size();
	hog.winSize = imgSize;

	std::cout<< "WinSize of HOGdescriptors is: "<<hog.winSize<<std::endl;
	std::cout<< "BlockSize of HOGdescriptors is: "<<hog.blockSize<<std::endl;
	std::cout<< "CellSize of HOGdescriptors is: "<<hog.cellSize<<std::endl;
	std::cout<< "BlockStride of HOGdescriptors is: "<<hog.blockStride<<std::endl;
	std::cout<< "Nbins of HOGdescriptors is: "<<hog.nbins<<std::endl;
	std::vector<float> descriptors;
	cv::Mat gray;
	cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	hog.compute(gray, descriptors, cv::Size(0,0), cv::Size(0,0));
	std::cout<< "Size of HOGdescriptors is: "<<descriptors.size()<<std::endl;

	// visualize hog descriptors 
	// std::string windowName = "Task1 image";
	// cv::namedWindow(windowName);
	// cv::imshow(windowName, image);
	// cv::destroyWindow(windowName);
	visualizeHOG(image, descriptors, hog, scale_factor);

	cv::waitKey(0);
	return 0;
}