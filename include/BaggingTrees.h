#ifndef BAGGINGTREES
#define BAGGINGTREES
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <sstream>

#include <opencv2/opencv.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/ml.hpp"

class BaggingTrees{
public:
	std::vector< cv::Ptr<cv::ml::DTrees> > dtrees;

	void create(int n_trees, int mDepth, int mSample, int mCategories);

	void train(cv::Mat train_data, cv::Mat train_labels);

	void predict(cv::Mat test_data, cv::Mat &votes);

private:
	int num_trees;
    int maxDepth;
    int minSample;
    int maxCategories;
    int CVFolds = 1;
};


#endif