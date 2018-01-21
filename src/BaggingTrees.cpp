#include "BaggingTrees.h"
#include <random>
#include <iostream>

void BaggingTrees::create(int n_trees, int mDepth, int mSample, int mCategories){
	num_trees = n_trees;
	maxDepth = mDepth; 			
	minSample = mSample; 
	maxCategories = mCategories;
	CVFolds = 1;
	for(unsigned i_tree = 0; i_tree<num_trees ; ++i_tree){
		cv::Ptr<cv::ml::DTrees> dtree = cv::ml::DTrees::create();  
	    dtree->setMaxCategories(maxCategories);  
	    dtree->setMaxDepth(maxDepth);  
	    dtree->setMinSampleCount(minSample);  
	    dtree->setCVFolds(CVFolds); 
	    dtrees.push_back(dtree);
	} 
}

void BaggingTrees::train(cv::Mat train_data, cv::Mat train_labels){
	for(unsigned i_tree = 0; i_tree<num_trees ; ++i_tree){
		// get boostrap from training set
		int num_sample = train_data.rows;
		cv::Mat subset_data;
		cv::Mat subset_labels;
		std::random_device rd;  //Will be used to obtain a seed for the random number engine
	    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	    gen.seed(i_tree);
	    std::uniform_int_distribution<> dis(0, num_sample-1);

	    for (int idx_sample=0; idx_sample<num_sample; ++idx_sample){
	    	// std::cout<<dis(gen)<<std::endl;
	    	int idx = int(dis(gen));
	    	subset_data.push_back(train_data.row(idx));
	    	subset_labels.push_back(train_labels.row(idx));
	    }
	    // if (i_tree == 0 || i_tree == 1 || i_tree == 2) std::cout<<subset_labels<<std::endl;
		cv::Ptr<cv::ml::TrainData> data = cv::ml::TrainData::create(subset_data, cv::ml::ROW_SAMPLE, subset_labels);
		dtrees[i_tree]->train(data);
	}
	 
}

void BaggingTrees::predict(cv::Mat test_data, cv::Mat &votes){

	int predict_samples_number = test_data.rows;
	cv::Mat Votes = cv::Mat::zeros(predict_samples_number, maxCategories, CV_32SC1);
	for(unsigned i_tree = 0; i_tree<num_trees; ++i_tree){
		cv::Mat results;
	    dtrees[i_tree]->predict(test_data, results); 
	    for(unsigned i_sample = 0; i_sample<predict_samples_number; ++i_sample){
	    	// std::cout<<results.at<float>(i_sample)<<std::endl;
	    	Votes.row(i_sample).at<int>(results.at<float>(i_sample))++;
	    }
	}
	// std::cout<<Votes<<std::endl;
	Votes.copyTo(votes);
}