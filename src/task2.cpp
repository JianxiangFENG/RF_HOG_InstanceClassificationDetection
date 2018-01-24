#include <iostream>
#include <vector>
#include <string>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/ml.hpp"
#include "helper.h"
#include "BaggingTrees.h"

using namespace std;

int main(int argc, char **argv){

	int num_Clas = 3;
	cv::HOGDescriptor hog;
	cv::Size blockStride(8, 8);
	cv::Size blockSize(16, 16);
	cv::Size cellSize(8, 8);
	hog.blockStride = blockStride;
	hog.blockSize = blockSize;
	hog.cellSize = cellSize;
	hog.nbins = 9; 
	cv::Size reSize(64, 64);
	hog.winSize =  reSize;


	//************* construct training set ***************
	// number of rows is number of sample, number of cols is number of features
	string trainingSetPath = "../data/task2/train";
	cv::Mat train_data;
	cv::Mat train_labels;
    for (unsigned idx_class = 0; idx_class<num_Clas; idx_class++){
    	vector<string> files = vector<string>();
    	string idx_class_str;
    	stringstream ss;
    	ss << idx_class;
    	ss >> idx_class_str;
    	getdir(trainingSetPath+"/0"+idx_class_str,files);
    	// cout<< "trainingSetPath is: "<<trainingSetPath+"/0"+idx_class_str<<endl;
    	// cout<< "files size is: "<<files.size()<<endl;	


    	for(unsigned idx_img = 0; idx_img<files.size(); idx_img++){
    		// cout<< "files are: "<<files[idx_img]<<endl;
    		string img_path = trainingSetPath+"/0"+idx_class_str+"/"+files[idx_img];
	    	cv::Mat image = cv::imread(img_path);
	    	cv::resize(image, image, reSize);
			vector<float> descriptors;
			cv::Mat gray;
			cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
			if(gray.empty()){
				cout<<"empty training image exists!"<<idx_img<<",img_path:"<< img_path<<"image size:"<<gray.size()<<endl;
				return -1;
			} 
			hog.compute(gray, descriptors, cv::Size(0,0), cv::Size(0,0));
			//convert vector to matrix
			int cols = descriptors.size();
			cv::Mat train_dataTemp = cv::Mat(1, cols, CV_32FC1);
			memcpy(train_dataTemp.data, descriptors.data(), descriptors.size()*sizeof(float));
			train_data.push_back(train_dataTemp);
			train_labels.push_back(int(idx_class));
    	}
    }
    
    //cout<<"train_data rows:"<<train_data.rows<<". train_data cols:"<<train_data.cols<<endl;
    cout<<"train_data size(width,height):"<<train_data.size()<<endl;	
    cout<<"train_labels size(width,height):"<<train_labels.size()<<endl;	
    // for(unsigned check_idx=0;check_idx<train_labels.rows; check_idx += 40){
    // 	cout<<train_labels.at<float>(check_idx,0)<<endl;
    // }
    cout<<"after shuffling!"<<endl;
    shuffleRows(train_data, train_labels);
    // for(unsigned check_idx=0;check_idx<train_labels.rows; check_idx += 40){
    // 	cout<<train_labels.at<float>(check_idx,0)<<endl;
    // }


    //****************************************************
	//******************** Training **********************
    //****************************************************

    int DT_MaxDepth = 30;
    int DT_MinSample = 4;

    int RF_MaxDepth = 30;
    int RF_MinSample = 3;
    int num_tree = 50;
	

    // Decision Tree
    cout<<"Define DTrees!"<<endl;
    cv::Ptr<cv::ml::DTrees> dtree = cv::ml::DTrees::create();  
    dtree->setMaxCategories(num_Clas);  
    dtree->setMaxDepth(DT_MaxDepth);  
    dtree->setMinSampleCount(DT_MinSample);  
    dtree->setCVFolds(1);  
  	cout<<"Start training DTrees!"<<endl;
    dtree->train(cv::ml::TrainData::create(train_data, cv::ml::ROW_SAMPLE, train_labels)); 
  	cout<<"Finish training DTrees!"<<endl;  
    

    // Bagging Tree
   //  cout<<"Define BaggingTrees("<<num_tree<<" trees)!"<<endl;
   //  BaggingTrees BTrees;
   //  BTrees.create(num_tree, DT_MaxDepth, DT_MinSample, num_Clas);  
  	// cout<<"Start training BaggingTrees!"<<endl;
  	// BTrees.train(train_data, train_labels);
  	// cout<<"Finish training BaggingTrees!"<<endl;


  	// Random Forest
    cout<<"Define RandomForest("<<num_tree<<" trees)!"<<endl;
    cv::Ptr<cv::ml::RTrees> RForest = cv::ml::RTrees::create();  
   //  RForest->setMaxCategories(num_Clas);  
   //  RForest->setMaxDepth(RF_MaxDepth);  
   //  RForest->setMinSampleCount(RF_MinSample);  
   //  RForest->setCVFolds(1);  
   //  RForest->setActiveVarCount(0);  
   //  RForest->setTermCriteria(cv::TermCriteria(cv::TermCriteria::COUNT, num_tree, 0.f));  
  	// cout<<"Start training RForest!"<<endl;
   //  RForest->train(cv::ml::TrainData::create(train_data, cv::ml::ROW_SAMPLE, train_labels)); 
  	// cout<<"Finish training RForest!"<<endl;
   // 	RForest->save("../model/best_task2_RF.xml");

    cout<<"Load Previous trained RandomForest"<<endl;
    std::string model_path = argv[1];
    RForest = RForest->load(model_path);
    cout<<"Load Previous trained RandomForest successfully"<<endl;



 	//****************************************************
	//******************** Testing ***********************
    //****************************************************


    //*************** construct test set ***************
    string testSetPath = "../data/task2/test";
	cv::Mat test_data;
	cv::Mat test_labels;
	for (unsigned idx_class = 0; idx_class<num_Clas; idx_class++){
		vector<string> files = vector<string>();
		string idx_class_str;
		stringstream ss;
		ss << idx_class;
		ss >> idx_class_str;
		getdir(testSetPath+"/0"+idx_class_str,files);
		cout<< "testSetPath is: "<<testSetPath+"/0"+idx_class_str<<endl;
		cout<< "files size is: "<<files.size()<<endl;	

		for(unsigned idx_img = 0; idx_img<files.size(); idx_img++){
			string img_path = testSetPath+"/0"+idx_class_str+"/"+files[idx_img];
	    	cv::Mat image = cv::imread(img_path);
	    	cv::resize(image, image, reSize);
			vector<float> descriptors;
			cv::Mat gray;
			cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
			if(gray.empty()){
				cout<<"empty image exists!"<<endl;
				return -1;
			} 
			hog.compute(gray, descriptors, cv::Size(0,0), cv::Size(0,0));
			//convert vector to matrix
			int cols = descriptors.size();
			cv::Mat testdataTemp = cv::Mat(1, cols, CV_32FC1);
			memcpy(testdataTemp.data, descriptors.data(), descriptors.size()*sizeof(float));
			test_data.push_back(testdataTemp);
			test_labels.push_back(float(idx_class));
		}
	}
	cout<<"test_data size(width,height):"<<test_data.size()<<endl;	
    cout<<"test_labels size(width,height):"<<test_labels.size()<<endl;	

	//*************** evaluating performance of DTree***************
	cv::Mat DTtest_results;
	int predict_samples_number = test_labels.rows;
    dtree->predict(test_data, DTtest_results);  
    if(DTtest_results.rows != predict_samples_number){
    	cout<<"number of DTtest_results is not identical with number of test_samples!"<<endl;
    	return -1;
    }
    
    int count{0};  
    for (int i = 0; i < predict_samples_number; ++i) {  
        float value1 = ((float*)test_labels.data)[i];  
        float value2 = ((float*)DTtest_results.data)[i];  
       
        if (int(value1) == int(value2)) ++count;  
    }  
    cout<<"DTree's accuracy: "<<count * 1.f / predict_samples_number<<endl;  
	// cout<<DTtest_results<<endl;


    /*
    //*************** evaluating performance of BaggingTrees ***************
	cv::Mat BTtest_results = cv::Mat::zeros(predict_samples_number, 2, CV_32FC1);
	cv::Mat BTtest_votes;
	BTrees.predict(test_data, BTtest_votes); 
	// cout<<BTtest_votes<<endl;
    // get the major vote and compute the probability and accuracy
    count = 0;  
    double max;
    int* max_idx = new int;
    for (int i = 0; i < predict_samples_number; ++i) {  
        float value1 = ((float*)test_labels.data)[i];  
		cv::minMaxIdx(BTtest_votes.row(i), NULL, &max, NULL, max_idx);
		// cout<<"max: "<<max << ". max_idx: "<< (max_idx)[0]<<(max_idx)[1]<<endl;
		BTtest_results.row(i).at<float>(0) = max_idx[1]; 
		float score = max/num_tree; 
		BTtest_results.row(i).at<float>(1) = score;
        float value2 = max_idx[1];  
        if (int(value1) == int(value2)) ++count;  
    }  
    cout<<"BaggingTrees's accuracy: "<<count * 1.f / predict_samples_number<<endl;  
    
    */

    //*************** evaluating performance of RForest ***************
	cv::Mat RTtest_counts = cv::Mat::zeros(predict_samples_number, num_Clas, CV_32FC1);
	cv::Mat RTtest_results = cv::Mat::zeros(predict_samples_number, 2, CV_32FC1);
	// cout<<"RTtest_counts: "<<RTtest_counts<<endl;
	cv::Mat RTtest_votes;
	RForest->predict(test_data); 
    RForest->getVotes(test_data, RTtest_votes, 0);
    // cout<<"RTtest_votes: "<<RTtest_votes.size()<<endl;
    // if(RTtest_votes.rows != predict_samples_number){
    // 	cout<<"number of RTtest_votes is not identical with number of test_samples!"<<endl;
    // 	return -1;
    // }
    // count the votes from each tree for each class
    // for(unsigned i_row = 0; i_row<RTtest_votes.rows; ++i_row){
    // 	for(unsigned i_col = 0; i_col<RTtest_votes.cols; ++i_col){
    // 		RTtest_counts.row(i_row).at<float>(RTtest_votes.row(i_row).at<float>(i_col))++;
    // 	}
    // } 

    // get the major vote and compute the probability and accuracy
    num_tree = 300;
    count = 0;  
    double max;
    int* max_idx = new int;
    for (int i = 1; i < predict_samples_number+1; ++i) {  
        float value1 = ((float*)test_labels.data)[i-1];  
		cv::minMaxIdx(RTtest_votes.row(i), NULL, &max, NULL, max_idx);
		// cout<<"max: "<<max << ". max_idx: "<< (max_idx)[0]<<(max_idx)[1]<<endl;
		RTtest_results.row(i-1).at<float>(0) = max_idx[1]; 
		float score = max/num_tree; 
		RTtest_results.row(i-1).at<float>(1) = score;
        float value2 = max_idx[1];  
        cout<<"confidence:"<<score<<", class:"<<value2<<", ground truth:"<< value1<<endl; 
        if (int(value1) == int(value2)){
            ++count; 
        }   
    }  
    cout<<"RandomForest's accuracy: "<<count * 1.f / predict_samples_number<<endl;  
    
   
    // cout<<"RTtest_results: "<<RTtest_results<<endl;
    // cout<<"RTtest_counts: "<<RTtest_counts<<endl;
    // cout<<"RTtest_votes: "<<RTtest_votes<<endl;
	
	return 0;
}