#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip> // setprecision
#include <fstream>
#include <stdlib.h>
#include <cmath>
#include <ctime>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/ximgproc/segmentation.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "helper.h"
#include "BaggingTrees.h"

using namespace std;

int main(int argc, char** argv){
	

    // algorithm params
    float counter = 0;
    float true_pos = 0;
    float total_gt = 44*3; 
	int num_Clas = 4;
    float conf_thresholds[9] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}; //(float)atoi(argv[2])/10;
    float Iou_threshold = 0.25;
    int searchStrategys[3] = {0,1,2}; // atoi(argv[1]);

    // file params
    string trainingSetPath = "../data/task3/train";
    string testingSetPath = "../data/task3/test";
    string gtPath = "../data/task3/gt";

    // Random Forest
    int num_tree = 300;
    int RF_MaxDepth = 100;
    int RF_MinSample = 3;
    double pi = 3.1415926535897;
    
	// initialize hog descriptors
	cv::HOGDescriptor hog;
	cv::Size blockStride(8, 8);
	cv::Size blockSize(16, 16);
	cv::Size cellSize(8, 8);
	hog.blockStride = blockStride;
	hog.blockSize = blockSize;
	hog.cellSize = cellSize;
	hog.nbins = 9; 
	// cv::Size reSize(128, 112);
    cv::Size reSize(64, 64);
	hog.winSize =  reSize;

	


    //*********************************************
    //*                 training                  *
    //*********************************************
    	
    /*
	//************* construct training set ***************
	// number of rows is number of sample, number of cols is number of features
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
    		// cout<< files[idx_img]<<endl;
    		string img_path = trainingSetPath+"/0"+idx_class_str+"/"+files[idx_img];
	    	cv::Mat image = cv::imread(img_path);
	    	cv::resize(image, image, reSize);
			vector<float> descriptors;
			cv::Mat gray;
			cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
			if(gray.empty()){
				cout<<"empty image exists!"<<endl;
				continue;
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
    //************* finish constructing training set ***************
    */
    
    
    cv::Ptr<cv::ml::RTrees> RForest = cv::ml::RTrees::create(); 

    // use pre-trained RF
    cout<<"Load Previous trained RandomForest"<<endl;
    RForest = RForest->load("../model/task3_RF_64win_origData.xml");

    // train new RF
   //  cout<<"Define RandomForest("<<num_tree<<" trees)!"<<endl; 
   //  RForest->setMaxCategories(num_Clas);  
   //  RForest->setMaxDepth(RF_MaxDepth);  
   //  RForest->setMinSampleCount(RF_MinSample);  
   //  RForest->setCVFolds(1);  
   //  RForest->setActiveVarCount(train_data.cols);  
   //  RForest->setTermCriteria(cv::TermCriteria(cv::TermCriteria::COUNT, num_tree, 0.f));  
  	// cout<<"Start training RForest!"<<endl;
   //  RForest->train(cv::ml::TrainData::create(train_data, cv::ml::ROW_SAMPLE, train_labels)); 
  	// cout<<"Finish training RForest!"<<endl;
   // 	RForest->save("../model/task3_RF_64window_allFeat.xml");



    // ############################ start testing ##########################
    for(int i_strategy = 3; i_strategy<4; i_strategy++){
        for(int i_conf_thres = 0; i_conf_thres<9; i_conf_thres++){
            
            // !!!reset to 0
            counter = 0;
            true_pos = 0;

            float conf_threshold = conf_thresholds[i_conf_thres];
            int searchStrategy = searchStrategys[i_strategy];
            clock_t time_stt = clock();
            clock_t initial_time = clock();    

            for(unsigned idx_test = 0; idx_test<44; ++idx_test){
                cout<<"Processing image "<<idx_test<<endl;
                cout<<"Strategy "<<i_strategy<<endl;
                cout<<"Conf_threshold is "<<conf_threshold<<endl;
                cout<<"Iou_threshold is "<<Iou_threshold<<endl;
            	// read image
            	string idx;
            	stringstream ss;
            	ss << idx_test;
            	ss >> idx;
                ifstream gtfile;
                string img_path;

                if (idx_test<10){
                    img_path = testingSetPath+"/000"+idx+".jpg";
                    gtfile.open(gtPath+"/000"+idx+".gt.txt",ifstream::in);
                }
                else
                {
                    img_path = testingSetPath+"/00"+idx+".jpg";
                    gtfile.open(gtPath+"/00"+idx+".gt.txt",ifstream::in);
                }
                	

                //*********************************************
                //*             read ground truth             *
                //*********************************************
        	    string line;
        	    cv::Mat gtMat = cv::Mat::zeros(3,4,CV_32SC1);
        		if (gtfile.is_open())
        		{
        		    while ( getline (gtfile,line) )
        		    {
        		    	std::istringstream iss(line);
        		    	int label, x, y, bottom_right_x, bottom_right_y;
        				if (!(iss >> label >> x >> y >> bottom_right_x >> bottom_right_y)) { break; } 
        				gtMat.row(label).at<int>(0) = x;
        				gtMat.row(label).at<int>(1) = y;
        				gtMat.row(label).at<int>(2) = bottom_right_x - x;
        				gtMat.row(label).at<int>(3) = bottom_right_y - y;
        		    }
        		    gtfile.close();
        		}
        		else cout << "Unable to open file"<<endl; 
        	    // cout<<gtMat<<endl;



        		//*********************************************
                //*       generate region proposals           *
                //*********************************************
                cv::Mat imIn = cv::imread(img_path);
                std::vector<cv::Rect> rects;
            	cout<<"imageIn'size is: "<<imIn.size()<<endl;
            	cv::Mat imOut = imIn.clone();

                //generate region proposals
                if(i_strategy <3){
                    time_stt = clock();
                	cv::Ptr<cv::ximgproc::segmentation::SelectiveSearchSegmentation> 
                		ssbb = cv::ximgproc::segmentation::createSelectiveSearchSegmentation();
            	    ssbb->setBaseImage(imIn);
            	    switch(searchStrategy){
            	    	case 0:ssbb->switchToSingleStrategy();
            	    		   	break;
            	    	case 1:ssbb->switchToSelectiveSearchFast();
            	    			break;
            	    	case 2:ssbb->switchToSelectiveSearchQuality();
            	    			break;
            	    	default:cout<<"Input Search Strategy in first argument: \n0:switchToSingleStrategy;\n1:switchToSelectiveSearchFast;\n2:switchToSelectiveSearchQuality"<<endl;
            	    			return -1;
            	    }
                	ssbb->process(rects); 
                    cout <<"Time use in generating region proposals is " <<1000*  (clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl;
            	   cout << "Total Number of Region Proposals: " << rects.size() << endl;
                }
                else if(i_strategy == 3){
                 //generate sliding windows
                    time_stt = clock();
                    int scale[3] = {80,100,120};
                    for(unsigned idx_scale = 0; idx_scale<3; ++idx_scale){
                        for(unsigned x=0; x<(imIn.cols-scale[idx_scale]); x=x+1){
                            for(unsigned y=0; y<(imIn.rows-scale[idx_scale]); y=y+1){
                                rects.push_back(cv::Rect(x,y,scale[idx_scale],scale[idx_scale]));
                            }
                        }
                	}
                    cout <<"Time use in generating sliding windows is " <<1000*  (clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl;
                    cout << "Total Number of sliding windows: " << rects.size() << endl;
                }



                
                //*********************************************
                //*        classify region proposals          *
                //*********************************************
                int predict_samples_number = rects.size();
                //RTtest_results: contain results without low confidence
                //first col contains predicted label, sceond col contains prob. of this predicted label,
        		//third row contains idx of rectangles among potential windows, number of row is number of test sample
        		cv::Mat RTtest_results; 
                time_stt = clock();
                for(int i = 0; i < rects.size(); i++) {
            		cv::Mat test_data;
                    if(rects[i].width>=80 && rects[i].height>=80&&rects[i].width<160 && rects[i].height<160 ){
                    	//resize image to specified size and extract HOG features
                    	cv::Mat image(imIn, rects[i]);
                    	cv::resize(image, image, reSize);
        				vector<float> descriptors;
        				cv::Mat gray;
        				cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        				hog.compute(gray, descriptors, cv::Size(0,0), cv::Size(0,0));
        				//convert vector to matrix
        				int cols = descriptors.size();
        				cv::Mat testdataTemp = cv::Mat(1, cols, CV_32FC1);
        				memcpy(testdataTemp.data, descriptors.data(), descriptors.size()*sizeof(float));
        				test_data.push_back(testdataTemp);
        				// feed test_data into classifier

        			 	//RTtest_votes: number of (row-1) is number of test sample, each col represents the votes of this label in the forest
        				cv::Mat RTtest_votes;
        				
        				RForest->predict(test_data); 
        			    RForest->getVotes(test_data, RTtest_votes, 0);
        			    double max_vote;        // votes for predicted class label
        			    int* max_idx = new int; //predicted class label
        				cv::minMaxIdx(RTtest_votes.row(1), NULL, &max_vote, NULL, max_idx);
        				cv::Mat temp = cv::Mat::zeros(1, 3, CV_32FC1);
        				if( float(max_vote/num_tree) > conf_threshold && max_idx[1]!=3){
                            temp.at<float>(0) = max_idx[1];
                            temp.at<float>(1) = float(max_vote/num_tree);
                            temp.at<float>(2) = i;
                            RTtest_results.push_back(temp);
                        }

                        // check recall of poetential bbs
                        for(int label = 0; label<3; label++){
                            cv::Rect gtRect;
                            gtRect = cv::Rect(gtMat.row(label).at<int>(0),gtMat.row(label).at<int>(1), gtMat.row(label).at<int>(2),gtMat.row(label).at<int>(3));
                            if(compute_Iou(rects[i], gtRect) > 0.6){
                                stringstream ss0; 
                                ss0<<fixed << setprecision(2) <<idx_test;
                                stringstream ss1; 
                                ss1<<fixed << setprecision(2) <<i;
                                stringstream ss2; 
                                ss2<<fixed << setprecision(2) <<i_strategy;
                                stringstream ss3; 
                                ss3<<fixed << setprecision(2) <<label;
                                cv::imwrite(trainingSetPath+"/Iou0_6_strategy"+ss2.str()+"/"+ss0.str()+"_"+ss3.str()+".png", imIn(rects[i]));
                            }
                        }
                        
                    }
                }
                cout <<"Time use in classification is " <<1000*  (clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl;
                // cout<<"RTtest_results.size: "<<RTtest_results.size()<<endl;
        		

                
                //*********************************************
                //*        non maximum suppression            *
                //*********************************************
                cout << "Non maximum suppression!"<< endl;
                time_stt = clock();
                sort(RTtest_results); // reorder RTtest_results into decreasing order according to the score
                // cout<<"Sorted!"<<endl;
                std::vector<cv::Rect> BB_results;
                std::vector<float> BB_results_confidence;
                int numBoxes = RTtest_results.rows;
                std::vector< int > is_suppressed(numBoxes);
                // initialize flag(is_suppressed)
                for(int i = 0; i<numBoxes; i++){
                    is_suppressed[i] = 0;
                    // cout<<"RTtest_results score: "<< RTtest_results.row(i).at<float>(0)<<","<<RTtest_results.row(i).at<float>(1) << endl;
                }
                for(int i = 0; i < numBoxes; i++) {
                    if (!is_suppressed[i]){
                        for(int j = i+1; j < numBoxes; j++){
                            if(!is_suppressed[j]){
                                int idx1 = RTtest_results.row(i).at<float>(2);
                                int idx2 = RTtest_results.row(j).at<float>(2);
                                cv::Rect m1 = rects[idx1];
                                cv::Rect m2 = rects[idx2];
                                float Iou = compute_Iou(m1, m2);
                                if(Iou > Iou_threshold){
                                    is_suppressed[j] = 1;
                                }
                            }
                        }
                    }
                	
                }
                cout <<"Time use in non_maximum_suppresion is " <<1000*  (clock() - time_stt)/(double)CLOCKS_PER_SEC <<"ms" << endl;
                



                //*********************************************
                //*       generate qualified outputs          *
                //*********************************************
                // compute Intersection over Union and visualize results   
                cv::Rect gtRect;
                cout << "Display results!"<< endl;  
                for(int i_bb = 0; i_bb<numBoxes; i_bb++){
                    if(!is_suppressed[i_bb]){

                        counter ++; // accumulate positive prediction
                        cout<<"counter:"<<counter<<endl;
                        float conf = RTtest_results.row(i_bb).at<float>(1);
                        int label = RTtest_results.row(i_bb).at<float>(0);
                        
                        cv::Rect currRect = rects[RTtest_results.row(i_bb).at<float>(2)];
                        gtRect = cv::Rect(gtMat.row(label).at<int>(0),gtMat.row(label).at<int>(1), gtMat.row(label).at<int>(2),gtMat.row(label).at<int>(3));
                        
                        if(compute_Iou(currRect, gtRect) > 0.5){
                            true_pos++;
                            cout<<"true_pos: "<<true_pos<<endl;
                        }
                        else
                        {// add false positive into training set
                            cout<<"false_positive!"<<endl;
                            stringstream ss1; 
                            ss1<<fixed << setprecision(2) <<(rand()%10000+1);
                            cv::imwrite(trainingSetPath+"/false_negative_SlidingWindows/"+ss1.str()+".jpg", imOut(currRect));
                        }
                        // display_output(imOut, currRect, label, conf);
                        
                    }
                }

                // cout << "Display output!"<< endl;  
                // imshow("Output", imOut);
                // cout<<"############### single image ################"<<endl;
                // cout<<"true_pos is:"<<true_pos<<endl;
                // cout<<"############### single image ################"<<endl;
                // cv::imshow("output", imOut);
                // cv::waitKey();
            } // end of image iteration
            

            cout<<"###############################"<<endl;
            cout<<"precision is:"<<true_pos/counter<<endl;
            cout<<"recall is:"<<true_pos/total_gt<<endl;
            cout<<"###############################"<<endl;
            string textToSave;
            ofstream saveFile ("../Results/task3_Iou0.25_64win_origData.txt", std::ios_base::app);
            stringstream ss1;
            ss1<<fixed << setprecision(2) <<i_strategy;
            saveFile <<"Selective strategy: "<<ss1.str()<<"\n";
            ss1.str(string());
            ss1<<fixed << setprecision(2) <<Iou_threshold;
            saveFile <<"Iou_threshold: "<<ss1.str()<<"\n";
            ss1.str(string());
            ss1<<fixed << setprecision(2) <<conf_threshold;
            saveFile <<"conf_threshold: "<<ss1.str()<<"\n";
            ss1.str(string());
            ss1<<fixed << setprecision(4) <<(true_pos/counter);
            saveFile <<"precision: "<<ss1.str()<<"\n";
            ss1.str(string());
            ss1<<fixed << setprecision(4) <<(true_pos/total_gt);
            saveFile <<"recall: "<<ss1.str()<<"\n";
            ss1.str(string());
            ss1<<fixed << setprecision(4) <<(1000*  (clock() - initial_time)/(double)CLOCKS_PER_SEC);
            saveFile <<"process 43 images cost(ms): "<<ss1.str()<<"\n";
            ss1.str(string());
            saveFile <<"\n\n";
            saveFile.close();
            

            
        }// end of conf_threshold interation
    } // end of strategy iteration
    return 0;
}