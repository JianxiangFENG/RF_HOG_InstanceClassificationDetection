#include "helper.h"

using namespace std;

int getdir (string dir, vector<string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << dir << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
      // cout<<(dirp->d_type & DT_DIR)<<endl;
      // cout<<(string(dirp->d_name) != "..")<<endl;
      // if (dirp->d_type & DT_DIR){
      //     if ((string(dirp->d_name) != "..") && (string(dirp->d_name) != ".")) getdir(dir+"/"+string(dirp->d_name), files);
      // } 
      bool found = (string(dirp->d_name).find(".jpg")!= std::string::npos)||(string(dirp->d_name).find(".JPEG")!= std::string::npos);
    	// cout<<dirp->d_name<<endl;
      // cout<<found<<endl;
      if (found){
          // if((string(dirp->d_name) != ".")&&(string(dirp->d_name) != "..")) 
          files.push_back(string(dirp->d_name));
          // cout<<dirp->d_name<<endl;
    	}
    }
    closedir(dp);
    return 0;
}

void shuffleRows(cv::Mat &feats, cv::Mat &labels)
{
  std::vector <int> seeds;
  for (int cont = 0; cont < feats.rows; cont++)
    seeds.push_back(cont);

  cv::randShuffle(seeds);

  cv::Mat featsTemp;
  cv::Mat labelsTemp;
  for (int cont = 0; cont < feats.rows; cont++){
  	featsTemp.push_back(feats.row(seeds[cont]));
	labelsTemp.push_back(labels.row(seeds[cont]));
  }
    
  feats = featsTemp;
  labels = labelsTemp;
}



float compute_Iou(cv::Rect rect1, cv::Rect rect2, int debug){
    float abs_Iou;
    float x = rect1.x;
    float y = rect1.y;
    float w = rect1.width;
    float h = rect1.height;
    float xp = rect2.x;
    float yp = rect2.y;
    float wp = rect2.width;
    float hp = rect2.height;
    float w_iou;
    float h_iou; 
    if ((xp+wp)<x || (x+w)<xp || (yp+hp)<y || (y+h)<yp){
        abs_Iou = 0;
    }
    else{
        w_iou = ((xp+wp)<(x+w)?(xp+wp):(x+w)) - (x>xp?x:xp);
        h_iou = ((yp+hp)<(y+h)?(yp+hp):(y+h)) - (y>yp?y:yp); 
        abs_Iou= w_iou*h_iou;
    }
    if (debug){
        cout<<"w*h:"<<w*h<<endl;
        cout<<"wp*hp:"<<wp*hp<<endl;
        cout<<"abs_Iou:"<<abs_Iou<<endl;
    }
    return abs_Iou/(w*h + wp*hp - abs_Iou);
}

void sort(cv::Mat &mat){
    // sorte row matrix according to decreasing order of its second column values
    for(int i=0; i<mat.rows; i++){
        for(int j=i+1; j<mat.rows; j++){
            float si = mat.row(i).at<float>(1);
            float sj = mat.row(j).at<float>(1);
            if(si<sj){
                cv::Mat temp = mat.row(j).clone();
                mat.row(i).copyTo(mat.row(j));
                temp.copyTo(mat.row(i));
            }
        }
    }
}

void display_output(cv::Mat imOut, cv::Rect currRect, int i_label,  float conf){     
    stringstream ss;
    ss<<i_label;
    string label = ss.str();
    ss.str(std::string());

    stringstream ss1;
    ss1<<fixed << setprecision(2) <<conf;
    string prob = ss1.str();
    ss1.str(std::string());

    if(i_label == 0){
        // gtRect = cv::Rect(gtMat.row(i_label).at<int>(0),gtMat.row(i_label).at<int>(1), gtMat.row(i_label).at<int>(2),gtMat.row(i_label).at<int>(3));
        // if (compute_Iou(gtRect, currRect) < 0.2){
        //     stringstream ss1;
        //     ss1<<fixed << setprecision(2) <<(rand()%10000+1);
        //     string index = ss1.str();
        //     cv::imwrite(trainingSetPath+"/03/"+index+".jpg", imOut(currRect));
        // }
        // cv::rectangle(imOut, gtRect, cv::Scalar(0, 0, 255));
        cv::rectangle(imOut, currRect, cv::Scalar(0, 0, 255));
        cv::putText(imOut,"label "+label+": "+prob, cv::Point(currRect.x-15,currRect.y-3), 
                 cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1.4);
     }
     else if(i_label == 1){
         // cv::rectangle(imOut, gtRect, cv::Scalar(0, 255, 0));
        // gtRect = cv::Rect(gtMat.row(i_label).at<int>(0),gtMat.row(i_label).at<int>(1), gtMat.row(i_label).at<int>(2),gtMat.row(i_label).at<int>(3));
        // if (compute_Iou(gtRect, currRect) < 0.2){
        //     stringstream ss1;
        //     ss1<<fixed << setprecision(2) <<(rand()%10000+1);
        //     string index = ss1.str();
        //     cv::imwrite(trainingSetPath+"/03/"+index+".jpg", imOut(currRect));
        // }
        // std::vector<cv::Rect> r;
        // r.push_back(currRect);
        // if (r.size()>1){
        //     cout<<compute_Iou(r[0],r[1],1)<<endl;
        // }
         cv::rectangle(imOut, currRect, cv::Scalar(0, 255, 0));
         cv::putText(imOut,"label "+label+": "+prob, cv::Point(currRect.x-15,currRect.y-3), 
             cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1.4);
     }
     else if(i_label == 2){
        // gtRect = cv::Rect(gtMat.row(i_label).at<int>(0),gtMat.row(i_label).at<int>(1), gtMat.row(i_label).at<int>(2),gtMat.row(i_label).at<int>(3));
        // if (compute_Iou(gtRect, currRect) < 0.2){
        //     stringstream ss1;
        //     ss1<<fixed << setprecision(2) <<(rand()%10000+1);
        //     string index = ss1.str();
        //     cv::imwrite(trainingSetPath+"/03/"+index+".jpg", imOut(currRect));
        // }
         // cv::rectangle(imOut, gtRect, cv::Scalar(255, 0, 0));
         cv::rectangle(imOut, currRect, cv::Scalar(255, 0, 0));
         cv::putText(imOut,"label "+label+": "+prob, cv::Point(currRect.x-15,currRect.y-3), 
                 cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1.4);
    }

}
