#include "fish_eye_det.hpp"
#include "gettarget.h"
extern struct tgt L_tgt[MAX_NUM_L_tgt];
extern struct tgt L_tgt_0[MAX_NUM_TARGET];



FishEyeDet::FishEyeDet(int width, int height,
                       int channels)
    :width_(width), height_(height),channels_(channels) {

        if(channels_ == 1){
            init_calib_data();
        }else{
            init_calib_data_visibleLight();
        }
        undistort_obj_img_ = std::make_unique<uint8_t[]>(UNDISTORT_H * UNDISTORT_W);
        undistort_scene_img_ = std::make_unique<uint8_t[]>(UNDISTORT_H * UNDISTORT_W);
        test();

        GetFrame("/home/fitz/Downloads/img_data/kejian1/50/Video_20230509163621861.avi");
        // GetFrame("/home/fitz/Downloads/img_data/kejian1/60/Video_20230509163223562.avi");
        // GetFrame("/home/fitz/Downloads/img_data/kejian1/70/Video_20230509162748564.avi");
        // GetFrame("/home/fitz/Downloads/img_data/kejian1/80/Video_20230509162450783.avi");
        // GetFrame("/home/fitz/Downloads/img_data/kejian1/90/Video_20230509162050071.avi");
        // GetFrame("/home/fitz/project/tests/fif2.avi");
    
        
    }

FishEyeDet::~FishEyeDet()
{
}

cv::Mat FishEyeDet::ReadRawImg(std::string filename, int width, int height)
{
    // open raw data
    // const std::string file_path = "D:/E_Dragon/OPENCV/testpictures/1.raw";
    std::ifstream fin;
    // 注意，这里要指定binary读取模式
    fin.open(filename, std::ios::binary);
    if (!fin) {
        std::cerr << "open failed: " << filename << std::endl;
    }
    // seek函数会把标记移动到输入流的结尾
    fin.seekg(0, fin.end);
    // tell会告知整个输入流（从开头到标记）的字节数量
    int length = fin.tellg();
    // 再把标记移动到流的开始位置
    fin.seekg(0, fin.beg);
    std::cout << "file length: " << length << std::endl;

    // load buffer
    // char* buffer = new char[length];
    std::unique_ptr<char[]> buffer(new char[length]);
   
    
    // read函数读取（拷贝）流中的length各字节到buffer
    fin.read(buffer.get(), length);
  
    // uint8_t *img = new uint8_t[width*height];
    std::unique_ptr<uint8_t[]> img(new uint8_t[width*height]);

    imgstrech16_to_8((unsigned short *)buffer.get(), img.get(), width, height);

    // construct opencv mat and show image
    cv::Mat image(cv::Size(width, height), CV_8UC1, img.get());
    // cv::imshow("test", image);
    // cv::waitKey(0);
    return image.clone();

}

void FishEyeDet::LoadCameraParameters(std::string filename, cv::Mat &K, cv::Mat &distcoef)
{
    cv::FileStorage fSettings(filename, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat k = cv::Mat::eye(3,3,CV_32F);
    k.at<float>(0,0) = fx;
    k.at<float>(1,1) = fy;
    k.at<float>(0,2) = cx ;
    k.at<float>(1,2) = cy;
    K = k;

    std::cout<<fx<<std::endl;

    cv::Mat DistCoef(1,4, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.k3"];
    DistCoef.at<float>(3) = fSettings["Camera.k4"];

    distcoef = DistCoef;



}


cv::Mat FishEyeDet::Resize(cv::Mat img, int scale)
{
    cv::Mat out;
    int width = static_cast<int>(img.cols * scale / 100);
    int height = static_cast<int>(img.rows * scale / 100);

    cv::Size size = cv::Size2d(width, height);

    cv::resize(img, out, size, cv::INTER_LINEAR);
    return out;

}

void FishEyeDet::GetFrame(std::string filename) 
{
    cv::VideoCapture capture;
    capture.open(filename);
    if(!capture.isOpened()){
        std::cout << "could not open this video capture" << std::endl;
        return;
    }
    cv::Mat first_frame, frame, first_frame_gray, frame_gray;
    bool first_flag = true;
    cv::Mat obj_image;
    cv::Mat source_img;
    cv::Mat obj_image_resize, scene_image_resize;
    std::unique_ptr<uint8_t[]> undistort_obj_img(new uint8_t[UNDISTORT_H * UNDISTORT_W]);
    std::unique_ptr<uint8_t[]> undistort_scene_img(new uint8_t[UNDISTORT_H * UNDISTORT_W]);
    
    while(1){
        auto tik_b = std::chrono::system_clock::now();
        if(first_flag){
            capture >> first_frame;

            // capture >> source_img;
            // cv::resize(source_img, first_frame, re_size_);

            // std::cout << "size"<<first_frame.size() <<std::endl;
            // cv::imshow("first_frame", first_frame);
            // cv::waitKey(0);

            cv::cvtColor(first_frame,first_frame_gray, cv::COLOR_RGB2GRAY);
            if(channels_ == 1){
                undistort(first_frame_gray.cols, first_frame_gray.rows, first_frame_gray.data, undistort_obj_img.get(), &hw_calib_data, 6.0);
                // undistort(2448, 2048, first_frame_gray.data,undistort_obj_img.get(), &hw_calib_data, 6.0);
            }else{
                undistort_visble(first_frame_gray.cols, first_frame_gray.rows, first_frame_gray.data, undistort_obj_img.get(), &kj_calib_data, 6.0);
            }
            obj_image.create(cv::Size(UNDISTORT_W, UNDISTORT_H), CV_8UC1);
            obj_image.data = undistort_obj_img.get();
            first_flag = false;
        }
        
        capture >> frame ;
        // capture >> source_img ;

         if(frame.empty()){
            break;
        }

        // cv::resize(source_img, frame, re_size_);
        auto tik_undis = std::chrono::system_clock::now();


        cv::cvtColor(frame, frame_gray, cv::COLOR_RGB2GRAY);
        if(channels_ == 1){
            undistort(frame_gray.cols, frame_gray.rows, frame_gray.data, undistort_scene_img.get(), &hw_calib_data, 6.0);
        }else{
            undistort_visble(frame_gray.cols, frame_gray.rows, frame_gray.data, undistort_scene_img.get(), &kj_calib_data, 6.0);
        }
        cv::Mat scene_image(cv::Size(UNDISTORT_W, UNDISTORT_H), CV_8UC1, undistort_scene_img.get());
        std::cout<<"obj image: "<<frame.channels()<<std::endl;

        auto tik = std::chrono::system_clock::now();


        // cv::imshow("resize fisrt img",obj_image);
        // cv::imshow("resie",scene_image);
        // cv::waitKey(0);
        

        // cv::resize(obj_image, obj_image_resize,cv::Size(824,624));
        // cv::resize(scene_image, scene_image_resize,cv::Size(824,624));
        // obj_image = scene_image;
        // scene_image = imgTranslate(scene_image, 50, 0, false);

        ExtractORBFeature(obj_image, scene_image);
        // H_ = ComputeHMatrix();
        // cv::Mat foreground_img = GetForegroundImage(obj_image, scene_image ,H_);
        // Detect(foreground_img, scene_image);

        cv::Mat black_mat = cv::Mat::zeros(obj_image.size(), CV_8UC1);
        if(!less_keypoints_flag_ && !less_keypoints_ransac_flag_){        //if keypoints less than 4
            H_ = ComputeHMatrix(rr_keypoints_obj_, rr_keypoints_scene_, rr_matches_);
            std::vector<cv::Mat> H_vec;
            for(int i= 0; i < matches_filter_.size(); i++){
                if(matches_filter_[i].size()<5){
                    H_vec.push_back(H_);
                }else{
                    H_vec.push_back(ComputeHMatrix(obj_kp_n_filter_[i], scene_kp_n_filter_[i], matches_filter_[i]));
                }
            }

            cv::Mat foreground_img = GetForegroundImage(obj_image, scene_image ,H_, H_vec);
            Detect(foreground_img, scene_image);
        }else{
            Detect(black_mat, scene_image);
        }
        auto tok = std::chrono::system_clock::now();

        double duration_ms = std::chrono::duration<double,std::milli>(tok - tik).count();
        double total_ms = std::chrono::duration<double,std::milli>(tok - tik_b).count();
        double undist_ms = std::chrono::duration<double,std::milli>(tik - tik_undis).count();
        double first_ms = std::chrono::duration<double, std::milli>(tik_undis - tik_b).count();
        std::cout << "it takes  " << duration_ms << " ms" << std::endl;
        std::cout << "fist use time:" << first_ms << " ms"<<std::endl;
        std::cout << "undistort use time :"<< undist_ms << "ms" << std::endl;
        std::cout << "total use time :"<< total_ms << "ms" << std::endl;


    }
    capture.release();

}

cv::Mat FishEyeDet::GetDetectImg(unsigned char* src_data) 
{ 
    // if(channels_ == 1){
    //     src_img_.create(cv::Size(UNDISTORT_W, UNDISTORT_H), CV_8UC1);
    // }else{
    //     src_img_.create(cv::Size(UNDISTORT_W, UNDISTORT_H), CV_8UC3);
    // }
    // src_img_.data = src_data;
    // if(first_flag_){
    //     first_frame_ = src_img_;
    //     if(channels_ == 1){
    //         undistort(first_frame_.cols, first_frame_.rows, first_frame_.data,undistort_obj_img_.get(),&hw_calib_data, 6.0 );
    //     }else{
    //         cv::Mat first_frame_gray;
    //         cv::cvtColor(first_frame_, first_frame_gray, cv::COLOR_RGB2GRAY);
    //         undistort_visble(first_frame_gray.cols, first_frame_gray.rows, first_frame_gray.data,undistort_obj_img_.get(),&kj_calib_data, 6.0 );
    //     }
    //     obj_image_.create(cv::Size(UNDISTORT_W, UNDISTORT_H), CV_8UC1);
    //     obj_image_.data = undistort_obj_img_.get();
    //     first_flag_ = false;
    // }
    // cv::Mat gray_frame;                                     // all image convert to gray img
    // if(channels_ == 3){
    //     cv::cvtColor(src_img_, gray_frame,cv::COLOR_RGB2GRAY);
    // }else{
    //     gray_frame = src_img_;
    // }

    // frame_ = gray_frame;

    // if(channels_ == 1){
    //     undistort(frame_.cols, frame_.rows, frame_.data, undistort_scene_img_.get(),&hw_calib_data, 6.0 );
    // }else{
    //     undistort_visble(frame_.cols, frame_.rows, frame_.data, undistort_scene_img_.get(),&kj_calib_data, 6.0 );
    // }
    // scene_image_.create(cv::Size(UNDISTORT_W, UNDISTORT_H), CV_8UC1);
    // scene_image_.data = undistort_scene_img_.get();

    // auto tik = std::chrono::system_clock::now();

    // ExtractORBFeature(obj_image_, scene_image_);
    
    // cv::Mat black_mat = cv::Mat::zeros(obj_image_.size(), CV_8UC1);
    // if(!less_keypoints_flag_ && !less_keypoints_ransac_flag_){        //if keypoints less than 4
    //     H_ = ComputeHMatrix(rr_keypoints_obj_, rr_keypoints_scene_, rr_matches_);
    //     cv::Mat foreground_img = GetForegroundImage(obj_image_, scene_image_ ,H_);
    //     Detect(foreground_img, scene_image_);
    // }else{
    //     Detect(black_mat, scene_image_);
    // }
    // auto tok = std::chrono::system_clock::now();

    // double duration_ms = std::chrono::duration<double,std::milli>(tok - tik).count();
    // std::cout << "it takes  " << duration_ms << " ms" << std::endl;


    return cv::Mat(); 
}

void FishEyeDet::ExtractORBFeature(cv::Mat obj_img, cv::Mat scene_img ){
    // initialize
    int features  = 2500;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create(features);

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    // // step 1: detect Oriented corners positions
    detector->detect(obj_img, keypoints_obj_);
    detector->detect(scene_img, keypoints_scene_);

    // step 2: compute BRIRF descriptors according to corners positions
    descriptor->compute(obj_img, keypoints_obj_, descriptors_obj_);
    descriptor->compute(scene_img, keypoints_scene_, descriptors_scene_);
    std::cout<<"kp size = " << keypoints_obj_.size() << "des size = "<< descriptors_obj_.type()<<std::endl;
    std::cout<<"kp scene size = " << keypoints_scene_.size() << "des scene size = "<< descriptors_scene_.type()<<std::endl;

    std::vector<std::vector<cv::KeyPoint>> obj_kp_n, scene_kp_n;
    std::vector<cv::Mat> obj_desc_n, scene_desc_n;
    SplitFeatureNGrid(2, obj_img.rows, obj_img.cols, keypoints_obj_, descriptors_obj_, keypoints_scene_, descriptors_scene_,
                        obj_kp_n, obj_desc_n, scene_kp_n, scene_desc_n);
    
    std::vector<std::vector<cv::KeyPoint>> obj_kp_n_filter, scene_kp_n_filter;
    std::vector<std::vector<cv::DMatch>> matches_filter;

    MatchFeatureNGrid(obj_kp_n,obj_desc_n, scene_kp_n, scene_desc_n, matches_filter, obj_kp_n_filter, scene_kp_n_filter);
    obj_kp_n_filter_ = obj_kp_n_filter;
    scene_kp_n_filter_= scene_kp_n_filter;
    matches_filter_ = matches_filter;

    // for(int i=0 ;i< 9;i++){
        
    //     std::cout <<i << "-th block ::"<<" obj kp filter size: " << obj_kp_n_filter[i].size()<< " scene kp filter size: " << scene_kp_n_filter[i].size();
    //     std::cout<<"match size: "<<matches_filter[i].size()<<std::endl;
    // }
    cv::Mat grid_img;
    // for(int i=0; i< obj_kp_n_filter.size(); i++){
    // }
    cv::drawMatches(obj_img, obj_kp_n_filter[3],scene_img, scene_kp_n_filter[3] , matches_filter[3], grid_img);
    cv::imshow("grid_img", grid_img);
    // cv::waitKey(0);

    // cv::Mat out_img;
    // // cv::drawKeypoints(obj_img, keypoints_obj_, out_img, cv::Scalar::all(-1));
    // cv::drawKeypoints(scene_img, keypoints_scene_, out_img, cv::Scalar::all(-1));
    // cv::imshow("ORB feature keypoints", out_img);
    // cv::waitKey(0);

    // step 3: match keypoints using Hamming distance
    std::vector<cv::DMatch> matches;
    matcher->match(descriptors_obj_, descriptors_scene_, matches);


    // --------------SIFT--------------
    // int numFeatures = 500;
    // int layers = 3;
    // cv::Ptr<cv::SiftFeatureDetector> detector = cv::SIFT::create(numFeatures,layers);
    // cv::Ptr<cv::SiftDescriptorExtractor> descriptor = cv::SiftDescriptorExtractor::create(numFeatures, layers);
    // // cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    // detector->detect(obj_img, keypoints_obj_);
    // detector->detect(scene_img, keypoints_scene_);  

    // descriptor->compute(obj_img, keypoints_obj_, descriptors_obj_);
    // descriptor->compute(scene_img, keypoints_scene_, descriptors_scene_);

    // cv::BFMatcher matcher(cv::NORM_L2);

    // std::vector<cv::DMatch> matches;
    // matcher.match(descriptors_obj_, descriptors_scene_, matches);
    // --------------SIFT--------------

    
    // step 4: filter matches keypoints 

    double min_dist = 100000, max_dist =0;

    for(int i = 0; i < descriptors_obj_.rows; i++)
    {
        double dist = matches[i].distance;
        if(dist < min_dist) min_dist = dist;
        if(dist > max_dist) max_dist = dist;
    }

    std::cout<<"min_distance = "<< min_dist<<std::endl;
    std::cout<<"max_distance = "<< max_dist<<std::endl;


    std::vector<cv::DMatch> good_matches;
    for(int i = 0; i < descriptors_obj_.rows; i++)
    {
        if(matches[i].distance <= std::max(3 * min_dist, 10.0))
        {
            good_matches.push_back(matches[i]);
        }
        // if(matches[i].distance <= std::max(3 * min_dist, 30.0))
        // {
        //     good_matches.push_back(matches[i]);
        // }
        // good_matches.push_back(matches[i]);

        // if(matches[i].distance <= 3*min_dist)
        // {
        //     good_matches.push_back(matches[i]);
        // }
    }
    std::cout<<" = "<< std::endl;
    good_matches_ = good_matches;

    std::vector<cv::DMatch> temp_matches;
    temp_matches = good_matches;

    int ptcount = good_matches.size();
    std::cout<<"ptcount = "<< ptcount<<std::endl;
    if(ptcount < 10)
    {
        std::cout<<"Don't find enough match points" << std::endl;
        less_keypoints_flag_ = true;
        return;
    }else{
        less_keypoints_flag_ = false;
    }

    // convert type to float
    std::vector<cv::KeyPoint> ran_keypoint_obj, ran_keypoint_scene;
    for(size_t i = 0; i < temp_matches.size(); i++)
    {
        ran_keypoint_obj.push_back(keypoints_obj_[temp_matches[i].queryIdx]);
        ran_keypoint_scene.push_back(keypoints_scene_[temp_matches[i].trainIdx]);
    }

    // change to cordinate
    std::vector<cv::Point2f> point_obj, point_scene;
    for(size_t i = 0; i<temp_matches.size(); i++)
    {
        point_obj.push_back(ran_keypoint_obj[i].pt);
        point_scene.push_back(ran_keypoint_scene[i].pt);
    }

    // undistort point
    // std::vector<cv::Point2f> undist_point_obj, undist_point_scene;
    // cv::undistortPoints(point_obj, undist_point_obj, intrinsics_, coeff_);
    // cv::undistortPoints(point_scene, undist_point_scene, intrinsics_, coeff_);
    

    // compute fundamental matrix
    std::vector<uchar> RansacStatus;
    // cv::Mat fundamental_matrix = cv::findFundamentalMat(undist_point_obj, undist_point_scene, RansacStatus, cv::FM_RANSAC, 0.1, 0.99);
    // cv::Mat fundamental_matrix = cv::findFundamentalMat(point_obj, point_scene, RansacStatus, cv::FM_RANSAC);
    cv::Mat fundamental_matrix = cv::findFundamentalMat(point_obj, point_scene, RansacStatus, cv::FM_RANSAC, 0.05, 0.99);

    // redifine  keypoints rr_keypoint and rr_matches to store new keypoints and fundamental matrix, 
    //deleting mismatched keypoints through  RansacStatus

    std::vector<cv::KeyPoint> rr_keypoints_obj, rr_keypoints_scene;
    std::vector<cv::DMatch> rr_matches;
    int index = 0;
    for(size_t i = 0; i< temp_matches.size(); i++)
    {
        if(RansacStatus[i] != 0)
        {
            rr_keypoints_obj.push_back(ran_keypoint_obj[i]);
            rr_keypoints_scene.push_back(ran_keypoint_scene[i]);
            temp_matches[i].queryIdx = index;
            temp_matches[i].trainIdx = index;
            rr_matches.push_back(temp_matches[i]);
            index++;
        }
    }

    rr_keypoints_obj_ = rr_keypoints_obj;
    rr_keypoints_scene_ = rr_keypoints_scene;
    rr_matches_ = rr_matches;


    std::cout<<"matches count after RANSAC: "<< rr_matches.size() << std::endl;
    if(rr_matches.size() < 4){                 // keypoints less than 4 , not compute H matrix
        less_keypoints_ransac_flag_ = true;
    }else{
        less_keypoints_ransac_flag_ = false;
    }

    cv::Mat img_rr_matches;
    // cv::Mat img_matche;
    // cv::Mat img_goodmatche;


    // cv::drawMatches(obj_img, rr_keypoints_obj,scene_img, rr_keypoints_scene, rr_matches, img_rr_matches);
    // cv::imshow("After RANSAC", img_rr_matches);
    // cv::drawMatches(obj_img, keypoints_obj_, scene_img, keypoints_scene_, matches, img_matche);
    // cv::drawMatches(obj_img, keypoints_obj_, scene_img, keypoints_scene_, good_matches, img_goodmatche);
    // cv::imshow("matche img", img_matche);
    // cv::imshow("good matche img", img_goodmatche);
    // cv::waitKey(0);


}

cv::Mat FishEyeDet::ComputeHMatrix(std::vector<cv::KeyPoint> obj_kp, std::vector<cv::KeyPoint> scene_kp, std::vector<cv::DMatch> match)
{
    //-- Localize the object
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;
    
    // cv::Point2f srcTri[rr_matches_.size()];
    // cv::Point2f dstTri[rr_matches_.size()];


    //-- Get the keypoints from the good matches
    for(size_t i = 0; i < match.size(); i++){
        obj.push_back(obj_kp[match[i].queryIdx].pt);
        scene.push_back(scene_kp[match[i].trainIdx].pt);

        // srcTri[i] = rr_keypoints_obj_[rr_matches_[i].queryIdx].pt;
        // dstTri[i] = rr_keypoints_scene_[rr_matches_[i].trainIdx].pt;
    }
    // cv::Mat h = cv::Mat::eye(3,3, CV_32F);

    // cv::findHomography(scene, obj);
    // cv::getAffineTransform(srcTri, dstTri);
    // std::cout<<"affice"<< cv::getAffineTransform(srcTri, dstTri);
    cv::Mat result = cv::findHomography(obj, scene);
    // cv::Mat result = cv::getAffineTransform(srcTri,dstTri);

    return result;

}

cv::Mat FishEyeDet::TransformFromObjToScene(cv::Mat obj_image, cv::Mat H)
{
    cv::Mat img_T_obj_to_scene ;
    // cv::perspectiveTransform(obj_image, img_T_obj_to_scene, H);
    cv::warpPerspective(obj_image, img_T_obj_to_scene, H, obj_image.size());
    // cv::warpAffine(obj_image, img_T_obj_to_scene,H, obj_image.size());
    return img_T_obj_to_scene;

}

void FishEyeDet::SplitFeatureNGrid(int n,int h, int w, std::vector<cv::KeyPoint> obj_kp, 
                        cv::Mat obj_desc, std::vector<cv::KeyPoint> scene_kp, cv::Mat scene_desc,
                          std::vector<std::vector<cv::KeyPoint>> &obj_kp_n, std::vector<cv::Mat> &obj_desc_n,
                          std::vector<std::vector<cv::KeyPoint>> &scenen_kp_n, std::vector<cv::Mat> &scene_desc_n)
{
    int slide_w = w / n, slide_h = h / n;
    
    obj_kp_n.resize(n * n);
    scenen_kp_n.resize(n * n);
    std::vector<std::vector<cv::Mat>> temp_obj_desc_n(n * n);
    std::vector<std::vector<cv::Mat>> temp_scene_desc_n(n * n);

    for(int index = 0; index < obj_kp.size(); index++){
        int num_block = 0;
        float xx = obj_kp[index].pt.x;
        float yy = obj_kp[index].pt.y;
        
        for(int i=0 ; i < n ; i++){
            for(int j = 0; j <n ; j++){
                int x = j * slide_w;
                int y = i * slide_h;
                if(xx>= x && xx < x + slide_w && yy>= y && yy < y + slide_h){
                    obj_kp_n[num_block].push_back(obj_kp[index]);
                    temp_obj_desc_n[num_block].push_back(obj_desc.rowRange(index,index + 1).clone());

                }
                num_block++;
                
            }
        }
    }

    for(int index = 0; index < scene_kp.size(); index++){
        int num_block = 0;
        float xx = scene_kp[index].pt.x;
        float yy = scene_kp[index].pt.y;
        
        for(int i=0 ; i < n ; i++){
            for(int j = 0; j <n ; j++){
                int x = j * slide_w;
                int y = i * slide_h;
                
                if(xx>= x && xx < x + slide_w && yy>= y && yy < y + slide_h){
                    scenen_kp_n[num_block].push_back(scene_kp[index]);
                    temp_scene_desc_n[num_block].push_back(scene_desc.rowRange(index,index + 1).clone());
                }
                num_block++;
                
            }
        }
    }
    

    cv::Mat  result_desc;
    
    for(int i=0; i < temp_obj_desc_n.size(); i++){
        cv::vconcat(temp_obj_desc_n[i],result_desc);
        obj_desc_n.push_back(result_desc);
        result_desc.release();
    }
    for(int i=0; i < temp_scene_desc_n.size(); i++){
        cv::vconcat(temp_scene_desc_n[i],result_desc);
        scene_desc_n.push_back(result_desc);
        result_desc.release();
    }
    
}

cv::Mat FishEyeDet::GetForegroundImage(cv::Mat obj_img, cv::Mat scene_img, cv::Mat g_H, std::vector<cv::Mat> H_vec) 
{   
    std::cout << "H = " << g_H << std::endl;

    auto split_img = SplitImageN(obj_img, 2);

    std::vector<cv::Mat> img_T_scene_vec;
    for(int i=0; i < split_img.size(); i++){
        cv::Mat img_T_scene = TransformFromObjToScene(split_img[i], H_vec[i]);
        img_T_scene_vec.push_back(img_T_scene);
    }
    std::cout << "H _matric =  " << H_vec[0] << std::endl;

    // cv::Mat img_T_scene = TransformFromObjToScene(obj_img, g_H);
    
    
    cv::Mat img_T_scene_concat = ConcatImage(img_T_scene_vec);
    

    cv::imshow("Transform img scene concat", img_T_scene_concat);
    // cv::imshow("undistort obj ", obj_img);
    // cv::imshow("undistort scene ",scene_img);
   
    cv::Mat diff, direct_diff;
    cv::Mat foreground_img, fore_out;
    cv::absdiff(img_T_scene_concat, scene_img, diff);               //转换后相减
    // cv::absdiff(obj_img, scene_img, direct_diff);  //两幅图直接相减（对比）
    cv::threshold(diff, foreground_img, 15, 255, cv::THRESH_BINARY);
    // // cv::adaptiveThreshold(diff, foreground_img, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV,11, 4);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2,3));
    cv::morphologyEx(foreground_img, fore_out, cv::MORPH_OPEN, kernel);
    
    // cv::imshow("direct diff", direct_diff);
    cv::imshow("diff", diff);
    cv::imshow("foreground", foreground_img);
    cv::imshow("open op img", fore_out);
    // cv::waitKey(0);


    return fore_out;
}


cv::Mat  FishEyeDet::Detect(cv::Mat img, cv::Mat background){
    cv::Mat detec_img ;
    cv::cvtColor(background, detec_img ,cv::COLOR_GRAY2RGB);
    
    // PickDotTarget(img.data, img.cols, img.rows, 60, 1);
    Detectingtarget(img.data, NULL);
    
  
    int count = 0;
    for(int i = 0; i <MAX_NUM_TARGET; i++){
        if(L_tgt_0[i].flag==1){
            count++;
            cv::rectangle(detec_img, cv::Point(L_tgt_0[i].x-L_tgt_0[i].w/2, L_tgt_0[i].y-L_tgt_0[i].h/2),
            cv::Point(L_tgt_0[i].x+L_tgt_0[i].w/2, L_tgt_0[i].y+L_tgt_0[i].h/2),cv::Scalar(0,0,255),2);
            
        }
    }
    cv::imshow("detect show",detec_img);
    cv::waitKey(0);

    // bbox.print_info();
    // std::cout<<"x="<<L_tgt[0].x<<"y = "<<L_tgt[0].y<<"w="<<L_tgt[0].w<<"h="<<L_tgt[0].h<<std::endl;
    // std::cout<<"img_width ="<< img.cols << " img_height= "<<img.rows <<std::endl;
    std::cout<<"target count = "<<count<<std::endl;
    return detec_img;

}

std::vector<cv::Mat> FishEyeDet::SplitImageN(cv::Mat img, int n) {
    std::vector<cv::Mat> res;

    int h = img.rows;
    int w = img.cols;
    int slide_h = h / n;
    int slide_w = w / n;

    for(int i = 0; i < n ;i++){
        for(int j = 0; j < n; j++){
            int x = j * slide_w;
            int y = i * slide_h;

            int x2 = x + slide_w;
            int y2 = y + slide_h;
            if(j ==  n - 1){
                x2 = w;
            }
            if(i == n - 1){
                y2 = h;
            }

            cv::Mat img_sub = img(cv::Rect(x, y, x2 - x, y2 - y));
            res.push_back(img_sub);

        }
    }
    return res;

}

cv::Mat FishEyeDet::ConcatImage(std::vector<cv::Mat> grid_img) {
    int n = grid_img.size();
    int n_s = std::sqrt(n);
    std::vector<std::vector<cv::Mat>> img_h(n_s);
    std::vector<cv::Mat> img_v;
    cv::Mat res;
    
    for(int i = 0; i < n; i++){
        img_h[i/n_s].push_back(grid_img[i]);
    }

    for(int i = 0; i< n_s; i++){
        cv::Mat temp;
        cv::hconcat(img_h[i], temp);
        img_v.push_back(temp);

    }
    cv::vconcat(img_v, res);
    return res;
}

void FishEyeDet::imgstrech16_to_8(unsigned short* img, uint8_t* out, int w,
                                  int h) {
    int i;
    int hisimg[16384]={0};
    unsigned short maxvalue=0,minvalue=0xffff;
    long long sumhisleft=0,sumhisright=0;
    unsigned short leftvalue=0,rightvalue=0;

    for(i=100;i<w*h;i++)
    {
        maxvalue=img[i]>=maxvalue?img[i]:maxvalue;
        minvalue=img[i]<=minvalue?img[i]:minvalue;
        if(img[i] > 16384) continue;
        hisimg[img[i]]=hisimg[img[i]]+1;
    }
    for(i=0;i<16384;i++)
    {
        sumhisleft=sumhisleft+hisimg[i];
        if(sumhisleft>w*h*0.2)  // 0.0002
            {leftvalue=i;break;}
    }
    for(i=16383;i>0;i--)
    {
        sumhisright=sumhisright+hisimg[i];
        if(sumhisright>w*h*0.2 )
            {rightvalue=i;break;}
    }

    for(i=100;i<w*h;i++)
    {
        img[i] = img[i]<leftvalue?255*0/16383.:img[i]>rightvalue?(255.*rightvalue)/16383:255*0/16384.+((255*rightvalue/16383.-255*0/16383.)*(img[i]-leftvalue))/(rightvalue-leftvalue);
        out[i] = img[i];
    }
}

cv::Mat FishEyeDet::imgTranslate(cv::Mat& matSrc, int xOffset, int yOffset,
                                 bool bScale) {
    
    int nRows = matSrc.rows;
    int nCols = matSrc.cols;
    int nRowsRet = 0;
    int nColsRet = 0;
    cv::Rect rectSrc;
    cv::Rect rectRet;
    if (bScale)
    {
        nRowsRet = nRows + abs(yOffset);
        nColsRet = nCols + abs(xOffset);
        rectSrc.x = 0;
        rectSrc.y = 0;
        rectSrc.width = nCols;
        rectSrc.height = nRows;
    }
    else
    {
        nRowsRet = matSrc.rows;
        nColsRet = matSrc.cols;
        if (xOffset >= 0)
        {
            rectSrc.x = 0;
        }
        else
        {
            rectSrc.x = abs(xOffset);
        }
        if (yOffset >= 0)
        {
            rectSrc.y = 0;
        }
        else
        {
            rectSrc.y = abs(yOffset);
        }
        rectSrc.width = nCols - abs(xOffset);
        rectSrc.height = nRows - abs(yOffset);
    }
    // 修正输出的ROI
    if (xOffset >= 0)
    {
        rectRet.x = xOffset;
    }
    else
    {
        rectRet.x = 0;
    }
    if (yOffset >= 0)
    {
        rectRet.y = yOffset;
    }
    else
    {
        rectRet.y = 0;
    }
    rectRet.width = rectSrc.width;
    rectRet.height = rectSrc.height;
    // 复制图像
    cv::Mat matRet(nRowsRet, nColsRet, matSrc.type(), cv::Scalar(0));
    matSrc(rectSrc).copyTo(matRet(rectRet));
    return matRet;

    return cv::Mat();
}

void FishEyeDet::MatchFeatureNGrid(
                        std::vector<std::vector<cv::KeyPoint>> obj_kp_n, std::vector<cv::Mat> obj_desc_n,
                          std::vector<std::vector<cv::KeyPoint>> scenen_kp_n, std::vector<cv::Mat> scene_desc_n,
                          std::vector<std::vector<cv::DMatch>> &match,
                          std::vector<std::vector<cv::KeyPoint>> &obj_kp_out, std::vector<std::vector<cv::KeyPoint>> &scene_kp_out) 
{
    int n = obj_kp_n.size();
    match.resize(n );
    obj_kp_out.resize(n);
    scene_kp_out.resize(n);

    for(int i = 0; i < obj_kp_n.size(); i++){
        std::vector<cv::KeyPoint> obj_kp = obj_kp_n[i];
        std::vector<cv::KeyPoint> scene_kp = scenen_kp_n[i];
        cv::Mat obj_desc = obj_desc_n[i];
        cv::Mat scene_desc = scene_desc_n[i];
        // std::cout << " obj keypoint size: "<<obj_kp.size() << "obj desc size: " << obj_desc.size() << std::endl;
        // std::cout << " scene keypoint size: "<<scene_kp.size() << "scene desc size: " << scene_desc.size() << std::endl;

        if(obj_kp.size() < 11 || scene_kp.size() < 11){
            continue;
        }
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
        std::vector<cv::DMatch> matchs_first;
        matcher->match(obj_desc, scene_desc, matchs_first);

        double min_dist = 1000000, max_dist = 0;
        for(int i = 0; i < obj_desc.rows; i++){
            double dist = matchs_first[i].distance;
            if(dist < min_dist) min_dist = dist;
            if(dist > max_dist) max_dist = dist;

        }
        std::vector<cv::DMatch> matchs_second;
        for(int i = 0; i < obj_desc.rows; i++){
            if(matchs_first[i].distance <= std::max(3 * min_dist, 10.0))
            {
                matchs_second.push_back(matchs_first[i]);
            }
        }
        if(matchs_second.size() < 10){
            continue;
        }
        std::vector<cv::DMatch> temp_matches;
        temp_matches = matchs_second;
        std::vector<cv::KeyPoint> ran_keypoint_obj, ran_keypoint_scene;
        for(size_t i = 0; i < temp_matches.size(); i++)
        {
            ran_keypoint_obj.push_back(obj_kp[temp_matches[i].queryIdx]);
            ran_keypoint_scene.push_back(scene_kp[temp_matches[i].trainIdx]);
        }

        std::vector<cv::Point2f> point_obj, point_scene;
        for(size_t i = 0; i<temp_matches.size(); i++)
        {
            point_obj.push_back(ran_keypoint_obj[i].pt);
            point_scene.push_back(ran_keypoint_scene[i].pt);
        }
        
        std::vector<uchar> RansacStatus;
        cv::Mat fundamental_matrix = cv::findFundamentalMat(point_obj, point_scene, RansacStatus, cv::FM_RANSAC, 0.05, 0.99);
        std::vector<cv::KeyPoint> rr_keypoints_obj, rr_keypoints_scene;
        std::vector<cv::DMatch> rr_matches;
        int index = 0;
        for(size_t i = 0; i< temp_matches.size(); i++)
        {
            if(RansacStatus[i] != 0)
            {
                rr_keypoints_obj.push_back(ran_keypoint_obj[i]);
                rr_keypoints_scene.push_back(ran_keypoint_scene[i]);
                temp_matches[i].queryIdx = index;
                temp_matches[i].trainIdx = index;
                rr_matches.push_back(temp_matches[i]);
                index++;
            }
        }
        match[i] = rr_matches;
        obj_kp_out[i] = rr_keypoints_obj;
        scene_kp_out[i] = rr_keypoints_scene;



        
    }

}

void FishEyeDet::init_calib_data() 
{
    hw_calib_data.ocam_model.ss[0] = -234.455515456019;
    hw_calib_data.ocam_model.ss[1] = 0;
    hw_calib_data.ocam_model.ss[2] = 0.0030;
    hw_calib_data.ocam_model.ss[3] = -8.57522958386482e-06;
    hw_calib_data.ocam_model.ss[4] = 4.34725816712150e-08;
    hw_calib_data.ocam_model.xc = 261.487846573450;
    hw_calib_data.ocam_model.yc = 3.194800174063079e+02;
    hw_calib_data.ocam_model.c = 1;
    hw_calib_data.ocam_model.d = 2.695243643555315e-04;
    hw_calib_data.ocam_model.e = -3.743486308392548e-05;
    hw_calib_data.ocam_model.width = 640;
    hw_calib_data.ocam_model.height = 512;

    hw_calib_data.ocam_model.pol[0] = 0.218927480852244;
    hw_calib_data.ocam_model.pol[1] = 1.17864149941501;
    hw_calib_data.ocam_model.pol[2] = 0.519483663949700;
    hw_calib_data.ocam_model.pol[3] = -3.94106292461532;
    hw_calib_data.ocam_model.pol[4] = -1.18420100990138;
    hw_calib_data.ocam_model.pol[5] = 7.70148652179621;
    hw_calib_data.ocam_model.pol[6] = 0.784799998460775;
    hw_calib_data.ocam_model.pol[7] = -1.79013790587332;
    hw_calib_data.ocam_model.pol[8] = 5.41646126305526;
    hw_calib_data.ocam_model.pol[9] = 0.0910907750862856;
    hw_calib_data.ocam_model.pol[10] = 25.1105926745008;
    hw_calib_data.ocam_model.pol[11] = -8.58448459026404;
    hw_calib_data.ocam_model.pol[12] = 93.5347699270376;
    hw_calib_data.ocam_model.pol[13] = 254.376895673036;

    hw_calib_data.ocam_model.N = 13;
}

void FishEyeDet::init_calib_data_visibleLight() 
{
    kj_calib_data.ocam_model.ss[0] = -5.381681644584942e+02;
	kj_calib_data.ocam_model.ss[1] = 0;
	kj_calib_data.ocam_model.ss[2] = 6.848166676889272e-04;
	kj_calib_data.ocam_model.ss[3] = -3.010664029115079e-07;
	kj_calib_data.ocam_model.ss[4] = 5.394883867073358e-10;
	
	kj_calib_data.ocam_model.xc = 1.023350467833255e+03;
	kj_calib_data.ocam_model.yc = 1.150138160359617e+03;
	
	kj_calib_data.ocam_model.c = 1.001233959935347;
	kj_calib_data.ocam_model.d = 0.005424486188633;
	kj_calib_data.ocam_model.e = -0.005776568857300;
	
	kj_calib_data.ocam_model.width = 2448;
	kj_calib_data.ocam_model.height = 2048;
	kj_calib_data.ocam_model.pol[0] = 0.437058618694313;
	kj_calib_data.ocam_model.pol[1] = 1.55981545560405;
	kj_calib_data.ocam_model.pol[2] = 0.427377970963078;
	kj_calib_data.ocam_model.pol[3] = -2.60588389617504;
	kj_calib_data.ocam_model.pol[4] = 0.926288669715811;
	kj_calib_data.ocam_model.pol[5] = 3.94705409281064;
	kj_calib_data.ocam_model.pol[6] = -0.358301604624634;
	kj_calib_data.ocam_model.pol[7] = 9.11725674246931;
	kj_calib_data.ocam_model.pol[8] = 12.1259555424670;
	kj_calib_data.ocam_model.pol[9] = -0.350335005904443;
	kj_calib_data.ocam_model.pol[10] = 39.3444533603722;
	kj_calib_data.ocam_model.pol[11] = 68.0439368213430;
	kj_calib_data.ocam_model.pol[12] = -5.52458740328598;
	kj_calib_data.ocam_model.pol[13] = 480.835533129717;
	kj_calib_data.ocam_model.pol[14] = 819.872411599417;

						 		
	kj_calib_data.ocam_model.N = 14;

}

void FishEyeDet::undistort(int distorted_image_w, int distorted_image_h,
                           unsigned char *distorted_image,
                           unsigned char *undistorted_image,
                           calib_data *calib_data, float fc) 
{
    int Nxc = UNDISTORT_H / 2;
	int Nyc = UNDISTORT_W / 2;
	int Nzc = -UNDISTORT_W / fc;
	float* Nx = (float*)calloc(UNDISTORT_H * UNDISTORT_W, sizeof(float));
	float* Ny = (float*)calloc(UNDISTORT_H * UNDISTORT_W, sizeof(float));
	float* Nz = (float*)calloc(UNDISTORT_H * UNDISTORT_W, sizeof(float));
	float* NORM = (float*)calloc(UNDISTORT_H * UNDISTORT_W, sizeof(float));
	float* theta = (float*)calloc(UNDISTORT_H * UNDISTORT_W, sizeof(float));
	float* rho = (float*)calloc(UNDISTORT_H * UNDISTORT_W, sizeof(float));
	float* x = (float*)calloc(UNDISTORT_H * UNDISTORT_W, sizeof(float));
	float* y = (float*)calloc(UNDISTORT_H * UNDISTORT_W, sizeof(float));
	float* mx = (float*)calloc(UNDISTORT_H * UNDISTORT_W, sizeof(float));
	float* my = (float*)calloc(UNDISTORT_H * UNDISTORT_W, sizeof(float));

	for (int i = 0; i < UNDISTORT_H; i++)
		for (int j = 0; j < UNDISTORT_W; j++)
		{
			Nx[i * UNDISTORT_W + j] = i - Nxc;
			Ny[i * UNDISTORT_W + j] = j - Nyc;
			Nz[i * UNDISTORT_W + j] = Nzc;
		}
	for (int i = 0; i < UNDISTORT_H; i++)
		for (int j = 0; j < UNDISTORT_W; j++)
		{
			NORM[i * UNDISTORT_W + j] = sqrt(Nx[i * UNDISTORT_W + j]* Nx[i * UNDISTORT_W + j]+ Ny[i * UNDISTORT_W + j] * Ny[i * UNDISTORT_W + j]);
			theta[i * UNDISTORT_W + j] = atan(Nz[i * UNDISTORT_W + j]/ (NORM[i * UNDISTORT_W + j]+0.000000000000000001));
		}
	for (int i = 0; i < UNDISTORT_H; i++)
		for (int j = 0; j < UNDISTORT_W; j++)
		{
			rho[i * UNDISTORT_W + j] = calib_data->ocam_model.pol[0];
		}
	for (int nc = 1; nc < 14; nc++)
	{
		for (int i = 0; i < UNDISTORT_H; i++)
			for (int j = 0; j < UNDISTORT_W; j++)
			{
				rho[i * UNDISTORT_W + j] = rho[i * UNDISTORT_W + j] * theta[i * UNDISTORT_W + j] + calib_data->ocam_model.pol[nc];
			}
	}
	for (int i = 0; i < UNDISTORT_H; i++)
		for (int j = 0; j < UNDISTORT_W; j++)
		{
			x[i * UNDISTORT_W + j] = Nx[i * UNDISTORT_W + j] / NORM[i * UNDISTORT_W + j] * rho[i * UNDISTORT_W + j];
			y[i * UNDISTORT_W + j] = Ny[i * UNDISTORT_W + j] / NORM[i * UNDISTORT_W + j] * rho[i * UNDISTORT_W + j];
		}
	for (int i = 0; i < UNDISTORT_H; i++)
		for (int j = 0; j < UNDISTORT_W; j++)
		{
			mx[i * UNDISTORT_W + j] = x[i * UNDISTORT_W + j] * calib_data->ocam_model.c + y[i * UNDISTORT_W + j] * calib_data->ocam_model.d + calib_data->ocam_model.xc;
			my[i * UNDISTORT_W + j] = x[i * UNDISTORT_W + j] * calib_data->ocam_model.e + y[i * UNDISTORT_W + j]                            + calib_data->ocam_model.yc;

		}
	for (int i = 0; i < UNDISTORT_H; i++)
		for (int j = 0; j < UNDISTORT_W; j++)
		{
			int px = mx[i * UNDISTORT_W + j];
			int py = my[i * UNDISTORT_W + j];
			if(px>=0&&px< distorted_image_h&& py >= 0 && py < distorted_image_w)
			undistorted_image[i * UNDISTORT_W + j] = distorted_image[px* distorted_image_w + py];
		}


	free(Nx);
	free(Ny);
	free(Nz);
	free(NORM);
	free(theta);
	free(rho);
	free(x);
	free(y);
	free(mx);
	free(my);
}

void FishEyeDet::undistort_visble(int distorted_image_w, int distorted_image_h,
                                  unsigned char *distorted_image,
                                  unsigned char *undistorted_image,
                                  calib_data *calib_data, float fc) 
{
    TicToc tt;
    int Nxc = UNDISTORT_H / 2;
	int Nyc = UNDISTORT_W / 2;
	int Nzc = -UNDISTORT_W / fc;

    float* Nx = (float*)malloc(UNDISTORT_H * UNDISTORT_W * sizeof(float));
    float* Ny = (float*)malloc(UNDISTORT_H * UNDISTORT_W * sizeof(float));
	float* Nz = (float*)malloc(UNDISTORT_H * UNDISTORT_W * sizeof(float));
	float* NORM = (float*)malloc(UNDISTORT_H * UNDISTORT_W * sizeof(float));
	float* theta = (float*)malloc(UNDISTORT_H * UNDISTORT_W * sizeof(float));
	float* rho = (float*)malloc(UNDISTORT_H * UNDISTORT_W * sizeof(float));
	float* x = (float*)malloc(UNDISTORT_H * UNDISTORT_W * sizeof(float));
	float* y = (float*)malloc(UNDISTORT_H * UNDISTORT_W * sizeof(float));
	float* mx = (float*)malloc(UNDISTORT_H * UNDISTORT_W * sizeof(float));
	float* my = (float*)malloc(UNDISTORT_H * UNDISTORT_W * sizeof(float));
   

    TicToc t_for;
	for (int i = 0; i < UNDISTORT_H; i++)
		for (int j = 0; j < UNDISTORT_W; j++)
		{
			Nx[i * UNDISTORT_W + j] = i - Nxc;
			Ny[i * UNDISTORT_W + j] = j - Nyc;
			Nz[i * UNDISTORT_W + j] = Nzc;
            NORM[i * UNDISTORT_W + j] = sqrt(Nx[i * UNDISTORT_W + j]* Nx[i * UNDISTORT_W + j]+ Ny[i * UNDISTORT_W + j] * Ny[i * UNDISTORT_W + j]);
			theta[i * UNDISTORT_W + j] = atan(Nz[i * UNDISTORT_W + j]/ (NORM[i * UNDISTORT_W + j]+0.000000000000000001));
            rho[i * UNDISTORT_W + j] = calib_data->ocam_model.pol[0];

		}

	for (int nc = 1; nc < 15; nc++)
	{
		for (int i = 0; i < UNDISTORT_H; i++)
			for (int j = 0; j < UNDISTORT_W; j++)
			{
				rho[i * UNDISTORT_W + j] = rho[i * UNDISTORT_W + j] * theta[i * UNDISTORT_W + j] + calib_data->ocam_model.pol[nc];
			}
	}
	for (int i = 0; i < UNDISTORT_H; i++)
		for (int j = 0; j < UNDISTORT_W; j++)
		{
			x[i * UNDISTORT_W + j] = Nx[i * UNDISTORT_W + j] / NORM[i * UNDISTORT_W + j] * rho[i * UNDISTORT_W + j];
			y[i * UNDISTORT_W + j] = Ny[i * UNDISTORT_W + j] / NORM[i * UNDISTORT_W + j] * rho[i * UNDISTORT_W + j];
            mx[i * UNDISTORT_W + j] = x[i * UNDISTORT_W + j] * calib_data->ocam_model.c + y[i * UNDISTORT_W + j] * calib_data->ocam_model.d + calib_data->ocam_model.xc;
			my[i * UNDISTORT_W + j] = x[i * UNDISTORT_W + j] * calib_data->ocam_model.e + y[i * UNDISTORT_W + j]                            + calib_data->ocam_model.yc;
            int px = mx[i * UNDISTORT_W + j];
			int py = my[i * UNDISTORT_W + j];
			if(px>=0&&px< distorted_image_h&& py >= 0 && py < distorted_image_w)
			undistorted_image[i * UNDISTORT_W + j] = distorted_image[px* distorted_image_w + py];

		}
        std::cout << "for use time" << t_for.toc() << std::endl;


	free(Nx);
	free(Ny);
	free(Nz);
	free(NORM);
	free(theta);
	free(rho);
	free(x);
	free(y);
	free(mx);
	free(my);
    std::cout << "all use time:"<< tt.toc() <<std::endl;
}

void FishEyeDet::test() 
{
    std::vector<int> a(10);
    
    for(int i = 11; i > 0; i--){
        a[i] = i-10;
    }
    for(int i=0; i<a.size();i++){
        std::cout<<"test = "<<a[i] <<std::endl;
    }
}
