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

        GetFrame("/home/fitz/Downloads/img_data/kejian1/50/Video_20230509163621861.avi");
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

        cv::cvtColor(frame, frame_gray, cv::COLOR_RGB2GRAY);
        if(channels_ == 1){
            undistort(frame_gray.cols, frame_gray.rows, frame_gray.data, undistort_scene_img.get(), &hw_calib_data, 6.0);
        }else{
            undistort_visble(frame_gray.cols, frame_gray.rows, frame_gray.data, undistort_scene_img.get(), &kj_calib_data, 6.0);
        }
        cv::Mat scene_image(cv::Size(UNDISTORT_W, UNDISTORT_H), CV_8UC1, undistort_scene_img.get());
        std::cout<<"obj image: "<<frame.channels()<<std::endl;

        auto tik = std::chrono::system_clock::now();
        

        cv::resize(obj_image, obj_image_resize,cv::Size(824,624));
        cv::resize(scene_image, scene_image_resize,cv::Size(824,624));
        

        ExtractORBFeature(obj_image_resize, scene_image_resize);
        // H_ = ComputeHMatrix();
        // cv::Mat foreground_img = GetForegroundImage(obj_image, scene_image ,H_);
        // Detect(foreground_img, scene_image);

        cv::Mat black_mat = cv::Mat::zeros(obj_image_resize.size(), CV_8UC1);
        if(!less_keypoints_flag_ && !less_keypoints_ransac_flag_){        //if keypoints less than 4
            H_ = ComputeHMatrix();
            cv::Mat foreground_img = GetForegroundImage(obj_image_resize, scene_image_resize ,H_);
            Detect(foreground_img, scene_image_resize);
        }else{
            Detect(black_mat, scene_image_resize);
        }
        auto tok = std::chrono::system_clock::now();

        double duration_ms = std::chrono::duration<double,std::milli>(tok - tik).count();
        std::cout << "it takes  " << duration_ms << " ms" << std::endl;

    }
    capture.release();

}

cv::Mat FishEyeDet::GetDetectImg(unsigned char* src_data) 
{ 
    if(channels_ == 1){
        src_img_.create(cv::Size(UNDISTORT_W, UNDISTORT_H), CV_8UC1);
    }else{
        src_img_.create(cv::Size(UNDISTORT_W, UNDISTORT_H), CV_8UC3);
    }
    src_img_.data = src_data;
    if(first_flag_){
        first_frame_ = src_img_;
        if(channels_ == 1){
            undistort(first_frame_.cols, first_frame_.rows, first_frame_.data,undistort_obj_img_.get(),&hw_calib_data, 6.0 );
        }else{
            cv::Mat first_frame_gray;
            cv::cvtColor(first_frame_, first_frame_gray, cv::COLOR_RGB2GRAY);
            undistort_visble(first_frame_gray.cols, first_frame_gray.rows, first_frame_gray.data,undistort_obj_img_.get(),&kj_calib_data, 6.0 );
        }
        obj_image_.create(cv::Size(UNDISTORT_W, UNDISTORT_H), CV_8UC1);
        obj_image_.data = undistort_obj_img_.get();
        first_flag_ = false;
    }
    cv::Mat gray_frame;                                     // all image convert to gray img
    if(channels_ == 3){
        cv::cvtColor(src_img_, gray_frame,cv::COLOR_RGB2GRAY);
    }else{
        gray_frame = src_img_;
    }

    frame_ = gray_frame;

    if(channels_ == 1){
        undistort(frame_.cols, frame_.rows, frame_.data, undistort_scene_img_.get(),&hw_calib_data, 6.0 );
    }else{
        undistort_visble(frame_.cols, frame_.rows, frame_.data, undistort_scene_img_.get(),&kj_calib_data, 6.0 );
    }
    scene_image_.create(cv::Size(UNDISTORT_W, UNDISTORT_H), CV_8UC1);
    scene_image_.data = undistort_scene_img_.get();

    auto tik = std::chrono::system_clock::now();

    ExtractORBFeature(obj_image_, scene_image_);
    
    cv::Mat black_mat = cv::Mat::zeros(obj_image_.size(), CV_8UC1);
    if(!less_keypoints_flag_ && !less_keypoints_ransac_flag_){        //if keypoints less than 4
        H_ = ComputeHMatrix();
        cv::Mat foreground_img = GetForegroundImage(obj_image_, scene_image_ ,H_);
        Detect(foreground_img, scene_image_);
    }else{
        Detect(black_mat, scene_image_);
    }
    auto tok = std::chrono::system_clock::now();

    double duration_ms = std::chrono::duration<double,std::milli>(tok - tik).count();
    std::cout << "it takes  " << duration_ms << " ms" << std::endl;


    return cv::Mat(); 
}

void FishEyeDet::ExtractORBFeature(cv::Mat obj_img, cv::Mat scene_img ){
    // initialize
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    // // step 1: detect Oriented corners positions
    detector->detect(obj_img, keypoints_obj_);
    detector->detect(scene_img, keypoints_scene_);

    // step 2: compute BRIRF descriptors according to corners positions
    descriptor->compute(obj_img, keypoints_obj_, descriptors_obj_);
    descriptor->compute(scene_img, keypoints_scene_, descriptors_scene_);

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

cv::Mat FishEyeDet::ComputeHMatrix()
{
    //-- Localize the object
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;
    
    cv::Point2f srcTri[rr_matches_.size()];
    cv::Point2f dstTri[rr_matches_.size()];


    //-- Get the keypoints from the good matches
    for(size_t i = 0; i < rr_matches_.size(); i++){
        obj.push_back(rr_keypoints_obj_[rr_matches_[i].queryIdx].pt);
        scene.push_back(rr_keypoints_scene_[rr_matches_[i].trainIdx].pt);

        srcTri[i] = rr_keypoints_obj_[rr_matches_[i].queryIdx].pt;
        dstTri[i] = rr_keypoints_scene_[rr_matches_[i].trainIdx].pt;
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

cv::Mat FishEyeDet::GetForegroundImage(cv::Mat obj_img, cv::Mat scene_img, cv::Mat H) 
{   
    std::cout << "H = " << H << std::endl;

    cv::Mat img_T_scene = TransformFromObjToScene(obj_img, H);

    // cv::imshow("Transform img scene", img_T_scene);
    // cv::imshow("undistort obj ", obj_img);
    // cv::imshow("undistort scene ",scene_img);
   
    cv::Mat diff, direct_diff;
    cv::Mat foreground_img, fore_out;
    cv::absdiff(img_T_scene, scene_img, diff);               //转换后相减
    cv::absdiff(obj_img, scene_img, direct_diff);  //两幅图直接相减（对比）
    cv::threshold(diff, foreground_img, 15, 255, cv::THRESH_BINARY);
    // cv::adaptiveThreshold(diff, foreground_img, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV,11, 4);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2,3));
    cv::morphologyEx(foreground_img, fore_out, cv::MORPH_OPEN, kernel);
    
    // cv::imshow("direct diff", direct_diff);
    // cv::imshow("diff", diff);
    // cv::imshow("foreground", foreground_img);
    // cv::imshow("open op img", fore_out);
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
    cv::waitKey(1);

    // bbox.print_info();
    // std::cout<<"x="<<L_tgt[0].x<<"y = "<<L_tgt[0].y<<"w="<<L_tgt[0].w<<"h="<<L_tgt[0].h<<std::endl;
    // std::cout<<"img_width ="<< img.cols << " img_height= "<<img.rows <<std::endl;
    std::cout<<"target count = "<<count<<std::endl;
    return detec_img;

}

void FishEyeDet::imgstrech16_to_8(unsigned short * img,uint8_t* out,int w,int h)
{
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

void FishEyeDet::Image2Video() 
{
    cv::VideoWriter writer;
    std::string path_src = "/home/fitz/project/fishEyeDect/image/";
    std::string s_image_name;
    int isColor = 0; 
    int frame_fps = 20;  
    int frame_width = 640;  
    int frame_height = 512;
    std::string video_name = "./out2.mp4";    
    writer = cv::VideoWriter(video_name, CV_FOURCC('m', 'p', '4', 'v'),frame_fps,cv::Size(frame_width,frame_height),true); 
    if(!writer.isOpened())
    {
        std::cout<< "Error : fail to open video writer\n"<<std::endl;
    } 
    cv::namedWindow("image to video", CV_WINDOW_AUTOSIZE);
    int num = 43;
    int i = 0;
    cv::Mat img;
    while(i<num){
        s_image_name = path_src + std::to_string(++i)+".png";
        img = cv::imread(s_image_name);
        if (!img.data)//判断图片调入是否成功  
        {  
            std::cout <<s_image_name<<std::endl;
            std::cout << "Could not load image file...\n" << std::endl;  
        }  
        std::cout << "image sieze"<<img.size() << std::endl;
        cv::imshow("image to video",img);
        cv::Mat image = cv::Mat::zeros(640, 512, CV_8UC3);
        image.setTo(cv::Scalar(100, 0, 0));

        //写入  
        writer.write(image);  
        // if (cv::waitKey(3) == 27 || i > 43)  
        // {  
        //     std::cout << "touch ESC" << std::endl;  
        //     break;  
        // } 

    }
    writer.release();
}
