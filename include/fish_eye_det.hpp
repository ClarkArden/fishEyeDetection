#pragma once
#include <string>
#include <iostream>
#include <memory>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "gettarget.h"

#define UNDISTORT_H (512)
#define UNDISTORT_W (640)

struct Bbox {
    int x;
    int y;
    int width;
    int height;
    void print_info(){
        std::cout << "x: " << x << "y:" <<y << "width: "<< width << " height: "<< height <<std::endl;
    }
};
struct yx_addr{
    float y;
    float x;
};
struct ocam_model{
    float ss[5];
    float xc;
    float yc;
    float c;
    float d;
    float e;
    int width;
    int height;
    float pol[14];
    float err[272];
    int N;
};
struct calib_data{
    struct ocam_model ocam_model;
};

class FishEyeDet
{
private:
    /* data */
    int scale_;          // percentage of original image
    int width_; 
    int height_;
    int channels_;
    cv::Mat obj_src_img_;
    cv::Mat scene_src_img_;

    cv::Mat obj_scale_img_;
    cv::Mat scene_scale_img_;
    
    cv::Mat undist_obj_img_;
    cv::Mat undist_scene_img_;

    // features
    std::vector<cv::KeyPoint> keypoints_obj_, keypoints_scene_;
    cv::Mat descriptors_obj_, descriptors_scene_;

    // matches
    std::vector<cv::DMatch> good_matches_;
    cv::Mat H_;

    // RANSAC
    std::vector<cv::KeyPoint> rr_keypoints_obj_, rr_keypoints_scene_;
    std::vector<cv::DMatch> rr_matches_;

    // flags
    bool less_keypoints_flag_ = false;
    bool less_keypoints_ransac_flag_ = false;

    // undistort 
    cv::Mat intrinsics_, coeff_;
    cv::Mat mapx_, mapy_;
   
    
    cv::Size corrected_size_ = cv::Size(3000,3000);


    struct calib_data hw_calib_data;
    struct calib_data kj_calib_data;



    // detect
    cv::Mat src_img_;
    std::unique_ptr<uint8_t[]> undistort_obj_img_;
    std::unique_ptr<uint8_t[]> undistort_scene_img_;

    cv::Mat first_frame_, frame_, first_frame_gray_, frame_gray_;
    bool first_flag_ = true;
    cv::Mat obj_image_, scene_image_;



public:
    FishEyeDet(int width, int height, int scale);
    ~FishEyeDet();

private:
    cv::Mat ReadRawImg(std::string filename, int width, int height);
    void LoadCameraParameters(std::string filename, cv::Mat &K, cv::Mat &distcoef);
    cv::Mat Resize(cv::Mat img, int scale);
    void GetFrame(std::string filename);
    cv::Mat GetDetectImg(unsigned char* src_data);
    void ExtractORBFeature(cv::Mat obj_img, cv::Mat scene_img);
    cv::Mat ComputeHMatrix();
    cv::Mat TransformFromObjToScene(cv::Mat obj_image, cv::Mat H);
    cv::Mat GetForegroundImage(cv::Mat obj_img, cv::Mat scene_img, cv::Mat H);
    cv::Mat Detect(cv::Mat img, cv::Mat background);
    void imgstrech16_to_8(unsigned short * img,uint8_t* out,int w,int h);
    void init_calib_data();
    void init_calib_data_visibleLight();
    void undistort(int distorted_image_w, int distorted_image_h,unsigned char* distorted_image, unsigned char* undistorted_image, struct calib_data* calib_data, float fc);
    void undistort_visble(int distorted_image_w, int distorted_image_h,unsigned char* distorted_image, unsigned char* undistorted_image, struct calib_data* calib_data, float fc);
    void Image2Video();
};


