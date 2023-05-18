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
#include "tic_toc.h"

#define UNDISTORT_W (1024)
#define UNDISTORT_H (824)

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
   
    
    cv::Size re_size_ = cv::Size(824, 624);


    struct calib_data hw_calib_data;
    struct calib_data kj_calib_data;



    // detect
    cv::Mat src_img_;
    std::unique_ptr<uint8_t[]> undistort_obj_img_;
    std::unique_ptr<uint8_t[]> undistort_scene_img_;

    cv::Mat first_frame_, frame_, first_frame_gray_, frame_gray_;
    bool first_flag_ = true;
    cv::Mat obj_image_, scene_image_;

    // grid
    std::vector<std::vector<cv::KeyPoint>> obj_kp_n_filter_, scene_kp_n_filter_;
    std::vector<std::vector<cv::DMatch>> matches_filter_;



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
    void SplitFeatureNGrid(int n,int h, int w, std::vector<cv::KeyPoint> obj_kp, cv::Mat obj_desc, std::vector<cv::KeyPoint> scene_kp, cv::Mat scene_desc,
                          std::vector<std::vector<cv::KeyPoint>> &obj_kp_n, std::vector<cv::Mat> &obj_desc_n,
                          std::vector<std::vector<cv::KeyPoint>> &scenen_kp_n, std::vector<cv::Mat> &scene_desc_n);
    void MatchFeatureNGrid(std::vector<std::vector<cv::KeyPoint>> obj_kp_n, std::vector<cv::Mat> obj_desc_n,
                          std::vector<std::vector<cv::KeyPoint>> scenen_kp_n, std::vector<cv::Mat> scene_desc_n,
                          std::vector<std::vector<cv::DMatch>> &match,
                          std::vector<std::vector<cv::KeyPoint>> &obj_kp_out, std::vector<std::vector<cv::KeyPoint>> &scene_kp_out);
    cv::Mat ComputeHMatrix(std::vector<cv::KeyPoint> obj_kp, std::vector<cv::KeyPoint> scene_kp, std::vector<cv::DMatch> match);
    cv::Mat TransformFromObjToScene(cv::Mat obj_image, cv::Mat H);
    cv::Mat GetForegroundImage(cv::Mat obj_img, cv::Mat scene_img, cv::Mat g_H, std::vector<cv::Mat> H_vec);
    cv::Mat Detect(cv::Mat img, cv::Mat background);
    std::vector<cv::Mat> SplitImageN(cv::Mat img, int n);
    cv::Mat ConcatImage(std::vector<cv::Mat> grid_img);
    void imgstrech16_to_8(unsigned short * img,uint8_t* out,int w,int h);
    void init_calib_data();
    void init_calib_data_visibleLight();
    void undistort(int distorted_image_w, int distorted_image_h,unsigned char* distorted_image, unsigned char* undistorted_image, struct calib_data* calib_data, float fc);
    void undistort_visble(int distorted_image_w, int distorted_image_h,unsigned char* distorted_image, unsigned char* undistorted_image, struct calib_data* calib_data, float fc);
    void test();
    cv::Mat imgTranslate(cv::Mat &matSrc, int xOffset, int yOffset, bool bScale);

};


