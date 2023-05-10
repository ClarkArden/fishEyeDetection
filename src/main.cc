#include <iostream>
#include "fish_eye_det.hpp"

int main(int argc, char** argv )
{
    std::unique_ptr<FishEyeDet>  ptr ;
    // ptr = std::make_unique<FishEyeDet>("/home/fitz/Pictures/obj.jpg","/home/fitz/Pictures/scene.jpg", 100); 
    // ptr = std::make_unique<FishEyeDet>("/home/fitz/Documents/fishEyeImage/dahudy/000000.bmp","/home/fitz/Documents/fishEyeImage/dahudy/000001.bmp", 100); 
    ptr = std::make_unique<FishEyeDet>(640, 512, 3); 
    std::cout << "done" << std::endl;
    // ptr = std::make_unique<FishEyeDet>("../data/000000.raw","../data/000001.raw", 100); 

    return 0;
}