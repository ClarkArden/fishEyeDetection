#pragma once
#define SIZEW_SEARCH (640) 
#define SIZEH_SEARCH (512)
#define MAX_NUM_L_tgt 2000
#define MAX_NUM_TARGET 200
#define MAX_ANGLE_CHANGE 0.175
#define CONFIDENCE_NUM 12


struct tgt
{
	short left,right,up,down,h,w;
	int x,y,gray,area;
	unsigned char flag;
};  
struct tl
{
	unsigned int x,gray;
};
struct tt1
{
	unsigned short f,id;
	unsigned short l,r;
	unsigned int x,y,g,a;
};
struct tzh1
{
	unsigned int gray;
	unsigned char confidence;
	unsigned short centroid_x,centroid_y;//,length,width;
	short up,down,left,right;
	unsigned short w,h;
	unsigned int area;	
	short velocity_x, velocity_y, accelerate_x, accelerate_y;
	short trace_deviation_x, trace_deviation_y;
	float angle_change;
	int history_x[CONFIDENCE_NUM], history_y[CONFIDENCE_NUM];
	int history_velocity_x[CONFIDENCE_NUM], history_velocity_y[CONFIDENCE_NUM];
}; 
struct tzh
{
	unsigned char confidence;
	unsigned short centroid_x,centroid_y;
	unsigned short w,h;	
}; 



void Midfilter(unsigned char * src,short Width,short gatecount,unsigned char * dst);
void Detectingtarget(unsigned char * formerimage, unsigned char* srcimg);//0xc1000 0xc8000����֡�
void Picktarget(unsigned char * buf,int w,int h,int threshold,int flag);
void IMG_Del_Col(unsigned char *  inptr,unsigned char *  outptr,int x_dim,int  width,const char * mask,int Del );
void Correlation(unsigned char* img, int w, int h);
void Coalition(short col);
int checkdotfeature(unsigned char* img, int imgw, int imgh, int x, int y, int w, int h);
void PickDotTarget(unsigned char * buf, int w, int h, int threshold, int flag);

