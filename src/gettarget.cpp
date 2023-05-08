#include <iostream>
#include <math.h>
#include <string.h>
#include "gettarget.h"

using namespace std;

char mask_del_col[9]={1,1,1,1,1,1,1,1,1};
volatile short tp_num[2]={0,0},ptr=0,pr;
volatile short temp_num=0; 
volatile int tal=0,tal1=0;
volatile short track_num=0,object_counter[2]={0,0};
volatile int targetnum=0; 
//struct tzh Track[30];
struct tzh1 target[MAX_NUM_TARGET];
struct tt1 tp[MAX_NUM_L_tgt][2];
struct tl temp[MAX_NUM_L_tgt];
struct tgt L_tgt[MAX_NUM_L_tgt];
struct tgt L_tgt_0[MAX_NUM_TARGET];


 
unsigned char Midtemp[2* SIZEW_SEARCH];
unsigned char Midtemp1[2* SIZEW_SEARCH];
unsigned char Midtemp2[2* SIZEW_SEARCH];

//extern unsigned char * pbuf_fchose;
//extern unsigned char * pbuf_temp;


void Detectingtarget(unsigned char * formerimage/*,unsigned char *latterimage,float *Param*/,unsigned char *srcimg)
{
	int m=0,n=0,i=0,j=0;
	int p=0,k=0,t=0,s=0;
	int graysum=0,graynum=0;

	int maxdistance=9999999;
	unsigned char * pbuf= formerimage;
	tal=tal1=0;
	
	targetnum = 0; 
	Picktarget(formerimage, SIZEW_SEARCH,SIZEH_SEARCH,0,1);//��ȡ��Ŀ��
	k=0;
	for(i=0;i<targetnum;i++)
	{
		if(L_tgt[i].area>=4&&L_tgt[i].down - L_tgt[i].up>=2&& L_tgt[i].right - L_tgt[i].left >=2&&L_tgt[i].gray<2147483647)
		{
   			L_tgt[k].x=(L_tgt[i].x)/L_tgt[i].area;
			L_tgt[k].y=(L_tgt[i].y)/L_tgt[i].area;
			L_tgt[k].up=L_tgt[i].up;
			L_tgt[k].down=L_tgt[i].down;
			L_tgt[k].left=L_tgt[i].left;
			L_tgt[k].right=L_tgt[i].right;
			L_tgt[k].area=L_tgt[i].area;
			L_tgt[k].gray=L_tgt[i].gray/L_tgt[i].area;
			L_tgt[k].w = L_tgt[i].right- L_tgt[i].left+1;
			L_tgt[k].h = L_tgt[i].down- L_tgt[i].up+1;

			k+=1;
		}
	}
	targetnum=k;
	k = 0;
	for(i=0;i<targetnum-1;i++)//���Ͻ�����ϲ�
	{	
		if(L_tgt[i].gray<2147483647)
		{
			L_tgt[k].up=L_tgt[i].up;
			L_tgt[k].down=L_tgt[i].down;
			L_tgt[k].left=L_tgt[i].left;
			L_tgt[k].right=L_tgt[i].right;
			L_tgt[k].area=L_tgt[i].area;
			tal=L_tgt[i].x*L_tgt[i].area;
			tal1=L_tgt[i].y*L_tgt[i].area;
			for(j=i+1;j<targetnum;j++)
			{
				n=L_tgt[i].x-L_tgt[j].x;m=L_tgt[i].y-L_tgt[j].y;
				n=n*n+m*m;
				if(n<0&&L_tgt[j].gray<2147483647)
				{
		   			tal+=L_tgt[j].x*L_tgt[j].area;
					tal1+=L_tgt[j].y*L_tgt[j].area;
					if(L_tgt[k].up>L_tgt[j].up) L_tgt[k].up=L_tgt[j].up;
					if(L_tgt[k].down<L_tgt[j].down) L_tgt[k].down=L_tgt[j].down;
					if(L_tgt[k].left>L_tgt[j].left) L_tgt[k].left=L_tgt[j].left;
					if(L_tgt[k].right<L_tgt[j].right) L_tgt[k].right=L_tgt[j].right;
					L_tgt[k].area+=L_tgt[j].area;
					L_tgt[j].gray=2147483647;
				}
			}					
			L_tgt[k].x=tal/L_tgt[k].area;
			L_tgt[k].y=tal1/L_tgt[k].area;
			L_tgt[k].h=L_tgt[k].down-L_tgt[k].up+1;
			L_tgt[k].w=L_tgt[k].right-L_tgt[k].left+1;
			k+=1;
		}
	}
	if(targetnum==0) object_counter[0]=0;
	else if(targetnum==1) object_counter[0]=1;
	else 
	{
		if(L_tgt[targetnum-1].gray==2147483647) object_counter[0]=k;
		else
		{
			L_tgt[k].x=L_tgt[targetnum-1].x;
			L_tgt[k].y=L_tgt[targetnum-1].y;
			L_tgt[k].h = L_tgt[targetnum - 1].h;
			L_tgt[k].w = L_tgt[targetnum - 1].w;
			L_tgt[k].up = L_tgt[targetnum - 1].up;
			L_tgt[k].down = L_tgt[targetnum - 1].down;
			L_tgt[k].left = L_tgt[targetnum - 1].left;
			L_tgt[k].right = L_tgt[targetnum - 1].right;
			L_tgt[k].flag = L_tgt[targetnum - 1].flag;
			L_tgt[k].gray = L_tgt[targetnum - 1].gray;
			L_tgt[k].area = L_tgt[targetnum - 1].area;
			object_counter[0]=k+1;
		}
	}
	Correlation(srcimg, SIZEW_SEARCH, SIZEH_SEARCH);
	memset((unsigned char*)L_tgt_0,0, MAX_NUM_TARGET*sizeof(struct tgt));
	int count = 0;
	if(object_counter[1]>0)
	{
		for(i=0;i<object_counter[1];i++)
		{
			if (target[i].confidence > CONFIDENCE_NUM-1 /*&& target[i].angle_change<MAX_ANGLE_CHANGE*/)
			{
				float average_velocity_x = 0;
				float average_velocity_y = 0;
				int k=0;
				for ( k = 1; k < CONFIDENCE_NUM; k++)//ƽ���ٶ�
				{
					average_velocity_x += abs(target[i].history_velocity_x[k]);
					average_velocity_y += abs(target[i].history_velocity_y[k]);
				}
				average_velocity_x = average_velocity_x / (CONFIDENCE_NUM - 1);
				average_velocity_y = average_velocity_y / (CONFIDENCE_NUM - 1);
				float trace_deviation_x = fabs(average_velocity_x * (CONFIDENCE_NUM - 1) - abs(target[i].history_x[CONFIDENCE_NUM - 1] - target[i].history_x[0]));
				float trace_deviation_y = fabs(average_velocity_y * (CONFIDENCE_NUM - 1) - abs(target[i].history_y[CONFIDENCE_NUM - 1] - target[i].history_y[0]));
	//			if (trace_deviation_x<12&& trace_deviation_y<12&&((average_velocity_x>=2&&average_velocity_x>target[i].w/5)||(average_velocity_y >= 2&&average_velocity_y>target[i].h/5)))
				{
					L_tgt_0[count].x = ((target[i].centroid_x));
					L_tgt_0[count].y = ((target[i].centroid_y));
					L_tgt_0[count].w = target[i].w ;
					L_tgt_0[count].h = target[i].h ;
					L_tgt_0[count].flag = 1;
				}
			}
			else
			{
				L_tgt_0[count].flag = 0;
			}
			count++;

		}
	}
	//return L_tgt_0;
}

void PickDotTarget(unsigned char * buf, int w, int h, int threshold, int flag)
{
	int i = 0, j = 0;
	int k = 0, n = 0, m = 0;
	targetnum  = 0;
	Picktarget(buf, w, h, threshold, 1); //��ȡ��Ŀ��?---?Ĭ����ֵΪ?0
    for(int i = 0; i < 100; i++){
        L_tgt[i].flag = 0;
    }
	k = 0;
	for (i = 0; i < targetnum; i++)
	{
		if (/*L_tgt[i].area<62500&&*/L_tgt[i].area >= 4 && L_tgt[i].down  - L_tgt[i].up >= 2 && L_tgt[i].right  - L_tgt[i].left  >= 2 && L_tgt[i].gray < 2147483647)
		{
			L_tgt[k].x = (L_tgt[i].x) / L_tgt[i].area;
			L_tgt[k].y = (L_tgt[i].y) / L_tgt[i].area;
			L_tgt[k].up = L_tgt[i].up;
			L_tgt[k].down = L_tgt[i].down;
			L_tgt[k].left = L_tgt[i].left;
			L_tgt[k].right = L_tgt[i].right;
			L_tgt[k].area = L_tgt[i].area;
			L_tgt[k].gray = L_tgt[i].gray / L_tgt[i].area;
			L_tgt[k].w = L_tgt[i].right - L_tgt[i].left + 1;
			L_tgt[k].h = L_tgt[i].down - L_tgt[i].up + 1;

			k += 1;
		}
	}
	targetnum = k;
	k = 0;
	for (i = 0; i < targetnum; i++)//���Ͻ�����ϲ�??targetnum-1
	{
		if (L_tgt[i].gray < 2147483647)
		{
			L_tgt[k].up = L_tgt[i].up;
			L_tgt[k].down = L_tgt[i].down;
			L_tgt[k].left = L_tgt[i].left;
			L_tgt[k].right = L_tgt[i].right;
			L_tgt[k].area = L_tgt[i].area;
			tal = L_tgt[i].x * L_tgt[i].area;
			tal1 = L_tgt[i].y * L_tgt[i].area;
			for (j = i + 1; j < targetnum; j++)
			{
				n = L_tgt[i].x - L_tgt[j].x;
				m = L_tgt[i].y - L_tgt[j].y;
				n = n * n + m * m;
				if (n < 0 && L_tgt[j].gray < 2147483647)//����
				{
					tal += L_tgt[j].x * L_tgt[j].area;
					tal1 += L_tgt[j].y * L_tgt[j].area;
					if (L_tgt[k].up > L_tgt[j].up)
						L_tgt[k].up = L_tgt[j].up;
					if (L_tgt[k].down < L_tgt[j].down)
						L_tgt[k].down = L_tgt[j].down;
					if (L_tgt[k].left > L_tgt[j].left)
						L_tgt[k].left = L_tgt[j].left;
					if (L_tgt[k].right < L_tgt[j].right)
						L_tgt[k].right = L_tgt[j].right;
					L_tgt[k].area += L_tgt[j].area;
					L_tgt[j].gray = 2147483647;
				}
			}
			L_tgt[k].x = tal / L_tgt[k].area;
			L_tgt[k].y = tal1 / L_tgt[k].area;
			L_tgt[k].h = L_tgt[k].down - L_tgt[k].up + 1;
			L_tgt[k].w = L_tgt[k].right - L_tgt[k].left + 1;
			L_tgt[k].flag = 1;
			k += 1;
		}


	}

	// if (k == 0)
	// {
	// 	m_edit_x = 0;
	// 	m_edit_y = 0;
	// }
	// else 
	// {
	// 	m_edit_x = L_tgt[0].x;
	// 	m_edit_y = L_tgt[0].y;
	// }

		
		//m_edit_w = 1;


	
	 
	//cout << "X = " << m_edit_x << "  Y = " << m_edit_y << "\n" << endl;

	////���ƺ����Ŀ������?---?200��
	//memcpy((unsigned char*)L_tgt_0, (unsigned char*)L_tgt, MAX_NUM_TARGET * sizeof(struct tgt));
	//memset(&g_send_serial_info, 0, sizeof(g_send_serial_info));//set?all?data?0x00

	////����ⷢ����
	//g_send_serial_info.data.frame_head[0] = 0x55;
	//g_send_serial_info.data.frame_head[1] = 0xAA;
	//g_send_serial_info.data.cmd_u2 = 0;
	//g_send_serial_info.data.Img_ID = 0;
	//g_send_serial_info.data.target_SN = 0;
	//g_send_serial_info.data.angle_Xcom_in_motion = 0;
	//g_send_serial_info.data.angle_Ycom_in_motion = 0;
	////����У��
	//int checkNum = 0, t_i = 0;
	//for (t_i = 0; t_i < 19; t_i++)
	//{
	//	checkNum += g_send_serial_info.buffer[t_i];
	//}
	//g_send_serial_info.data.add_check = (unsigned char)(checkNum & 0x00FF);

	//g_send_serial_info.data.target_head[0] = 0x77;

	//g_send_serial_info.data.target_head[1] = 0xAA;
	//g_send_serial_info.data.target_yaw = 0;
	//g_send_serial_info.data.target_pitch = 0;

	//if (k > 16)//?Ŀ�������λ
	//{
	//	g_send_serial_info.data.target_num = 16;
	//}
	//else
	//{
	//	g_send_serial_info.data.target_num = k;
	//}

	////g_send_serial_info.data.target_num?=?testTargetNum;??//����

	//for (t_i = 0; t_i < g_send_serial_info.data.target_num; t_i++)
	//{
	//	g_send_serial_info.data.targetInfo[t_i].Img_ID = 0;
	//	g_send_serial_info.data.targetInfo[t_i].target_SN = 0;
	//	g_send_serial_info.data.targetInfo[t_i].target_width = L_tgt[t_i].w;
	//	g_send_serial_info.data.targetInfo[t_i].target_high = L_tgt[t_i].h;
	//	g_send_serial_info.data.targetInfo[t_i].target_pixel_x = L_tgt[t_i].x;
	//	g_send_serial_info.data.targetInfo[t_i].target_pixel_y = L_tgt[t_i].y;
	//}

	//checkNum = 0;
	//for (t_i = 20; t_i < (31 + g_send_serial_info.data.target_num * 20); t_i++)
	//{
	//	checkNum += g_send_serial_info.buffer[t_i];
	//}
	//g_send_serial_info.data.tail_add_check = (unsigned char)(checkNum & 0x00FF);
	//g_send_serial_info.buffer[t_i] = g_send_serial_info.data.tail_add_check;
}




	

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Picktarget(unsigned char * buf,int w,int h,int threshold,int flag)
{
	unsigned char *Orig_data;            
	int i,j,r=0,ttpp; 
		  
	Orig_data=(unsigned char*)(buf);
	if(flag==1)
	{
		for(i=2;i<h-2;i++)//154
		{    
			temp_num=0;	 
			r=i*w;
			for(j=3;j<w-3;j++)//128
			{       
				if(Orig_data[r+j]>threshold)
				{                        
					temp[temp_num].gray=Orig_data[r+j];
					temp[temp_num].x=j;
					temp_num+=1;
				} 		
			}
			if(temp_num>0)
			{
				tp_num[ptr]=0;
				pr=!ptr;
				tp[0][ptr].x=temp[0].x;
				tp[0][ptr].l=temp[0].x;
				tp[0][ptr].g=temp[0].gray;
				tp[0][ptr].a=1;
				tp_num[ptr]=1;
				ttpp=0;
				for(j=1;j<temp_num;j++)
				{
					ttpp=tp_num[ptr]-1;
					if(temp[j].x-temp[j-1].x==1)
					{				
						tp[ttpp][ptr].x+=temp[j].x;
						tp[ttpp][ptr].g+=temp[j].gray;
						tp[ttpp][ptr].a+=1;
					}
					else
					{
						tp[ttpp][ptr].r=temp[j-1].x;
						tp_num[ptr]+=1;
						ttpp=tp_num[ptr]-1;
						tp[ttpp][ptr].x=temp[j].x;
						tp[ttpp][ptr].l=temp[j].x;
						tp[ttpp][ptr].g=temp[j].gray;
						tp[ttpp][ptr].a=1;				
					}
				}
				tp[ttpp][ptr].r=temp[j-1].x;
				tp[0][ptr].y=i;
				Coalition(tp[0][ptr].y); 
				ptr=pr;
			}
		}
	}

	else if(flag==0)
	{
		for(i=3;i<h-3;i++)//154
		{    
			temp_num=0;	 
			r=i*w;
			for(j=3;j<w-3;j++)//128
			{       
				if(Orig_data[r+j]<threshold)
				{                        
					temp[temp_num].gray=Orig_data[r+j];
					temp[temp_num].x=j;
					temp_num+=1;
				} 		
			}
			if(temp_num>0)
			{
				tp_num[ptr]=0;
				pr=!ptr;
				tp[0][ptr].x=temp[0].x;
				tp[0][ptr].l=temp[0].x;
				tp[0][ptr].g=temp[0].gray;
				tp[0][ptr].a=1;
				tp_num[ptr]=1;
				ttpp=0;
				for(j=1;j<temp_num;j++)
				{
					ttpp=tp_num[ptr]-1;
					if(temp[j].x-temp[j-1].x==1)
					{				
						tp[ttpp][ptr].x+=temp[j].x;
						tp[ttpp][ptr].g+=temp[j].gray;
						tp[ttpp][ptr].a+=1;
					}
					else
					{
						tp[ttpp][ptr].r=temp[j-1].x;
						tp_num[ptr]+=1;
						ttpp=tp_num[ptr]-1;
						tp[ttpp][ptr].x=temp[j].x;
						tp[ttpp][ptr].l=temp[j].x;
						tp[ttpp][ptr].g=temp[j].gray;
						tp[ttpp][ptr].a=1;				
					}
				}
				tp[ttpp][ptr].r=temp[j-1].x;
				tp[0][ptr].y=i;
				Coalition(tp[0][ptr].y); 
				ptr=pr;
			}
		}
	}
	return;    
}

void IMG_Del_Col(unsigned char *  inptr,unsigned char *  outptr,int x_dim,int  width,const char * mask,int Del )
{
		   unsigned char   *IN1,*IN2,*IN3;                         
           unsigned char           *out;                                   
                                           
           unsigned char    pix10,  pix20,  pix30;                                 
           unsigned char    mask10, mask20, mask30;                                
                                                                           
           short      sum,   sum00,  sum11,sum22;                       
           int      i;                                                     
           int      j;                                           
                                                                           
           IN1      =   inptr;                                             
           IN2      =   IN1 + width;                                       
           IN3      =   IN2 + width;                                       
           out      =   outptr+width+1;                                            
           if(Del==2)                                                                
           {
	           for (j = 0; j < x_dim ; j++)                                    
	           {                                                               
	               sum = 0;                                                    
	                                                                           
	               for (i = 0; i < 3; i++)                                     
	               {                                                           
	                   pix10  =   IN1[i];                                      
	                   pix20  =   IN2[i];                                      
	                   pix30  =   IN3[i];                                      
	                                                                          
	                   mask10 =   mask[i];                                     
	                   mask20 =   mask[i + 3];                                 
	                   mask30 =   mask[i + 6];                                 
	                                                                           
	                   sum00  =   pix10 * mask10;                              
	                   sum11  =   pix20 * mask20;                              
	                   sum22  =   pix30 * mask30;                              
	                                                                           
	                   sum   +=   sum00 + sum11+ sum22;                        
	               }                                                           
	                                                                          
	               IN1++;                                                      
	               IN2++;                                                      
	               IN3++;
	               *out=sum/9;
	               out++;
	             }
		   }
		   if(Del==1)                                                                
           {
	           for (j = 0; j < x_dim ; j++)                                    
	           {                                                               
	               sum = 0;                                                    
	                                                                           
	               for (i = 0; i < 3; i++)                                     
	               {                                                           
	                   pix10  =   IN1[i];                                      
	                   pix20  =   IN2[i];                                      
	                   pix30  =   IN3[i];                                      
	                                                                          
	                   mask10 =   mask[i];                                     
	                   mask20 =   mask[i + 3];                                 
	                   mask30 =   mask[i + 6];                                 
	                                                                           
	                   sum00  =   pix10 * mask10;                              
	                   sum11  =   pix20 * mask20;                              
	                   sum22  =   pix30 * mask30;                              
	                                                                           
	                   sum   +=   sum00 + sum11+ sum22;                        
	               }                                                           
	                                                                          
	               IN1++;                                                      
	               IN2++;                                                      
	               IN3++;
	               *out=(sum>=1275&&*IN2==255)?255:0;
	               out++;
	             }
		   }
		   if(Del==0)                                                                
           {
	           for (j = 0; j < x_dim ; j++)                                    
	           {                                                               
	               sum = 0;                                                    
	                                                                           
	               for (i = 0; i < 3; i++)                                     
	               {                                                           
	                   pix10  =   IN1[i];                                      
	                   pix20  =   IN2[i];                                      
	                   pix30  =   IN3[i];                                      
	                                                                          
	                   mask10 =   mask[i];                                     
	                   mask20 =   mask[i + 3];                                 
	                   mask30 =   mask[i + 6];                                 
	                                                                           
	                   sum00  =   pix10 * mask10;                              
	                   sum11  =   pix20 * mask20;                              
	                   sum22  =   pix30 * mask30;                              
	                                                                           
	                   sum   +=   sum00 + sum11+ sum22;                        
	               }                                                           
	                                                                          
	               IN1++;                                                      
	               IN2++;                                                      
	               IN3++;
	               *out=(sum>=1275)?255:0;
			   		out++;
			   }
		   }

}


void Coalition(short col)
{
	int j,k,ttpp,t1;
	
	for(j=0;j<tp_num[ptr];j++) 
	{
		tp[j][ptr].id=32767;
		if(tp[0][pr].y==(col-1))
		{
			t1=0;
			for(k=0;k<tp_num[pr];k++)
			{
				if(tp[j][ptr].l<=(tp[k][pr].r+1)&&(tp[j][ptr].r+1)>=tp[k][pr].l)
				{	
					tp[j][ptr].id=0;
					if(!t1)	
					{						
						ttpp=tp[k][pr].f;
						L_tgt[ttpp].x+=tp[j][ptr].x;
						L_tgt[ttpp].y+=col*tp[j][ptr].a;
						L_tgt[ttpp].gray+=tp[j][ptr].g;
						L_tgt[ttpp].area+=tp[j][ptr].a; 
						tp[j][ptr].f=ttpp;
						if(L_tgt[ttpp].left>tp[j][ptr].l) L_tgt[ttpp].left=tp[j][ptr].l;
						if(L_tgt[ttpp].right<tp[j][ptr].r) L_tgt[ttpp].right=tp[j][ptr].r;
						L_tgt[ttpp].down=col;
					}
					else
					{
						if(ttpp!=tp[k][pr].f&&L_tgt[tp[k][pr].f].gray<2147483647)
						{
							L_tgt[ttpp].x+=L_tgt[tp[k][pr].f].x;
							L_tgt[ttpp].y+=L_tgt[tp[k][pr].f].y;
							L_tgt[ttpp].gray+=L_tgt[tp[k][pr].f].gray;
							L_tgt[ttpp].area+=L_tgt[tp[k][pr].f].area;
							if(L_tgt[ttpp].left>L_tgt[tp[k][pr].f].left) L_tgt[ttpp].left=L_tgt[tp[k][pr].f].left;
							if(L_tgt[ttpp].right<L_tgt[tp[k][pr].f].right) L_tgt[ttpp].right=L_tgt[tp[k][pr].f].right;
							if(L_tgt[ttpp].up>L_tgt[tp[k][pr].f].up) L_tgt[ttpp].up=L_tgt[tp[k][pr].f].up;
							L_tgt[tp[k][pr].f].gray=2147483647;
							tp[k][pr].f=ttpp;
						}
					}
					t1+=1;
				}				
			}
		}
		if(tp[j][ptr].id)
		{
			tp[j][ptr].f=targetnum;
			L_tgt[targetnum].x=tp[j][ptr].x;
			L_tgt[targetnum].y=col*tp[j][ptr].a;//y accumulate in line
			L_tgt[targetnum].gray=tp[j][ptr].g;
			L_tgt[targetnum].area=tp[j][ptr].a;
			L_tgt[targetnum].left=tp[j][ptr].l;
			L_tgt[targetnum].right=tp[j][ptr].r;
			L_tgt[targetnum].up=col;
			L_tgt[targetnum].down=col;			
			targetnum=targetnum>MAX_NUM_L_tgt? MAX_NUM_L_tgt :targetnum+1;
		}
	}
	return;
}


void Correlation(unsigned char * img,int w,int h)
{
	short i,j,mini,num,flag1,diff,ensure_num=3;	
	short k[6];int hnum = 0;
	num=0;
	for(i=0;i<object_counter[1];i++)
	{
		mini=100;		
		k[2]=target[i].centroid_x;
		k[3]=target[i].centroid_y;
//		if(target[i].confidence<5) diff=50;
//		else 
		diff=16;
		for(j=0;j<object_counter[0];j++)
		{
			if(L_tgt[j].x!=32767)
			{
				k[0]=abs(k[2]-L_tgt[j].x);
				k[1]=abs(k[3]-L_tgt[j].y);
				k[1]=sqrt(k[0] * k[0] + k[1] * k[1]);
				k[4] = fabs(target[i].area - L_tgt[j].area); 
				if(k[1]<=diff&&k[4]<=2*(target[i].area> L_tgt[j].area? L_tgt[j].area: target[i].area))
				{
					if(mini>k[1])
					{
						mini=k[1];flag1=j;
					}
				}
			}
		}
		if(mini==100)//|| (!checkdotfeature(img,w,h, target[i].centroid_x, target[i].centroid_y, target[i].w, target[i].h)))
		{
			target[i].confidence-=1;
			if(target[i].confidence>0)
			{
				target[num].confidence=target[i].confidence;
				target[num].area=target[i].area;
				target[num].centroid_x=k[2];
				target[num].centroid_y=k[3];
				target[num].up=target[i].up;
				target[num].trace_deviation_x = target[i].trace_deviation_x;
				target[num].trace_deviation_y = target[i].trace_deviation_y;
				target[num].accelerate_x = target[i].accelerate_x;
				target[num].accelerate_y = target[i].accelerate_y;
				target[num].angle_change = target[i].angle_change;
				target[num].velocity_x = target[i].velocity_x;
				target[num].velocity_y = target[i].velocity_y;
				target[num].down=target[i].down;
				target[num].left=target[i].left;			
				target[num].right=target[i].right;
				for (hnum = 0; hnum < CONFIDENCE_NUM; hnum++)
				{
					target[num].history_velocity_x[hnum] = target[i].history_velocity_x[hnum];
					target[num].history_velocity_y[hnum] = target[i].history_velocity_y[hnum];
					target[num].history_x[hnum] = target[i].history_x[hnum];
					target[num].history_y[hnum] = target[i].history_y[hnum];

				}
		//////���ӿ��͸�
				target[num].h=target[i].down-target[i].up+1;
				target[num].w=target[i].right-target[i].left+1;
				num=num< (MAX_NUM_TARGET-1)?(num+1): (MAX_NUM_TARGET - 1);
			} 
		}
		else
		{ 	
			k[0]=L_tgt[flag1].x-target[i].centroid_x;
			k[1]=L_tgt[flag1].y-target[i].centroid_y;
			//k[2]=target[i].confidence+1;
			/*if(k[2]>=ensure_num&&track_num==0)
			{
				targettrack.area=(target[i].area+L_tgt[flag1].area)>>1;
				targettrack.centroid_x=(target[i].centroid_x+L_tgt[flag1].x)>>1;
				targettrack.centroid_y=(target[i].centroid_y+L_tgt[flag1].y)>>1;
				targettrack.velocity_xrat=target[i].velocity_x-k[0];
				targettrack.velocity_yrat=target[i].velocity_y-k[1];
				targettrack.velocity_x=k[0];			
				targettrack.velocity_y=k[1];
				targettrack.confidence=8;
				track_num=1;
			}
			else*/
			{		
				target[num].confidence=target[i].confidence< CONFIDENCE_NUM+8?target[i].confidence+1:target[i].confidence;
				target[num].area=(/*target[i].area+*/L_tgt[flag1].area)/*>>1*/;
				target[num].trace_deviation_x = abs(L_tgt[flag1].x - (target[i].centroid_x + target[i].velocity_x + target[i].accelerate_x));
				target[num].trace_deviation_y = abs(L_tgt[flag1].y - (target[i].centroid_y + target[i].velocity_y + target[i].accelerate_y));
				target[num].centroid_x=(/*target[i].centroid_x+*/L_tgt[flag1].x);//>>1;
				target[num].centroid_y=(/*target[i].centroid_y+*/L_tgt[flag1].y);//>>1;
				if ((k[0] * k[0] + k[1] * k[1]) == 0 || (target[i].velocity_x * target[i].velocity_x + target[i].velocity_y * target[i].velocity_y) == 0)
				target[num].angle_change = 0;
				else target[num].angle_change = fabs(asin((k[1]* target[i].velocity_x*1.0- k[0] * target[i].velocity_y*1.0)/(sqrt(k[0] * k[0] + k[1] * k[1])*sqrt(target[i].velocity_x * target[i].velocity_x + target[i].velocity_y * target[i].velocity_y))));
				target[num].accelerate_x = k[0] - target[i].velocity_x;
				target[num].accelerate_y = k[1] - target[i].velocity_y;
				target[num].velocity_x = k[0];
				target[num].velocity_y = k[1];
				target[num].up= L_tgt[flag1].up;
				target[num].down= L_tgt[flag1].down;
				target[num].left= L_tgt[flag1].left;
				target[num].right= L_tgt[flag1].right;
				//���ӿ��͸�
				target[num].h= L_tgt[flag1].down- L_tgt[flag1].up+1;
				target[num].w= L_tgt[flag1].right- L_tgt[flag1].left+1;
				for (hnum = 0; hnum < CONFIDENCE_NUM - 1; hnum++)
				{
					target[i].history_velocity_x[hnum] = target[i].history_velocity_x[hnum + 1];
					target[i].history_velocity_y[hnum] = target[i].history_velocity_y[hnum + 1];
					target[i].history_x[hnum] = target[i].history_x[hnum + 1];
					target[i].history_y[hnum] = target[i].history_y[hnum + 1];
				}
				target[i].history_velocity_x[CONFIDENCE_NUM -1] = k[0];
				target[i].history_velocity_y[CONFIDENCE_NUM - 1] = k[1];
				target[i].history_x[CONFIDENCE_NUM - 1] = L_tgt[flag1].x;
				target[i].history_y[CONFIDENCE_NUM - 1] = L_tgt[flag1].y;
				for (hnum = 0; hnum < CONFIDENCE_NUM; hnum++)
				{
					target[num].history_velocity_x[hnum] = target[i].history_velocity_x[hnum];
					target[num].history_velocity_y[hnum] = target[i].history_velocity_y[hnum];
					target[num].history_x[hnum] = target[i].history_x[hnum];
					target[num].history_y[hnum] = target[i].history_y[hnum];

				}
				num = num < (MAX_NUM_TARGET - 1) ? (num + 1) : (MAX_NUM_TARGET - 1);
			}
			L_tgt[flag1].x=32767;
		}
	}		
	object_counter[1]=num;
	for(i=0;i<object_counter[0];i++)
	{
		if(L_tgt[i].x!=32767)
		{ 			
			target[object_counter[1]].area=L_tgt[i].area;
			target[object_counter[1]].up=L_tgt[i].up;
			target[object_counter[1]].down=L_tgt[i].down;
			target[object_counter[1]].left=L_tgt[i].left;
			target[object_counter[1]].right=L_tgt[i].right;
			target[object_counter[1]].centroid_x=L_tgt[i].x;
			target[object_counter[1]].centroid_y=L_tgt[i].y;
			//���ӿ��͸�
			target[object_counter[1]].h=L_tgt[i].down-L_tgt[i].up+1;
			target[object_counter[1]].w=L_tgt[i].right-L_tgt[i].left+1;
			target[object_counter[1]].trace_deviation_x = 0;
			target[object_counter[1]].trace_deviation_y = 0;
			target[object_counter[1]].accelerate_x = 0;
			target[object_counter[1]].accelerate_y = 0;
			target[object_counter[1]].velocity_x = 0;
			target[object_counter[1]].velocity_y = 0;
			target[object_counter[1]].confidence=1;
			for (hnum = 0; hnum < CONFIDENCE_NUM; hnum++)
			{
				target[object_counter[1]].history_velocity_x[hnum] = 0;
				target[object_counter[1]].history_velocity_y[hnum] = 0;
				target[object_counter[1]].history_x[hnum] = L_tgt[i].x;
				target[object_counter[1]].history_y[hnum] = L_tgt[i].y;

			}
			object_counter[1] = object_counter[1] < (MAX_NUM_TARGET - 1) ? (object_counter[1] + 1) : (MAX_NUM_TARGET - 1);
		}
	}
	return;
}

void Midfilter(unsigned char * src,short Width,short gatecount,unsigned char * dst)
{
	int x=0;
	unsigned char * addr=(unsigned char *)(src);
	unsigned char * Midaddr=(unsigned char *)(dst+(gatecount-1)*Width);
	unsigned short temp=0;
	unsigned char temp1=0;
	unsigned char temp2=0;
	
	for(x=0;x<Width;x++)
	{
		Midtemp1[x]=Midtemp2[x];
		Midtemp2[x]=Midtemp[x];
	}
	Midtemp[0]=addr[0];
	Midtemp[Width-1]=addr[Width-1];
	for(x=1;x<Width-1;x++)
	{	temp=addr[x]+addr[x-1];
		temp1=addr[x-1]>addr[x]?addr[x-1]:addr[x];
		temp2=temp-temp1;
		Midtemp[x]=temp1<addr[x+1]?temp1:temp2>addr[x+1]?temp2:addr[x+1];
	}
	for(x=0;x<Width;x++)
		Midaddr[x+Width]=Midtemp[x];		
	if(gatecount>1)
	{	for(x=0;x<Width;x++)
		{	temp=Midtemp1[x]+Midtemp2[x];
			temp1=Midtemp1[x]>Midtemp2[x]?Midtemp1[x]:Midtemp2[x];
			temp2=temp-temp1;
			Midaddr[x]=temp1<Midtemp[x]?temp1:temp2>Midtemp[x]?temp2:Midtemp[x];
		}
	}

}

int checkdotfeature(unsigned char* img, int imgw, int imgh, int x, int y, int w, int h)
{
	int i, j;
	int cx1 = 0; int cy1 = 0; int pixelnum1 = 0; int dotpos = 0;
	int cx = 0; int cy = 0; int pixelnum = 0;
	int maxgray = 0; int mingray = 0xffff;
	int averagegray = 0; int averagegray1 = 0;

	for (j = y - h / 2; j < y + h / 2; j++)
		for (i = x - w / 2; i < x + w / 2; i++)
		{
			if (i < 0 || i >= imgw || j < 0 || j >= imgh) continue;
			if (mingray > img[j * imgw + i]) mingray = img[j * imgw + i];
			if (maxgray < img[j * imgw + i]) maxgray = img[j * imgw + i];
			averagegray1 = averagegray1 + img[j * imgw + i]; pixelnum++;
		}
	averagegray = averagegray1 / pixelnum;//(averagegray1/(pixelnum)+(maxgray+mingray)/2)/2;
	pixelnum = 0;
	//for (j = y - h / 2; j < y + h / 2; j++)
	//	for (i = x - w / 2; i < x + w / 2; i++)
	//	{
	//		if (i < 0 || i >= imgw || j < 0 || j >= imgh) continue;
	//		if (img[j * imgw + i] > averagegray) { cx += i; cy += j; pixelnum++; }
	//		else { cx1 += i; cy1 += j; pixelnum1++; }
	//	}
	//cx = (cx / pixelnum); cy = (cy / pixelnum);
	//cx1 = (cx1 / pixelnum1); cy1 = (cy1 / pixelnum1);
//	dotpos = sqrt(((cx - cx1) * (cx - cx1) + (cy - cy1) * (cy - cy1)) / (1.0 * (w * w + h * h)));
	return ((maxgray-mingray)>24 && maxgray>180&&(w*1.0/h>0.20)&& (w * 1.0 / h < 5));
}
