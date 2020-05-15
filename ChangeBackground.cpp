#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>

using namespace  std;
using namespace cv;

Mat mat_to_sample(Mat &img);
int main(int argc, char** argv)
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    Mat src;

    //读取图片
    src = imread( argv[1], 1 );

    //判断图片是否存在
    if(src.empty())
    {
        cout<<"pic is not exist!!!!"<<endl;
        return -1;
    }

    if ( !src.data )
    {
        printf("No image data \n");
        return -1;
    }

    //显示图片
    string win_name = "pic";
    namedWindow(win_name,WINDOW_AUTOSIZE);
    imshow(win_name,src);


    //准备数据
    Mat points = mat_to_sample(src);

    //运行KMeans
    int num_cluster = 4;
    Mat lables;
    Mat centers;
    TermCriteria criteria =  TermCriteria(TermCriteria::EPS+TermCriteria::COUNT,10,0.1);
    kmeans(points,num_cluster,lables,criteria,3,KMEANS_PP_CENTERS,centers);

    //去背景 + 遮罩生成

    Mat mask = Mat::zeros(src.size(),CV_8UC1);
    int index = src.cols*2+2;
    int cindex = lables.at<int>(index,0);  //获取图片左上角(2,2)标签数据 证件照这个位置一般都是背景
    int height = src.rows;
    int width = src.cols;

    //遍历每个像素点
    for(int row=0;row <height;row++)
        for(int col=0;col<width;col++)
        {
            index = row*width+col;
            int lable = lables.at<int>(index,0);
            if(lable == cindex)   //如果当前点为背景点
            {
                mask.at<uchar>(row,col) = 0;    //背景部分为黑色
            }else
            {
                mask.at<uchar>(row,col) = 255;  //原图部分为黑色
            }
        }
    imshow("mask",mask);


    //腐蚀 + 高斯模糊   使边界点过渡更加自然 颜色过渡不会太生硬
    Mat kernel = getStructuringElement(MORPH_RECT,Size(3,3));
    erode(mask,mask,kernel);  //腐蚀 减少边界噪点
    imshow("erode mask",mask);

    GaussianBlur(mask,mask,Size(3,3),0,0);  //高斯模糊
    imshow("blur mask",mask);


    //通道混合
    Vec3b color(0,0,255);
    //color[0] = theRNG().uniform(0,255);
    //color[1] = theRNG().uniform(0,255);
    //color[2] = theRNG().uniform(0,255);

    //创建结果图片
    Mat result(src.size(),src.type());

    //背景与前景边界部分经过高斯处理 该部分的颜色需要渐变处理 看起来效果更加自然
    double w =0;
    int b=0,g=0,r=0;
    int b1=0,g1=0,r1=0;
    int b2=0,g2=0,r2=0;

    for(int row=0;row <height;row++)
        for(int col=0;col<width;col++)
        {
            int m = mask.at<uchar>(row,col);
            if(m == 255)   //前景部分 直接将原图拷贝
            {
                result.at<Vec3b>(row,col) = src.at<Vec3b>(row,col);
            }else if(m==0) //背景部分 使用自己定义的颜色
            {
                result.at<Vec3b>(row,col) = color;
            }else  //前景与背景 高斯过渡部分
            {
                w = m/255.0;
                b1 = src.at<Vec3b>(row,col)[0];
                g1 = src.at<Vec3b>(row,col)[1];
                r1 = src.at<Vec3b>(row,col)[2];

                b2 = color[0];
                g2 = color[1];
                r2 = color[2];

                //颜色随着高斯模糊强度 进行加权渐变
                b = b1*w + b2*(1.0-w);
                g = g1*w + g2*(1.0-w);
                r = r1*w + r2*(1.0-w);

                result.at<Vec3b>(row,col)[0] = b;
                result.at<Vec3b>(row,col)[1] = g;
                result.at<Vec3b>(row,col)[2] = r;
            }
        }
    imshow("result ",result);
    waitKey(0);
    destroyAllWindows();
    return 0;
}

Mat mat_to_sample(Mat &img)
{
    int width = img.cols;
    int height = img.rows;
    int sample_counts = width*height;
    int dims = img.channels();
    Mat points(sample_counts,dims,CV_32F);

    int index = 0;
    for(int row=0;row <img.rows;row++)
        for(int col=0;col<img.cols;col++)
        {
            index = row * width + col;
            Vec3b bgr = img.at<Vec3b>(row,col);
            points.at<float>(index,0) = bgr[0];
            points.at<float>(index,1) = bgr[1];
            points.at<float>(index,2) = bgr[2];

        }

    return points;
}

