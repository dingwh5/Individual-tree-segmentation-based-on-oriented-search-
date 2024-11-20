#include <iostream>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/common.h>
#include <pcl/common/eigen.h>
#include <Eigen/Dense>
#include <filesystem>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <Eigen/Eigenvalues>
#include <pcl/io/pcd_io.h>
#include <pcl/io/io.h>
#include <unsupported/Eigen/MatrixFunctions>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <algorithm>
#define _CRT_SECURE_NO_WARNINGS
#include <Windows.h>
#include <stdio.h>
#include <math.h>
#include "string.h"
#include <minwindef.h>
#include <pcl/PolygonMesh.h>
#include <pcl/conversions.h>
#include <pcl/common/centroid.h>
#include<opencv2/opencv.hpp>
#include"opencv2/highgui/highgui.hpp"  
#include <string>
#include <vector>





// #pragma comment(lib, "gdal_i.lib")
using namespace std;
using namespace pcl;
using namespace Eigen;
#define MAX_LINE 1000   //定义txt中最大行数。可调整更改
#define WITHOUT_NUMPY//添加此宏定义来明确告诉后续代码不使用numpy
#define FLT_MAX 3.402823466e+38F 




struct Lei {
    typedef pcl::PointCloud<pcl::PointXYZI>::Ptr PointCloudPtr;
    PointCloudPtr cloud; // 点云
    PointXYZI high;
    PointXYZI mean;               //mean中的intensity表示平均半径
    PointXYZI low;
    int valid;
    std::vector<float> features;
    double area;
    double eccentricity;
};






void HighS(const PointCloud<PointXYZI>::Ptr& cloudxyzi, KdTreeFLANN<PointXYZI>::Ptr& kdtree,
    PointXYZI Core, const float& k, const char& search, double& KnnHighZ, int& KnnHighZ_indices)
{
    vector<int> KNNIndices(k); // 存放搜索到的 k 邻近点的索引值
    vector<float> KNNSquareDistance(k); //存放搜索到的 k 邻近点的对应到查询点的距离平方  
    PointXYZI KnnHigh;
    KnnHigh.x = -FLT_MAX;   KnnHigh.y = -FLT_MAX;  KnnHigh.z = -FLT_MAX;
    if (search == 'k') {
        if (kdtree->nearestKSearch(Core, k, KNNIndices, KNNSquareDistance) > 0) {          //nearestK
            for (int m = 0; m < KNNIndices.size(); m++) {
                if ((cloudxyzi->points[KNNIndices.at(m)].z > KnnHighZ)) {        // && (dis < 3 )
                    KnnHigh = cloudxyzi->points[KNNIndices.at(m)];
                    KnnHighZ_indices = KNNIndices.at(m);
                }
            }
        }
    }

    if (search == 'r') {
        if (kdtree->radiusSearch(Core, k, KNNIndices, KNNSquareDistance) > 0) {          //nearestK
            for (int m = 0; m < KNNIndices.size(); m++) {
                if ((cloudxyzi->points[KNNIndices.at(m)].z > KnnHigh.z)) {
                    KnnHigh = cloudxyzi->points[KNNIndices.at(m)];
                    KnnHighZ_indices = KNNIndices.at(m);
                }
                else if ((cloudxyzi->points[KNNIndices.at(m)].z == KnnHigh.z)) {
                    if (cloudxyzi->points[KNNIndices.at(m)].x > KnnHigh.x) {
                        KnnHigh = cloudxyzi->points[KNNIndices.at(m)];
                        KnnHighZ_indices = KNNIndices.at(m);
                    }
                    else if (cloudxyzi->points[KNNIndices.at(m)].x == KnnHigh.x) {
                        if (cloudxyzi->points[KNNIndices.at(m)].y > KnnHigh.y) {
                            KnnHigh = cloudxyzi->points[KNNIndices.at(m)];
                            KnnHighZ_indices = KNNIndices.at(m);
                        }
                    }
                }
            }
        }
    }
    KnnHighZ = KnnHigh.z;
}




void calculateEllipse(const PointCloud<PointXYZI>::Ptr& input_cloud, double &area, double & eccentricity) {
    // 读取点云数据
    std::vector<cv::Point2f> points;
    // 将点云数据填充到points中
    for (int i = 0; i < input_cloud->points.size(); i++) {
        points.push_back(cv::Point2f(input_cloud->points[i].x, input_cloud->points[i].y));
    }

    // 拟合椭圆
    if (input_cloud->points.size() < 10) {
        area = 0;
        eccentricity = 1;
    }
    else {
        cv::RotatedRect ellipse = cv::fitEllipse(points);

        // 计算椭圆面积
        area = M_PI * ellipse.size.width / 2 * ellipse.size.height / 2;
        eccentricity = std::sqrt(1 - std::pow(ellipse.size.width / 2.0f / ellipse.size.height / 2.0f, 2));
    }
}




void calculateCorvariance(const PointCloud<PointXYZI>::Ptr& input_cloud, Eigen::Matrix3f* output_cor, float& variance_Z)
{
    //Settings 
    pcl::PointXYZ point_can;	// Candidate point
    Eigen::Vector3f key_coor = Eigen::Vector3f::Zero(3, 1);	// Coordinates of key point 
    Eigen::Vector3f can_coor = Eigen::Vector3f::Zero(3, 1);	// Coordinates of candidate point
    Eigen::Vector3f sum_coor = Eigen::Vector3f::Zero(3, 1);	// Sum of coordinates
    Eigen::Vector3f diff_coor = Eigen::Vector3f::Zero(3, 1);	// Coordinates difference
    Eigen::Matrix3f cor_single = Eigen::Matrix3f::Zero(3, 3);	// CTC for a point
    Eigen::Matrix3f cor_sum = Eigen::Matrix3f::Zero(3, 3);		// Sum of all CTC

    std::vector<int> points_id_support; 	// Neighbors within radius search
    std::vector<float> points_dis_support; 	// Distance of these neighbors 

    int num_support = input_cloud->points.size();	//Num of input point
    int num_support_min = 3;
    float point_dis = 0;
    float sum_dis = 0;

    //Tranverse in the support region
    if (num_support > num_support_min)
    {
        for (size_t i = 0; i < num_support; i++)
        {
            //point_can=input_cloud->points[i];
            sum_coor[0] = sum_coor[0] + input_cloud->points[i].x;
            sum_coor[1] = sum_coor[1] + input_cloud->points[i].y;
            sum_coor[2] = sum_coor[2] + input_cloud->points[i].z;
        }

        //key point
        key_coor[0] = sum_coor[0] / num_support;
        key_coor[1] = sum_coor[1] / num_support;
        key_coor[2] = sum_coor[2] / num_support;

        for (size_t j = 0; j < num_support; j++)
        {
            //Get candidate point in support
            //point_can=input_cloud->points[j]; 
            can_coor[0] = input_cloud->points[j].x;
            can_coor[1] = input_cloud->points[j].y;
            can_coor[2] = input_cloud->points[j].z;

            //Coordinate differences
            diff_coor = can_coor - key_coor;

            //CTC
            cor_single = diff_coor * diff_coor.transpose();
            cor_sum = cor_sum + cor_single;
        }
        variance_Z = (cor_sum)(2, 2) / num_support;
    }
    else
    {
        sum_dis = 1;
        cor_sum = Eigen::Matrix3f::Zero(3, 3);
        variance_Z = 0;
    }

    //Final covariance matrix
    *output_cor = cor_sum;
}




std::vector<float>
calculateEigenFeatures(const PointCloud<PointXYZI>::Ptr& input_cloud)//Eigen features
{
    //Parameter setting
    Eigen::Vector3f eig_values;
    Eigen::Matrix3f eig_vectors;
    Eigen::Matrix3f* cor_matrix = new Eigen::Matrix3f;
    std::vector<float> output_features;

    int point_num = 0;
    point_num = input_cloud->points.size();
    float eig_e1 = 0, eig_e2 = 0, eig_e3 = 0;
    float varianceZ;
    //float *output_features=new float[8];

    //Weighted corvarance matrix
    //this->calculateWeightedCorvariance(input_cloud,cor_matrix);
    calculateCorvariance(input_cloud, cor_matrix, varianceZ);

    //EVD
    pcl::eigen33(*cor_matrix, eig_vectors, eig_values);

    //cout << eig_values[0] << '\t' << eig_values[1] << '\t' << eig_values[2] << '\n';


    //Eigen values (normalized)
    if (eig_values[0] == 0 && eig_values[1] == 0 && eig_values[2] == 0)
    {
        output_features.push_back(float(0)); output_features.push_back(float(1)); output_features.push_back(float(1)); output_features.push_back(float(0));
        output_features.push_back(float(0)); output_features.push_back(float(1)); output_features.push_back(float(1)); output_features.push_back(float(1));

    }
    else
    {
        // 求前两个较大的特征值
        Eigen::Vector3d::Index maxRow, maxCol, minRow, minCol;
        eig_values.maxCoeff(&maxRow, &maxCol);
        eig_values.minCoeff(&minRow, &minCol);
        // l1 > l2 > l3
        float eig_e1s = (float)eig_values[maxRow];
        float eig_e2s = (float)eig_values[3 - maxRow - minRow]; // 这个是巧算，基于 0 + 1 + 2 = 3
        float eig_e3s = (float)eig_values[minRow];

        eig_e1 = (float)eig_e1s / sqrt(pow(eig_e1s, 2) + pow(eig_e2s, 2) + pow(eig_e3s, 2));
        eig_e2 = (float)eig_e2s / sqrt(pow(eig_e1s, 2) + pow(eig_e2s, 2) + pow(eig_e3s, 2));
        eig_e3 = (float)eig_e3s / sqrt(pow(eig_e1s, 2) + pow(eig_e2s, 2) + pow(eig_e3s, 2));


        //Feature calculation
        if (eig_e1 == 0)
        {
            output_features.push_back(float(0));//Linearity
            output_features.push_back(float(1));//Planarity
            output_features.push_back(float(0));//Scattering
        }
        else
        {
            output_features.push_back(float(eig_e1 - eig_e2) / eig_e1);//Linearity
            output_features.push_back(float(eig_e2 - eig_e3) / eig_e1);//Planarity
            output_features.push_back(float(eig_e3) / eig_e1);//Scattering
        }

        output_features.push_back(float(eig_e1) / (eig_e1 + eig_e2 + eig_e3));//Change of curvature

        if (eig_e2 == 0)
        {
            output_features.push_back(float(0));//Anisotropy
        }
        else
        {
            output_features.push_back(float(eig_e1 - eig_e3) / eig_e2);//Anisotropy
        }

        if (eig_e1 * eig_e2 * eig_e3 == 0)
        {
            output_features.push_back(float(0));//Eigenentropy
        }
        else
        {
            output_features.push_back(-1 * (eig_e1 * log(eig_e1) + eig_e2 * log(eig_e2) + eig_e3 * log(eig_e3)));//Eigenentropy
        }

        output_features.push_back(eig_e1 + eig_e2 + eig_e3);//Sum of eigen values
        output_features.push_back(pow(float(eig_e1 * eig_e2 * eig_e3), float(1.0 / 3)));//Omnivariance
    }
    return(output_features);
}





void Cha(const PointCloud<PointXYZ>::Ptr& cloudxyz, const float& k, const char& search, int& numSeg, string in)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr  cloudxyzi(new pcl::PointCloud<pcl::PointXYZI>);
    PointXYZI Point;
    for (int i = 0; i < cloudxyz->size(); i++) {
        Point.x = cloudxyz->points[i].x - cloudxyz->points[0].x;
        Point.y = cloudxyz->points[i].y - cloudxyz->points[0].y;
        Point.z = cloudxyz->points[i].z;
        Point.intensity = -1;
        cloudxyzi->points.push_back(Point);
    }


    //*****************************************************************查询*************************************************************************************                
    vector<int> Tujing;           //存储搜索路径上的所有点的索引序列
    int i = 0;                   // 存放当前处理点的下标
    double KnnHighZ;              //存放临近点点云的最大Z值
    int KnnHighZ_indices, Core_indices;
    PointXYZI Core;
    KdTreeFLANN<PointXYZI>::Ptr kdtree(new KdTreeFLANN<PointXYZI>);
    kdtree->setInputCloud(cloudxyzi);
    for (const auto& point : *cloudxyzi) {
        Core = point; // 要查询的点 
        Core_indices = i;
        Tujing.push_back(i);
        HighS(cloudxyzi, kdtree, Core, k, search, KnnHighZ, KnnHighZ_indices);
        while (KnnHighZ > Core.z && cloudxyzi->points[KnnHighZ_indices].intensity == -1) {
            Core = cloudxyzi->points[KnnHighZ_indices];
            Core_indices = KnnHighZ_indices;
            HighS(cloudxyzi, kdtree, Core, k, search, KnnHighZ, KnnHighZ_indices);
            Tujing.push_back(Core_indices);
        }

        if (cloudxyzi->points[KnnHighZ_indices].intensity != -1) {
            for (int q = 0; q < Tujing.size(); q++) {
                cloudxyzi->points[Tujing.at(q)].intensity = cloudxyzi->points[KnnHighZ_indices].intensity;
            }
        }
        else {
            for (int q = 0; q < Tujing.size(); q++) {
                cloudxyzi->points[Tujing.at(q)].intensity = numSeg;
            }
            numSeg = numSeg + 1;
        }
        i = i + 1;
        Tujing.clear();
    }


    //****************************************************************输出查询点的XYZI*******************************************************************************   
    ofstream S(in);
    S.precision(10);


    int chul;
    for (int i = 0; i < cloudxyzi->points.size(); i++) {
        chul = cloudxyzi->points[i].intensity;
        S << cloudxyzi->points[i].x + cloudxyz->points[0].x << "\t"
            << cloudxyzi->points[i].y + cloudxyz->points[0].y << "\t"
            << cloudxyzi->points[i].z << "\t" << chul << '\n';

    }
    S.close();
}




std::vector<Lei> calculateFeatures(const PointCloud<PointXYZI>::Ptr& cloudxyzi, string& in, vector<vector<int>> Color) {
    //*****************************************************************导入，计算numSeg*************************************************************************************
    pcl::PointXYZI Core;                                            //找到总类数
    int numSeg = 0;
    pcl::PointXYZI lowest;                                  //高程归一化，计算最低点
    lowest.z = FLT_MAX;
    for (int n = 0; n < cloudxyzi->points.size(); n++) {
        Core = cloudxyzi->points[n]; // 要查询的点 
        int chul = static_cast<int>(Core.intensity);
        if (chul > numSeg) {
            numSeg = chul;
        }
        if (Core.z < lowest.z) {
            lowest = Core;
        }
    }
    numSeg = numSeg + 1;



    //*****************************************************************归类，属性计算*************************************************************************************
    std::vector<Lei> seg(numSeg);
    for (int n3 = 0; n3 < numSeg; n3++) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr  cloudsu(new pcl::PointCloud<pcl::PointXYZI>);
        for (int n4 = 0; n4 < cloudxyzi->points.size(); n4++) {            // Seg赋值，遍历每个点
            Core = cloudxyzi->points[n4]; // 要查询的点 
            int chul = static_cast<int>(Core.intensity);
            if (chul == n3) {
                cloudsu->points.push_back(Core);
            }
        }


        seg[n3].low.z = FLT_MAX;
        int size = cloudsu->points.size();
        double sum_x = 0; double sum_y = 0; double sum_z = 0;
        if (size > 0) {
            PointXYZ high, mean;
            for (int n5 = 0; n5 < size; n5++) {
                sum_x += cloudsu->points[n5].x;
                sum_y += cloudsu->points[n5].y;
                sum_z += cloudsu->points[n5].z;
                if (cloudsu->points[n5].z > seg[n3].high.z) {
                    seg[n3].high = cloudsu->points[n5];
                }
                if (cloudsu->points[n5].z < seg[n3].low.z) {
                    seg[n3].low = cloudsu->points[n5];
                }
            }
            seg[n3].mean.x = sum_x / size;
            seg[n3].mean.y = sum_y / size;
            seg[n3].mean.z = sum_z / size;

            double dis; double sum_dis = 0;
            for (int n7 = 0; n7 < size; n7++) {
                dis = sqrt(pow(cloudsu->points[n7].x - seg[n3].mean.x, 2) + pow(cloudsu->points[n7].y - seg[n3].mean.y, 2));
                sum_dis = sum_dis + dis;
            }
            seg[n3].mean.intensity = sum_dis / cloudsu->points.size();


            std::vector<float> output_features = calculateEigenFeatures(cloudsu);
            seg[n3].features = output_features;
            double area, eccentricity;
            calculateEllipse(cloudsu, area, eccentricity);

            seg[n3].area = area;
            seg[n3].eccentricity = eccentricity;



            if ( area > 5 ||  seg[n3].high.z - seg[n3].low.z > 3) {           //area > 3 &&  seg[n3].low.z - lowest.z < 2
                seg[n3].valid = 1;
            }
            else {
                seg[n3].valid = 0;
            }


        }
        seg[n3].cloud = cloudsu;
        cloudsu.reset();
    }




    //*****************************************************************分类结果*************************************************************************************
    stringstream qian, s, hou, ss;                             // 计算、输出中心位置
    qian << in << "_lei.txt";
    s << in << "_point.txt";
    ofstream  Q(qian.str()), S(s.str());
    Q.precision(10);



    int chul;
    int yu = 0;                         //; cloudxyzi->points.size() * 0.005
    for (int n1 = 0; n1 < numSeg; n1++) {
        if (seg[n1].valid > 0) {                //!seg[n1].cloud->empty()
            Q //<< n1 << "\t" << seg[n1].valid << "\t"
                << seg[n1].high.x << "\t" << seg[n1].high.y << "\t" << seg[n1].high.z << "\t" 
                //<< seg[n1].low.z << "\t"
                //<< seg[n1].cloud->points.size() << "\t"// << seg[n1].features[4] << "\t"
                //<< seg[n1].area << "\t" << seg[n1].eccentricity << "\t"
                //<< seg[n1].features[0] << "\t" << seg[n1].features[1] << "\t" << seg[n1].features[2] << "\t" << seg[n1].features[3] << "\t"
                //<< seg[n1].features[4] << "\t" << seg[n1].features[5] << "\t" << seg[n1].features[6] << "\t" << seg[n1].features[7] << "\t"
                //<< Color[n1][0] << "\t" << Color[n1][1] << "\t" << Color[n1][2] 
                << endl;
        }
    }
    for (int i = 0; i < cloudxyzi->points.size(); i++) {
        chul = static_cast<int>(cloudxyzi->points[i].intensity);
        if (seg[chul].cloud->points.size() > yu) {
            S << std::fixed << std::setprecision(6) << cloudxyzi->points[i].x << "\t"
                << cloudxyzi->points[i].y << "\t"
                << cloudxyzi->points[i].z << "\t"
                << chul << "\t" << Color[chul][0] << "\t" << Color[chul][1] << "\t" << Color[chul][2] << '\n';
        }

    }
    Q.close();  S.close();
    return seg;
}







pcl::PointCloud<pcl::PointXYZI>::Ptr generate(const PointCloud<PointXYZI>::Ptr& cloudxyzi, std::vector<Lei> seg) {
    //*****************************************************************所属点更新*************************************************************************************
    int numSeg = seg.size();
    KdTreeFLANN<PointXYZI>::Ptr kdtree(new KdTreeFLANN<PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_validmean(new pcl::PointCloud<pcl::PointXYZI>);
    for (int n3 = 0; n3 < numSeg; n3++) {
        cloud_validmean->points.push_back(seg[n3].mean);
    }
    kdtree->setInputCloud(cloud_validmean);


    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ge(new pcl::PointCloud<pcl::PointXYZI>);
    for (int j1 = 0; j1 < numSeg; j1++) {
        int size = seg[j1].cloud->points.size();
        for (int j2 = 0; j2 < size; j2++) {
            PointXYZI point_ge;
            point_ge.x = seg[j1].cloud->points[j2].x;
            point_ge.y = seg[j1].cloud->points[j2].y;
            point_ge.z = seg[j1].cloud->points[j2].z;

            int k;
            if (numSeg > 50) {
                k = 50;
            }
            else {
                k = numSeg;
            }


            vector<int> KNNIndices(k);
            vector<float> KNNSquareDistance(k);
            kdtree->nearestKSearch(seg[j1].mean, k, KNNIndices, KNNSquareDistance);

            float dis_mean_min = FLT_MAX;
            int dis_mean_min_idx = 10000;
            for (int j3 = 0; j3 < k; j3++) {                                            //计算到临近点的包围圈的距离
                int idx = KNNIndices[j3];
                float dis_mean = sqrt(pow(seg[j1].cloud->points[j2].x - seg[idx].mean.x, 2) + pow(seg[j1].cloud->points[j2].y - seg[idx].mean.y, 2)) - seg[idx].mean.intensity;
                if (dis_mean < dis_mean_min && seg[idx].valid == 1 && seg[j1].cloud->points[j2].z < seg[idx].high.z) {           //距离最小且有效
                    dis_mean_min = dis_mean;
                    dis_mean_min_idx = idx;
                }
                if (seg[j1].valid == 1) {                  //最好情况下保留它本来的点
                    double dis = sqrt(pow(seg[j1].cloud->points[j2].x - seg[j1].mean.x, 2) + pow(seg[j1].cloud->points[j2].y - seg[j1].mean.y, 2)) - seg[j1].mean.intensity;
                    if (dis - dis_mean_min < 1) {
                        dis_mean_min_idx = j1;
                    }
                }
            }
            point_ge.intensity = dis_mean_min_idx;
            cloud_ge->points.push_back(point_ge);
        }
    }
    return cloud_ge;
}






namespace fs = std::filesystem;


int main() {
    std::string folder_path = "F:\\Single_tree\\\Discussion_NEWFOR\\0905_nolayer"; // 指定文件夹路径
    std::vector<std::string> pcd_files;

    // 遍历文件夹中的所有文件
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        std::string file_path = entry.path().string();

        // 检查文件是否是PCD文件
        if (file_path.substr(file_path.find_last_of(".") + 1) == "pcd") {
            pcd_files.push_back(file_path);
        }
    }


    /*string color_str = "F:\\FORinstance_dataset\\color.txt";
    ofstream  color_ofstream(color_str);
    srand((unsigned int)time(NULL));
    for (int n3 = 0; n3 <= 50000; n3++) {               // 置空
        vector<int> row;
        for (int n4 = 0; n4 < 3; n4++) {
            row.push_back(rand() % 255);               //颜色取值0-254
        }
        color_ofstream << row[0] << "\t" << row[1] << "\t" << row[2] << "\n";
        //Color.push_back(row);
    }
    color_ofstream.close();*/
    string color_in = "F:\\Single_tree\\FORinstance\\color.txt";
    std::ifstream file_color(color_in);
    vector<vector<int>> Color;                  // 颜色初始化。每一行代表每一类，三列分别表示RGB
    std::string line_color;
    while (std::getline(file_color, line_color))
    {
        std::istringstream iss(line_color);
        vector<int> row;
        int r, g, b;
        iss >> r >> g >> b;
        row.push_back(r); row.push_back(g); row.push_back(b);
        Color.push_back(row);
    }


    // 输出找到的所有PCD文件路径
    for (const auto& file : pcd_files) {
        std::cout << file << std::endl;
        PointCloud<PointXYZ>::Ptr cloudpre(new PointCloud<PointXYZ>);
        pcl::io::loadPCDFile<PointXYZ>(file, *cloudpre);

        // 设置参数循环
        for (float k = 1.0; k <= 2.5; k = k + 0.1) {
            char search = 'r';
            int numJu = 0;                       // 聚合后类数目
            int numSeg = 0;                     // 存放最高点的簇号
            string in = file.substr(0, file.length() - 4) + "_" + search + to_string(k).substr(0, 3) + "_cha.txt";                       //"F:\\NEWFOR\\03_Cotolivier\\03_ALS_CSFgui
            Cha(cloudpre, k, search, numSeg, in);


            clock_t time0 = clock();
            std::ifstream file(in);
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloudxyzi(new pcl::PointCloud<pcl::PointXYZI>);
            std::string line;
            while (std::getline(file, line))
            {
                std::istringstream iss(line);
                pcl::PointXYZI point;
                iss >> point.x >> point.y >> point.z >> point.intensity;
                cloudxyzi->push_back(point);
            }



            //std::sort(cloudxyzi->points.begin(), cloudxyzi->points.end(), [](pcl::PointXYZI p1, pcl::PointXYZI p2) {return p1.z < p2.z; });
            string in_sub = in.substr(0, in.length() - 4);
            std::vector<Lei> seg = calculateFeatures(cloudxyzi, in_sub, Color);    //+"_0005"
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ge = generate(cloudxyzi, seg);
            std::vector<Lei> seg_ge = calculateFeatures(cloud_ge, in_sub + "_ge", Color);

            clock_t time1 = clock();//记录处理结束的时间
            cout <<'k='<<k<< ",共耗时： " << (time1 - time0) / 1000.0 / 60.0 << ".min" << endl;
        }
    }
}