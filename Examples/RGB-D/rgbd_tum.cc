#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<unistd.h>
#include<opencv2/core/core.hpp>
#include<System.h>

using namespace std;

/**
 * @brief 加载图像
 * 
 * @param[in] strAssociationFilename    关联文件的访问路径
 * @param[out] vstrImageFilenamesRGB     彩色图像路径序列
 * @param[out] vstrImageFilenamesD       深度图像路径序列
 * @param[out] vTimestamps               时间戳
 */
void LoadImages(
    const string &strAssociationFilename, 
    vector<string> &vstrImageFilenamesRGB,
    vector<string> &vstrImageFilenamesD, 
    vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 5)
    {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association" << endl;
        return 1;
    }

    vector<string> vstrImageFilenamesRGB;   // 放需要读取的彩色图像
    vector<string> vstrImageFilenamesD;     // 深度图像的路径
    vector<double> vTimestamps;             // 时间戳的变量

    // 从命令行输入参数中得到关联文件的路径
    string strAssociationFilename = string(argv[4]);

    // Step 1 从关联文件中加载这些信息
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    //彩色图像和深度图像数据的一致性检查
    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    // Step 2 初始化 ORB-SLAM 工程类
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);


    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imRGB, imD;

    // Step 3 对图像序列中的每张图像展开遍历
    for(int ni=0; ni<nImages; ni++)
    {
        // Step 3.1 读取: 图像-imRGB  深度信息-imD  时间戳信息-tframe
        imRGB   = cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[ni],CV_LOAD_IMAGE_UNCHANGED);
        imD     = cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        // 确定图像合法性
        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

        // 计算耗时(begin)
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    
        // Step 3.2 将图像和深度信息传入封装的 SLAM 工程类
        SLAM.TrackRGBD(imRGB,imD,tframe);

        // 计算耗时(end)
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        vTimesTrack[ni]=ttrack; // 第 ni帧的 SLAM处理耗时

        // Step 3.3 根据时间戳,准备加载下一张图片
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Step 4 终止SLAM过程
    SLAM.Shutdown();

    // // Tracking time statistics
    // //统计分析追踪耗时
    // sort(vTimesTrack.begin(),vTimesTrack.end());
    // float totaltime = 0;
    // for(int ni=0; ni<nImages; ni++)
    // {
    //     totaltime+=vTimesTrack[ni];
    // }
    // cout << "-------" << endl << endl;
    // cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    // cout << "mean tracking time: " << totaltime/nImages << endl;


    // Step 5 保存最终的相机轨迹
    SLAM.SaveTrajectoryTUM("./CameraTrajectory2.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");   

    return 0;
}

//从关联文件中提取这些需要加载的图像的路径和时间戳
void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    //输入文件流
    ifstream fAssociation;
    //打开关联文件
    fAssociation.open(strAssociationFilename.c_str());
    //一直读取,知道文件结束
    while(!fAssociation.eof())
    {
        string s;
        //读取一行的内容到字符串s中
        getline(fAssociation,s);
        //如果不是空行就可以分析数据了
        if(!s.empty())
        {
            //字符串流
            stringstream ss;
            ss << s;
            //字符串格式:  时间戳 rgb图像路径 时间戳 深度图像路径
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}
