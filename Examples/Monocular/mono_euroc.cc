#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<opencv2/core/core.hpp>
#include<System.h>
#include<unistd.h>


using namespace std;


/**
 * @brief 获取图像文件的信息
 * @param[in]  strImagePath     图像文件存放路径
 * @param[in]  strPathTimes     时间戳文件的存放路径
 * @param[out] vstrImages       outout：图像文件名 vector容器
 * @param[out] vTimeStamps      output：时间戳 vector容器
 */
void LoadImages(const string &strImagePath, const string &strPathTimes,vector<string> &vstrImages, vector<double> &vTimeStamps);


int main(int argc, char **argv)
{
    // step 0 检查输入参数个数是否足够
    // if(argc != 5)
    // {
    //     cerr << endl << "Usage: ./mono_tum path_to_vocabulary path_to_settings path_to_image_folder path_to_times_file" << endl;
    //     return 1;
    // }

    // step 1 加载图像
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    LoadImages(string(argv[3]),             
               string(argv[4]),           
               vstrImageFilenames,         
               vTimestamps);               

    // 图像数量 nImages
    int nImages = vstrImageFilenames.size();
    if(nImages<=0)
    {
        cerr << "ERROR: Failed to load images" << endl;
        return 1;
    }

    // step 2 初始化SLAM系统
    ORB_SLAM2::System SLAM(
        argv[1],                            // path_to_vocabulary
        argv[2],                            // path_to_settings
        ORB_SLAM2::System::MONOCULAR,       // 模式选择：单目
        true);                              // 可视化选择：开



    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;


    // step 3 依次追踪序列中的每一张图像
    cv::Mat im;
    for(int ni=0; ni<nImages; ni++)
    {
        // step 3.1 读取每张图像及其时间戳（读取过程中不改变图像的格式 ）
        im = cv::imread(vstrImageFilenames[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];


        if(im.empty())
        {
            cerr << endl << "Failed to load image at: "
                 <<  vstrImageFilenames[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif


        // step 3.2 把图像及其时间戳传给SLAM系统，开始跟踪
        SLAM.TrackMonocular(im,tframe);


#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif
        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    } // end of for() 所有图像进行 SLAM 后结束


    // step 4 终止SLAM系统
    SLAM.Shutdown();


    // step 5 计算平均耗时
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;


    // step 6 保存 TUM 格式的相机轨迹
    // 估计是单目时有尺度漂移, 而LGA GBA都只能优化关键帧使尺度漂移最小, 普通帧所产生的轨迹漂移这里无能为力, 我猜作者这样就只
    // 保存了关键帧的位姿,从而避免普通帧带有尺度漂移的位姿对最终误差计算的影响
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}




void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps)
{
    // 打开文件
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);
    // 遍历文件
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        // 只有在当前行不为空的时候执行
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            // 生成当前行所指出的RGB图像的文件名称
            vstrImages.push_back(strImagePath + "/" + ss.str() + ".png");
            double t;
            ss >> t;
            // 记录该图像的时间戳
            vTimeStamps.push_back(t/1e9);

        }
    }
}
