#ifndef SYSTEM_H
#define SYSTEM_H

#include <string>
#include <thread>
#include <opencv2/core/core.hpp>
#include "Tracking.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"
#include "Viewer.h"


namespace ORB_SLAM2
{


class Viewer;
class FrameDrawer;
class Map;
class Tracking;
class LocalMapping;
class LoopClosing;


class System
{
public:

    // 系统所使用的传感器类型
    enum eSensor{
        MONOCULAR=0,    // 0：单目
        STEREO=1,       // 1：双目
        RGBD=2          // 2：RGBD
    };

public:


    /**
     * @brief 初始化SLAM系统，包含四个线程的初始化
     * @param[in] strVocFile        ORB字典文件的路径
     * @param[in] strSettingsFile   配置文件的路径
     * @param[in] sensor            使用的传感器类型
     * @param[in] bUseViewer        是否使用可视化界面
     */
    System(const string &strVocFile,            
           const string &strSettingsFile, 
           const eSensor sensor,            
           const bool bUseViewer = true);    


    // 下面是针对三种不同类型的传感器所设计的三种运动追踪接口。彩色图像为CV_8UC3类型，并且都将会被转换成为灰度图像。
    // 追踪接口返回估计的相机位姿，如果追踪失败则返回NULL

    // Proccess the given stereo frame. Images must be synchronized and rectified（校正）.
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    // NOTE 注意这里英文注释的说法，双目图像有同步和校准的概念。
    cv::Mat TrackStereo(const cv::Mat &imLeft,          //左目图像
                        const cv::Mat &imRight,         //右目图像
                        const double &timestamp);       //时间戳

    // Process the given rgbd frame. Depthmap must be registered to the RGB frame.
    // Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Input depthmap: Float (CV_32F).
    // Returns the camera pose (empty if tracking fails).
    // NOTE 而在这里对RGBD图像的说法则是“配准”
    cv::Mat TrackRGBD(const cv::Mat &im,                //彩色图像
                      const cv::Mat &depthmap,          //深度图像
                      const double &timestamp);         //时间戳

    // Proccess the given monocular frame
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    /**
     * @brief 
     * @param[in] im            传入的图像
     * @param[in] timestamp     该图像的时间戳
     */
    cv::Mat TrackMonocular(const cv::Mat &im,           
                           const double &timestamp);    



    // 开启定位模式，此时仅有运动追踪部分在工作，局部建图功能则不工作
    void ActivateLocalizationMode();
    // 取消定位模式
    void DeactivateLocalizationMode();


    // 获取从上次调用本函数后是否发生了比较大的地图变化
    bool MapChanged();


    // 复位系统
    void Reset();
    //关闭系统，这将会关闭所有线程并且丢失曾经的各种数据
    void Shutdown();

    // 以TUM格式保存相机的运动轨迹
    void SaveTrajectoryTUM(const string &filename);        
    // 以TUM格式保存关键帧位姿
    void SaveKeyFrameTrajectoryTUM(const string &filename);    
    // 以KITTI格式保存相机的运行轨迹
    void SaveTrajectoryKITTI(const string &filename);

    // 在这里可以实现自己的地图保存和加载函数
    // SaveMap(const string &filename);
    // LoadMap(const string &filename);

    //获取最近的运动追踪状态、地图点追踪状态、特征点追踪状态（）
    int GetTrackingState();
    std::vector<MapPoint*> GetTrackedMapPoints();
    std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();

private:

    // 全局的变量 ⭐
    eSensor mSensor;    // 传感器类型
    Map* mpMap;         // 全局地图 = 关键帧 + 地图点，只有在插入关键帧的时候才增加
    ORBVocabulary* mpVocabulary;     // ORB字典
    KeyFrameDatabase* mpKeyFrameDatabase;   // 关键帧数据库，用于重定位和回环检测
    std::vector<MapPoint*> mTrackedMapPoints;       // 地图点容器
    std::vector<cv::KeyPoint> mTrackedKeyPointsUn;  // 关键点容器

    // 前端、后端、回环检测
    Tracking* mpTracker;         // 追踪器，除了进行运动追踪外还要负责创建关键帧、创建新地图点和进行重定位的工作。详细信息还得看相关文件
    LocalMapping* mpLocalMapper; // 局部建图器。局部BA由它进行。
    LoopClosing* mpLoopCloser;   // 回环检测器，它会执行位姿图优化并且开一个新的线程进行全局BA
    
    // 可视化
    Viewer* mpViewer;           // 可视化器
    FrameDrawer* mpFrameDrawer; // 帧绘制器
    MapDrawer* mpMapDrawer;     // 地图绘制器


    //复位标志
    std::mutex mMutexReset;
    bool mbReset;

    //模式改变标志
    std::mutex mMutexMode;
    bool mbActivateLocalizationMode;    // 开启定位模式
    bool mbDeactivateLocalizationMode;  // 关闭定位模式

    // 追踪状态标志，注意前三个的类型和上面的函数类型相互对应
    std::mutex mMutexState;
    int mTrackingState;
   
    // 局部建图线程、回环检测线程和查看器线程。
    std::thread* mptLocalMapping;
    std::thread* mptLoopClosing;
    std::thread* mptViewer;

};

}// namespace ORB_SLAM

#endif // SYSTEM_H
