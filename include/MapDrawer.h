#ifndef MAPDRAWER_H
#define MAPDRAWER_H

#include"Map.h"
#include"MapPoint.h"
#include"KeyFrame.h"
#include<pangolin/pangolin.h>

#include<mutex>

namespace ORB_SLAM2
{

class MapDrawer
{
public:
    /**
     * @brief 构造函数
     * 
     * @param[in] pMap              地图句柄
     * @param[in] strSettingPath    配置文件的路径
     */
    MapDrawer(Map* pMap, const string &strSettingPath);
    
    //地图句柄
    Map* mpMap;

    /** @brief 绘制地图点 */
    void DrawMapPoints();
    /**
     * @brief 绘制关键帧
     * 
     * @param[in] bDrawKF       是否绘制关键帧
     * @param[in] bDrawGraph    是否绘制共视图
     */
    void DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph);
    /**
     * @brief 绘制当前相机
     * 
     * @param[in] Twc 相机的位姿矩阵
     */
    void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);
    /**
     * @brief 设置当前帧的相机位姿
     * 
     * @param[in] Tcw 位姿矩阵
     */
    void SetCurrentCameraPose(const cv::Mat &Tcw);
    /**
     * @brief 设置参考关键帧
     * 
     * @param[in] pKF 参考关键帧的句柄
     */
    void SetReferenceKeyFrame(KeyFrame *pKF);
    /**
     * @brief 将相机位姿mCameraPose由Mat类型转化为OpenGlMatrix类型
     * 
     * @param[out] M 
     */
    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M);

private:

    //绘制这些部件的参数
    ///关键帧-大小
    float mKeyFrameSize;
    ///关键帧-线宽
    float mKeyFrameLineWidth;
    ///共视图的线宽
    float mGraphLineWidth;
    ///地图点的大小
    float mPointSize;
    ///绘制的相机的大小
    float mCameraSize;
    ///绘制相机的线宽
    float mCameraLineWidth;

    ///相机位置
    cv::Mat mCameraPose;

    ///线程互斥量
    std::mutex mMutexCamera;
};

} //namespace ORB_SLAM

#endif // MAPDRAWER_H
