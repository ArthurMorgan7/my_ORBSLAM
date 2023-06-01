#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "LoopClosing.h"
#include "Frame.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

namespace ORB_SLAM2
{

class LoopClosing;


class Optimizer
{
public:

    /**
     * @brief BA优化
     *         
     * @param[in]  vpKFs        参与BA的所有关键帧
     * @param[in]  vpMP         参与BA的所有地图点
     * @param[in]  nIterations  优化迭代次数（20次）
     * @param[in]  pbStopFlag   是否强制暂停
     * @param[in]  nLoopKF      关键帧的个数 -- 但是我觉得形成了闭环关系的当前关键帧的id
     * @param[in]  bRobust      是否使用核函数
     */
    void static BundleAdjustment(const std::vector<KeyFrame*> &vpKF, const std::vector<MapPoint*> &vpMP,
                                 int nIterations = 5, bool *pbStopFlag=NULL, const unsigned long nLoopKF=0,
                                 const bool bRobust = true);

    /**
     * @brief 对pMap全局地图的关键帧和地图点做全局BA优化，但主要功能还是调用 BundleAdjustment,这个函数相当于加了一个壳.
     * 
     * @param[in] pMap          地图对象的指针
     * @param[in] nIterations   迭代次数
     * @param[in] pbStopFlag    外界给的控制GBA停止的标志位
     * @param[in] nLoopKF       当前回环关键帧的id，其实也就是参与GBA的关键帧个数
     * @param[in] bRobust       是否使用鲁棒核函数
     */
    void static GlobalBundleAdjustemnt(Map* pMap, int nIterations=5, bool *pbStopFlag=NULL,
                                       const unsigned long nLoopKF=0, const bool bRobust = true);

    
    /**
     * @brief 局部 BA
     * 
     * @param pKF        KeyFrame
     * @param pbStopFlag 是否停止优化的标志
     * @param pMap       在优化后，更新状态时需要用到Map的互斥量mMutexMapUpdate
     * @note 由局部建图线程调用,对局部地图进行优化的函数
     */
    void static LocalBundleAdjustment(KeyFrame* pKF, bool *pbStopFlag, Map *pMap);

    /**
     * @brief 仅位姿优化
     *
     * @param   pFrame Frame
     * @return  inliers数量
     */
    int static PoseOptimization(Frame* pFrame);


    /**
     * @brief 闭环检测后，EssentialGraph优化 
     *
     * @param pMap               全局地图
     * @param pLoopKF            闭环匹配上的关键帧
     * @param pCurKF             当前关键帧
     * @param NonCorrectedSim3   未经过Sim3传播调整过的关键帧位姿
     * @param CorrectedSim3      经过Sim3传播调整过的关键帧位姿
     * @param LoopConnections    因闭环时MapPoints调整而新生成的边
     */
    void static OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections,
                                       const bool &bFixScale);

    
    /**
     * @brief 形成闭环时,对当前关键帧和闭环关键帧的Sim3位姿进行优化
     *         
     * @param pKF1        KeyFrame
     * @param pKF2        KeyFrame
     * @param vpMatches1  两个关键帧的匹配关系
     * @param g2oS12      两个关键帧间的Sim3变换，方向是从2到1
     * @param th2         卡方检验是否为误差边用到的阈值
     * @param bFixScale   是否优化尺度，弹目进行尺度优化，双目不进行尺度优化
     * @return int                  优化之后匹配点中内点的个数
     */
    static int OptimizeSim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches1,
                            g2o::Sim3 &g2oS12, const float th2, const bool bFixScale);
};

} //namespace ORB_SLAM

#endif // OPTIMIZER_H
