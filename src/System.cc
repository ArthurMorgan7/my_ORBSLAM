#include <thread>					
#include <pangolin/pangolin.h>		
#include <iomanip>					
#include <unistd.h>
#include "System.h"
#include "Converter.h"		


namespace ORB_SLAM2
{

// --------------------- 构造函数 ------------------------------

System::System(const string &strVocFile,					//词典文件路径
			   const string &strSettingsFile,				//配置文件路径
			   const eSensor sensor,						//传感器类型
               const bool bUseViewer):						//是否使用可视化界面

					 mSensor(sensor), 							//初始化传感器类型
					 mpViewer(static_cast<Viewer*>(NULL)),		//空。。。对象指针？  TODO 
					 mbReset(false),							//无复位标志
					 mbActivateLocalizationMode(false),			//没有这个模式转换标志
        			 mbDeactivateLocalizationMode(false)		//没有这个模式转换标志
{
    // 输出当前传感器类型
    cout << "Input sensor was set to: ";
    if(mSensor==MONOCULAR)
        cout << "Monocular" << endl;
    else if(mSensor==STEREO)
        cout << "Stereo" << endl;
    else if(mSensor==RGBD)
        cout << "RGB-D" << endl;

    // 检验配置文件
    cv::FileStorage fsSettings(strSettingsFile.c_str(), 	//将配置文件名转换成为字符串
    						   cv::FileStorage::READ);		//只读
    if(!fsSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       exit(-1);
    }
    
    // Step 1 创建基本的对象
    cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;
    mpVocabulary = new ORBVocabulary();
    // 获取字典加载状态
    bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
    // 如果加载失败，就输出调试信息
    if(!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << strVocFile << endl;
        // 然后退出
        exit(-1);
    }
    // 否则则说明加载成功
    cout << "Vocabulary loaded!" << endl << endl;

    // 创建一个关键帧数据库
    mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

    // 创建全局地图
    mpMap = new Map();

    // 创建可视化相关
    mpFrameDrawer = new FrameDrawer(mpMap);
    mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

    // Step 2 建立追踪线程
    mpTracker = new Tracking(this,						// ？？？  
    						 mpVocabulary,				// 字典
    						 mpFrameDrawer, 			
    						 mpMapDrawer,				
                             mpMap, 					// 地图
                             mpKeyFrameDatabase, 		// 关键帧数据库
                             strSettingsFile, 			// 设置文件路径
                             mSensor);					// 传感器类型

    // Step 3 创建局部地图类
    mpLocalMapper = new LocalMapping(mpMap, 				//指定使iomanip
    								 mSensor==MONOCULAR);	// TODO 为什么这个要设置成为MONOCULAR？？？
    mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run,	//这个线程会调用的函数
    							 mpLocalMapper);				//这个调用函数的参数

    // Stpe 4 创建回环检测类和线程
    mpLoopCloser = new LoopClosing(mpMap, 						//地图
    							   mpKeyFrameDatabase, 			//关键帧数据库
    							   mpVocabulary, 				//ORB字典
    							   mSensor!=MONOCULAR);			//当前的传感器是否是单目
    mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run,	//线程的主函数
    							mpLoopCloser);					//该函数的参数

    // Step 5 创建可视化类和线程
    if(bUseViewer)
    {
    	//如果指定了，程序的运行过程中需要运行可视化部分
    	//新建viewer
        mpViewer = new Viewer(this, 			//又是这个
        					  mpFrameDrawer,	//帧绘制器
        					  mpMapDrawer,		//地图绘制器
        					  mpTracker,		//追踪器
        					  strSettingsFile);	//配置文件的访问路径
        //新建viewer线程
        mptViewer = new thread(&Viewer::Run, mpViewer);
        //给运动追踪器设置其查看器
        mpTracker->SetViewer(mpViewer);
    }

    // Step 6 设置进程间的指针
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);
}

// --------------------- 单目,双目,RGBD的跟踪主程序 ------------------------------

cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp)
{
    if(mSensor!=MONOCULAR)
    {
        cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular." << endl;
        exit(-1);
    }

    // 检查模式变化
    {
        // 独占锁，主要是为了mbActivateLocalizationMode和mbDeactivateLocalizationMode不会发生混乱
        unique_lock<mutex> lock(mMutexMode);
        // mbActivateLocalizationMode为true会关闭局部地图线程
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            // 局部地图关闭以后，只进行追踪的线程，只计算相机的位姿，没有对局部地图进行更新
            // 设置mbOnlyTracking为真
            mpTracker->InformOnlyTracking(true);
            // 关闭线程可以使得别的线程得到更多的资源
            mbActivateLocalizationMode = false;
        }
        // 如果mbDeactivateLocalizationMode是true，局部地图线程就被释放, 关键帧从局部地图中删除.
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // 检查重置
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    //获取相机位姿的估计结果
    cv::Mat Tcw = mpTracker->GrabImageMonocular(im,timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

    return Tcw;
}

cv::Mat System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp)
{
	//检查输入数据类型是否合法
    if(mSensor!=STEREO)
    {
    	//不合法那就退出
        cerr << "ERROR: you called TrackStereo but input sensor was not set to STEREO." << endl;
        exit(-1);
    }   

    //检查是否有运行模式的改变
    // Check mode change
    {
    	// TODO 锁住这个变量？防止其他的线程对它的更改？
        unique_lock<mutex> lock(mMutexMode);
        //如果激活定位模式
        if(mbActivateLocalizationMode)
        {
        	//调用局部建图器的请求停止函数
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }
            //运行到这里的时候，局部建图部分就真正地停止了
            //告知追踪器，现在 只有追踪工作
            mpTracker->InformOnlyTracking(true);// 定位时，只跟踪
            //同时清除定位标记
            mbActivateLocalizationMode = false;// 防止重复执行
        }//如果激活定位模式

        if(mbDeactivateLocalizationMode)
        {
        	//如果取消定位模式
        	//告知追踪器，现在地图构建部分也要开始工作了
            mpTracker->InformOnlyTracking(false);
            //局部建图器要开始工作呢
            mpLocalMapper->Release();
            //清楚标志
            mbDeactivateLocalizationMode = false;// 防止重复执行
        }//如果取消定位模式
        
    }//检查是否有模式的改变

    // Check reset，检查是否有复位的操作
    {
    	//上锁
	    unique_lock<mutex> lock(mMutexReset);
	    //是否有复位请求？
	    if(mbReset)
	    {
	    	//有，追踪器复位
	        mpTracker->Reset();
	        //清除标志
	        mbReset = false;
	    }//是否有复位请求
    }//检查是否有复位的操作

    //用矩阵Tcw来保存估计的相机 位姿，运动追踪器的GrabImageStereo函数才是真正进行运动估计的函数
    cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft,imRight,timestamp);

    //给运动追踪状态上锁
    unique_lock<mutex> lock2(mMutexState);
    //获取运动追踪状态
    mTrackingState = mpTracker->mState;
    //获取当前帧追踪到的地图点向量指针
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    //获取当前帧追踪到的关键帧特征点向量的指针
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    //返回获得的相机运动估计
    return Tcw;
}
 
cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp)
{
    // Step 1: 检查工作

	//判断输入数据类型是否合法
    if(mSensor!=RGBD)
    {
        cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
        exit(-1);
    }    

    //检查模式改变
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    //检查是否有复位请求
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    // Step 2: 主要SLAM工程
    cv::Mat Tcw = mpTracker->GrabImageRGBD(im,depthmap,timestamp);
    
    // Step 3: 获取SLAM结果的相关信息
    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;    // 跟踪状态
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;  // 地图点
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;    // 关键点
    return Tcw; // 返回什么不重要，重要的是System类里包含的地图点和位姿
}

// --------------------- 纯定位与SLAM切换 ------------------------------

void System::ActivateLocalizationMode()
{
	//上锁
    unique_lock<mutex> lock(mMutexMode);
    //设置标志
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

// --------------------- 保存结果数据 ------------------------------

void System::SaveTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    
    //只有在传感器为双目或者RGBD时才可以工作
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
        return;
    }

    //从地图中获取所有的关键帧
    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    //根据关键帧生成的先后顺序（id）进行排序
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    // 到原点的转换，获取这个转换矩阵
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    //文件写入的准备工作
    ofstream f;
    f.open(filename.c_str());
    //这个可以理解为，在输出浮点数的时候使用0.3141592654这样的方式而不是使用科学计数法
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.
    // 之前的帧位姿都是基于其参考关键帧的，现在我们把它恢复

    // 列表: 参考关键帧
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    
    // 列表: 每帧对应的时间戳
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    
    // 列表: 每帧的追踪状态组成       
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    
    //对于每一个mlRelativeFramePoses中的帧lit
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(), // lit：开始
        lend=mpTracker->mlRelativeFramePoses.end();      // lend：结尾
        
        lit!=lend;  // 如果没有遍历到结尾

        lit++,      // 相对帧位姿 递增
        lRit++,     // 参考关键帧 递增
        lT++,       // 时间戳    递增
        lbL++       // 追踪状态  递增
        )		
    {
    	// 如果该帧追踪失败，不管它，进行下一个
        if(*lbL)
            continue;

        // 追踪成功 👇

        // 创建变换矩阵，初始化为一个单位阵
        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

       	//获取其对应的参考关键帧
        KeyFrame* pKF = *lRit;

        // If the reference keyframe was culled（剔除）, traverse（扫描？） the spanning tree to get a suitable keyframe.
        while(pKF->isBad())
        {
        	//更新关键帧变换矩阵的初始值，
            Trw = Trw*pKF->mTcp;
            //并且更新到原关键帧的父关键帧
            pKF = pKF->GetParent();
        }//查看当前使用的参考关键帧是否为bad

        // TODO 其实我也是挺好奇，为什么在这里就能够更改掉不合适的参考关键帧了呢

        // TODO 这里的函数GetPose()和上面的mTcp有什么不同？
        //最后一个Two是原点校正

        //最终得到的是参考关键帧相对于世界坐标系的变换（原始数据）
        Trw = Trw*pKF->GetPose()*Two;

        // 在此基础上得到相机当前帧相对于世界坐标系的变换（Tcw在此创建）（数据来源 lit 和 Trw）
        cv::Mat Tcw = (*lit)*Trw;

        //然后分解出旋转矩阵 (Rwc在此创建)（数据来源 Tcw）
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();

        //以及平移向量 （twc在此创建）（数据来源 Tcw）
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        //用四元数表示旋转（q在此创建）
        vector<float> q = Converter::toQuaternion(Rwc);

        // 然后按照给定的格式输出到文件中
        // setprecision(n): 保留小数点后 n位 
        f << setprecision(6) << *lT << " " <<  setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
        // *lT：时间戳信息
        // twc：平移信息
        // q：旋转信息

    } // end_of_for

    // 操作完毕，关闭文件并且输出调试信息
    f.close();

    cout << endl << "trajectory saved!" << endl;
}

void System::SaveKeyFrameTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    //获取关键帧vector并按照生成时间对其进行排序
    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    //本来这里需要进行原点校正，但是实际上没有做
    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    //文件写入的准备操作
    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    //对于每个关键帧
    for(size_t i=0; i<vpKFs.size(); i++)
    {
    	//获取该 关键帧
        KeyFrame* pKF = vpKFs[i];

        //原本有个原点校正，这里注释掉了
       // pKF->SetPose(pKF->GetPose()*Two);

        //如果这个关键帧是bad那么就跳过
        if(pKF->isBad())
            continue;

        //抽取旋转部分和平移部分，前者使用四元数表示
        cv::Mat R = pKF->GetRotation().t();
        vector<float> q = Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        //按照给定的格式输出到文件中
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

    }

    //关闭文件
    f.close();
    cout << endl << "trajectory saved!" << endl;
}

void System::SaveTrajectoryKITTI(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    //检查输入数据的类型
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
        return;
    }

    //下面的操作和前面TUM数据集格式的非常相似，因此不再添加注释
    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(), lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    {
        ORB_SLAM2::KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        while(pKF->isBad())
        {
          //  cout << "bad parent" << endl;
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        f << setprecision(9) << Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1)  << " " << Rwc.at<float>(0,2) << " "  << twc.at<float>(0) << " " <<
             Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1)  << " " << Rwc.at<float>(1,2) << " "  << twc.at<float>(1) << " " <<
             Rwc.at<float>(2,0) << " " << Rwc.at<float>(2,1)  << " " << Rwc.at<float>(2,2) << " "  << twc.at<float>(2) << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}

// --------------------- 获取信息 ------------------------------

int System::GetTrackingState()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

vector<MapPoint*> System::GetTrackedMapPoints()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedMapPoints;
}

vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}

// --------------------- 其他 ------------------------------

bool System::MapChanged()
{
    static int n=0;
    //其实整个函数功能实现的重点还是在这个GetLastBigChangeIdx函数上
    int curn = mpMap->GetLastBigChangeIdx();
    if(n<curn)
    {
        n=curn;
        return true;
    }
    else
        return false;
}

void System::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::Shutdown()
{
	//对局部建图线程和回环检测线程发送终止请求
    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();
    //如果使用了可视化窗口查看器
    if(mpViewer)
    {
    	//向查看器发送终止请求
        mpViewer->RequestFinish();
        //等到，知道真正地停止
        while(!mpViewer->isFinished())
            usleep(5000);
    }

    // Wait until all thread have effectively stopped
    while(!mpLocalMapper->isFinished() || 
    	  !mpLoopCloser->isFinished()  || 
    	   mpLoopCloser->isRunningGBA())			
    {
        usleep(5000);
    }

    if(mpViewer)
    	//如果使用了可视化的窗口查看器执行这个
    	// TODO 但是不明白这个是做什么的。如果我注释掉了呢？
        pangolin::BindToContext("ORB-SLAM2: Map Viewer");
}


} //namespace ORB_SLAM
