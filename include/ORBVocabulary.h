#ifndef ORBVOCABULARY_H
#define ORBVOCABULARY_H

#include "Thirdparty/DBoW2/DBoW2/FORB.h"
#include "Thirdparty/DBoW2/DBoW2/TemplatedVocabulary.h"

namespace ORB_SLAM2
{

// 只是增加了一个自定义的数据类型罢了
typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>   ORBVocabulary;

} 

#endif 
