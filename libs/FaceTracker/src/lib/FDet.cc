///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2010, Jason Mora Saragih, all rights reserved.
//
// This file is part of FaceTracker.
//
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
//
//     * The software is provided under the terms of this licence stricly for
//       academic, non-commercial, not-for-profit purposes.
//     * Redistributions of source code must retain the above copyright notice, 
//       this list of conditions (licence) and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright 
//       notice, this list of conditions (licence) and the following disclaimer 
//       in the documentation and/or other materials provided with the 
//       distribution.
//     * The name of the author may not be used to endorse or promote products 
//       derived from this software without specific prior written permission.
//     * As this software depends on other libraries, the user must adhere to 
//       and keep in place any licencing terms of those libraries.
//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite the following work:
//
//       J. M. Saragih, S. Lucey, and J. F. Cohn. Face Alignment through 
//       Subspace Constrained Mean-Shifts. International Conference of Computer 
//       Vision (ICCV), September, 2009.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR IMPLIED 
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO 
// EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF 
// THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///////////////////////////////////////////////////////////////////////////////
#include <FaceTracker/FDet.h>
// cv
#include "opencv2/opencv.hpp"

using namespace FACETRACKER;
using namespace std;
//===========================================================================
FDet& FDet::operator= (FDet const& rhs)
{
  this->_min_neighbours = rhs._min_neighbours;
  this->_min_size = rhs._min_size;
  this->_img_scale = rhs._img_scale;
  this->_scale_factor = rhs._scale_factor;
  this->_window = rhs._window;
  this->_stages = rhs._stages;
  this->_classifier = rhs._classifier;
  this->_haar_count = rhs._haar_count;
  this->_haar_rect = rhs._haar_rect;
  this->small_img_ = rhs.small_img_.clone(); return *this;
}
//===========================================================================
void FDet::Init(const char* fname,
		const double img_scale,
		const double scale_factor,
		const int    min_neighbours,
		const int    min_size)
{
  if(!_classifier.load(fname)){
    printf("ERROR(%s,%d) : Failed loading classifier cascade!\n",
	   __FILE__,__LINE__); abort();
  }
  _stages.clear();
  _img_scale      = img_scale;
  _scale_factor   = scale_factor;
  _min_neighbours = min_neighbours;
  _min_size       = min_size; return;
}
//===========================================================================
cv::Rect FDet::Detect(cv::Mat im)
{
  assert(im.type() == CV_8U);
  cv::Mat gray; cv::Rect R;
  int w = cvRound(im.cols/_img_scale);
  int h = cvRound(im.rows/_img_scale);
  if((small_img_.rows!=h) || (small_img_.cols!=w))small_img_.create(h,w,CV_8U);
  if(im.channels() == 1)gray = im;
  else{gray=cv::Mat(im.rows,im.cols,CV_8U);cv::cvtColor(im,gray,cv::COLOR_BGR2GRAY);}
  cv::resize(gray,small_img_,cv::Size(w,h),0,0,cv::INTER_LINEAR);
  cv::equalizeHist(small_img_,small_img_);
  std::vector<cv::Rect> faces;
  int minSize = cvRound(_min_size/_img_scale);
  _classifier.detectMultiScale(small_img_,faces,_scale_factor,_min_neighbours,0,
			       cv::Size(minSize,minSize));
  _haar_count = (int)faces.size();
  if(faces.empty())return cv::Rect(0,0,0,0);
  // Use the largest face.
  int maxv = -1;
  for(size_t i = 0; i < faces.size(); i++){
    const cv::Rect & r = faces[i];
    if(maxv < r.width*r.height){
      maxv = r.width*r.height; R.x = r.x*_img_scale; R.y = r.y*_img_scale;
      R.width  = r.width*_img_scale; R.height = r.height*_img_scale;
    }
  }
  _haar_rect = R; return R;
}
//===========================================================================
void FDet::Load(const char* fname)
{
  ifstream s(fname); assert(s.is_open()); this->Read(s); s.close(); return;
}
//===========================================================================
void FDet::Save(const char* fname)
{
  ofstream s(fname); assert(s.is_open()); this->Write(s);s.close(); return;
}
//===========================================================================
void FDet::Write(ofstream &s)
{
  if(_stages.empty()){
    printf("ERROR(%s,%d) : Only cascades read from a FaceTracker model can be written\n",
	   __FILE__,__LINE__); return;
  }
  s << IO::FDET          << " "
    << _min_neighbours   << " " 
    << _min_size         << " "
    << _img_scale        << " "
    << _scale_factor     << " "
    << _stages.size()    << " "
    << _window.width     << " " 
    << _window.height    << " ";
  for(size_t i = 0; i < _stages.size(); i++){
    const HaarStage & st = _stages[i];
    s << st.parent << " " << st.next << " " << st.child << " "
      << st.threshold << " " << st.trees.size() << " "; 
    for(size_t j = 0; j < st.trees.size(); j++){
      const HaarTree & t = st.trees[j];
      s << t.nodes.size() << " ";
      for(size_t k = 0; k < t.nodes.size(); k++){
	const HaarNode & n = t.nodes[k];
	s << n.threshold << " " << n.left << " " << n.right << " "
	  << n.alpha << " " << n.tilted << " ";
	for(int l = 0; l < 3; l++){
	  s << n.rect[l].weight << " " << n.rect[l].x << " " << n.rect[l].y << " "
	    << n.rect[l].width << " " << n.rect[l].height << " ";
	}
      }
      s << t.lastAlpha << " ";
    }
  }return;
}
//===========================================================================
void FDet::Read(ifstream &s,bool readType)
{ 
  int n;
  if(readType){int type; s >> type; assert(type == IO::FDET);}
  s >> _min_neighbours >> _min_size >> _img_scale >> _scale_factor >> n;
  s >> _window.width >> _window.height;
  _stages.assign(n, HaarStage());
  for(int i = 0; i < n; i++){
    HaarStage & st = _stages[i];
    int ntrees;
    s >> st.parent >> st.next >> st.child >> st.threshold >> ntrees;
    st.trees.assign(ntrees, HaarTree());
    for(int j = 0; j < ntrees; j++){
      HaarTree & t = st.trees[j];
      int nnodes; s >> nnodes;
      t.nodes.assign(nnodes, HaarNode());
      for(int k = 0; k < nnodes; k++){
	HaarNode & nd = t.nodes[k];
	s >> nd.threshold >> nd.left >> nd.right >> nd.alpha >> nd.tilted;
	for(int l = 0; l < 3; l++){
	  s >> nd.rect[l].weight >> nd.rect[l].x >> nd.rect[l].y
	    >> nd.rect[l].width >> nd.rect[l].height;
	}
      }
      s >> t.lastAlpha;
    }
  }
  BuildClassifier(); return;
}
//===========================================================================
// Writes the cascade in OpenCV's current (traincascade) format and loads it.
// Both formats share the tree encoding: a child index > 0 is another node,
// and <= 0 means leaf number -index.
void FDet::BuildClassifier()
{
  cv::FileStorage fs(".xml", cv::FileStorage::WRITE | cv::FileStorage::MEMORY);
  int maxWeak = 0;
  for(size_t i = 0; i < _stages.size(); i++)
    maxWeak = std::max(maxWeak, (int)_stages[i].trees.size());
  fs << "cascade" << "{";
  fs << "stageType" << "BOOST";
  fs << "featureType" << "HAAR";
  fs << "height" << _window.height;
  fs << "width" << _window.width;
  fs << "stageParams" << "{" << "maxWeakCount" << maxWeak << "}";
  fs << "featureParams" << "{" << "maxCatCount" << 0 << "}";
  fs << "stageNum" << (int)_stages.size();
  fs << "stages" << "[";
  std::vector<const HaarNode*> features;
  for(size_t i = 0; i < _stages.size(); i++){
    const HaarStage & st = _stages[i];
    fs << "{";
    fs << "maxWeakCount" << (int)st.trees.size();
    fs << "stageThreshold" << st.threshold;
    fs << "weakClassifiers" << "[";
    for(size_t j = 0; j < st.trees.size(); j++){
      const HaarTree & t = st.trees[j];
      std::vector<double> internalNodes;
      std::vector<double> leafValues;
      for(size_t k = 0; k < t.nodes.size(); k++){
	const HaarNode & nd = t.nodes[k];
	internalNodes.push_back(nd.left);
	internalNodes.push_back(nd.right);
	internalNodes.push_back((double)features.size());
	internalNodes.push_back(nd.threshold);
	leafValues.push_back(nd.alpha);
	features.push_back(&nd);
      }
      leafValues.push_back(t.lastAlpha);
      fs << "{" << "internalNodes" << internalNodes
	 << "leafValues" << leafValues << "}";
    }
    fs << "]" << "}";
  }
  fs << "]";
  fs << "features" << "[";
  for(size_t f = 0; f < features.size(); f++){
    fs << "{" << "rects" << "[";
    for(int l = 0; l < 3; l++){
      const HaarRect & r = features[f]->rect[l];
      if(r.weight == 0.f) continue;
      fs << "[:" << r.x << r.y << r.width << r.height << r.weight << "]";
    }
    fs << "]" << "tilted" << features[f]->tilted << "}";
  }
  fs << "]";
  fs << "}";
  std::string xml = fs.releaseAndGetString();
  cv::FileStorage in(xml, cv::FileStorage::READ | cv::FileStorage::MEMORY);
  if(!_classifier.read(in.getFirstTopLevelNode())){
    printf("ERROR(%s,%d) : Failed converting the face detector cascade!\n",
	   __FILE__,__LINE__); abort();
  }
}
//===========================================================================
