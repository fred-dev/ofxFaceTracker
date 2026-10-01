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
#ifndef __FDet_h_
#define __FDet_h_
#include <FaceTracker/IO.h>
namespace FACETRACKER
{
  //===========================================================================
  /**
      A wrapper for OpenCV's face detector.

      The model file stores a Haar cascade in the layout of OpenCV's old C
      structures (CvHaarClassifierCascade), which OpenCV 4 removed. The cascade
      is read into the plain structures below and converted in memory to the
      current cascade format for cv::CascadeClassifier.
  */
  class FDet{
  public:
    struct HaarRect { float weight; int x, y, width, height; };
    struct HaarNode {
      float threshold; int left, right; float alpha; int tilted;
      HaarRect rect[3];
    };
    struct HaarTree {
      std::vector<HaarNode> nodes;
      float lastAlpha;              // leaf value after the last node's alpha
    };
    struct HaarStage {
      int parent, next, child; float threshold;
      std::vector<HaarTree> trees;
    };

    int                      _haar_count;
    cv::Rect                 _haar_rect;
    int                      _min_neighbours; /**< see OpenCV documentation */
    int                      _min_size;       /**< ...                      */
    double                   _img_scale;      /**< ...                      */
    double                   _scale_factor;   /**< ...                      */

    FDet(){_haar_count=0;}
    FDet(const char* fname){this->Load(fname);}
    FDet(const char*  cascFile,
	 const double img_scale = 1.3,
	 const double scale_factor = 1.1,
	 const int    min_neighbours = 2,
	 const int    min_size = 30){
      this->Init(cascFile,img_scale,scale_factor,min_neighbours,min_size);
    }
    ~FDet() {}
    FDet& operator=(FDet const&rhs);
    /** Loads a cascade from an OpenCV cascade XML file. */
    void Init(const char* fname,
	      const double img_scale = 1.3,
	      const double scale_factor = 1.1,
	      const int    min_neighbours = 2,
	      const int    min_size = 30);
    cv::Rect Detect(cv::Mat im);
    void Load(const char* fname);
    void Save(const char* fname);
    void Write(std::ofstream &s);
    void Read(std::ifstream &s,bool readType = true);

  private:
    void BuildClassifier();
    cv::Size                 _window;
    std::vector<HaarStage>   _stages;
    cv::CascadeClassifier    _classifier;
    cv::Mat small_img_;
  };
  //===========================================================================
}
#endif
