/*
Copyright (c) 2013, Kai Klindworth
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef INITIAL_DISPARITY_H
#define INITIAL_DISPARITY_H

#include <vector>
#include <functional>
#include <memory>

namespace cv {
	class Mat;
}

class StereoTask;
class StereoSingleTask;
class DisparityRegion;
class InitialDisparityConfig;
class RefinementConfig;
class RegionContainer;
class segmentation_algorithm;
class RegionInterval;
class RegionDescriptor;

void fillRegionContainer(RegionContainer& result, StereoSingleTask& task, std::shared_ptr<segmentation_algorithm>& algorithm);
void getRegionEntropy(cv::Mat& base, DisparityRegion &cregion);
void getAllRegionEntropies(cv::Mat& base, std::vector<DisparityRegion>& regions);
void generateRegionInformation(RegionContainer& left, RegionContainer& right);

//void getRegionDisparity(SegRegion& pixel_idx, const cv::Mat &base, const cv::Mat &match, int dispMin, int dispMax, unsigned int dilate_grow);
std::pair<cv::Mat, cv::Mat> segment_based_disparity_it(StereoTask& task, const InitialDisparityConfig& config, const RefinementConfig& refconfig, std::shared_ptr<segmentation_algorithm>& algorithm, int subsampling);
std::pair<cv::Mat, cv::Mat> segment_based_disparity_lss(StereoTask& task, const InitialDisparityConfig &config, std::shared_ptr<segmentation_algorithm>& algorithm);

#endif // INITIAL_DISPARITY_H
