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

#ifndef SLIDINGHIMIGRADIENT_H
#define SLIDINGHIMIGRADIENT_H

namespace cv {
	class Mat;
}
#include <functional>

class single_stereo_task;

void scaleUpCostMap(cv::Mat& cost_map_src, cv::Mat& cost_map_dst, float oldscale);

cv::Mat onestepSlidingInfoGradient(single_stereo_task task, std::function<cv::Mat(single_stereo_task)> func, int windowsize);
std::function<cv::Mat(single_stereo_task)> gradient_enhancer_bind(std::function<cv::Mat(single_stereo_task)> func, int windowsize);

cv::Mat genericScaledProcessing(single_stereo_task task, int border, std::function<cv::Mat(single_stereo_task)> func);

#endif // SLIDINGHIMIGRADIENT_H
