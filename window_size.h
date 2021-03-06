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

#ifndef WINDOW_SIZE_H
#define WINDOW_SIZE_H

#include <opencv2/core/core.hpp>

class disparity_region;

/**
 * @brief adaptive_window_size
 * @param image The original image
 * @param labels Segmentation labels
 * @param threshold
 * @param minsize Sets the minimum possible window size
 * @param maxsize Sets the maximum possible window size
 * @param regions Container with region_descriptors
 * @return Matrix with the suggested window sizes. cv::Vec2b, [0]=y, [1]=x
 */
cv::Mat_<cv::Vec2b> adaptive_window_size(const cv::Mat& image, const cv::Mat_<int>& labels, float threshold, int minsize, int maxsize, const std::vector<disparity_region>& regions);
void showWindowSizes(cv::Mat& sizes);

#endif // WINDOW_SIZE_H
