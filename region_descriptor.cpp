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

#include "region_descriptor.h"

#include "region.h"

cv::Mat RegionDescriptor::getMask(int margin)
{
	assert(bounding_box.width > 0 && bounding_box.height > 0);

	cv::Mat mask = cv::Mat::zeros(bounding_box.height + 2*margin, bounding_box.width + 2*margin, CV_8UC1);
	for(const RegionInterval& cinterval : lineIntervals)
	{
		assert(cinterval.y + margin - bounding_box.y >= 0 && cinterval.y + margin - bounding_box.y < bounding_box.height + 2*margin);
		assert(cinterval.lower + margin - bounding_box.x >= 0 && cinterval.lower + margin - bounding_box.x + cinterval.length() < bounding_box.width + 2*margin);

		memset(mask.ptr<unsigned char>(cinterval.y + margin - bounding_box.y, cinterval.lower + margin - bounding_box.x), 255, cinterval.length());
	}
	return mask;
}

cv::Mat RegionDescriptor::getAsMat(const cv::Mat& src, int d)
{
	return getRegionAsMat(src, lineIntervals, d);
}

int RegionDescriptor::size() const
{
	return getSizeOfRegion(lineIntervals);
}
