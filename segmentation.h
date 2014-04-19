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

#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <opencv2/core/core.hpp>
#include <memory>

class RegionContainer;
class RegionDescriptor;
class fusion_work_data;
class StereoSingleTask;

class segmentation_settings
{
public:
	std::string algorithm;
	int spatial_var;
	float color_var;
	int superpixel_size;
	float superpixel_compactness;

	bool enable_regionsplit;
	bool enable_fusion;
	bool enable_color_fusion;
};

class segmentation_algorithm {
public:
	/**
	 * @brief operator () Segments an image
	 * @param image Image, which should be segmented
	 * @param labels cv::Mat image with int as datatype. Counting from zero, out parameter
	 * @return Number of segments
	 */
	virtual int operator()(const cv::Mat& image, cv::Mat_<int>& labels) = 0;

	/**
	 * @brief cacheAllowed Returns, if caching for this segmentation algorithm is allowed
	 * @return
	 */
	virtual bool cacheAllowed() const { return true; }

	/**
	 * @brief cacheName Name of algorithm for filenames
	 * @return
	 */
	virtual std::string cacheName() const = 0;

	/**
	 * @brief refinementPossible Returns if it is possible to obtain a finer segmentation
	 * @return
	 */
	virtual bool refinementPossible() { return false; }

	/**
	 * @brief refine Returns a segmentation with smaller segments. The smaller segments lay in a bigger segments and not in more than one!
	 */
	virtual void refine(RegionContainer&) {}
};

class slic_segmentation : public segmentation_algorithm {
public:
	slic_segmentation(const segmentation_settings& psettings) : settings(psettings) {}
	virtual int operator()(const cv::Mat& image, cv::Mat_<int>& labels);
	virtual std::string cacheName() const;

private:
	segmentation_settings settings;
};

class crslic_segmentation : public segmentation_algorithm {
public:
	crslic_segmentation(const segmentation_settings& psettings) : settings(psettings) {}
	virtual int operator()(const cv::Mat& image, cv::Mat_<int>& labels);
	virtual std::string cacheName() const;

private:
	segmentation_settings settings;
};

class meanshift_segmentation : public segmentation_algorithm {
public:
	meanshift_segmentation(const segmentation_settings& psettings) : settings(psettings) {}
	virtual int operator()(const cv::Mat& image, cv::Mat_<int>& labels);
	virtual std::string cacheName() const;

private:
	segmentation_settings settings;
};

class mssuperpixel_segmentation : public segmentation_algorithm {
public:
	mssuperpixel_segmentation(const segmentation_settings& psettings) : settings(psettings) {}
	virtual int operator()(const cv::Mat& image, cv::Mat_<int>& labels);
	virtual std::string cacheName() const;
	virtual bool cacheAllowed() const;
	virtual bool refinementPossible();
	virtual void refine(RegionContainer &container);

private:
	std::shared_ptr<fusion_work_data> fusion_data;
	segmentation_settings settings;
	cv::Mat_<int> superpixel;
	int regions_count_superpixel;
};

std::shared_ptr<segmentation_algorithm> getSegmentationClass(const segmentation_settings& settings);

class fusion_work_data
{
public:
	fusion_work_data(std::size_t size) :
		visited(std::vector<unsigned char>(size, 0)),
		active(std::vector<unsigned char>(size, 1)),
		fused(std::vector<std::vector<std::size_t>>(size)),
		fused_with(std::vector<std::size_t>(size, 0))
	{
	}

	void visit_reset()
	{
		std::fill(visited.begin(), visited.end(), 0);
	}

	std::vector<unsigned char> visited;
	std::vector<unsigned char> active;
	std::vector<std::vector<std::size_t> > fused;
	std::vector<std::size_t> fused_with;
};

//void runFusion(cv::Mat& labels, std::vector<SegRegion>& regions, std::function<bool(const SegRegion& master_seg, const SegRegion& slave_seg, const SegRegion& fusion_seg)> check_func);
cv::Mat getWrongColorSegmentationImage(cv::Mat_<int>& labels, int labelcount);
cv::Mat getWrongColorSegmentationImage(RegionContainer& container);
//void fuse(fusion_work_data& data, std::vector<SegRegion>& regions, cv::Mat& labels);
//int mean_shift_segmentation(const cv::Mat& src, cv::Mat& labels_dst, int spatial_variance, float color_variance, int minsize);
//int ms_slic(const cv::Mat& image, cv::Mat& labels, const segmentation_settings &config);

const cv::FileNode& operator>>(const cv::FileNode& stream, segmentation_settings& config);
cv::FileStorage& operator<<(cv::FileStorage& stream, const segmentation_settings& config);

int cachedSegmentation(StereoSingleTask& task, cv::Mat_<int>& labels, std::shared_ptr<segmentation_algorithm>& algorithm);

//int split_region(const RegionDescriptor& descriptor, int min_size, std::back_insert_iterator<std::vector<RegionDescriptor>> it);
void split_region_test();

#endif // SEGMENTATION_H
