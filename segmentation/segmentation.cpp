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

#include "segmentation.h"

#include "initial_disparity.h"
#include "region.h"
#include "intervals.h"
#include "intervals_algorithms.h"

#include "region_descriptor.h"
#include "region_descriptor_algorithms.h"

#include <stdexcept>
#include <fstream>

#include "segmentation_cr.h"
#ifdef USE_MEANSHIFT
#include "segmentation_ms.h"
#endif

#ifdef USE_SLIC
#include "segmentation_slic.h"
#endif

#ifdef USE_MSSLIC
#include "segmentation_msslic.h"
#endif

#include "segmentation_algorithms.h"

cv::Mat_<cv::Vec3b> getWrongColorSegmentationImage(cv::Mat_<int>& labels, int labelcount)
{
	std::vector<cv::Vec3b> colors;
	colors.reserve(labelcount);

	std::srand(0);

	for(int i = 0; i < labelcount; ++i)
	{
		cv::Vec3b ccolor;
		ccolor[0] = std::rand() % 256;
		ccolor[1] = std::rand() % 256;
		ccolor[2] = std::rand() % 256;
		colors.push_back(ccolor);
	}

	cv::Mat result(labels.size(), CV_8UC3);

	cv::Vec3b *dst_ptr = result.ptr<cv::Vec3b>(0);
	int *src_ptr = labels[0];

	for(std::size_t i = 0; i < labels.total(); ++i)
		*dst_ptr++ = colors[*src_ptr++];
	return result;
}

/*int ms_cr(const cv::Mat& image, cv::Mat_<int>& labels, const segmentation_settings& config)
{
	crslic_segmentation seg_obj(config);
	int regions_count = seg_obj(image, labels);
	//int regions_count = slicSuperpixels(image, labels, config.superpixel_size, config.superpixel_compactness);
	//std::vector<SegRegion> regions = getRegionVector(labels, regions_count);
	std::vector<DisparityRegion> regions(regions_count);//getRegionVector(result.labels, regions_count);
	fillRegionDescriptors(regions.begin(), regions.end(), labels);

	calculate_all_average_colors(image, regions);

	std::pair<int,int> slic_size = get_slic_seeds(image.cols, image.rows, config.superpixel_size);
	cv::Mat slic_image_lab(slic_size.first, slic_size.second, CV_64FC3);
	cv::Vec3d *dst_ptr = slic_image_lab.ptr<cv::Vec3d>(0);

	std::cout << "slic_size : width:" << slic_size.second << ", height: " << slic_size.first << std::endl;
	std::cout << "complete: " << slic_image_lab.total() << std::endl;
	std::cout << "regions: " << regions.size() << std::endl;

	int yex = labels.rows / (slic_size.first);
	int xex = labels.cols / (slic_size.second);

	for(int i = 0; i < slic_size.first; ++i)
	{
		for(int j = 0; j < slic_size.second; ++j)
		{
			int idx = labels.at<int>(i*yex+yex/2,j*xex+xex/2);
			*dst_ptr++ = regions[idx].average_color;
		}
	}

	checkLabelsIntervalsInvariant(regions, labels, regions.size());

	cv::Mat slic_image = lab_to_bgr(slic_image_lab);
	cv::Mat msResult;
	int mscount = segmentImage2(slic_image, msResult, config.spatial_var, config.color_var);
	//cv::imshow("ms", getWrongColorSegmentationImage(msResult, mscount));
	std::cout << "mscount: " << mscount << std::endl;

	fusion_work_data data(regions.size()); //TODO check
	int *src_ptr = msResult.ptr<int>(0);
	std::vector<int> mapping(mscount, -1);
	for(int i = 0; i < slic_size.first; ++i)
	{
		for(int j = 0; j < slic_size.second; ++j)
		{
			int idx = labels.at<int>(i*yex+yex/2,j*xex+xex/2);
			int label = *src_ptr++;
			if(mapping[label] == -1)
				mapping[label] = idx;
			else if(idx != mapping[label] && data.active[idx])
			{
				int fusion_idx = mapping[label];
				data.fused_with[idx] = fusion_idx;
				data.active[idx] = false;
				data.fused[fusion_idx].push_back(idx);
			}
		}
	}

	fuse(data, regions, labels);

	return regions.size();
}*/

std::shared_ptr<segmentation_algorithm> getSegmentationClass(const segmentation_settings& settings) {
#ifdef USE_MEANSHIFT
	if(settings.algorithm == "meanshift")
		return std::make_shared<meanshift_segmentation>(settings);
#endif
#ifdef USE_SLIC
	if(settings.algorithm == "superpixel")
		return std::make_shared<slic_segmentation>(settings);
#endif
#ifdef USE_MSSLIC
	if(settings.algorithm == "ms_superpixel")
		return std::make_shared<mssuperpixel_segmentation>(settings);
#endif
	if(settings.algorithm == "cr_superpixel")
		return std::make_shared<crslic_segmentation>(settings);

	throw std::invalid_argument("unknown segmentation algorithm");
}

const cv::FileNode& operator>>(const cv::FileNode& stream, segmentation_settings& config)
{
	stream["regionsplit"] >> config.enable_regionsplit;
	stream["fusion"] >> config.enable_fusion;
	stream["color_fusion"] >> config.enable_color_fusion;
	stream["spatial_var"] >> config.spatial_var;
	stream["color_var"] >> config.color_var;
	stream["superpixel_size"] >> config.superpixel_size;
	stream["superpixel_compactness"] >> config.superpixel_compactness;
	stream["segmentation"] >> config.algorithm;

	return stream;
}

cv::FileStorage& operator<<(cv::FileStorage& stream, const segmentation_settings& config)
{
	stream << "segmentation" << config.algorithm;
	stream << "spatial_var" << config.spatial_var << "color_var" << config.color_var;
	stream << "superpixel_size" << config.superpixel_size << "superpixel_compactness" << config.superpixel_compactness;
	stream << "regionsplit" << config.enable_regionsplit;
	stream << "fusion" << config.enable_fusion;
	stream << "color_fusion" << config.enable_color_fusion;
	return stream;
}

template<typename T, typename InsertIterator>
void insert_pair(T pair, InsertIterator it)
{
	*it = pair.first;
	++it;
	*it = pair.second;
	++it;
}

template<typename T, typename InsertIterator>
int insert_pair_regions(T pair, InsertIterator it)
{
	int counter = 0;
	if(pair.first.m_size > 0)
	{
		*it = pair.first;
		++it;
		++counter;
	}
	if(pair.second.m_size > 0)
	{
		*it = pair.second;
		++it;
		++counter;
	}
	return counter;
}

template<typename T, typename func_type>
std::pair<T,T> split_region_proxy(const T& descriptor, int split_threshold, func_type func)
{
	std::pair<T,T> result;
	result.first = descriptor;
	result.second = descriptor;

	func(descriptor, result.first, result.second, split_threshold);

	return result;
}

void hsplit_region(const RegionDescriptor& descriptor, RegionDescriptor& first, RegionDescriptor& second, int split_threshold)
{
	first.lineIntervals.clear();
	first.bounding_box.height = split_threshold - descriptor.bounding_box.y;

	second.lineIntervals.clear();
	second.bounding_box.y = split_threshold;
	second.bounding_box.height = descriptor.bounding_box.y + descriptor.bounding_box.height - split_threshold;

	for(const RegionInterval& cinterval : descriptor.lineIntervals)
	{
		if(cinterval.y < split_threshold)
			first.lineIntervals.push_back(cinterval);
		else
			second.lineIntervals.push_back(cinterval);
	}

	first.m_size = getSizeOfRegion(first.lineIntervals);
	second.m_size = getSizeOfRegion(second.lineIntervals);
}

void vsplit_region(const RegionDescriptor& descriptor, RegionDescriptor& first, RegionDescriptor& second, int split_threshold)
{
	first.lineIntervals.clear();
	first.bounding_box.width = split_threshold - descriptor.bounding_box.x;

	second.lineIntervals.clear();
	second.bounding_box.x = split_threshold;
	second.bounding_box.width = descriptor.bounding_box.x + descriptor.bounding_box.width - split_threshold;

	for(const RegionInterval& cinterval : descriptor.lineIntervals)
	{
		if(cinterval.upper > split_threshold && cinterval.lower <= split_threshold)
		{
			first.lineIntervals.push_back(RegionInterval(cinterval.y, cinterval.lower, split_threshold));
			second.lineIntervals.push_back(RegionInterval(cinterval.y, split_threshold, cinterval.upper));
		}
		else if(cinterval.upper <= split_threshold)
			first.lineIntervals.push_back(cinterval);
		else
			second.lineIntervals.push_back(cinterval);
	}

	first.m_size = getSizeOfRegion(first.lineIntervals);
	second.m_size = getSizeOfRegion(second.lineIntervals);
}

cv::Point region_avg_point(const RegionDescriptor& descriptor)
{
	long long x_avg = 0;
	long long y_avg = 0;

	for(const RegionInterval& cinterval : descriptor.lineIntervals)
	{
		x_avg += cinterval.upper - (cinterval.upper - cinterval.lower)/2 - 1;
		y_avg += cinterval.y;
	}
	x_avg /= descriptor.lineIntervals.size();
	y_avg /= descriptor.lineIntervals.size();

	return cv::Point(x_avg+1,y_avg+1);//because we want open intervalls +1
}

template<typename T>
int split_region(const T& descriptor, int min_size, std::back_insert_iterator<std::vector<T>> it)
{
	cv::Point avg_pt = region_avg_point(descriptor);

	bool split_h = false;
	if( ((avg_pt.y - descriptor.bounding_box.y) >= min_size)
			&& ((descriptor.bounding_box.y + descriptor.bounding_box.height - avg_pt.y) >= min_size))
		split_h = true;

	bool split_v = false;
	if( ((avg_pt.x - descriptor.bounding_box.x) >= min_size)
			&& ((descriptor.bounding_box.x + descriptor.bounding_box.width - avg_pt.x) >= min_size))
		split_v = true;

	int counter;
	if(split_h && split_v)
	{
		auto temp = split_region_proxy(descriptor, avg_pt.y, hsplit_region);
		counter = insert_pair_regions(split_region_proxy(temp.first, avg_pt.x, vsplit_region), it);
		counter += insert_pair_regions(split_region_proxy(temp.second, avg_pt.x, vsplit_region), it);
	}
	else if(split_h)
		counter = insert_pair_regions(split_region_proxy(descriptor, avg_pt.y, hsplit_region), it);
	else if(split_v)
		counter = insert_pair_regions(split_region_proxy(descriptor, avg_pt.x, vsplit_region), it);
	else
	{
		*it = descriptor;
		++it;
		counter = 1;
	}

	//return (split_h ? 2 : 1) * (split_v ? 2 : 1);
	return counter;
}

class defusion_data
{
public:
	defusion_data(int pidx_old, int psplit_factor) : idx_old(pidx_old), split_factor(psplit_factor) {}
	int idx_old;
	int split_factor;
};

cv::Mat_<int> segmentation_iteration(std::vector<DisparityRegion>& regions, cv::Size size)
{
	int min_size = 10;
	std::vector<DisparityRegion> created_regions;
	created_regions.reserve(regions.size());

	std::vector<defusion_data> data;
	for(std::size_t i = 0; i < regions.size(); ++i)
	{
		regions[i].base_disparity = regions[i].disparity;
		int ret = split_region(regions[i], min_size, std::back_inserter(created_regions));

		for(int i = 0; i < ret; ++i)
			data.push_back(defusion_data(i, ret));
	}
	std::swap(regions, created_regions);


	return generate_label_matrix(size, regions);
//	regionWiseSet<int
	//for(DisparityRegion& cregion : regions)


	/*std::vector<T> regions(newsegcount);// = getRegionVector(newlabels, newsegcount);
	fillRegionDescriptors(regions.begin(), regions.end(), newlabels);

	const std::size_t regions_count = regions.size();
	std::vector<std::size_t> inverse_mapping;
	inverse_mapping.reserve(fused_regions.size());
	for(std::size_t i = 0; i < regions_count; ++i)
	{
		if(data.active[i])
			inverse_mapping.push_back(i);
	}
	assert(fused_regions.size() == inverse_mapping.size());
	std::cout << fused_regions.size() << " vs " << inverse_mapping.size() << std::endl;

	for(std::size_t i = 0; i < regions_count; ++i)
	{
		std::size_t master_unfused_idx = i;

		while(!data.active[master_unfused_idx])
			master_unfused_idx = data.fused_with[master_unfused_idx];
		auto it = std::find(inverse_mapping.begin(), inverse_mapping.end(), master_unfused_idx);
		if(it != inverse_mapping.end())
		{
			std::size_t master_fused_idx = std::distance(inverse_mapping.begin(), it);
			regions[i].disparity = fused_regions[master_fused_idx].disparity;
			regions[i].base_disparity = regions[i].disparity;
		}
		else
			std::cerr << "missed mapping" << std::endl;
	}

	std::swap(fused_regions, regions);*/
}

RegionDescriptor create_rectangle(cv::Rect rect)
{
	RegionDescriptor result;
	for(int i = 0; i < rect.height; ++i)
		result.lineIntervals.push_back(RegionInterval(i+rect.y, rect.x, rect.x + rect.width));

	result.bounding_box = rect;
	result.m_size = rect.area();

	return result;
}

void split_region_test()
{
	//quadratic case
	RegionDescriptor test = create_rectangle(cv::Rect(5,5,10,10));
	std::vector<RegionDescriptor> res1;
	int ret1 = split_region(test, 5, std::back_inserter(res1));

	assert(res1.size() == 4);
	assert(res1[0].size() == 25);
	assert(res1[1].size() == 25);
	assert(res1[2].size() == 25);
	assert(res1[3].size() == 25);
	assert(ret1 == 4);

	//too small case (both directions)
	std::vector<RegionDescriptor> res2;
	int ret2 = split_region(test, 6, std::back_inserter(res2));
	assert(ret2 == 1);
	assert(res2.size() == 1);
	assert(res2[0].size() == 100);

	//rectangular case
	RegionDescriptor test3 = create_rectangle(cv::Rect(10,15, 10,20));
	std::vector<RegionDescriptor> res3;
	int ret3 = split_region(test3, 4, std::back_inserter(res3));
	assert(ret3 == 4);
	assert(res3.size() == 4);
	assert(res3[0].size() == 50);
	assert(res3[1].size() == 50);
	assert(res3[2].size() == 50);
	assert(res3[3].size() == 50);

	//to small case (one direction)
	std::vector<RegionDescriptor> res4;
	int ret4 = split_region(test3, 10, std::back_inserter(res4));
	assert(ret4 == 2);
	assert(res4.size() == 2);
	assert(res4[0].size() == 100);
	assert(res4[1].size() == 100);

	//irregular shaped case
	RegionDescriptor test5 = test3;
	test5.lineIntervals.push_back(RegionInterval(30,15,40));
	test5.bounding_box.width = 25;
	test5.bounding_box.height += 1;
	std::vector<RegionDescriptor> res5;
	int ret5 = split_region(test5, 5, std::back_inserter(res5));
	assert(ret5 == 4);
	assert(res5.size() == 4);

	return;
}

