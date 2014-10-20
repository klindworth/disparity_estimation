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
#include "misc.h"
#include "region_descriptor.h"
#include "region_descriptor_algorithms.h"

#include <stdexcept>
#include <fstream>

#include "segmentation_cr.h"
#include "segmentation_ms.h"
#include "segmentation_slic.h"

#include "segmentation_algorithms.h"



cv::Mat getWrongColorSegmentationImage(cv::Mat_<int>& labels, int labelcount)
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

cv::Mat getWrongColorSegmentationImage(RegionContainer& container)
{
	std::srand(0);
	return regionWiseSet<cv::Vec3b>(container.task, container.regions, [&](const DisparityRegion&){
		cv::Vec3b ccolor;
		ccolor[0] = std::rand() % 256;
		ccolor[1] = std::rand() % 256;
		ccolor[2] = std::rand() % 256;
		return ccolor;
	});
}

/*int segmentImage2(cv::Mat& src, cv::Mat& labels_dst, int spatial_variance, float color_variance)
{
	msImageProcessor proc;
	proc.DefineImage(src.data, (src.channels() == 3 ? COLOR : GRAYSCALE), src.rows, src.cols);
	proc.Segment(spatial_variance,color_variance, 1, NO_SPEEDUP);//HIGH_SPEEDUP, MED_SPEEDUP, NO_SPEEDUP; high: speedupThreshold setzen!
	//cv::Mat res = cv::Mat(src.size(), src.type());
	//proc.GetResults(res.data);

	labels_dst = cv::Mat(src.size(), CV_32SC1);
	int regions_count = proc.GetRegionsModified(labels_dst.data);

	return regions_count;
}

int ms_slic(const cv::Mat& image, cv::Mat& labels, const segmentation_settings& config)
{
	int regions_count = slicSuperpixels(image, labels, config.superpixel_size, config.superpixel_compactness);
	std::vector<SegRegion> regions = getRegionVector(labels, regions_count);

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

	fusion_work_data data(regions);
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
}
*/

/*

int mssuperpixel_segmentation::operator()(const cv::Mat& image, cv::Mat_<int>& labels) {
	int regions_count = slicSuperpixels(image, labels, settings.superpixel_size, settings.superpixel_compactness);
	std::vector<RegionDescriptor> regions(regions_count);
	fillRegionDescriptors(regions.begin(), regions.end(), labels);

	superpixel = labels.clone();
	regions_count_superpixel = regions_count;

	calculate_all_average_colors(image, regions);

	std::pair<int,int> slic_size = get_slic_seeds(image.cols, image.rows, settings.superpixel_size);
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
	int mscount = segmentImage2(slic_image, msResult, settings.spatial_var, settings.color_var);
	//cv::imshow("ms", getWrongColorSegmentationImage(msResult, mscount));
	std::cout << "mscount: " << mscount << std::endl;

	std::shared_ptr<fusion_work_data> data = std::shared_ptr<fusion_work_data>( new fusion_work_data(regions.size()) );
	int *src_ptr = msResult.ptr<int>(0);
	std::vector<int> mapping(mscount, -1);
	for(int i = 0; i < slic_size.first; ++i)
	{
		for(int j = 0; j < slic_size.second; ++j)
		{
			int idx = labels.at<int>(i*yex+yex/2,j*xex+xex/2);
			int label = *src_ptr++;
			int fusion_idx = mapping[label];

			if(fusion_idx == -1 && data->active[idx])
				mapping[label] = idx;
			else if(idx != fusion_idx && data->active[idx])
			{
				while(!data->active[fusion_idx])
					fusion_idx = data->fused_with[fusion_idx];

				assert(fusion_idx >= 0 && fusion_idx < regions.size());
				data->fused_with[idx] = fusion_idx;
				if(! data->active[fusion_idx])
					std::cout << "master of master: " << data->fused_with[fusion_idx] << std::endl;
				assert(data->active[fusion_idx]);
				data->active[idx] = false;
				data->fused[fusion_idx].push_back(idx);
			}
		}
	}
	fusion_data = data;

	fuse(*data, regions, labels);

	return regions.size();
}

bool mssuperpixel_segmentation::cacheAllowed() const {
	return false;
}

std::string mssuperpixel_segmentation::cacheName() const
{
	return "mssuperpixel";
}

bool mssuperpixel_segmentation::refinementPossible() {
	return true;
}*/

/*void mssuperpixel_segmentation::refine(RegionContainer& container) {
	defuse(container.regions, superpixel, regions_count_superpixel, *fusion_data);
	container.labels = superpixel;

	generate_neighborhood(container.labels, container.regions);
}*/



std::shared_ptr<segmentation_algorithm> getSegmentationClass(const segmentation_settings& settings) {
	/*if(settings.algorithm == "meanshift")
		return std::shared_ptr<segmentation_algorithm>( new meanshift_segmentation(settings) );
	else if(settings.algorithm == "superpixel")
		return std::shared_ptr<segmentation_algorithm>( new slic_segmentation(settings) );
	else if(settings.algorithm == "ms_superpixel")
		return std::shared_ptr<segmentation_algorithm>( new mssuperpixel_segmentation(settings) );
	else*/ if(settings.algorithm == "cr_superpixel")
		return std::make_shared<crslic_segmentation>(settings);
	else
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

