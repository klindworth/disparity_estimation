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
//include "slic_adaptor.h"



#include <stdexcept>
#include <fstream>

#include "segmentation_cr.h"
#include "segmentation_ms.h"

/**
 * @brief fusion
 * @param regions
 * @param labels
 * @param idx Region to fuse
 * @param fusion_idx Region to fuse width
 * @param segcount Number of active segments
 * @param check_func Function to check if the combination of segments fullfills conditions
 */
template<typename T>
void fusion(fusion_work_data& data, std::vector<T>& regions, std::size_t idx, std::size_t fusion_idx, std::function<bool(const T& master_seg, const T& slave_seg, const T& fusion_seg)> check_func)
{
	T& cregion = regions[idx];
	data.visited[idx] = 1;
	for(std::pair<std::size_t, std::size_t>& cpair : cregion.neighbors)
	{
		std::size_t neighbor = cpair.first;
		if(neighbor != fusion_idx && data.active[neighbor])
		{
			T& cneighbor = regions[neighbor];
			if(check_func(cregion, cneighbor, regions[fusion_idx]))
			{
				assert(data.active[fusion_idx]);
				data.fused_with[neighbor] = fusion_idx;
				data.fused[fusion_idx].push_back(neighbor);
				//if the neighbor already fused with some other segments
				std::copy(data.fused[neighbor].begin(), data.fused[neighbor].end(), std::back_inserter(data.fused[fusion_idx]));
				data.active[neighbor] = 0;

				if(!(data.visited[neighbor]))
					fusion(data, regions, neighbor, fusion_idx, check_func);
			}
		}
	}
}

template<typename T>
void fuse(fusion_work_data& data, std::vector<T>& regions, cv::Mat& labels)
{
	const std::size_t regions_count = regions.size();
	#pragma omp parallel for default(none) shared(regions, data)
	for(std::size_t master_idx = 0; master_idx < regions_count; ++master_idx)
	{
		//regions[master_idx].active = data.active[master_idx];
		if(data.active[master_idx])
		{
			//remove doubles
			std::sort(data.fused[master_idx].begin(), data.fused[master_idx].end());
			data.fused[master_idx].erase(std::unique(data.fused[master_idx].begin(), data.fused[master_idx].end()), data.fused[master_idx].end());

			for(std::size_t slave_idx : data.fused[master_idx])
			{
				assert(!data.active[slave_idx]);
				T& cregion = regions[slave_idx];
				std::copy(cregion.lineIntervals.begin(), cregion.lineIntervals.end(), std::back_inserter(regions[master_idx].lineIntervals));
				regions[master_idx].m_size += cregion.m_size;
				cregion.m_size = 0;
			}
		}
	}

	for(std::size_t i = 0; i < regions_count; ++i)
	{
		if(data.active[i])
			assert(regions[i].m_size > 0);
		else
		{
			if(regions[i].m_size != 0)
			{
				std::cout << i << std::endl;
				std::cout << "size: "  << regions[i].m_size << std::endl;
				std::cout << "master: " << data.fused_with[i] << std::endl;
				std::cout << "master-active: " << (int)data.active[data.fused_with[i]] << std::endl;
				std::copy(data.fused[data.fused_with[i]].begin(), data.fused[data.fused_with[i]].end(), std::ostream_iterator<int>(std::cout));
				std::cout << std::endl;
				//assert(regions[i].size == 0); //TODO fixme
			}
		}
			//
	}

	regions.erase(std::remove_if(regions.begin(), regions.end(), [](const T& cregion){return (cregion.m_size == 0);}), regions.end());

	//regenerate labels image, sort region intervals
	/*parallel_region(regions.begin(), regions.end(), [&](RegionDescriptor region) {
		std::sort(region.lineIntervals.begin(), region.lineIntervals.end());
		intervals::setRegionValue<int>(labels, region.lineIntervals, i);
	});*/
	const std::size_t regions_count2 = regions.size();
	#pragma omp parallel for default(none) shared(labels, regions)
	for(std::size_t i = 0; i < regions_count2; ++i)
	{
		std::sort(regions[i].lineIntervals.begin(), regions[i].lineIntervals.end());
		intervals::setRegionValue<int>(labels, regions[i].lineIntervals, i);
	}

	//assert(checkLabelsIntervalsInvariant(regions, labels, regions.size()));
	//assert(std::count(data.active.begin(), data.active.end(), 1) == regions.size()); //TODO: fixme

	generate_neighborhood(labels, regions);
}

template<typename T>
void runFusion(cv::Mat& labels, std::vector<T>& regions, std::function<bool(const T& master_seg, const T& slave_seg, const T& fusion_seg)> check_func)
{
	std::size_t segcount = regions.size();
	fusion_work_data data(segcount);

	for(std::size_t i = 0; i < segcount; ++i)
	{
		data.visit_reset();
		if(data.active[i])
			fusion(data, regions, i, i, check_func);
		/*else
		{
			int last_j = i;
			while(data.active[last_j])
				last_j = data.fused_with[last_j];
			fusion(data, regions, labels, i, data.fused_with[last_j], check_func);
		}*/
	}
	std::cout << "fuse" << std::endl;
	fuse(data, regions, labels);
	std::cout << "fusion finished, regions: " << regions.size() << std::endl;

	refreshBoundingBoxes(regions.begin(), regions.end(), labels);
	generate_neighborhood(labels, regions);
}

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

int slic_segmentation::operator()(const cv::Mat& image, cv::Mat& labels) {
	return slicSuperpixels(image, labels, settings.superpixel_size, settings.superpixel_compactness);
}

std::string slic_segmentation::cacheName() const
{
	std::stringstream stream;
	stream << "superpixel_" << settings.superpixel_size;
	return stream.str();
}*/


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

template<typename T>
void defuse(std::vector<T>& fused_regions, cv::Mat_<int>& newlabels, int newsegcount, const fusion_work_data& data)
{
	std::vector<T> regions(newsegcount);// = getRegionVector(newlabels, newsegcount);
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

	std::swap(fused_regions, regions);
}

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
		return std::shared_ptr<segmentation_algorithm>( new crslic_segmentation(settings) );
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

