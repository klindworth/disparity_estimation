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
#include "genericfunctions.h"
#include "misc.h"
#include "region_descriptor.h"
#include "region_descriptor_algorithms.h"
#include "slic_adaptor.h"
#include "ms_cv.h"

#include "msImageProcessor.h"

#include <stdexcept>
#include <fstream>

#include "contourRelaxation/FeatureType.h"
#include "contourRelaxation/ContourRelaxation.h"
#include "contourRelaxation/InitializationFunctions.h"

int cachedSegmentation(StereoSingleTask& task, cv::Mat_<int>& labels, std::shared_ptr<segmentation_algorithm>& algorithm)
{
	int regions_count = 0;
	if(algorithm->cacheAllowed())
	{
		std::string filename = "cache/" + task.fullname + "_" + algorithm->cacheName() + ".cache.cvmat";
		std::ifstream istream(filename, std::ifstream::binary);
		if(istream.is_open())
		{
			std::cout << "use cachefile: " << filename << std::endl;
			istream.read((char*)&regions_count, sizeof(int));
			labels = streamToMat(istream);
			istream.close();
		}
		else
		{
			std::cout << "create cachefile: " << filename << std::endl;
			regions_count = (*algorithm)(task.base, labels);

			std::ofstream ostream(filename, std::ofstream::binary);
			ostream.write((char*)&regions_count, sizeof(int));
			matToStream(labels, ostream);
			ostream.close();
		}
	}
	else
		regions_count = (*algorithm)(task.base, labels);
	return regions_count;
}

/**
 * @param src Image to segment
 * @param labels_dst cv::Mat where the (int) labels will be written in
 * @return Number of different labels
 */
//7,4.0
/*int mean_shift_segmentation(const cv::Mat& src, cv::Mat& labels_dst, int spatial_variance, float color_variance, int minsize)
{
	msImageProcessor proc;
	proc.DefineImage(src.data, (src.channels() == 3 ? COLOR : GRAYSCALE), src.rows, src.cols);
	proc.Segment(spatial_variance,color_variance, minsize, MED_SPEEDUP);//HIGH_SPEEDUP, MED_SPEEDUP, NO_SPEEDUP; high: speedupThreshold setzen!
	//cv::Mat res = cv::Mat(src.size(), src.type());
	//proc.GetResults(res.data);

	labels_dst = cv::Mat(src.size(), CV_32SC1);
	int regions_count = proc.GetRegionsModified(labels_dst.data);

	return regions_count;
}*/

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
void fuse(fusion_work_data& data, std::vector<T>& regions, cv::Mat_<int>& labels)
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

int segmentImage2(cv::Mat& src, cv::Mat& labels_dst, int spatial_variance, float color_variance)
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

int ms_slic(const cv::Mat& image, cv::Mat_<int>& labels, const segmentation_settings& config)
{
	int regions_count = slicSuperpixels(image, labels, config.superpixel_size, config.superpixel_compactness);
	std::vector<DisparityRegion> regions(regions_count);
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

	fusion_work_data data(regions.size());
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

int slic_segmentation::operator()(const cv::Mat& image, cv::Mat_<int>& labels) {
	return slicSuperpixels(image, labels, settings.superpixel_size, settings.superpixel_compactness);
}

std::string slic_segmentation::cacheName() const
{
	std::stringstream stream;
	stream << "superpixel_" << settings.superpixel_size;
	return stream.str();
}

int ms_cr(const cv::Mat& image, cv::Mat_<int>& labels, const segmentation_settings& config)
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
}


int crslic_segmentation::operator()(const cv::Mat& image, cv::Mat_<int>& labels)
{
	float directCliqueCost = 0.3;
	unsigned int const iterations = 3;
	double const diagonalCliqueCost = directCliqueCost / sqrt(2);

	bool isColorImage = (image.channels() == 3);
	std::vector<FeatureType> features;
	if (isColorImage)
		features.push_back(Color);
	else
		features.push_back(Grayvalue);

	features.push_back(Compactness);

	ContourRelaxation<int> crslic_obj(features);
	cv::Mat labels_temp = createBlockInitialization<int>(image.size(), settings.superpixel_size, settings.superpixel_size);

	crslic_obj.setCompactnessData(settings.superpixel_compactness);

	if (isColorImage)
	{
		cv::Mat imageYCrCb;
		cv::cvtColor(image, imageYCrCb, CV_BGR2YCrCb);
		std::vector<cv::Mat> imageYCrCbChannels;
		cv::split(imageYCrCb, imageYCrCbChannels);

		crslic_obj.setColorData(imageYCrCbChannels[0], imageYCrCbChannels[1], imageYCrCbChannels[2]);
	}
	else
		crslic_obj.setGrayvalueData(image.clone());

	crslic_obj.relax(labels_temp, directCliqueCost, diagonalCliqueCost, iterations, labels);
	return 1+*(std::max_element(labels.begin(), labels.end()));
}

std::string crslic_segmentation::cacheName() const
{
	std::stringstream stream;
	stream << "crsuperpixel_" << settings.superpixel_size;
	return stream.str();
}

int meanshift_segmentation::operator()(const cv::Mat& image, cv::Mat_<int>& labels) {
	return mean_shift_segmentation(image, labels, settings.spatial_var, settings.color_var, 20);
}

std::string meanshift_segmentation::cacheName() const
{
	std::stringstream stream;
	stream << "meanshift_" << settings.spatial_var;
	return stream.str();
}

int mssuperpixel_segmentation::operator()(const cv::Mat& image, cv::Mat_<int>& labels) {
	int regions_count = slicSuperpixels(image, labels, settings.superpixel_size, settings.superpixel_compactness);
	std::vector<DisparityRegion> regions(regions_count);
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
}

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

void mssuperpixel_segmentation::refine(RegionContainer& container) {
	defuse(container.regions, superpixel, regions_count_superpixel, *fusion_data);
	container.labels = superpixel;

	generate_neighborhood(container.labels, container.regions);
}



std::shared_ptr<segmentation_algorithm> getSegmentationClass(const segmentation_settings& settings) {
	if(settings.algorithm == "meanshift")
		return std::shared_ptr<segmentation_algorithm>( new meanshift_segmentation(settings) );
	else if(settings.algorithm == "superpixel")
		return std::shared_ptr<segmentation_algorithm>( new slic_segmentation(settings) );
	else if(settings.algorithm == "ms_superpixel")
		return std::shared_ptr<segmentation_algorithm>( new mssuperpixel_segmentation(settings) );
	else if(settings.algorithm == "cr_superpixel")
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

template<typename T, typename InsertIterator>
void insert_pair(T pair, InsertIterator it)
{
	*it = pair.first;
	++it;
	*it = pair.second;
	++it;
}

std::pair<RegionDescriptor, RegionDescriptor> hsplit_region(const RegionDescriptor& descriptor, int split_threshold)
{
	std::pair<RegionDescriptor, RegionDescriptor> result;
	result.first.bounding_box = descriptor.bounding_box;
	result.first.bounding_box.height = split_threshold - descriptor.bounding_box.y;

	result.second.bounding_box = descriptor.bounding_box;
	result.second.bounding_box.y = split_threshold;
	result.second.bounding_box.height = descriptor.bounding_box.y + descriptor.bounding_box.height - split_threshold;

	for(const RegionInterval& cinterval : descriptor.lineIntervals)
	{
		if(cinterval.y < split_threshold)
			result.first.lineIntervals.push_back(cinterval);
		else
			result.second.lineIntervals.push_back(cinterval);
	}

	result.first.m_size = getSizeOfRegion(result.first.lineIntervals);
	result.second.m_size = getSizeOfRegion(result.second.lineIntervals);

	return result;
}

std::pair<RegionDescriptor, RegionDescriptor> vsplit_region(const RegionDescriptor& descriptor, int split_threshold)
{
	std::pair<RegionDescriptor, RegionDescriptor> result;
	result.first.bounding_box = descriptor.bounding_box;
	result.first.bounding_box.width = split_threshold - descriptor.bounding_box.x;

	result.second.bounding_box = descriptor.bounding_box;
	result.second.bounding_box.x = split_threshold;
	result.second.bounding_box.width = descriptor.bounding_box.x + descriptor.bounding_box.width - split_threshold;

	for(const RegionInterval& cinterval : descriptor.lineIntervals)
	{
		if(cinterval.upper > split_threshold && cinterval.lower <= split_threshold)
		{
			result.first.lineIntervals.push_back(RegionInterval(cinterval.y, cinterval.lower, split_threshold));
			result.second.lineIntervals.push_back(RegionInterval(cinterval.y, split_threshold, cinterval.upper));
		}
		else if(cinterval.upper <= split_threshold)
			result.first.lineIntervals.push_back(cinterval);
		else
			result.second.lineIntervals.push_back(cinterval);
	}

	result.first.m_size = getSizeOfRegion(result.first.lineIntervals);
	result.second.m_size = getSizeOfRegion(result.second.lineIntervals);

	return result;
}

int split_region(const RegionDescriptor& descriptor, int min_size, std::back_insert_iterator<std::vector<RegionDescriptor>> it)
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

	++x_avg; //because we want open intervalls
	++y_avg;

	bool split_h = false;
	if( ((y_avg - descriptor.bounding_box.y) >= min_size)
			&& ((descriptor.bounding_box.y + descriptor.bounding_box.height - y_avg) >= min_size))
		split_h = true;

	bool split_v = false;
	if( ((x_avg - descriptor.bounding_box.x) >= min_size)
			&& ((descriptor.bounding_box.x + descriptor.bounding_box.width - x_avg) >= min_size))
		split_v = true;

	if(split_h && split_v)
	{
		auto temp = hsplit_region(descriptor, y_avg);
		insert_pair(vsplit_region(temp.first, x_avg), it);
		insert_pair(vsplit_region(temp.second, x_avg), it);
	}
	else if(split_h)
		insert_pair(hsplit_region(descriptor, y_avg), it);
	else if(split_v)
		insert_pair(vsplit_region(descriptor, x_avg), it);
	else
	{
		*it = descriptor;
		++it;
	}

	return (split_h ? 2 : 1) * (split_v ? 2 : 1);
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
	RegionDescriptor test = create_rectangle(cv::Rect(5,5,10,10));

	//quadratic case
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

