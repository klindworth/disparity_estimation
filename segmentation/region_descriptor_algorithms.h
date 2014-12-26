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

#ifndef REGION_DESCRIPTOR_ALGORITHMS_H
#define REGION_DESCRIPTOR_ALGORITHMS_H

#include "region_descriptor.h"
#include "intervals_algorithms.h"
#include "segmentation_image.h"

//! Applies a function in parallel (via OpenMP) to a range of elements
template<typename Iterator, typename T>
inline void parallel_region(Iterator begin, Iterator end, T func)
{
	const std::size_t regions_count = std::distance(begin, end);
	#pragma omp parallel for default(none) shared(begin, func)
	for(std::size_t i = 0; i < regions_count; ++i)
		func(*(begin + i));
}

/*template<typename T, typename reg_type>
cv::Mat_<T> regionWiseSet(cv::Size size, const std::vector<reg_type>& regions, std::function<T(const reg_type& region)> func)
{
	cv::Mat_<T> result(size, 0);

	parallel_region(regions.begin(), regions.end(), [&](const reg_type& region) {
		intervals::setRegionValue<T>(result, region.lineIntervals, func(region));
	});

	return result;
}*/

namespace region_descriptors
{

//! Moves a range of RegionIntervals in x-direction (amount: offset). The intervals will be capped to stay in the range [0..width)
template<typename Iterator>
inline void move_x(Iterator it, Iterator end, int offset, int width)
{
	for(; it != end; ++it)
		it->move(offset, width);
}

/**
 * @brief Creates a cv::Mat with the size, that's defined via parameter. The content will be defined by a passed function, that returns for each region value.
 * T defines type of the resulting matrix. Therefore the passed function must return T. The function will be called for each region in the container passed as parameter.
 * @param size Size of the resulting matrix.
 * @param regions The regions, on which the function will be applied.
 * @param func Function that will be called foreach region. The function must have const reg_type& as parameter and must return T.
 */
template<typename T, typename reg_type, typename lambda_type>
cv::Mat_<T> set_regionwise(cv::Size size, const std::vector<reg_type>& regions, lambda_type func)
{
	cv::Mat_<T> result(size, 0);

	parallel_region(regions.begin(), regions.end(), [&](const reg_type& region) {
		intervals::set_region_value<T>(result, region.lineIntervals, func(region));
	});

	return result;
}

template<typename T, typename reg_type, typename lambda_type>
cv::Mat_<T> set_regionwise(const segmentation_image<reg_type>& image, lambda_type func)
{
	return set_regionwise<T>(image.image_size, image.regions, func);
}

//! Returns an image, that will show each region with a random color.
template<typename reg_type>
cv::Mat_<cv::Vec3b> wrong_color_segmentation_image(cv::Size size, const std::vector<reg_type>& regions)
{
	std::srand(0);
	return set_regionwise<cv::Vec3b>(size, regions, [&](const reg_type&){
		cv::Vec3b ccolor;
		ccolor[0] = std::rand() % 256;
		ccolor[1] = std::rand() % 256;
		ccolor[2] = std::rand() % 256;
		return ccolor;
	});
}

template<typename reg_type>
cv::Mat_<cv::Vec3b> wrong_color_segmentation_image(const segmentation_image<reg_type>& image)
{
	return wrong_color_segmentation_image(image.image_size, image.regions);
}

//! Creates a matrix (int) with the segmentation labels as content.
template<typename reg_type>
cv::Mat_<int> generate_label_matrix(cv::Size size, const std::vector<reg_type>& regions)
{
	cv::Mat_<int> result(size, 0);

	for(std::size_t i = 0; i < regions.size(); ++i)
	{
		intervals::set_region_value<int>(result, regions[i].lineIntervals, (int)i);
	}

	return result;
}

inline cv::Point bounding_middle_point(const cv::Rect& rect)
{
	cv::Point res;
	res.x = rect.width / 2 + rect.x;
	res.y = rect.height /2 + rect.y;
	return res;
}

/**
 * @brief Updates all bounding boxes of region_descriptors in a container
 * @param begin begin Iterator of the container with region_descriptors in it
 * @param end end Iterator of the container with region_descriptors in it
 * @param labels Matrix with segment labels
 */
template<typename Iterator, typename label_type>
void refresh_bounding_boxes(Iterator begin, Iterator end, const cv::Mat_<label_type>& labels)
{
	parallel_region(begin, end, [&](region_descriptor& region){
		region.bounding_box.x = labels.cols;
		region.bounding_box.y = labels.rows;
		region.bounding_box.height = 0;
		region.bounding_box.width = 0;
	});

	const label_type* clabel = labels[0];
	for(int y = 0; y < labels.rows; ++y)
	{
		for(int x = 0; x < labels.cols; ++x)
		{
			label_type i = *clabel++;//labels.at<int>(y,x);
			cv::Rect& crect = (*(begin + i)).bounding_box;

			assert(i < std::distance(begin, end) && i >= 0);

			crect.x = std::min(x, crect.x);
			crect.y = std::min(y, crect.y);
			crect.width  = std::max(x, crect.width); //save temp the maximal x coordinate
			crect.height = std::max(y, crect.height);
		}
	}

	parallel_region(begin, end, [&](region_descriptor& region){
		region.bounding_box.width  -= region.bounding_box.x - 1;
		region.bounding_box.height -= region.bounding_box.y - 1;
		region.avg_point = bounding_middle_point(region.bounding_box);
	});
}

//! Fills a vector of RegionDescriptors with the correct lineIntervals and sizes
template<typename Iterator, typename label_type>
void fill(Iterator begin, Iterator end, const cv::Mat_<label_type>& labels)
{
	auto factory = [&](std::size_t y, std::size_t lower, std::size_t upper, label_type value) {
		assert(value < std::distance(begin, end) && value >= 0);
		(*(begin + value)).lineIntervals.push_back(region_interval(y, lower, upper));
	};

	intervals::convert_differential<label_type>(labels, factory);

	parallel_region(begin, end, [&](region_descriptor& region){
		//compute size
		region.m_size = region.size();
		assert(region.m_size != 0);
	});

	refresh_bounding_boxes(begin, end, labels);
}

inline neighbor_vector::iterator find_neighbor(neighbor_vector& container, std::size_t idx)
{
	return std::find_if(container.begin(), container.end(), [=](const std::pair<std::size_t, std::size_t>& cpair){return cpair.first == idx;});
}

/**
 * Adds a neighbor relation to the regions. If that relation is already established, only the common boundary counter is increased
 * @param regions Vector with all regions, which contains the both indices that are passed as parameters too
 * @param idx1 Vector index for one of the two neighboring regions
 * @param idx2 Vector index for one of the two neighboring regions
 */
template<typename T>
inline void save_neighbors(std::vector<T>& regions, std::size_t idx1, std::size_t idx2)
{
	assert(idx1 < regions.size());
	assert(idx2 < regions.size());

	if(idx1 != idx2)
	{
		neighbor_vector& reg1 = regions[idx1].neighbors;
		neighbor_vector& reg2 = regions[idx2].neighbors;
		neighbor_vector::iterator it = find_neighbor(reg1, idx2);
		if(it == reg1.end())
		{
			reg1.push_back(std::make_pair(idx2, 1));
			reg2.push_back(std::make_pair(idx1, 1));
		}
		else
		{
			it->second += 1;
			neighbor_vector::iterator it2 = find_neighbor(reg2, idx1);
			it2->second += 1;
		}
	}
}

template<typename T, typename label_type>
void generate_neighborhood(const cv::Mat_<label_type> &labels, std::vector<T> &regions)
{
	for(T& cregion : regions)
		cregion.neighbors.clear();

	for(int i = 0; i < labels.rows; ++i)
	{
		for(int j = 1; j < labels.cols; ++j)
			save_neighbors(regions, labels(i,j), labels(i, j-1));
	}
	for(int i = 1; i < labels.rows; ++i)
	{
		for(int j = 0; j < labels.cols; ++j)
			save_neighbors(regions, labels(i,j), labels(i-1, j));
	}
}



template<typename Iterator>
void calculate_all_average_colors(const cv::Mat& image, Iterator begin, Iterator end)
{
	cv::Mat lab_image = bgr_to_lab(image);
	cv::Mat lab_double_image;
	lab_image.convertTo(lab_double_image, CV_64FC3);

	parallel_region(begin, end, [&](region_descriptor& region)
	{
		calculate_average_color(region, lab_double_image);
	});
}

/**
 * @brief Calculates an average value of the neighboring segments.
 * @param container Container with the regions
 * @param neighbors Vector with the neighbors of a region
 * @param init_value Initial value for calculating the sum for the average
 * @param func A function which returns a value for a given region, which will be summed up by this function.
 * Therefore it must be a function that returns a value of type T and accepts a parameter of tyoe region_type
 */
template<typename T, typename region_type, typename lambda_type>
T neighbors_average(const std::vector<region_type>& container, const neighbor_vector& neighbors, const T& init_value, lambda_type func)
{
	T result = init_value;
	for(const std::pair<std::size_t, std::size_t>& cpair : neighbors)
	{
		result += func(container[cpair.first]);
	}
	return result/(int)neighbors.size();
}

/*template<typename lambda_type>
void index_foreach_neighborhood(const neighbor_vector& neighbors, lambda_func)
{
	for(const std::pair<std::size_t, std::size_t>& cpair : neighbors)
	{
		result += func(cpair.first);
	}
}*/

/**
 * Saves the value of each neighbor in a vector
 * @param cache The vector, where the gathered values from the neighbors will be saved in. Therefore the vector will have the size of the neighborhood
 * @param container Vector with the regions
 * @param neighbors Vector with the neighbors
 * @param gather_caching_value Function that accepts a region of type region_type and returns a value of type cache_type
 */
template<typename cache_type, typename region_type, typename lambda_type>
void gather_neighbor_values(std::vector<cache_type>& cache, const std::vector<region_type>& container, const neighbor_vector& neighbors, lambda_type gather_caching_value)
{
	std::size_t nsize = neighbors.size();
	cache.resize(nsize);

	for(std::size_t i = 0; i < nsize; ++i)
		cache[i] = gather_caching_value(container[neighbors[i].first]);
}

/**
 * This function works like gather_neighbor_value, but the passed function must accept a vector index/segment id as parameter instead a reference to the region itself @see gather_neighbor_value
 */
template<typename cache_type, typename lambda_type>
void gather_neighbor_values_idx(std::vector<cache_type>& cache, const neighbor_vector& neighbors, lambda_type gather_caching_value)
{
	std::size_t nsize = neighbors.size();
	cache.resize(nsize);

	for(std::size_t i = 0; i < nsize; ++i)
		cache[i] = gather_caching_value(neighbors[i].first);
}

/**
 * Like gather_neighbor_values, but the values are weighted by the size of the shared boundary between the region and that neighbor
 */
template<typename T, typename reg_type, typename lambda_type>
T weighted_neighbors_average(const std::vector<reg_type>& container, const neighbor_vector& neighbors, const T& initVal, lambda_type func)
{
	T result = initVal;
	float sum_weights = 0.0f;
	for(const std::pair<std::size_t, std::size_t>& cpair : neighbors)
	{
		result += cpair.second*func(container[cpair.first]);
		sum_weights += cpair.second;
	}
	return result/sum_weights;
}

namespace invariants
{
/**
 * Checks, if the sum of all sizes equals the image size and if the numbers in the labels matrix
 *  equals the position of the corresponding RegionDescriptor
 */
template<typename Iterator, typename label_type>
bool check_labels_Intervals(Iterator begin, Iterator end, const cv::Mat_<label_type>& labels)
{
	int pixelcount = 0;
	for(Iterator it = begin; it != end; ++it)
	{
		const region_descriptor& cregion = *it;
		int segsize = cregion.size();
		assert(segsize == cregion.m_size);
		assert(segsize != 0);
		pixelcount += segsize;
		for(const region_interval& cinterval : cregion.lineIntervals)
		{
			for(int x = cinterval.lower; x < cinterval.upper; ++x)
			{
				label_type val = labels(cinterval.y, x);
				assert(val == std::distance(begin, it));
				if(val != std::distance(begin, it))
					return false;
			}
		}
	}
	assert(pixelcount == labels.rows * labels.cols);
	if(pixelcount != labels.rows * labels.cols)
		return false;
	else
		return true;
}

/**
 * Checks if all neighbor ids are existing regions, and it checks if the neighboring region has the
 * region as its neighbor. It parameter regions_count doesn't have to be equal to regions.size(),
 * if it's lower it means only the first regions_count regions will be checked
 */
template<typename T>
bool check_neighborhood(const std::vector<T> &regions, std::size_t regions_count)
{
	for(std::size_t i = 0; i < regions_count; ++i)
	{
		const region_descriptor& cregion = regions[i];
		const std::size_t neigh_count = cregion.neighbors.size();

		for(std::size_t j = 0; j < neigh_count; ++j)
		{
			std::size_t c_idx = cregion.neighbors[j].first;

			assert(c_idx < regions_count);

			const region_descriptor& cneighbor = regions[c_idx];

			bool found = false;
			const std::size_t inner_neigh_count = cneighbor.neighbors.size();
			for(std::size_t k = 0; k < inner_neigh_count; ++k)
			{
				if(cneighbor.neighbors[k].first == i)
				{
					found = true;
					break;
				}
			}

			assert(found);
		}
	}
	return true;
}
}


}

#endif // REGION_DESCRIPTOR_ALGORITHMS_H
