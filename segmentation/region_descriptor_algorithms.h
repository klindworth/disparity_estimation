#ifndef REGION_DESCRIPTOR_ALGORITHMS_H
#define REGION_DESCRIPTOR_ALGORITHMS_H

#include "genericfunctions.h"
#include "region_descriptor.h"
#include "intervals_algorithms.h"

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

template<typename T, typename reg_type, typename lambda_type>
cv::Mat_<T> regionWiseSet(cv::Size size, const std::vector<reg_type>& regions, lambda_type func)
{
	cv::Mat_<T> result(size, 0);

	parallel_region(regions.begin(), regions.end(), [&](const reg_type& region) {
		intervals::setRegionValue<T>(result, region.lineIntervals, func(region));
	});

	return result;
}

template<typename reg_type>
cv::Mat_<cv::Vec3b> getWrongColorSegmentationImage(cv::Size size, const std::vector<reg_type>& regions)
{
	std::srand(0);
	return regionWiseSet<cv::Vec3b>(size, regions, [&](const reg_type&){
		cv::Vec3b ccolor;
		ccolor[0] = std::rand() % 256;
		ccolor[1] = std::rand() % 256;
		ccolor[2] = std::rand() % 256;
		return ccolor;
	});
}

template<typename reg_type>
cv::Mat_<int> generate_label_matrix(cv::Size size, const std::vector<reg_type>& regions)
{
	cv::Mat_<int> result(size, 0);

	for(std::size_t i = 0; i < regions.size(); ++i)
	{
		intervals::setRegionValue<int>(result, regions[i].lineIntervals, (int)i);
	}

	return result;
}

/**
 * Fills a vector of RegionDescriptors with the correct lineIntervals and sizes
 */
template<typename Iterator, typename label_type>
void fillRegionDescriptors(Iterator begin, Iterator end, const cv::Mat_<label_type>& labels)
{
	auto factory = [&](std::size_t y, std::size_t lower, std::size_t upper, label_type value) {
		assert(value < std::distance(begin, end) && value >= 0);
		(*(begin + value)).lineIntervals.push_back(RegionInterval(y, lower, upper));
	};

	intervals::convertDifferential<label_type>(labels, factory);

	parallel_region(begin, end, [&](RegionDescriptor& region){
		//compute size
		region.m_size = region.size();
		assert(region.m_size != 0);
	});

	refreshBoundingBoxes(begin, end, labels);
}

inline neighbor_vector::iterator find_neighbor(neighbor_vector& container, std::size_t val)
{
	return std::find_if(container.begin(), container.end(), [=](const std::pair<std::size_t, std::size_t>& cpair){return cpair.first == val;});
}

template<typename T>
inline void save_neighbors(std::vector<T>& regions, std::size_t val1, std::size_t val2)
{
	if(val1 != val2)
	{
		neighbor_vector& reg1 = regions[val1].neighbors;
		neighbor_vector& reg2 = regions[val2].neighbors;
		neighbor_vector::iterator it = find_neighbor(reg1, val2);
		if(it == reg1.end())
		{
			reg1.push_back(std::make_pair(val2, 1));
			reg2.push_back(std::make_pair(val1, 1));
		}
		else
		{
			it->second += 1;
			neighbor_vector::iterator it2 = find_neighbor(reg2, val1);
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

template<typename Iterator, typename label_type>
void refreshBoundingBoxes(Iterator begin, Iterator end, const cv::Mat_<label_type>& labels)
{
	parallel_region(begin, end, [&](RegionDescriptor& region){
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

	parallel_region(begin, end, [&](RegionDescriptor& region){
		region.bounding_box.width  -= region.bounding_box.x - 1;
		region.bounding_box.height -= region.bounding_box.y - 1;
	});
}

/**
 * Checks, if the sum of all sizes equals the image size and if the numbers in the labels matrix
 *  equals the position of the corresponding RegionDescriptor
 */
template<typename Iterator, typename label_type>
bool checkLabelsIntervalsInvariant(Iterator begin, Iterator end, const cv::Mat_<label_type>& labels)
{
	int pixelcount = 0;
	for(Iterator it = begin; it != end; ++it)
	{
		const RegionDescriptor& cregion = *it;
		int segsize = cregion.size();
		assert(segsize == cregion.m_size);
		assert(segsize != 0);
		pixelcount += segsize;
		for(const RegionInterval& cinterval : cregion.lineIntervals)
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
bool checkNeighborhoodInvariant(const std::vector<T> &regions, std::size_t regions_count)
{
	for(std::size_t i = 0; i < regions_count; ++i)
	{
		const RegionDescriptor& cregion = regions[i];
		const std::size_t neigh_count = cregion.neighbors.size();

		for(std::size_t j = 0; j < neigh_count; ++j)
		{
			std::size_t c_idx = cregion.neighbors[j].first;

			assert(c_idx < regions_count);

			const RegionDescriptor& cneighbor = regions[c_idx];

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

template<typename Iterator>
void calculate_all_average_colors(const cv::Mat& image, Iterator begin, Iterator end)
{
	cv::Mat lab_image = bgr_to_lab(image);
	cv::Mat lab_double_image;
	lab_image.convertTo(lab_double_image, CV_64FC3);

	parallel_region(begin, end, [&](RegionDescriptor& region)
	{
		calculate_average_color(region, lab_double_image);
	});
}

/*template<typename T, typename reg_type>
cv::Mat_<unsigned char> regionWiseImage(cv::Size size, std::vector<reg_type>& regions, std::function<T(const reg_type& region)> func)
{
	return getValueScaledImage<T, unsigned char>(regionWiseSet<T, reg_type>(size, regions, func));
}*/

template<typename T, typename reg_type, typename lambda_type>
cv::Mat_<unsigned char> regionWiseImage(cv::Size size, std::vector<reg_type>& regions, lambda_type func)
{
	return getValueScaledImage<T, unsigned char>(regionWiseSet<T, reg_type>(size, regions, func));
}

/*template<typename T, typename reg_type>
T getNeighborhoodsAverage(const std::vector<reg_type>& container, const neighbor_vector& neighbors, const T& initVal, std::function<T(const reg_type&)> func)
{
	T result = initVal;
	for(const std::pair<std::size_t, std::size_t>& cpair : neighbors)
	{
		result += func(container[cpair.first]);
	}
	return result/(int)neighbors.size();
}*/

template<typename T, typename reg_type, typename lambda_type>
T getNeighborhoodsAverage(const std::vector<reg_type>& container, const neighbor_vector& neighbors, const T& initVal, lambda_type func)
{
	T result = initVal;
	for(const std::pair<std::size_t, std::size_t>& cpair : neighbors)
	{
		result += func(container[cpair.first]);
	}
	return result/(int)neighbors.size();
}

template<typename T, typename reg_type>
T getWeightedNeighborhoodsAverage(const std::vector<reg_type>& container, const neighbor_vector& neighbors, const T& initVal, std::function<T(const reg_type&)> func)
{
	T result = initVal;
	float sum_weight = 0.0f;
	for(const std::pair<std::size_t, std::size_t>& cpair : neighbors)
	{
		result += cpair.second*func(container[cpair.first]);
		sum_weight += cpair.second;
	}
	return result/sum_weight;
}

#endif // REGION_DESCRIPTOR_ALGORITHMS_H
