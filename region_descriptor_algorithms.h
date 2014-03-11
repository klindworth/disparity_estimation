#ifndef REGION_DESCRIPTOR_ALGORITHMS_H
#define REGION_DESCRIPTOR_ALGORITHMS_H

#include "genericfunctions.h"
#include "region_descriptor.h"
#include "intervals_algorithms.h"

/**
 * Fills a vector of RegionDescriptors with the correct lineIntervals and sizes
 */
template<typename Iterator>
void fillRegionDescriptors(Iterator begin, Iterator end, const cv::Mat& labels)
{
	auto factory = [&](std::size_t y, std::size_t lower, std::size_t upper, int value) {
		assert(value < std::distance(begin, end) && value >= 0);
		(*(begin + value)).lineIntervals.push_back(RegionInterval(y, lower, upper));
	};

	const cv::Mat_<int> labels_typed = labels;
	intervals::convertDifferential<int>(labels_typed, factory);

	parallel_region(begin, end, [&](RegionDescriptor& region){
		//compute size
		region.m_size = region.size();
		assert(region.m_size != 0);
	});

	refreshBoundingBoxes(begin, end, labels);
}

inline std::vector<std::pair<std::size_t, std::size_t> >::iterator find_neighbor(std::vector<std::pair<std::size_t, std::size_t> >& container, std::size_t val)
{
	return std::find_if(container.begin(), container.end(), [=](const std::pair<std::size_t, std::size_t>& cpair){return cpair.first == val;});
}

template<typename T>
inline void save_neighbors(std::vector<T>& regions, std::size_t val1, std::size_t val2)
{
	typedef std::vector<std::pair<std::size_t, std::size_t> > neighbor_vector;
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

template<typename T>
void generate_neighborhood(cv::Mat &labels, std::vector<T> &regions)
{
	for(T& cregion : regions)
		cregion.neighbors.clear();

	for(int i = 0; i < labels.rows; ++i)
	{
		for(int j = 1; j < labels.cols; ++j)
			save_neighbors(regions, labels.at<int>(i,j), labels.at<int>(i, j-1));
	}
	for(int i = 1; i < labels.rows; ++i)
	{
		for(int j = 0; j < labels.cols; ++j)
			save_neighbors(regions, labels.at<int>(i,j), labels.at<int>(i-1, j));
	}
}

template<typename Iterator>
void refreshBoundingBoxes(Iterator begin, Iterator end, const cv::Mat& labels)
{
	parallel_region(begin, end, [&](RegionDescriptor& region){
		region.bounding_box.x = labels.cols;
		region.bounding_box.y = labels.rows;
		region.bounding_box.height = 0;
		region.bounding_box.width = 0;
	});

	const int* clabel = labels.ptr<int>(0);
	for(int y = 0; y < labels.rows; ++y)
	{
		for(int x = 0; x < labels.cols; ++x)
		{
			int i = *clabel++;//labels.at<int>(y,x);
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

template<typename Iterator>
bool checkLabelsIntervalsInvariant(Iterator begin, Iterator end, const cv::Mat& labels)
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
				int val = labels.at<int>(cinterval.y, x);
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

template<typename T, typename reg_type>
cv::Mat_<unsigned char> regionWiseImage(cv::Size size, std::vector<reg_type>& regions, std::function<T(const reg_type& region)> func)
{
	return getValueScaledImage<T, unsigned char>(regionWiseSet<T, reg_type>(size, regions, func));
}

template<typename T, typename reg_type>
T getNeighborhoodsAverage(const std::vector<reg_type>& container, const std::vector<std::pair<std::size_t, std::size_t>>& neighbors, const T& initVal, std::function<T(const reg_type&)> func)
{
	T result = initVal;
	for(const std::pair<std::size_t, std::size_t>& cpair : neighbors)
	{
		result += func(container[cpair.first]);
	}
	return result/(int)neighbors.size();
}

template<typename T, typename reg_type>
T getWeightedNeighborhoodsAverage(const std::vector<reg_type>& container, const std::vector<std::pair<std::size_t, std::size_t>>& neighbors, const T& initVal, std::function<T(const reg_type&)> func)
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
