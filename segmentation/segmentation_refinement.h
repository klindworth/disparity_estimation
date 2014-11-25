#ifndef SEGMENTATION_REFINEMENT_H
#define SEGMENTATION_REFINEMENT_H

class RegionDescriptor;
#include <opencv2/core/core.hpp>
#include <vector>

//int split_region(const RegionDescriptor& descriptor, int min_size, std::back_insert_iterator<std::vector<RegionDescriptor>> it);
//void split_region_test();
//cv::Mat_<int> segmentation_iteration(std::vector<DisparityRegion>& regions, cv::Size size);


//interally used ones
void hsplit_region(const RegionDescriptor& descriptor, RegionDescriptor& first, RegionDescriptor& second, int split_threshold);
void vsplit_region(const RegionDescriptor& descriptor, RegionDescriptor& first, RegionDescriptor& second, int split_threshold);
cv::Point region_avg_point(const RegionDescriptor& descriptor);

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

//disparity specific
template<typename T>
cv::Mat_<int> segmentation_iteration(std::vector<T>& regions, cv::Size size)
{
	int min_size = 10;
	std::vector<T> created_regions;
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
}

#endif
