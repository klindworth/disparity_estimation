#include "segmentation_refinement.h"

#include "region.h"

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
