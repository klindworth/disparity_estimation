#ifndef REGION_GROUND_TRUTH_H_
#define REGION_GROUND_TRUTH_H_

#include <cmath>
#include <array>
#include <opencv2/core/core.hpp>
#include <segmentation/intervals_algorithms.h>

class disparity_region;
class region_container;

class result_eps_calculator
{
private:
	static const int bins = 11;
	std::array<unsigned int, bins> counters;
	unsigned int total = 0;
public:
	result_eps_calculator();
	void operator()(short gt, short estimated);
	float epsilon_result(unsigned int eps) const;
	void print_to_stream(std::ostream& stream) const;
	void operator+=(const result_eps_calculator& other);
};

template<typename region_type, typename InsertIterator>
void region_ground_truth(const std::vector<region_type>& regions, const cv::Mat_<short>& gt, InsertIterator it)
{
	for(const region_type& cregion : regions)
	{
		int sum = 0;
		int count = 0;

		intervals::foreach_region_point(cregion.lineIntervals.begin(), cregion.lineIntervals.end(), [&](cv::Point pt){
			short value = gt(pt);
			if(value != 0)
			{
				sum += value;
				++count;
			}
		});

		*it = count > 0 ? std::round(sum/count) : 0;
		++it;
	}
}

result_eps_calculator get_region_comparision(const std::vector<disparity_region>& regions, const std::vector<short>& gt);
cv::Mat_<unsigned char> get_region_gt_error_image(const region_container& container, const std::vector<short>& gt);

std::ostream& operator<<(std::ostream& stream, const result_eps_calculator& res);

#endif
