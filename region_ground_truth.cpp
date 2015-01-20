#include "region_ground_truth.h"
#include "disparity_region.h"

result_eps_calculator::result_eps_calculator()
{
	std::fill(counters.begin(), counters.end(), 0);
	total = 0;
}

void result_eps_calculator::operator()(short gt, short estimated)
{
	++total;
	unsigned int diff = std::abs(std::abs(estimated) - std::abs(gt));

	for(unsigned int i = diff; i < bins; ++i)
		++counters[i];
}

float result_eps_calculator::epsilon_result(unsigned int eps) const
{
	if(eps < bins)
		return (float)counters[eps]/total;
	else
		return 1.0f;
}

void result_eps_calculator::print_to_stream(std::ostream& stream) const
{
	stream << "correct: " << (float)counters[0]/total << ", approx5: " << (float)counters[4]/total << ", approx10: " << (float)counters[9]/total;
}

void result_eps_calculator::operator+=(const result_eps_calculator& other)
{
	for(int i = 0; i < bins; ++i)
		counters[i] += other.counters[i];
	total += other.total;
}

result_eps_calculator get_region_comparision(const std::vector<disparity_region>& regions, const std::vector<short>& gt)
{
	result_eps_calculator diff_calc;
	for(std::size_t i = 0; i < regions.size(); ++i)
	{
		if(gt[i] != 0)
			diff_calc(gt[i], regions[i].disparity);
	}
	return diff_calc;
}

cv::Mat_<unsigned char> get_region_gt_error_image(const region_container& container, const std::vector<short>& gt)
{
	cv::Mat_<unsigned char> diff_image(container.image_size, 0);
	for(std::size_t i = 0; i < container.regions.size(); ++i)
	{
		if(gt[i] != 0)
			intervals::set_region_value<unsigned char>(diff_image, container.regions[i].lineIntervals, std::abs(std::abs(gt[i]) - std::abs(container.regions[i].disparity)));
	}

	return diff_image;
}

std::ostream& operator<<(std::ostream& stream, const result_eps_calculator& res)
{
	res.print_to_stream(stream);
	return stream;
}
