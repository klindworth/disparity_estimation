#ifndef SEGMENTATION_IMAGE_H
#define SEGMENTATION_IMAGE_H

#include <vector>
#include <opencv2/core/core.hpp>

template<typename reg_type>
struct segmentation_image {
	typedef reg_type regions_type;
	std::size_t segment_count;
	cv::Size image_size;
	std::vector<reg_type> regions;
	cv::Mat_<int> labels;
};

#endif
