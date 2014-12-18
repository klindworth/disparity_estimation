#ifndef SEGMENTATION_IMAGE_H
#define SEGMENTATION_IMAGE_H

#include <vector>
#include <opencv2/core/core.hpp>

/**
 * @brief Structure for saving a vector full with RegionDescriptor or a derivative and a matrix with the label for each pixel.
 *
 * The class is inteded for saving a bunch of reg_type (e.g. RegionDescriptors), all of those should occupy all pixels of an image.
 * The structure also saves a matrix with the size of the image. Each coordinate of that matrix represents a segment id.
 * Those segments id are also the indices of the vector. That means if a region is at position (3,2) and it's position in the regions vector is 5,
 * The matrix in labels should contain the value 5 at the position (3,2).
 */
template<typename reg_type>
struct segmentation_image {
	typedef reg_type regions_type;
	std::size_t segment_count;
	cv::Size image_size;
	std::vector<regions_type> regions; ///< Vector with some kind of RegionDescriptor. The position in the vector is also the segment id
	cv::Mat_<int> labels; ///< Matrix with the size of the image. Each pixel positions contains the segment id of the segment, that the pixel belongs to
};

#endif
