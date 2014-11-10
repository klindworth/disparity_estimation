#ifndef SEGMENTATION_REFINEMENT_H
#define SEGMENTATION_REFINEMENT_H

class DisparityRegion;
#include <opencv2/core/core.hpp>
#include <vector>

//int split_region(const RegionDescriptor& descriptor, int min_size, std::back_insert_iterator<std::vector<RegionDescriptor>> it);
void split_region_test();
cv::Mat_<int> segmentation_iteration(std::vector<DisparityRegion>& regions, cv::Size size);

#endif
