#include "segmentation_ms.h"

#include "meanshift_cv/msImageProcessor.h"
#include "meanshift_cv/ms_cv.h"

int meanshift_segmentation::operator()(const cv::Mat& image, cv::Mat_<int>& labels) {
	return mean_shift_segmentation(image, labels, settings.spatial_var, settings.color_var, 20);
}

std::string meanshift_segmentation::name() const
{
	return "meanshift";
}

