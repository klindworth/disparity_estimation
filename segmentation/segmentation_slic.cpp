#include "segmentation_slic.h"

#include "SLIC_CV/slic_adaptor.h"

int slic_segmentation::operator()(const cv::Mat& image, cv::Mat_<int>& labels) {
	return slicSuperpixels(image, labels, settings.superpixel_size, settings.superpixel_compactness);
}

std::string slic_segmentation::cacheName() const
{
	std::stringstream stream;
	stream << "superpixel_" << settings.superpixel_size;
	return stream.str();
}
