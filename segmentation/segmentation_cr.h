#ifndef SEGMENTATION_CR_H
#define SEGMENTATION_CR_H

#include "segmentation.h"

class crslic_segmentation : public segmentation_algorithm {
public:
	crslic_segmentation(const segmentation_settings& psettings) : settings(psettings) {}
	int operator()(const cv::Mat& image, cv::Mat_<int>& labels) override;
	std::string cacheName() const override;

private:
	segmentation_settings settings;
};

#endif

