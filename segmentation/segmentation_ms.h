#ifndef SEGMENTATION_MS_H
#define SEGMENTATION_MS_H

#include "segmentation.h"

class meanshift_segmentation : public segmentation_algorithm {
public:
	meanshift_segmentation(const segmentation_settings& psettings) : settings(psettings) {}
	int operator()(const cv::Mat& image, cv::Mat_<int>& labels) override;
	std::string name() const override;

private:
	segmentation_settings settings;
};

#endif
