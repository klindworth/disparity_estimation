#ifndef SEGMENTATION_SLIC_H
#define SEGMENTATION_SLIC_H

#include "segmentation.h"

class slic_segmentation : public segmentation_algorithm {
public:
	slic_segmentation(const segmentation_settings& psettings) : settings(psettings) {}
	int operator()(const cv::Mat& image, cv::Mat_<int>& labels) override;
	virtual std::string name() const override;

private:
	segmentation_settings settings;
};

#endif
