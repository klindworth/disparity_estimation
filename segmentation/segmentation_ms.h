#ifndef SEGMENTATION_MS_H
#define SEGMENTATION_MS_H

#include "segmentation.h"

class meanshift_segmentation : public segmentation_algorithm {
public:
	meanshift_segmentation(const segmentation_settings& psettings) : settings(psettings) {}
	virtual int operator()(const cv::Mat& image, cv::Mat_<int>& labels);
	virtual std::string cacheName() const;

private:
	segmentation_settings settings;
};

#endif
