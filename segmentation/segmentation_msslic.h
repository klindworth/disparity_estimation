#ifndef SEGMENTATION_MSSLIC_H
#define SEGMENTATION_MSSLIC_H

#include "segmentation.h"

class mssuperpixel_segmentation : public segmentation_algorithm {
public:
	mssuperpixel_segmentation(const segmentation_settings& psettings) : settings(psettings) {}
	int operator()(const cv::Mat& image, cv::Mat_<int>& labels) override;
	std::string name() const override;

private:
	std::shared_ptr<fusion_work_data> fusion_data;
	segmentation_settings settings;
	cv::Mat_<int> superpixel;
	int regions_count_superpixel;
};

#endif
