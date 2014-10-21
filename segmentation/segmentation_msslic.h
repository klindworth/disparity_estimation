#ifndef SEGMENTATION_MSSLIC_H
#define SEGMENTATION_MSSLIC_H

#include "segmentation.h"

class mssuperpixel_segmentation : public segmentation_algorithm {
public:
	mssuperpixel_segmentation(const segmentation_settings& psettings) : settings(psettings) {}
	virtual int operator()(const cv::Mat& image, cv::Mat_<int>& labels);
	virtual std::string cacheName() const;
	virtual bool cacheAllowed() const;
	virtual bool refinementPossible();
	virtual void refine(RegionContainer &container);

private:
	std::shared_ptr<fusion_work_data> fusion_data;
	segmentation_settings settings;
	cv::Mat_<int> superpixel;
	int regions_count_superpixel;
};

#endif
