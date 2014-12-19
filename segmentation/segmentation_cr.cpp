#include "segmentation_cr.h"

#include "contourRelaxation/FeatureType.h"
#include "contourRelaxation/ContourRelaxation.h"
#include "contourRelaxation/InitializationFunctions.h"

int crslic_segmentation::operator()(const cv::Mat& image, cv::Mat_<int>& labels)
{
	float directCliqueCost = 0.3;
	unsigned int const iterations = 3;
	double const diagonalCliqueCost = directCliqueCost / sqrt(2);

	bool isColorImage = (image.channels() == 3);
	std::vector<FeatureType> features;
	if (isColorImage)
		features.push_back(Color);
	else
		features.push_back(Grayvalue);

	features.push_back(Compactness);

	ContourRelaxation<int> crslic_obj(features);
	cv::Mat labels_temp = createBlockInitialization<int>(image.size(), settings.superpixel_size, settings.superpixel_size);

	crslic_obj.setCompactnessData(settings.superpixel_compactness);

	if (isColorImage)
	{
		cv::Mat imageYCrCb;
		cv::cvtColor(image, imageYCrCb, CV_BGR2YCrCb);
		std::vector<cv::Mat> imageYCrCbChannels;
		cv::split(imageYCrCb, imageYCrCbChannels);

		crslic_obj.setColorData(imageYCrCbChannels[0], imageYCrCbChannels[1], imageYCrCbChannels[2]);
	}
	else
		crslic_obj.setGrayvalueData(image.clone());

	crslic_obj.relax(labels_temp, directCliqueCost, diagonalCliqueCost, iterations, labels);
	return 1+*(std::max_element(labels.begin(), labels.end()));
}

std::string crslic_segmentation::name() const
{
	return "crsuperpixel";
}

