#include "segmentation_ms.h"

//include "msImageProcessor.h"

/**
 * @param src Image to segment
 * @param labels_dst cv::Mat where the (int) labels will be written in
 * @return Number of different labels
 */
//7,4.0
/*int mean_shift_segmentation(const cv::Mat& src, cv::Mat& labels_dst, int spatial_variance, float color_variance, int minsize)
{
	msImageProcessor proc;
	proc.DefineImage(src.data, (src.channels() == 3 ? COLOR : GRAYSCALE), src.rows, src.cols);
	proc.Segment(spatial_variance,color_variance, minsize, MED_SPEEDUP);//HIGH_SPEEDUP, MED_SPEEDUP, NO_SPEEDUP; high: speedupThreshold setzen!
	//cv::Mat res = cv::Mat(src.size(), src.type());
	//proc.GetResults(res.data);

	labels_dst = cv::Mat(src.size(), CV_32SC1);
	int regions_count = proc.GetRegionsModified(labels_dst.data);

	return regions_count;
}

int meanshift_segmentation::operator()(const cv::Mat& image, cv::Mat_<int>& labels) {
	return mean_shift_segmentation(image, labels, settings.spatial_var, settings.color_var, 20);
}

std::string meanshift_segmentation::cacheName() const
{
	std::stringstream stream;
	stream << "meanshift_" << settings.spatial_var;
	return stream.str();
}
*/
