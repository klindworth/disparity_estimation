#include "segmentation_msslic.h"

#include "meanshift_cv/msImageProcessor.h"
#include "SLIC_CV/slic_adaptor.h"
#include "region_descriptor.h"
#include "region_descriptor_algorithms.h"
#include "segmentation_algorithms.h"

#include <iostream>
#include <iterator>

int segmentImage2(cv::Mat& src, cv::Mat& labels_dst, int spatial_variance, float color_variance)
{
	msImageProcessor proc;
	proc.DefineImage(src.data, (src.channels() == 3 ? COLOR : GRAYSCALE), src.rows, src.cols);
	proc.Segment(spatial_variance,color_variance, 1, NO_SPEEDUP);//HIGH_SPEEDUP, MED_SPEEDUP, NO_SPEEDUP; high: speedupThreshold setzen!
	//cv::Mat res = cv::Mat(src.size(), src.type());
	//proc.GetResults(res.data);

	labels_dst = cv::Mat(src.size(), CV_32SC1);
	int regions_count = proc.GetRegionsModified(labels_dst.data);

	return regions_count;
}

int mssuperpixel_segmentation::operator()(const cv::Mat& image, cv::Mat_<int>& labels) {
	int regions_count = slicSuperpixels(image, labels, settings.superpixel_size, settings.superpixel_compactness);
	std::vector<RegionDescriptor> regions(regions_count);
	fillRegionDescriptors(regions.begin(), regions.end(), labels);

	superpixel = labels.clone();
	regions_count_superpixel = regions_count;

	calculate_all_average_colors(image, regions.begin(), regions.end());

	std::pair<int,int> slic_size = get_slic_seeds(image.cols, image.rows, settings.superpixel_size);
	cv::Mat slic_image_lab(slic_size.first, slic_size.second, CV_64FC3);
	cv::Vec3d *dst_ptr = slic_image_lab.ptr<cv::Vec3d>(0);

	std::cout << "slic_size : width:" << slic_size.second << ", height: " << slic_size.first << std::endl;
	std::cout << "complete: " << slic_image_lab.total() << std::endl;
	std::cout << "regions: " << regions.size() << std::endl;

	int yex = labels.rows / (slic_size.first);
	int xex = labels.cols / (slic_size.second);

	for(int i = 0; i < slic_size.first; ++i)
	{
		for(int j = 0; j < slic_size.second; ++j)
		{
			int idx = labels.at<int>(i*yex+yex/2,j*xex+xex/2);
			*dst_ptr++ = regions[idx].average_color;
		}
	}

	checkLabelsIntervalsInvariant(regions.begin(), regions.end(), labels);

	cv::Mat slic_image = lab_to_bgr(slic_image_lab);
	cv::Mat msResult;
	int mscount = segmentImage2(slic_image, msResult, settings.spatial_var, settings.color_var);
	//cv::imshow("ms", getWrongColorSegmentationImage(msResult, mscount));
	std::cout << "mscount: " << mscount << std::endl;

	std::shared_ptr<fusion_work_data> data = std::shared_ptr<fusion_work_data>( new fusion_work_data(regions.size()) );
	int *src_ptr = msResult.ptr<int>(0);
	std::vector<int> mapping(mscount, -1);
	for(int i = 0; i < slic_size.first; ++i)
	{
		for(int j = 0; j < slic_size.second; ++j)
		{
			int idx = labels.at<int>(i*yex+yex/2,j*xex+xex/2);
			int label = *src_ptr++;
			int fusion_idx = mapping[label];

			if(fusion_idx == -1 && data->active[idx])
				mapping[label] = idx;
			else if(idx != fusion_idx && data->active[idx])
			{
				while(!data->active[fusion_idx])
					fusion_idx = data->fused_with[fusion_idx];

				assert(fusion_idx >= 0 && fusion_idx < regions.size());
				data->fused_with[idx] = fusion_idx;
				if(! data->active[fusion_idx])
					std::cout << "master of master: " << data->fused_with[fusion_idx] << std::endl;
				assert(data->active[fusion_idx]);
				data->active[idx] = false;
				data->fused[fusion_idx].push_back(idx);
			}
		}
	}
	fusion_data = data;

	fuse(*data, regions, labels);

	return regions.size();
}

bool mssuperpixel_segmentation::cacheAllowed() const {
	return false;
}

std::string mssuperpixel_segmentation::cacheName() const
{
	return "mssuperpixel";
}

