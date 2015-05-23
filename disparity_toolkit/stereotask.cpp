/*
Copyright (c) 2013, Kai Klindworth
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "stereotask.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <boost/filesystem.hpp>
#include <algorithm>

#include "genericfunctions.h"
#include "disparity_utils.h"

cv::Mat_<unsigned char> estimated_occlusion_map(const disparity_map& disparity)
{
	cv::Mat_<unsigned char> occ = disparity::occlusion_stat<short>(disparity, 1.0f/disparity.sampling);

	unsigned char* ptr = occ.data;
	for(unsigned int i = 0; i < occ.total(); ++i)
	{
		*ptr = *ptr > 0 ? 255 : 0;
		++ptr;
	}
	return occ;
}

template<typename storage_type>
std::string determine_path(const storage_type& storage, const std::string& name, const boost::filesystem::path& base_path)
{
	std::string temp;
	storage[name] >> temp;
	if(temp == "")
		return "";
	else
	{
		boost::filesystem::path result_path = base_path / temp;
		return result_path.string();
	}
}

template<typename storage_type>
stereo_task create_from_storage(const storage_type& fs, const std::string& base_pathname = "")
{
	std::string name;
	int dispRange, groundSubsampling;

	fs["dispRange"] >> dispRange;
	fs["groundTruthSubsampling"] >> groundSubsampling;
	fs["name"] >> name;

	boost::filesystem::path base_path = base_pathname;

	std::string left  = determine_path(fs, "left",  base_path);
	std::string right = determine_path(fs, "right", base_path);
	std::string leftGround  = determine_path(fs, "groundLeft",  base_path);
	std::string rightGround = determine_path(fs, "groundRight", base_path);
	std::string leftOcc  = determine_path(fs, "occLeft",  base_path);
	std::string rightOcc = determine_path(fs, "occRight", base_path);

	return stereo_task(name, left, right, leftGround, rightGround, leftOcc, rightOcc, groundSubsampling, dispRange);
}

stereo_task stereo_task::load_from_file(const std::string& filename)
{
	cv::FileStorage fs(filename + ".yml", cv::FileStorage::READ);
	if(fs.isOpened())
		return create_from_storage(fs);
	else
		throw std::runtime_error("opening task failed: " + filename);
}

stereo_task stereo_task::load_from_filestorage(const cv::FileStorage& fs, const std::string& base_path)
{
	return create_from_storage(fs, base_path);
}

stereo_task stereo_task::load_from_filestorage(const cv::FileNode& fs, const std::string& base_path)
{
	return create_from_storage(fs, base_path);
}

void stereo_task::load_images(const std::string& nameLeft, const std::string& nameRight)
{
	left  = cv::imread(nameLeft);
	right = cv::imread(nameRight);

	if(!left.data || !right.data)
		throw std::runtime_error("image couldn't be loaded: " + nameLeft + " or " + nameRight);

	filenameLeft  = nameLeft;
	filenameRight = nameRight;

	cv::cvtColor(left,  leftGray,  CV_BGR2GRAY);
	cv::cvtColor(right, rightGray, CV_BGR2GRAY);
}

void stereo_task::load_ground_truth(const std::string& nameGroundLeft, const std::string& nameGroundRight, int subsampling)
{
	cv::Mat ground_temp_left  = cv::imread(nameGroundLeft,  CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat ground_temp_right = cv::imread(nameGroundRight, CV_LOAD_IMAGE_GRAYSCALE);

	filenameGroundLeft  = nameGroundLeft;
	filenameGroundRight = nameGroundRight;

	if(ground_temp_left.data)
	{
		ground_temp_left.convertTo(groundLeft, CV_16SC1);
		groundLeft *= -1;
		groundLeft.sampling = subsampling;
	}
	else
		std::cout << "no ground truth data for left image" << std::endl;

	if(ground_temp_right.data)
	{
		ground_temp_right.convertTo(groundRight, CV_16SC1);
		groundRight.sampling = subsampling;
	}
	else
	{
		if(groundLeft.data)
			groundRight = disparity::warp_disparity<short>(groundLeft);
		std::cout << "no ground truth data for right image" << std::endl;
	}
}

/*void StereoTask::write(cv::FileNode& node) const
{
	node << "left" << filenameLeft << "right" << filenameRight << "dispRange" << dispRange;
	node << "groundLeft" << filenameGroundLeft << "groundRight" << filenameGroundRight;
	node << "occLeft" << filenameOccLeft << "occRight" << filenameOccRight;
	node << "occ-left-valid" << (occLeft.data != 0) << "occ-right-valid" << (occRight.data != 0);
	node << "groundTruthSubsampling" << groundTruthSubsampling;
	return node;
}*/

void stereo_task::load_occ(const std::string& nameOccLeft, const std::string& nameOccRight)
{
	occLeft  = cv::imread(nameOccLeft, CV_LOAD_IMAGE_GRAYSCALE);
	occRight = cv::imread(nameOccRight, CV_LOAD_IMAGE_GRAYSCALE);

	filenameOccLeft = nameOccLeft;
	filenameOccRight = nameOccRight;

	if(!occLeft.data)
		std::cout << "no occ data for left image" << std::endl;
	if(!occRight.data)
		std::cout << "no occ data for right image" << std::endl;

	estimate_occ();
}

void stereo_task::estimate_occ()
{
	if(groundLeft.data && !occRight.data)
		occRight = estimated_occlusion_map(groundLeft);


	if(groundRight.data && !occLeft.data)
		occLeft = estimated_occlusion_map(groundRight);
}

stereo_task::stereo_task(const std::string& pname, const std::string& nameLeft, const std::string& nameRight, const std::string& nameGroundLeft, const std::string& nameGroundRight, const std::string& nameOccLeft, const std::string& nameOccRight, unsigned char subsamplingGroundTruth, int dispRange)
	: name(pname), disp_range(dispRange), forward(disparity_range(-dispRange+1, 0)), backward(disparity_range(0, dispRange-1))
{
	ground_truth_sampling = subsamplingGroundTruth;

	load_images(nameLeft, nameRight);
	load_ground_truth(nameGroundLeft, nameGroundRight, ground_truth_sampling);
	load_occ(nameOccLeft, nameOccRight);

	estimate_occ();

	init_single_tasks();
}

bool stereo_task::valid() const
{
	if(left.size[0] != right.size[0] || left.size[1] != right.size[1] || !right.data || !left.data)
		return false;
	else
		return true;
}

cv::FileStorage& operator<<(cv::FileStorage& stream, const stereo_task& task)
{
	stream << "name" << task.name;
	stream << "left" << task.filenameLeft << "right" << task.filenameRight << "dispRange" << task.disp_range;
	stream << "groundLeft" << task.filenameGroundLeft << "groundRight" << task.filenameGroundRight;
	stream << "occLeft" << task.filenameOccLeft << "occRight" << task.filenameOccRight;
	stream << "occ-left-valid" << (task.occLeft.data != 0) << "occ-right-valid" << (task.occRight.data != 0);
	stream << "groundTruthSubsampling" << task.ground_truth_sampling;
	return stream;
}

void stereo_task::init_single_tasks()
{
	//forward.name = name;
	//forward.fullname = name + "-forward";
	forward.base = left;
	forward.baseGray = leftGray;
	forward.match = right;
	forward.matchGray = rightGray;
	forward.occ = occLeft;
	forward.groundTruth = groundLeft;

	//backward.name = name;
	//backward.fullname = name + "-backward";
	backward.base = right;
	backward.baseGray = rightGray;
	backward.match = left;
	backward.matchGray = leftGray;
	backward.occ = occRight;
	backward.groundTruth = groundRight;
}



