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

cv::Mat estimated_occlusion_map(const cv::Mat& disparity, int subsample, bool invert = false)
{
	cv::Mat occ;
	if(!invert)
		occ = disparity::occlusion_stat<unsigned char>(disparity / subsample);
	else
	{
		cv::Mat dispN = cv::Mat(disparity.size(), CV_16SC1);
		const unsigned char* src = disparity.data;
		short* dst = dispN.ptr<short>(0);
		for(unsigned int i = 0; i < dispN.total(); ++i)
			*dst++ = - *src++ / subsample;

		occ  = disparity::occlusion_stat<short>(dispN);
	}

	unsigned char* ptr = occ.data;
	for(unsigned int i = 0; i < occ.total(); ++i)
	{
		if(*ptr > 0)
			*ptr = 255;
		++ptr;
	}
	return occ;

}

stereo_task::stereo_task(const std::string& filename)
{
	cv::FileStorage stream(filename + ".yml", cv::FileStorage::READ);
	if(stream.isOpened())
	{
		stream["name"] >> this->name;
		stream["dispRange"] >> this->dispRange;
		stream["groundTruthSubsampling"] >> this->groundTruthSubsampling;

		load_images((std::string)stream["left"], (std::string)stream["right"]);
		load_ground_truth((std::string)stream["groundLeft"], (std::string)stream["groundRight"]);
		load_occ((std::string)stream["occLeft"], (std::string)stream["occRight"]);

		init_single_tasks();
	}
	else
		std::cerr << "opening task failed: " << filename << std::endl;
}

stereo_task::stereo_task(const std::string& pname, const cv::Mat& pleft, const cv::Mat& pright, int dispRange) : left(pleft), right(pright), name(pname)
{
	cv::cvtColor(left,  leftGray,  CV_BGR2GRAY);
	cv::cvtColor(right, rightGray, CV_BGR2GRAY);
	this->dispRange = dispRange;

	init_single_tasks();
}

void stereo_task::load_images(const std::string& nameLeft, const std::string& nameRight)
{
	left  = cv::imread(nameLeft);
	right = cv::imread(nameRight);

	filenameLeft = nameLeft;
	filenameRight = nameRight;

	cv::cvtColor(left,  leftGray,  CV_BGR2GRAY);
	cv::cvtColor(right, rightGray, CV_BGR2GRAY);
}

void stereo_task::load_ground_truth(const std::string& nameGroundLeft, const std::string& nameGroundRight)
{
	cv::Mat ground_temp_left = cv::imread(nameGroundLeft, CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat ground_temp_right = cv::imread(nameGroundRight, CV_LOAD_IMAGE_GRAYSCALE);

	filenameGroundLeft = nameGroundLeft;
	filenameGroundRight =nameGroundRight;

	if(ground_temp_left.data)
	{
		ground_temp_left.convertTo(groundLeft, CV_16SC1);
		groundLeft *= -1;
	}
	else
		std::cout << "no ground truth data for left image" << std::endl;

	if(ground_temp_right.data)
		ground_temp_right.convertTo(groundRight, CV_16SC1);
	else
	{
		if(groundLeft.data)
			groundRight = disparity::warp_disparity<short>(groundLeft);
		std::cout << "no ground truth data for right image" << std::endl;
	}
}

stereo_task::stereo_task(const std::string& pname, const std::string& nameLeft, const std::string& nameRight, int dispRange) : name(pname)
{
	load_images(nameLeft, nameRight);
	this->dispRange = dispRange;

	init_single_tasks();
}

stereo_task::stereo_task(const std::string& pname, const std::string& nameLeft, const std::string& nameRight, const std::string& nameGroundLeft, const std::string& nameGroundRight, unsigned char subsamplingGroundTruth, int dispRange) : name(pname)
{
	groundTruthSubsampling = subsamplingGroundTruth;
	this->dispRange = dispRange;

	load_images(nameLeft, nameRight);
	load_ground_truth(nameGroundLeft, nameGroundRight);
	estimate_occ();

	init_single_tasks();
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
		occRight = estimated_occlusion_map(groundLeft, groundTruthSubsampling, true);


	if(groundRight.data && !occLeft.data)
		occLeft = estimated_occlusion_map(groundRight, groundTruthSubsampling, false);
}

stereo_task::stereo_task(const std::string& pname, const std::string& nameLeft, const std::string& nameRight, const std::string& nameGroundLeft, const std::string& nameGroundRight, const std::string& nameOccLeft, const std::string& nameOccRight, unsigned char subsamplingGroundTruth, int dispRange) : name(pname)
{
	groundTruthSubsampling = subsamplingGroundTruth;
	this->dispRange = dispRange;

	load_images(nameLeft, nameRight);
	load_ground_truth(nameGroundLeft, nameGroundRight);
	load_occ(nameOccLeft, nameOccRight);

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
	stream << "left" << task.filenameLeft << "right" << task.filenameRight << "dispRange" << task.dispRange;
	stream << "groundLeft" << task.filenameGroundLeft << "groundRight" << task.filenameGroundRight;
	stream << "occLeft" << task.filenameOccLeft << "occRight" << task.filenameOccRight;
	stream << "occ-left-valid" << (task.occLeft.data != 0) << "occ-right-valid" << (task.occRight.data != 0);
	stream << "groundTruthSubsampling" << task.groundTruthSubsampling;
	return stream;
}

void stereo_task::init_single_tasks()
{
	forward.name = name;
	forward.fullname = name + "-forward";
	forward.base = left;
	forward.baseGray = leftGray;
	forward.match = right;
	forward.matchGray = rightGray;
	forward.groundTruthSampling = groundTruthSubsampling;
	forward.occ = occLeft;
	forward.groundTruth = groundLeft;
	forward.dispMin = -dispRange+1;
	forward.dispMax = 0;
	forward.range = disparity_range(-dispRange+1, 0);

	backward.name = name;
	backward.fullname = name + "-backward";
	backward.base = right;
	backward.baseGray = rightGray;
	backward.match = left;
	backward.matchGray = leftGray;
	backward.groundTruthSampling = groundTruthSubsampling;
	backward.occ = occRight;
	backward.groundTruth = groundRight;
	backward.dispMin = 0;
	backward.dispMax = dispRange-1;
	backward.range = disparity_range(0, dispRange-1);
}

TaskTestSet::TaskTestSet(const std::string& filename) : task_collection(filename)
{
	cv::FileStorage stream(filename + ".yml", cv::FileStorage::READ);
	if(!stream.isOpened())
		std::cerr << "failed to open " << filename << std::endl;

	std::vector<std::string> taskFilenames;
	stream["tasks"] >> taskFilenames;

	for(std::string& cname : taskFilenames)
		tasks.push_back(stereo_task(cname));
}

cv::FileStorage& operator<<(cv::FileStorage& stream, const TaskTestSet& testset)
{
	stream << "tasks" << "[:";
	for(const stereo_task& ctask : testset.tasks)
		stream << ctask.name;
	stream << "]";

	return stream;
}

std::vector<std::string> get_filenames(const std::string& folder_name, const std::string& file_extension)
{
	std::vector<std::string> result;

	boost::filesystem::path cpath(folder_name);
	try
	{
		if (boost::filesystem::exists(cpath))
		{
			if (boost::filesystem::is_directory(cpath))
			{
				auto it = boost::filesystem::directory_iterator(cpath);
				auto end = boost::filesystem::directory_iterator();

				for(; it != end; ++it)
				{
					boost::filesystem::path cpath = *it;
					if(boost::filesystem::is_regular_file(cpath) && cpath.extension() == file_extension)
						result.push_back(cpath.filename().string());
				}
			}
		}
		else
			std::cerr << folder_name + " doesn't exist" << std::endl;
	}
	catch (const boost::filesystem::filesystem_error& ex)
	{
		std::cout << ex.what() << std::endl;
	}

	return result;
}

std::string build_filename(const boost::filesystem::path& fpath, const std::string& filename)
{
	boost::filesystem::path cpath = fpath;
	cpath /= filename;

	return cpath.string();
}

bool check_completeness(const std::vector<std::string>& filenames, const boost::filesystem::path& fpath)
{
	for(const std::string& cfilename : filenames)
	{
		boost::filesystem::path cpath = fpath;
		cpath /= cfilename;

		if(!boost::filesystem::exists(cpath))
			return false;
	}

	return true;
}

void enforce_completeness(std::vector<std::string>& filenames, const boost::filesystem::path& fpath)
{
	filenames.erase(std::remove_if(filenames.begin(), filenames.end(), [=](const std::string& cfilename) {
		boost::filesystem::path cpath = fpath;
		cpath /= cfilename;

		return !boost::filesystem::exists(cpath);
	}), filenames.end());
}

folder_testset::folder_testset(const std::string& filename) : task_collection(filename)
{
	cv::FileStorage stream(filename + ".yml", cv::FileStorage::READ);
	if(!stream.isOpened())
		std::cerr << "failed to open " << filename << std::endl;

	//file stuff
	stream["left"] >> left;
	stream["right"] >> right;
	stream["groundLeft"] >> dispLeft;
	stream["groundRight"] >> dispRight;
	stream["fileextension_images"] >> fileextension_images;
	stream["fileextension_gt"] >> fileextension_gt;

	std::vector<std::string> filenames = get_filenames(left, fileextension_images);

	bool require_completeness = true;
	if(require_completeness)
	{
		check_completeness(filenames, right);
		if(!dispLeft.empty())
			enforce_completeness(filenames, dispLeft);
		if(!dispRight.empty())
			enforce_completeness(filenames, dispRight);
	}
	else
	{
		if(! check_completeness(filenames, right))
			std::cerr << "right folder not complete" << std::endl;
		if(!dispLeft.empty())
		{
			if(!check_completeness(filenames, dispLeft))
				std::cerr << "dispLeft folder not complete" << std::endl;
		}
		if(!dispRight.empty())
		{
			if(!check_completeness(filenames, dispRight))
				std::cerr << "dispRight folder not complete" << std::endl;
		}
	}

	std::cout << filenames.size() << " files found" << std::endl;

	//settings
	stream["dispRange"] >> dispRange;
	stream["groundTruthSubsampling"] >> subsamplingGroundTruth;

	//create tasks
	for(const std::string& cfilename : filenames)
	{
		std::string file_left = build_filename(left, cfilename);
		std::string file_right = build_filename(right, cfilename);
		std::string file_dispLeft = !dispLeft.empty() ? build_filename(dispLeft, cfilename) : "";
		std::string file_dispRight = !dispRight.empty() ? build_filename(dispRight, cfilename) : "";

		std::cout << cfilename << std::endl;

		tasks.emplace_back(cfilename, file_left, file_right, file_dispLeft, file_dispRight, subsamplingGroundTruth, dispRange);
	}

	std::cout << tasks.size() << " tasks found" << std::endl;
}

