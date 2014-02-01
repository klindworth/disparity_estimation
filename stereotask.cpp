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

#include "genericfunctions.h"
#include "disparity_utils.h"

cv::Mat estimatedOcclusionMap(const cv::Mat& disparity, int subsample, bool invert = false)
{
	cv::Mat occ;
	if(!invert)
		occ = occlusionStat<unsigned char>(disparity / subsample);
	else
	{
		cv::Mat dispN = cv::Mat(disparity.size(), CV_16SC1);
		const unsigned char* src = disparity.data;
		short* dst = dispN.ptr<short>(0);
		for(unsigned int i = 0; i < dispN.total(); ++i)
			*dst++ = - *src++ / subsample;

		occ  = occlusionStat<short>(dispN);
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

StereoTask::StereoTask(const std::string& filename)
{
	cv::FileStorage stream(filename + ".yml", cv::FileStorage::READ);
	if(stream.isOpened())
	{
		stream["name"] >> this->name;
		stream["dispRange"] >> this->dispRange;
		stream["groundTruthSubsampling"] >> this->groundTruthSubsampling;

		loadImages((std::string)stream["left"], (std::string)stream["right"]);
		loadGroundTruth((std::string)stream["groundLeft"], (std::string)stream["groundRight"]);
		loadOcc((std::string)stream["occLeft"], (std::string)stream["occRight"]);

		initSingleTasks();
	}
	else
		std::cerr << "opening task failed: " << filename << std::endl;
}

StereoTask::StereoTask(const std::string& pname, const cv::Mat& pleft, const cv::Mat& pright, int dispRange) : left(pleft), right(pright), name(pname)
{
	cv::cvtColor(left,  leftGray,  CV_BGR2GRAY);
	cv::cvtColor(right, rightGray, CV_BGR2GRAY);
	this->dispRange = dispRange;

	initSingleTasks();
}

void StereoTask::loadImages(const std::string& nameLeft, const std::string& nameRight)
{
	left  = cv::imread(nameLeft);
	right = cv::imread(nameRight);

	filenameLeft = nameLeft;
	filenameRight = nameRight;

	cv::cvtColor(left,  leftGray,  CV_BGR2GRAY);
	cv::cvtColor(right, rightGray, CV_BGR2GRAY);
}

void StereoTask::loadGroundTruth(const std::string& nameGroundLeft, const std::string& nameGroundRight)
{
	cv::Mat tempLeft  = cv::imread(nameGroundLeft);
	cv::Mat tempRight = cv::imread(nameGroundRight);

	filenameGroundLeft = nameGroundLeft;
	filenameGroundRight =nameGroundRight;

	if(tempLeft.data)
		cv::cvtColor(tempLeft,  groundLeft,  CV_BGR2GRAY);
	else
		std::cout << "no ground truth data for left image" << std::endl;

	if(tempRight.data)
		cv::cvtColor(tempRight, groundRight, CV_BGR2GRAY);
	else
		std::cout << "no ground truth data for right image" << std::endl;
}

StereoTask::StereoTask(const std::string& pname, const std::string& nameLeft, const std::string& nameRight, int dispRange) : name(pname)
{
	loadImages(nameLeft, nameRight);
	this->dispRange = dispRange;

	initSingleTasks();
}

StereoTask::StereoTask(const std::string& pname, const std::string& nameLeft, const std::string& nameRight, const std::string& nameGroundLeft, const std::string& nameGroundRight, unsigned char subsamplingGroundTruth, int dispRange) : name(pname)
{
	groundTruthSubsampling = subsamplingGroundTruth;
	this->dispRange = dispRange;

	loadImages(nameLeft, nameRight);
	loadGroundTruth(nameGroundLeft, nameGroundRight);
	estimateOcc();

	initSingleTasks();
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

void StereoTask::loadOcc(const std::string& nameOccLeft, const std::string& nameOccRight)
{
	cv::Mat tempLeft  = cv::imread(nameOccLeft);
	cv::Mat tempRight = cv::imread(nameOccRight);

	filenameOccLeft = nameOccLeft;
	filenameOccRight = nameOccRight;

	if(tempLeft.data)
		cv::cvtColor(tempLeft,  occLeft,  CV_BGR2GRAY);
	else
		std::cout << "no occ data for left image" << std::endl;
	if(tempRight.data)
		cv::cvtColor(tempRight, occRight, CV_BGR2GRAY);
	else
		std::cout << "no occ data for right image" << std::endl;

	estimateOcc();
}

void StereoTask::estimateOcc()
{
	if(groundLeft.data && !occRight.data)
		occRight = estimatedOcclusionMap(groundLeft, groundTruthSubsampling, true);


	if(groundRight.data && !occLeft.data)
		occLeft = estimatedOcclusionMap(groundRight, groundTruthSubsampling, false);
}

StereoTask::StereoTask(const std::string& pname, const std::string& nameLeft, const std::string& nameRight, const std::string& nameGroundLeft, const std::string& nameGroundRight, const std::string& nameOccLeft, const std::string& nameOccRight, unsigned char subsamplingGroundTruth, int dispRange) : name(pname)
{
	groundTruthSubsampling = subsamplingGroundTruth;
	this->dispRange = dispRange;

	loadImages(nameLeft, nameRight);
	loadGroundTruth(nameGroundLeft, nameGroundRight);
	loadOcc(nameOccLeft, nameOccRight);

	initSingleTasks();
}

bool StereoTask::valid() const
{
	if(left.size[0] != right.size[0] || left.size[1] != right.size[1] || !right.data || !left.data)
		return false;
	else
		return true;
}

cv::FileStorage& operator<<(cv::FileStorage& stream, const StereoTask& task)
{
	stream << "name" << task.name;
	stream << "left" << task.filenameLeft << "right" << task.filenameRight << "dispRange" << task.dispRange;
	stream << "groundLeft" << task.filenameGroundLeft << "groundRight" << task.filenameGroundRight;
	stream << "occLeft" << task.filenameOccLeft << "occRight" << task.filenameOccRight;
	stream << "occ-left-valid" << (task.occLeft.data != 0) << "occ-right-valid" << (task.occRight.data != 0);
	stream << "groundTruthSubsampling" << task.groundTruthSubsampling;
	return stream;
}

void StereoTask::initSingleTasks()
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
}

TaskTestSet::TaskTestSet(const std::string& filename) : name(filename)
{
	cv::FileStorage stream(filename + ".yml", cv::FileStorage::READ);
	if(!stream.isOpened())
		std::cerr << "failed to open " << filename << std::endl;

	std::vector<std::string> taskFilenames;
	stream["tasks"] >> taskFilenames;

	for(std::string& cname : taskFilenames)
		tasks.push_back(StereoTask(cname));
}

cv::FileStorage& operator<<(cv::FileStorage& stream, const TaskTestSet& testset)
{
	stream << "tasks" << "[:";
	for(const StereoTask& ctask : testset.tasks)
		stream << ctask.name;
	stream << "]";

	return stream;
}

