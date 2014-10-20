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

#ifndef STEREOTASK_H
#define STEREOTASK_H

#include <opencv2/core/core.hpp>
#include <array>
#include <iterator>
#include <string>

class StereoSingleTask
{
public:
	std::string name, fullname;
	cv::Mat base, match, baseGray, matchGray, groundTruth, occ;
	int dispMin, dispMax;
	unsigned char groundTruthSampling;
};

/// Class for saving a StereoTask, forward and backward
class StereoTask
{
public:
	StereoTask(const std::string& name, const cv::Mat& pleft, const cv::Mat& pright, int dispRange);
	StereoTask(const std::string& name, const std::string &nameLeft, const std::string &nameRight, int dispRange);
	StereoTask(const std::string& name, const std::string &nameLeft, const std::string &nameRight, const std::string &nameGroundLeft, const std::string &nameGroundRight, unsigned char subsamplingGroundTruth, int dispRange);
	StereoTask(const std::string& name, const std::string& nameLeft, const std::string& nameRight, const std::string& nameGroundLeft, const std::string& nameGroundRight, const std::string& nameOccLeft, const std::string& nameOccRight, unsigned char subsamplingGroundTruth, int dispRange);
	StereoTask(const std::string& filename);
	//void write(cv::FileNode& node) const;
	bool valid() const;

	cv::Mat left, right, leftGray, rightGray, groundLeft, groundRight, occLeft, occRight, algoLeft, algoRight;
	int dispRange;
	unsigned char groundTruthSubsampling;
	std::string name, filenameLeft, filenameRight, filenameGroundLeft, filenameGroundRight, filenameOccLeft, filenameOccRight;

	StereoSingleTask forward, backward;
private:
	double errorMeasureInternal(const cv::Mat& disparity, const cv::Mat &groundTruth, const cv::Mat& occ, unsigned char subsamplingDisparity, unsigned char subsamplingGroundTruth);
	cv::Mat diffImageInternal(const cv::Mat& disparity, const cv::Mat& groundTruth, const cv::Mat& occ, unsigned char subsamplingDisparity, unsigned char subsamplingGroundTruth);
	void initSingleTasks();
	void loadImages(const std::string& nameLeft, const std::string& nameRight);
	void loadGroundTruth(const std::string &nameGroundLeft, const std::string &nameGroundRight);
	void loadOcc(const std::string& nameOccLeft, const std::string& nameOccRight);
	void estimateOcc();
};

template<typename charT, typename traits>
inline std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& stream, const StereoTask& task)
{
	stream << "left: " << task.filenameLeft << "\nright: " << task.filenameRight << "\ndispRange: " << task.dispRange;
	stream << "\ngroundLeft: " << task.filenameGroundLeft << "\ngroundRight: " << task.filenameGroundRight;
	stream << "\noccLeft: " << task.filenameOccLeft << "\noccRight: " << task.filenameOccRight;
	stream << "\nocc-left-valid: " << (task.occLeft.data != 0) << "\nocc-right-valid: " << (task.occRight.data != 0);
	return stream;
}

cv::FileStorage& operator<<(cv::FileStorage& stream, const StereoTask& task);

/// Container for StereoTasks. It can read a testset file, which contain the filenames for the single StereoTasks
class TaskTestSet
{
public:
	TaskTestSet () {}
	TaskTestSet(const std::string& filename);
	std::string name;
	std::vector<StereoTask> tasks;
};

cv::FileStorage& operator<<(cv::FileStorage& stream, const TaskTestSet& testset);

#endif // STEREOTASK_H
