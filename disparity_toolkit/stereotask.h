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

#include "disparity_map.h"
#include <opencv2/core/core.hpp>
#include <string>
#include "disparity_range.h"

class single_stereo_task
{
public:
	//single_stereo_task(const disparity_range& prange) : range(prange) {}
	//std::string name, fullname;
	cv::Mat base, match, baseGray, matchGray;
	disparity_map groundTruth;
	cv::Mat_<unsigned char> occ;
	disparity_range range;
};

/// Class for saving a specific stereo processing task. It can be loaded from a yaml file, or you construct one by passing all the image paths
class stereo_task
{
public:
	//stereo_task(const std::string& name, const cv::Mat& pleft, const cv::Mat& pright, int dispRange);
	//stereo_task(const std::string& name, const std::string &nameLeft, const std::string &nameRight, int dispRange);
	//stereo_task(const std::string& name, const std::string &nameLeft, const std::string &nameRight, const std::string &nameGroundLeft, const std::string &nameGroundRight, unsigned char subsamplingGroundTruth, int dispRange);
	stereo_task(const std::string& name, const std::string& nameLeft, const std::string& nameRight, const std::string& nameGroundLeft, const std::string& nameGroundRight, const std::string& nameOccLeft, const std::string& nameOccRight, unsigned char subsamplingGroundTruth, int dispRange);
	//stereo_task(const std::string& filename);
	//void write(cv::FileNode& node) const;
	bool valid() const;

	cv::Mat left, right, leftGray, rightGray, occLeft, occRight, algoLeft, algoRight;
	disparity_map groundLeft, groundRight;
	int dispRange;
	unsigned char groundTruthSubsampling;
	std::string name, filenameLeft, filenameRight, filenameGroundLeft, filenameGroundRight, filenameOccLeft, filenameOccRight;

	single_stereo_task forward, backward;

	//static construction functions
	//static stereo_task construct(const std::string& name, const cv::Mat& pleft, const cv::Mat& pright, int dispRange);
	//static stereo_task construct(const std::string& name, const std::string &nameLeft, const std::string &nameRight, int dispRange);
	static stereo_task load_from_file(const std::string& filename);
	static stereo_task load_from_filestorage(const cv::FileStorage& fs, const std::string& base_path = "");
	static stereo_task load_from_filestorage(const cv::FileNode& fs, const std::string& base_path = "");

private:
	//void load_from_storage(const cv::FileStorage& fs);

	void init_single_tasks();
	void load_images(const std::string& nameLeft, const std::string& nameRight);
	void load_ground_truth(const std::string &nameGroundLeft, const std::string &nameGroundRight, int subsampling);
	void load_occ(const std::string& nameOccLeft, const std::string& nameOccRight);
	void estimate_occ();
};

template<typename charT, typename traits>
inline std::basic_ostream<charT, traits>& operator<<(std::basic_ostream<charT, traits>& stream, const stereo_task& task)
{
	stream << "left: " << task.filenameLeft << "\nright: " << task.filenameRight << "\ndispRange: " << task.dispRange;
	stream << "\ngroundLeft: " << task.filenameGroundLeft << "\ngroundRight: " << task.filenameGroundRight;
	stream << "\noccLeft: " << task.filenameOccLeft << "\noccRight: " << task.filenameOccRight;
	stream << "\nocc-left-valid: " << (task.occLeft.data != 0) << "\nocc-right-valid: " << (task.occRight.data != 0);
	return stream;
}

cv::FileStorage& operator<<(cv::FileStorage& stream, const stereo_task& task);

class task_collection
{
public:
	task_collection(const std::string& filename) : name(filename) {}
	task_collection() = default;
	std::string name;
	std::vector<stereo_task> tasks;
};

/// Container for stereo_tasks. It can read a testset file, which contain the filenames for the single stereo_tasks
class TaskTestSet : public task_collection
{
public:
	TaskTestSet () {}
	TaskTestSet(const std::string& filename);
};

cv::FileStorage& operator<<(cv::FileStorage& stream, const TaskTestSet& testset);

///constructs a task_collection by reading a directory full of image files, so you don't have to create a yaml file per image.
class folder_testset : public task_collection
{
public:
	folder_testset(const std::string& filename);

	std::string left, right, dispLeft, dispRight;
	std::string fileextension_images, fileextension_gt;
	std::vector<std::string> filenames;

	int dispRange;
	unsigned char subsamplingGroundTruth;
};

#endif // STEREOTASK_H
