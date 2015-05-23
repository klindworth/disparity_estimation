#ifndef TASK_COLLECTION_H_
#define TASK_COLLECTION_H_

/*
Copyright (c) 2013, 2015 Kai Klindworth
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

#endif
