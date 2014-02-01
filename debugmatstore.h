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

#ifndef DEBUGMATSTORE_H
#define DEBUGMATSTORE_H

#include <opencv2/core/core.hpp>
#include <vector>
#include <memory>

class StereoSingleTask;
class StereoTask;
class RegionContainer;

struct viewerMat
{
	cv::Mat left;
	cv::Mat right;
	cv::Mat cost_map;
	cv::Mat windows;
	cv::Mat offset;
	std::string name;
	bool forward;
	int windowsize;
};

class TaskStore
{
public:
	TaskStore() {}
	TaskStore(const std::string& pname, StereoTask* ptask) : name(pname), task(ptask) {}
	std::string name;
	std::vector<std::pair<cv::Mat, std::string> > simpleMatrices;
	std::vector<struct viewerMat> costmaps;
	StereoTask *task;
	std::shared_ptr<RegionContainer> left, right;
};

class DebugMatStore
{
public:
	DebugMatStore();
	void startNewTask(const std::string &name, StereoTask& task);
	void setRegionContainer(std::shared_ptr<RegionContainer>& left, std::shared_ptr<RegionContainer>& right);
	void addMat(const cv::Mat &mat, const char* name);
	void addMat(const cv::Mat &left, const cv::Mat &right, cv::Mat &cost, const char* name, int windowsize, bool forward, cv::Mat windows = cv::Mat(), cv::Mat offset = cv::Mat());
	void addMat(const StereoSingleTask& task, cv::Mat &cost, const char* name, int windowsize, cv::Mat windows = cv::Mat(), cv::Mat offset = cv::Mat());

	std::vector<TaskStore> tasks;
};

extern DebugMatStore matstore;

#endif // DEBUGMATSTORE_H
