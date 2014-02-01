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

#include "debugmatstore.h"

#include "stereotask.h"

DebugMatStore matstore;

DebugMatStore::DebugMatStore()
{
}

void DebugMatStore::startNewTask(const std::string& name, StereoTask &task)
{
	tasks.push_back(TaskStore(name, &task));
}
void DebugMatStore::setRegionContainer(std::shared_ptr<RegionContainer>& left, std::shared_ptr<RegionContainer>& right)
{
	assert(!tasks.empty());

	tasks.back().left = left;
	tasks.back().right = right;
}

void DebugMatStore::addMat(const cv::Mat& mat, const char* name)
{
	assert(!tasks.empty());

	tasks.back().simpleMatrices.push_back(std::make_pair(mat, std::string(name)));
}

void DebugMatStore::addMat(const cv::Mat& left, const cv::Mat& right, cv::Mat &cost, const char* name, int windowsize, bool forward, cv::Mat windows, cv::Mat offset)
{
	assert(!tasks.empty());

	viewerMat temp;
	temp.left = left;
	temp.right = right;
	temp.cost_map = cost;
	temp.name = std::string(name);
	temp.windowsize = windowsize;
	temp.forward = forward;
	temp.windows = windows;
	temp.offset = offset;

	tasks.back().costmaps.push_back(temp);
}

void DebugMatStore::addMat(const StereoSingleTask& task, cv::Mat& cost, const char* name, int windowsize, cv::Mat windows, cv::Mat offset)
{
	assert(!tasks.empty());

	viewerMat temp;
	temp.left = task.base;
	temp.right = task.match;
	temp.cost_map = cost;
	temp.name = std::string(name);
	temp.windowsize = windowsize;
	temp.forward = task.dispMin < 0;
	temp.windows = windows;
	temp.offset = offset;

	tasks.back().costmaps.push_back(temp);
}


