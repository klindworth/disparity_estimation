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

class stereo_task;
class region_container;

class debug_task_store
{
public:
	debug_task_store() = default;
	debug_task_store(const std::string& pname, stereo_task* ptask) : name(pname), task(ptask) {}
	std::string name;
	std::vector<std::pair<cv::Mat, std::string> > simpleMatrices;
	stereo_task *task;
	std::shared_ptr<region_container> left, right;
};

class debug_store
{
public:
	debug_store() = default;
	void start_new_task(const std::string &name, stereo_task& task);
	void set_region_container(std::shared_ptr<region_container>& left, std::shared_ptr<region_container>& right);
	void add_mat(const cv::Mat &mat, const char* name);
	void add_mat(const char *name, const cv::Mat &mat);

	std::vector<debug_task_store> tasks;
};

extern debug_store matstore;

#endif // DEBUGMATSTORE_H
