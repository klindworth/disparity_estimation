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

#include "disparity_toolkit/stereotask.h"

debug_store matstore;

void debug_store::start_new_task(const std::string& name, stereo_task &task)
{
	tasks.push_back(debug_task_store(name, &task));
}
void debug_store::set_region_container(std::shared_ptr<region_container>& left, std::shared_ptr<region_container>& right)
{
	assert(!tasks.empty());

	tasks.back().left = left;
	tasks.back().right = right;
}

void debug_store::add_mat(const cv::Mat& mat, const char* name)
{
	assert(!tasks.empty());

	tasks.back().simpleMatrices.push_back(std::make_pair(mat, std::string(name)));
}

void debug_store::add_mat(const char *name, const cv::Mat &mat)
{
	add_mat(mat, name);
}

