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

#ifndef SEGMENTATION_ALGORITHMS_H
#define SEGMENTATION_ALGORITHMS_H

#include "segmentation.h"
#include "region_descriptor_algorithms.h"

#include <iterator>

class fusion_work_data
{
public:
	fusion_work_data(std::size_t size) :
		visited(std::vector<unsigned char>(size, 0)),
		active(std::vector<unsigned char>(size, 1)),
		fused(std::vector<std::vector<std::size_t>>(size)),
		fused_with(std::vector<std::size_t>(size, 0))
	{
	}

	void visit_reset()
	{
		std::fill(visited.begin(), visited.end(), 0);
	}

	std::vector<unsigned char> visited;
	std::vector<unsigned char> active;
	std::vector<std::vector<std::size_t> > fused;
	std::vector<std::size_t> fused_with;
};

/**
 * @brief fusion
 * @param regions
 * @param labels
 * @param idx Region to fuse
 * @param fusion_idx Region to fuse width
 * @param segcount Number of active segments
 * @param check_func Function to check if the combination of segments fullfills conditions
 */
template<typename T>
void fusion(fusion_work_data& data, std::vector<T>& regions, std::size_t idx, std::size_t fusion_idx, std::function<bool(const T& master_seg, const T& slave_seg, const T& fusion_seg)> check_func)
{
	T& cregion = regions[idx];
	data.visited[idx] = 1;
	for(std::pair<std::size_t, std::size_t>& cpair : cregion.neighbors)
	{
		std::size_t neighbor = cpair.first;
		if(neighbor != fusion_idx && data.active[neighbor])
		{
			T& cneighbor = regions[neighbor];
			if(check_func(cregion, cneighbor, regions[fusion_idx]))
			{
				assert(data.active[fusion_idx]);
				data.fused_with[neighbor] = fusion_idx;
				data.fused[fusion_idx].push_back(neighbor);
				//if the neighbor already fused with some other segments
				std::copy(data.fused[neighbor].begin(), data.fused[neighbor].end(), std::back_inserter(data.fused[fusion_idx]));
				data.active[neighbor] = 0;

				if(!(data.visited[neighbor]))
					fusion(data, regions, neighbor, fusion_idx, check_func);
			}
		}
	}
}

template<typename T>
void fuse(fusion_work_data& data, std::vector<T>& regions, cv::Mat_<int>& labels)
{
	const std::size_t regions_count = regions.size();
	#pragma omp parallel for
	for(std::size_t master_idx = 0; master_idx < regions_count; ++master_idx)
	{
		//regions[master_idx].active = data.active[master_idx];
		if(data.active[master_idx])
		{
			//remove doubles
			std::sort(data.fused[master_idx].begin(), data.fused[master_idx].end());
			data.fused[master_idx].erase(std::unique(data.fused[master_idx].begin(), data.fused[master_idx].end()), data.fused[master_idx].end());

			for(std::size_t slave_idx : data.fused[master_idx])
			{
				assert(!data.active[slave_idx]);
				T& cregion = regions[slave_idx];
				std::copy(cregion.lineIntervals.begin(), cregion.lineIntervals.end(), std::back_inserter(regions[master_idx].lineIntervals));
				regions[master_idx].m_size += cregion.m_size;
				cregion.m_size = 0;
			}
		}
	}

	for(std::size_t i = 0; i < regions_count; ++i)
	{
		if(data.active[i])
			assert(regions[i].m_size > 0);
		else
		{
			if(regions[i].m_size != 0)
			{
				std::cout << i << std::endl;
				std::cout << "size: "  << regions[i].m_size << std::endl;
				std::cout << "master: " << data.fused_with[i] << std::endl;
				std::cout << "master-active: " << (int)data.active[data.fused_with[i]] << std::endl;
				std::copy(data.fused[data.fused_with[i]].begin(), data.fused[data.fused_with[i]].end(), std::ostream_iterator<int>(std::cout));
				std::cout << std::endl;
				//assert(regions[i].size == 0); //TODO fixme
			}
		}
			//
	}

	regions.erase(std::remove_if(regions.begin(), regions.end(), [](const T& cregion){return (cregion.m_size == 0);}), regions.end());

	//regenerate labels image, sort region intervals
	/*parallel_region(regions.begin(), regions.end(), [&](RegionDescriptor region) {
		std::sort(region.lineIntervals.begin(), region.lineIntervals.end());
		intervals::setRegionValue<int>(labels, region.lineIntervals, i);
	});*/
	const std::size_t regions_count2 = regions.size();
	#pragma omp parallel for default(none) shared(labels, regions)
	for(std::size_t i = 0; i < regions_count2; ++i)
	{
		std::sort(regions[i].lineIntervals.begin(), regions[i].lineIntervals.end());
		intervals::set_region_value<int>(labels, regions[i].lineIntervals, i);
	}

	//assert(checkLabelsIntervalsInvariant(regions, labels, regions.size()));
	//assert(std::count(data.active.begin(), data.active.end(), 1) == regions.size()); //TODO: fixme

	region_descriptors::generate_neighborhood(labels, regions);
}

template<typename T>
void runFusion(cv::Mat& labels, std::vector<T>& regions, std::function<bool(const T& master_seg, const T& slave_seg, const T& fusion_seg)> check_func)
{
	std::size_t segcount = regions.size();
	fusion_work_data data(segcount);

	for(std::size_t i = 0; i < segcount; ++i)
	{
		data.visit_reset();
		if(data.active[i])
			fusion(data, regions, i, i, check_func);
		/*else
		{
			int last_j = i;
			while(data.active[last_j])
				last_j = data.fused_with[last_j];
			fusion(data, regions, labels, i, data.fused_with[last_j], check_func);
		}*/
	}
	std::cout << "fuse" << std::endl;
	fuse(data, regions, labels);
	std::cout << "fusion finished, regions: " << regions.size() << std::endl;

	refresh_bounding_boxes(regions.begin(), regions.end(), labels);
	generate_neighborhood(labels, regions);
}

template<typename T, typename lambda_type>
void defuse(std::vector<T>& fused_regions, cv::Mat_<int>& newlabels, int newsegcount, const fusion_work_data& data, lambda_type transfer_region)
{
	std::vector<T> regions(newsegcount);
	fill_region_descriptors(regions.begin(), regions.end(), newlabels);

	const std::size_t regions_count = regions.size();
	std::vector<std::size_t> inverse_mapping;
	inverse_mapping.reserve(fused_regions.size());
	for(std::size_t i = 0; i < regions_count; ++i)
	{
		if(data.active[i])
			inverse_mapping.push_back(i);
	}
	assert(fused_regions.size() == inverse_mapping.size());
	std::cout << fused_regions.size() << " vs " << inverse_mapping.size() << std::endl;

	for(std::size_t i = 0; i < regions_count; ++i)
	{
		std::size_t master_unfused_idx = i;

		while(!data.active[master_unfused_idx])
			master_unfused_idx = data.fused_with[master_unfused_idx];
		auto it = std::find(inverse_mapping.begin(), inverse_mapping.end(), master_unfused_idx);
		if(it != inverse_mapping.end())
		{
			std::size_t master_fused_idx = std::distance(inverse_mapping.begin(), it);

			transfer_region(fused_regions[master_fused_idx], regions[i]);
		}
		else
			std::cerr << "missed mapping" << std::endl;
	}

	std::swap(fused_regions, regions);
}

#endif
