#ifndef ML_REGION_OPTIMIZER_ALGORITHMS_H_
#define ML_REGION_OPTIMIZER_ALGORITHMS_H_

template<int vector_size, int vector_size_per_disp, typename dst_type, typename src_type>
void merge_with_corresponding_optimization_vector(dst_type *dst_ptr, const disparity_region& baseRegion, const std::vector<src_type>& optimization_vector_base, const std::vector<std::vector<src_type>>& optimization_vectors_match, const region_container& match, int delta, const single_stereo_task& task)
{
	const int crange = task.range_size();
	disparity_range drange = task_subrange(task, baseRegion.base_disparity, delta);

	std::vector<dst_type> disp_optimization_vector(vector_size_per_disp);
	for(short d = drange.start(); d <= drange.end(); ++d)
	{
		std::fill(disp_optimization_vector.begin(), disp_optimization_vector.end(), 0.0f);
		const int corresponding_disp_idx = -d - match.task.dispMin;
		foreach_corresponding_region(baseRegion.corresponding_regions[d-task.dispMin], [&](std::size_t idx, float percent) {
			const src_type* it = &(optimization_vectors_match[idx][corresponding_disp_idx*vector_size_per_disp]);
			for(int i = 0; i < vector_size_per_disp; ++i)
				disp_optimization_vector[i] += percent * *it++;
		});

		const src_type *base_ptr = optimization_vector_base.data() + (d-drange.start())*vector_size_per_disp;
		const dst_type *other_ptr = disp_optimization_vector.data();

		//dst_type *ndst_ptr = dst_ptr + (d-drange.start())*ml_region_optimizer::vector_size_per_disp*2;
		dst_type *ndst_ptr = dst_ptr + vector_size_per_disp*2*(int)std::abs(d);

		for(int j = 0; j < vector_size_per_disp; ++j)
			*ndst_ptr++ = *base_ptr++;

		for(int j = 0; j < vector_size_per_disp; ++j)
			*ndst_ptr++ = *other_ptr++;
	}

	const src_type *base_src_ptr = optimization_vector_base.data()+crange*vector_size_per_disp;

	dst_type *ndst_ptr = dst_ptr + crange*vector_size_per_disp*2;
	for(int i = 0; i < vector_size; ++i)
		*ndst_ptr++ = *base_src_ptr++;
}

#endif

