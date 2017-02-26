#ifndef ML_REGION_OPTIMIZER_ALGORITHMS_H_
#define ML_REGION_OPTIMIZER_ALGORITHMS_H_

#include <disparity_toolkit/disparity_range.h>
#include <vector>

/**
 * Merges two feature vectors into one per disparity value.
 * |base +0|match+0|base+1|match-1|base+2|match-2|...
 * Because this is a region based algoritm, and a region usually overlaps more than one region in the match image,
 * a weighted feature vector is calculated.
 *
 * vector_size: disparity-independet features of the base image.
 * vector_size_per_disp: size of the feature vector per disparity of one image
 * Therefore the complete feature will be: vector_size+disparity range size*2*vector_size_per_disp
 *
 * @param Destination array. Sufficient memory must already be allocated
 * @param feature_vector_base Feature vector, that is builded like |features for disparity 0|features for disparity 1|..|disparity independet features|
 * @param feature_vectors_match Feature vectors of all regions in the matching image
 * @param delta Only a certain epsilon range around the current estimated disparity should be considered
 */
template<int vector_size, int vector_size_per_disp, typename dst_type, typename src_type>
void merge_with_corresponding_feature_vector(dst_type *dst_ptr, const disparity_region& baseRegion, const std::vector<src_type>& feature_vector_base,
											 const std::vector<std::vector<src_type>>& feature_vectors_match, const region_container& match, int delta, const single_stereo_task& task)
{
	const int crange = task.range.size();
	disparity_range drange = task_subrange(task, baseRegion.base_disparity, delta);

	std::vector<dst_type> disp_optimization_vector(vector_size_per_disp);
	for(int d = drange.start(); d <= drange.end(); ++d)
	{
		//weighted sum of features of the corresponding feature vectors. we've to do this, because the warped region from the base image overlaps multiple regions in the other image
		std::fill(disp_optimization_vector.begin(), disp_optimization_vector.end(), 0.0f);
		const int corresponding_disp_idx = -d - match.task.range.start();
		foreach_corresponding_region(baseRegion.corresponding_regions[d-task.range.start()], [&](std::size_t idx, float percent) {
			const src_type* it = &(feature_vectors_match[idx][corresponding_disp_idx*vector_size_per_disp]);
			for(int i = 0; i < vector_size_per_disp; ++i)
				disp_optimization_vector[i] += percent * *it++;
		});

		const src_type *base_ptr = feature_vector_base.data() + (d-drange.start())*vector_size_per_disp;
		const dst_type *other_ptr = disp_optimization_vector.data();

		//dst_type *ndst_ptr = dst_ptr + (d-drange.start())*ml_region_optimizer::vector_size_per_disp*2;
		dst_type *ndst_ptr = dst_ptr + vector_size_per_disp*2*std::abs(d);

		//copy all features of the base region for this particular disparity
		for(int j = 0; j < vector_size_per_disp; ++j)
			*ndst_ptr++ = *base_ptr++;

		//copy the newly weighted features for the match image
		for(int j = 0; j < vector_size_per_disp; ++j)
			*ndst_ptr++ = *other_ptr++;
	}

	//now copy the disparity independet features
	const src_type *base_src_ptr = feature_vector_base.data()+crange*vector_size_per_disp;

	dst_type *ndst_ptr = dst_ptr + crange*vector_size_per_disp*2;
	for(int i = 0; i < vector_size; ++i)
		*ndst_ptr++ = *base_src_ptr++;
}

#endif

