// MiniMax-H3 VSA, correctness-first SM80 implementation.
// Q/K/V/gate/output use F32 [1,56,S,128]. Routing metadata and scratch stay device-resident.

extern "C" __global__ void h3_vsa_centroids_f32(
    float* q_centroids,
    float* k_centroids,
    float* v_centroids,
    const float* query,
    const float* key,
    const float* value,
    const int* block_offsets,
    const int* source_indices,
    int sequence,
    int blocks)
{
    int block = (int)blockIdx.x;
    int head = (int)blockIdx.y;
    int dim = (int)threadIdx.x;
    if (block >= blocks || head >= 56 || dim >= 128) return;

    int start = block_offsets[block];
    int stop = block_offsets[block + 1];
    float q_sum = 0.0f;
    float k_sum = 0.0f;
    float v_sum = 0.0f;
    for (int i = start; i < stop; ++i)
    {
        int row = source_indices[i];
        long index = ((long)head * sequence + row) * 128 + dim;
        q_sum += query[index];
        k_sum += key[index];
        v_sum += value[index];
    }
    float inverse = 1.0f / (float)(stop - start);
    long centroid = ((long)block * 56 + head) * 128 + dim;
    q_centroids[centroid] = q_sum * inverse;
    k_centroids[centroid] = k_sum * inverse;
    v_centroids[centroid] = v_sum * inverse;
}

// The ComfySOL proxy centers K block means before per-block int8 quantization. Subtracting one global K mean is
// rank-invariant in exact arithmetic, while matching the published quantized carrier semantics at tie boundaries.
extern "C" __global__ void h3_vsa_key_means_f32(
    float* key_means,
    const float* k_centroids,
    int blocks)
{
    int head = (int)blockIdx.x;
    int dim = (int)threadIdx.x;
    if (head >= 56 || dim >= 128) return;
    float sum = 0.0f;
    for (int block = 0; block < blocks; ++block)
    {
        sum += k_centroids[((long)block * 56 + head) * 128 + dim];
    }
    key_means[head * 128 + dim] = sum / (float)blocks;
}

__device__ __forceinline__ bool h3_vsa_better(float a_score, int a_index, float b_score, int b_index)
{
    return a_score > b_score || (a_score == b_score && a_index < b_index);
}

__device__ __forceinline__ float h3_vsa_centroid_score(
    const float* q,
    const float* k,
    const float* k_mean,
    int profile)
{
    if (profile != 1)
    {
        float score = 0.0f;
        #pragma unroll
        for (int dim = 0; dim < 128; ++dim) score += q[dim] * k[dim];
        return score;
    }

    float q_absmax = 0.0f;
    float k_absmax = 0.0f;
    #pragma unroll
    for (int dim = 0; dim < 128; ++dim)
    {
        q_absmax = fmaxf(q_absmax, fabsf(q[dim]));
        k_absmax = fmaxf(k_absmax, fabsf(k[dim] - k_mean[dim]));
    }
    float q_scale = q_absmax == 0.0f ? 1.0f : q_absmax / 127.0f;
    float k_scale = k_absmax == 0.0f ? 1.0f : k_absmax / 127.0f;
    int integer_dot = 0;
    #pragma unroll
    for (int dim = 0; dim < 128; ++dim)
    {
        int q_int = max(-127, min(127, __float2int_rn(q[dim] / q_scale)));
        int k_int = max(-127, min(127, __float2int_rn((k[dim] - k_mean[dim]) / k_scale)));
        integer_dot += q_int * k_int;
    }
    return (float)integer_dot * q_scale * k_scale;
}

extern "C" __global__ void h3_vsa_routes_f32(
    int* route_counts,
    int* route_blocks,
    const float* q_centroids,
    const float* k_centroids,
    const float* key_means,
    int blocks,
    int padded_blocks,
    int prefix_blocks,
    int keep_blocks,
    int max_routes,
    int profile)
{
    int query_block = (int)blockIdx.x;
    int head = (int)blockIdx.y;
    if (query_block >= blocks || head >= 56) return;
    int route_unit = query_block * 56 + head;
    if (query_block < prefix_blocks)
    {
        if (threadIdx.x == 0) route_counts[route_unit] = -1;
        return;
    }

    extern __shared__ unsigned char shared_bytes[];
    float* original = (float*)shared_bytes;
    float* sorted = original + padded_blocks;
    int* order = (int*)(sorted + padded_blocks);
    int* selected = order + padded_blocks;
    const float* q = q_centroids + ((long)query_block * 56 + head) * 128;
    const float* k_mean = key_means + head * 128;

    for (int candidate = (int)threadIdx.x; candidate < padded_blocks; candidate += (int)blockDim.x)
    {
        float score = -__int_as_float(0x7f800000);
        // Prefix blocks are exempt sinks and never consume the 10% video-tile budget.
        if (candidate >= prefix_blocks && candidate < blocks)
        {
            const float* k = k_centroids + ((long)candidate * 56 + head) * 128;
            score = h3_vsa_centroid_score(q, k, k_mean, profile);
        }
        original[candidate] = score;
        sorted[candidate] = score;
        order[candidate] = candidate;
        selected[candidate] = 0;
    }
    __syncthreads();

    // Deterministic bitonic ordering: score descending, lower block index wins ties.
    for (int width = 2; width <= padded_blocks; width <<= 1)
    {
        for (int stride = width >> 1; stride > 0; stride >>= 1)
        {
            for (int left = (int)threadIdx.x; left < padded_blocks; left += (int)blockDim.x)
            {
                int right = left ^ stride;
                if (right > left)
                {
                    bool descending = (left & width) == 0;
                    bool left_better = h3_vsa_better(sorted[left], order[left], sorted[right], order[right]);
                    if (left_better != descending)
                    {
                        float score_swap = sorted[left];
                        sorted[left] = sorted[right];
                        sorted[right] = score_swap;
                        int index_swap = order[left];
                        order[left] = order[right];
                        order[right] = index_swap;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (profile == 1)
    {
        int routeable_blocks = blocks - prefix_blocks;
        if (keep_blocks == routeable_blocks)
        {
            for (int candidate = prefix_blocks + (int)threadIdx.x;
                 candidate < blocks;
                 candidate += (int)blockDim.x)
            {
                selected[candidate] = 1;
            }
        }
        else
        {
            float threshold = sorted[keep_blocks];
            for (int candidate = prefix_blocks + (int)threadIdx.x;
                 candidate < blocks;
                 candidate += (int)blockDim.x)
            {
                if (original[candidate] > threshold) selected[candidate] = 1;
            }
        }
    }
    else
    {
        for (int rank = (int)threadIdx.x; rank < keep_blocks; rank += (int)blockDim.x)
        {
            selected[order[rank]] = 1;
        }
    }
    for (int candidate = (int)threadIdx.x; candidate < prefix_blocks; candidate += (int)blockDim.x)
    {
        selected[candidate] = 1;
    }
    if (threadIdx.x == 0)
    {
        if (profile == 1)
        {
            selected[query_block] = 1;
            if (query_block > 0) selected[query_block - 1] = 1;
            if (query_block + 1 < blocks) selected[query_block + 1] = 1;
        }
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        int count = 0;
        long route_base = (long)route_unit * max_routes;
        route_counts[route_unit] = 0;
        for (int candidate = 0; candidate < blocks; ++candidate)
        {
            if (selected[candidate] != 0)
            {
                if (count >= max_routes)
                {
                    route_counts[route_unit] = -2;
                    asm volatile("trap;");
                    return;
                }
                route_blocks[route_base + count] = candidate;
                ++count;
            }
        }
        route_counts[route_unit] = count;
    }
}

__device__ __forceinline__ float h3_vsa_warp_sum(float value)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

extern "C" __global__ void h3_vsa_attention_f32(
    float* output,
    const float* query,
    const float* key,
    const float* value,
    const float* gate,
    const float* q_centroids,
    const float* k_centroids,
    const float* v_centroids,
    const int* block_offsets,
    const int* source_indices,
    const int* source_block,
    const int* route_counts,
    const int* route_blocks,
    int sequence,
    int blocks,
    int max_routes)
{
    int row = (int)blockIdx.x;
    int head = (int)blockIdx.y;
    int lane = (int)threadIdx.x;
    if (row >= sequence || head >= 56 || lane >= 32) return;
    int query_block = source_block[row];
    long query_base = ((long)head * sequence + row) * 128;

    float fine0 = 0.0f, fine1 = 0.0f, fine2 = 0.0f, fine3 = 0.0f;
    float fine_max = -__int_as_float(0x7f800000);
    float fine_sum = 0.0f;
    if (query_block >= 0)
    {
        int route_unit = query_block * 56 + head;
        int route_count = route_counts[route_unit];
        int route_iterations = route_count < 0 ? blocks : route_count;
        for (int route = 0; route < route_iterations; ++route)
        {
            int key_block = route_count < 0 ? route
                : route_blocks[(long)route_unit * max_routes + route];
            int start = block_offsets[key_block];
            int stop = block_offsets[key_block + 1];
            for (int i = start; i < stop; ++i)
            {
                int key_row = source_indices[i];
                long key_base = ((long)head * sequence + key_row) * 128;
                float dot = query[query_base + lane]
                    * key[key_base + lane]
                    + query[query_base + lane + 32]
                    * key[key_base + lane + 32]
                    + query[query_base + lane + 64]
                    * key[key_base + lane + 64]
                    + query[query_base + lane + 96]
                    * key[key_base + lane + 96];
                float score = h3_vsa_warp_sum(dot) * 0.08838834764831845f;
                float next_max = fmaxf(fine_max, score);
                float old_weight = fine_sum == 0.0f ? 0.0f : expf(fine_max - next_max);
                float new_weight = expf(score - next_max);
                fine0 = fine0 * old_weight + value[key_base + lane] * new_weight;
                fine1 = fine1 * old_weight + value[key_base + lane + 32] * new_weight;
                fine2 = fine2 * old_weight + value[key_base + lane + 64] * new_weight;
                fine3 = fine3 * old_weight + value[key_base + lane + 96] * new_weight;
                fine_sum = fine_sum * old_weight + new_weight;
                fine_max = next_max;
            }
        }
    }

    float coarse0 = 0.0f, coarse1 = 0.0f, coarse2 = 0.0f, coarse3 = 0.0f;
    float coarse_max = -__int_as_float(0x7f800000);
    float coarse_sum = 0.0f;
    if (query_block >= 0)
    {
        long query_centroid = ((long)query_block * 56 + head) * 128;
        for (int block = 0; block < blocks; ++block)
        {
            long centroid = ((long)block * 56 + head) * 128;
            float dot = q_centroids[query_centroid + lane]
                * k_centroids[centroid + lane]
                + q_centroids[query_centroid + lane + 32]
                * k_centroids[centroid + lane + 32]
                + q_centroids[query_centroid + lane + 64]
                * k_centroids[centroid + lane + 64]
                + q_centroids[query_centroid + lane + 96]
                * k_centroids[centroid + lane + 96];
            float score = h3_vsa_warp_sum(dot) * 0.08838834764831845f;
            float next_max = fmaxf(coarse_max, score);
            float old_weight = coarse_sum == 0.0f ? 0.0f : expf(coarse_max - next_max);
            float new_weight = expf(score - next_max);
            coarse0 = coarse0 * old_weight + v_centroids[centroid + lane] * new_weight;
            coarse1 = coarse1 * old_weight + v_centroids[centroid + lane + 32] * new_weight;
            coarse2 = coarse2 * old_weight + v_centroids[centroid + lane + 64] * new_weight;
            coarse3 = coarse3 * old_weight + v_centroids[centroid + lane + 96] * new_weight;
            coarse_sum = coarse_sum * old_weight + new_weight;
            coarse_max = next_max;
        }
    }

    if (query_block < 0)
    {
        output[query_base + lane] = 0.0f;
        output[query_base + lane + 32] = 0.0f;
        output[query_base + lane + 64] = 0.0f;
        output[query_base + lane + 96] = 0.0f;
        return;
    }
    fine0 /= fine_sum; fine1 /= fine_sum; fine2 /= fine_sum; fine3 /= fine_sum;
    coarse0 /= coarse_sum; coarse1 /= coarse_sum; coarse2 /= coarse_sum; coarse3 /= coarse_sum;
    output[query_base + lane] = fine0 + gate[query_base + lane] * coarse0;
    output[query_base + lane + 32] = fine1 + gate[query_base + lane + 32] * coarse1;
    output[query_base + lane + 64] = fine2 + gate[query_base + lane + 64] * coarse2;
    output[query_base + lane + 96] = fine3 + gate[query_base + lane + 96] * coarse3;
}
