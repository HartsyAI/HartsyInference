// 1D convolution + transposed convolution for codec models.
// Layout: input/output channels-first [B, C, T]; Conv1d weight [C_out, C_in/groups, K];
// ConvTranspose1d weight [C_in, C_out, K]. Both kernels assume F32.
//
// Build:
//   nvcc --ptx -arch=sm_80 conv1d_f32.cu -o conv1d_f32.ptx
//   (move the .ptx into src/SharpInference.Cuda/Ptx/)

extern "C" __global__ void conv1d_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int batch,
    int cIn,
    int cOut,
    int tIn,
    int tOut,
    int kernel,
    int stride,
    int padLeft,
    int dilation,
    int groups,
    int hasBias)
{
    int outFlat = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * cOut * tOut;
    if (outFlat >= total) return;

    int j = outFlat % tOut;
    int oc = (outFlat / tOut) % cOut;
    int b = outFlat / (tOut * cOut);

    int outPerGroup = cOut / groups;
    int inPerGroup = cIn / groups;
    int group = oc / outPerGroup;
    int icStart = group * inPerGroup;

    int jSrcStart = j * stride - padLeft;
    float acc = (hasBias != 0) ? bias[oc] : 0.0f;

    int wRow = oc * inPerGroup * kernel;
    for (int ic = 0; ic < inPerGroup; ic++) {
        int inCh = icStart + ic;
        int inBase = (b * cIn + inCh) * tIn;
        int wBase = wRow + ic * kernel;
        #pragma unroll 4
        for (int k = 0; k < kernel; k++) {
            int src = jSrcStart + k * dilation;
            if ((unsigned)src < (unsigned)tIn) {
                acc += input[inBase + src] * weight[wBase + k];
            }
        }
    }
    output[outFlat] = acc;
}

extern "C" __global__ void conv_transpose1d_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int batch,
    int cIn,
    int cOut,
    int tIn,
    int tOut,
    int kernel,
    int stride,
    int padLeft,
    int dilation,
    int hasBias)
{
    int outFlat = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * cOut * tOut;
    if (outFlat >= total) return;

    int j = outFlat % tOut;
    int oc = (outFlat / tOut) % cOut;
    int b = outFlat / (tOut * cOut);

    float acc = (hasBias != 0) ? bias[oc] : 0.0f;

    // Output-driven gather: for target output j, walk all kernel taps k that produced it.
    int jShifted = j + padLeft;
    for (int k = 0; k < kernel; k++) {
        int num = jShifted - k * dilation;
        if (num < 0) continue;
        if (num % stride != 0) continue;
        int i = num / stride;
        if (i >= tIn) continue;
        for (int ic = 0; ic < cIn; ic++) {
            int inIdx = (b * cIn + ic) * tIn + i;
            int wIdx = (ic * cOut + oc) * kernel + k;
            acc += input[inIdx] * weight[wIdx];
        }
    }
    output[outFlat] = acc;
}
