// Audio-codec elementwise activations for SEANet/EnCodec/DAC/Mimi/Vocos/LSTM paths.
//
// Adds the F32 kernels missing from the original elementwise_f32.ptx:
//   * sigmoid       — 1 / (1 + exp(-x)), used by LSTM gates + streaming TTS EOS classifier
//   * tanh          — used by LSTM cell + output gate; also useful as a general activation
//   * elu           — x if x>=0 else alpha*(exp(x)-1); SEANet residual block activation
//   * snake         — x + sin(alpha*x)^2 / alpha           (vanilla, DAC/SNAC/Stable Audio)
//   * snake_beta    — x + sin(alpha*x)^2 / (beta + 1e-8)   (BigVGAN-v2 variant)
//
// Build:
//   nvcc --ptx -arch=sm_80 audio_activations_f32.cu -o audio_activations_f32.ptx
//   (move the .ptx into src/HartsyInference.Cuda/Ptx/)
//
// All kernels assume contiguous float buffers. Snake variants additionally take per-
// channel alpha (and optional beta) tensors of shape [channels].

extern "C" __global__ void audio_sigmoid_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float x = input[i];
    // Sign-aware to avoid overflow in exp.
    if (x >= 0.0f) {
        float ex = expf(-x);
        output[i] = 1.0f / (1.0f + ex);
    } else {
        float ex = expf(x);
        output[i] = ex / (1.0f + ex);
    }
}

extern "C" __global__ void audio_tanh_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    output[i] = tanhf(input[i]);
}

extern "C" __global__ void audio_elu_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    float alpha,
    int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float x = input[i];
    output[i] = (x >= 0.0f) ? x : (alpha * (expf(x) - 1.0f));
}

extern "C" __global__ void audio_snake_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ alpha,
    int batch,
    int channels,
    int timeDim)
{
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * timeDim;
    if (flat >= total) return;
    int c = (flat / timeDim) % channels;
    float x = input[flat];
    float a = alpha[c];
    float s = sinf(a * x);
    output[flat] = x + (s * s) / a;
}

extern "C" __global__ void audio_snake_beta_f32(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ alpha,
    const float* __restrict__ beta,
    int batch,
    int channels,
    int timeDim)
{
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * timeDim;
    if (flat >= total) return;
    int c = (flat / timeDim) % channels;
    float x = input[flat];
    float a = alpha[c];
    float divisor = beta[c] + 1e-8f;
    float s = sinf(a * x);
    output[flat] = x + (s * s) / divisor;
}
