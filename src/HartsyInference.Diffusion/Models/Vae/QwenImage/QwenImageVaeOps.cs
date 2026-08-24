using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae.QwenImage;

/// <summary>Low-level ops for the Qwen-Image (WAN 2.1 family) 3D causal autoencoder. The VAE was trained
/// jointly for video and image at 16-channel latent / 8× spatial downscale / 4× temporal downscale.
/// For image-only inference (Anima) we run the decoder with <c>T = 1</c> throughout, which collapses
/// every 3D causal convolution to a 2D convolution:
///
/// <para>A <c>CausalConv3d(kT=3, padT_front=2, padT_back=0)</c> on a <c>T=1</c> input zero-pads the
/// time axis to length 3 (two zero frames + the real frame), then runs a 3D conv of kernel size
/// <c>(3, kH, kW)</c>. Only the <c>kT=2</c> slice of the kernel multiplies real data; the other two
/// slices multiply zeros. So the output equals a plain <c>Conv2d</c> with weight
/// <c>weight_3d[:, :, -1, :, :]</c>. For <c>kT=1</c> convs (1×1×1, including the top-level <c>conv1/conv2</c>
/// and the attention <c>to_qkv/proj</c>), the temporal slot index is 0 (the only one). For
/// <c>(3, 1, 1)</c> time_conv layers — used to temporally upsample for video — we simply skip them
/// since the image-mode output has <c>T = 1</c>.</para>
///
/// <para>This file exposes static helpers used by <see cref="QwenImageVaeDecoder"/> and its block
/// classes. All ops produce <c>F32</c> output to match the rest of HartsyInference's VAE path.</para></summary>
public static unsafe class QwenImageVaeOps
{
    /// <summary>Slices a 5-D <c>[out, in, kT, kH, kW]</c> Conv3D weight tensor down to a 4-D
    /// <c>[out, in, kH, kW]</c> Conv2D weight by extracting one temporal slot. For Wan/Qwen
    /// CausalConv3d's image mode, that slot is the LAST one (<paramref name="temporalSlot"/> = kT - 1)
    /// for kT &gt; 1 kernels, or 0 for kT = 1 kernels. Always allocates an F32 result; if the source
    /// tensor is BF16/F16/F8, it's cast on the fly.</summary>
    public static Tensor SliceConv3dToConv2d(Tensor conv3dWeight, int temporalSlot = -1)
    {
        if (conv3dWeight.Shape.Rank != 5)
            throw new ArgumentException(
                $"Expected 5-D Conv3D weight [out, in, kT, kH, kW], got rank {conv3dWeight.Shape.Rank} shape {conv3dWeight.Shape}.",
                nameof(conv3dWeight));

        int outCh = (int)conv3dWeight.Shape[0];
        int inCh = (int)conv3dWeight.Shape[1];
        int kT = (int)conv3dWeight.Shape[2];
        int kH = (int)conv3dWeight.Shape[3];
        int kW = (int)conv3dWeight.Shape[4];
        if (temporalSlot < 0) temporalSlot += kT;
        if (temporalSlot < 0 || temporalSlot >= kT)
            throw new ArgumentOutOfRangeException(nameof(temporalSlot), $"Slot {temporalSlot} out of range [0, {kT}).");

        // Cast to F32 first (the existing backend Conv2D path expects F32 weights for these small VAE
        // ops; HartsyInference's mixed-precision conv is reserved for the bigger diffusion convs).
        Tensor sourceF32 = conv3dWeight.DType == DType.F32 ? conv3dWeight : conv3dWeight.CastTo(DType.F32);

        TensorShape outShape = new TensorShape(outCh, inCh, kH, kW);
        Tensor result = new Tensor(outShape, DType.F32);

        long perKwSlice = (long)kH * kW;  // floats per (kT) entry
        long srcStride_I = (long)kT * perKwSlice;
        long srcStride_O = (long)inCh * srcStride_I;
        long dstStride_I = perKwSlice;
        long dstStride_O = (long)inCh * dstStride_I;
        long bytesPerSlice = perKwSlice * sizeof(float);

        float* src = (float*)sourceF32.DataPointer;
        float* dst = (float*)result.DataPointer;
        for (int o = 0; o < outCh; o++)
        {
            for (int i = 0; i < inCh; i++)
            {
                float* srcRow = src + o * srcStride_O + i * srcStride_I + temporalSlot * perKwSlice;
                float* dstRow = dst + o * dstStride_O + i * dstStride_I;
                Buffer.MemoryCopy(srcRow, dstRow, bytesPerSlice, bytesPerSlice);
            }
        }

        if (!ReferenceEquals(sourceF32, conv3dWeight)) sourceF32.Dispose();
        return result;
    }

    /// <summary>Flattens a gamma weight of any rank to a 1-D <c>[C]</c> F32 tensor. WAN's RMSNorm
    /// stores gamma as either <c>[C, 1, 1, 1]</c> (residual / conv blocks operating on
    /// <c>[B, C, T, H, W]</c>) or <c>[C, 1, 1]</c> (attention blocks operating on
    /// <c>[B, C, H, W]</c>). Both reduce to a flat <c>[C]</c> for the per-channel L2 normalize.</summary>
    public static Tensor FlattenGamma(Tensor gamma)
    {
        // Flatten by total element count — the leading dim is C and every trailing dim is 1.
        long elements = gamma.Shape.ElementCount;
        Tensor f32 = gamma.DType == DType.F32 ? gamma : gamma.CastTo(DType.F32);
        if (f32.Shape.Rank == 1 && (int)f32.Shape[0] == elements)
        {
            return f32;
        }
        TensorShape outShape = new TensorShape(elements);
        Tensor result = new Tensor(outShape, DType.F32);
        long bytes = elements * sizeof(float);
        Buffer.MemoryCopy(f32.DataPointer, result.DataPointer, bytes, bytes);
        if (!ReferenceEquals(f32, gamma)) f32.Dispose();
        return result;
    }

    /// <summary>Per-pixel L2-normalize across the channel dim, then scale by <c>sqrt(C) * gamma</c>.
    /// This matches WAN's <c>RMS_norm</c> module (<c>channel_first=True, images=True</c>) which is
    /// equivalent to <c>F.normalize(x, dim=1) * sqrt(C) * gamma</c>.
    ///
    /// <para>Input and output are <c>[B, C, H, W]</c>. <paramref name="gamma"/> is a flat <c>[C]</c>
    /// F32 tensor (use <see cref="FlattenGamma"/> at load time to produce it from the file's
    /// <c>[C, 1, 1, 1]</c> or <c>[C, 1, 1]</c> shapes). Output is F32 even if input isn't.</para>
    ///
    /// <para><b>Host loop — DECODER MUST NOT USE THIS.</b> At 1024² each call D2H-drains an up-to-400 MB conv
    /// output, loops on the CPU, and re-uploads — this was ~20 s of a 27 s Krea2 gen until the decoder switched to
    /// <see cref="IBackend.WanRmsNormChannel"/> (identical math, GPU-resident). Only the ENCODER (img2img, one
    /// call per source image) still uses it; port it the same way when img2img perf matters.</para></summary>
    public static void RmsNormPerPixelAcrossChannels(Tensor output, Tensor input, Tensor gamma, float eps = 1e-12f)
    {
        if (input.Shape.Rank != 4)
            throw new ArgumentException($"RmsNorm: input must be 4-D [B, C, H, W], got {input.Shape}.", nameof(input));
        if (output.Shape != input.Shape)
            throw new ArgumentException($"RmsNorm: output shape {output.Shape} != input shape {input.Shape}.", nameof(output));
        int batch = (int)input.Shape[0];
        int channels = (int)input.Shape[1];
        int height = (int)input.Shape[2];
        int width = (int)input.Shape[3];
        if (gamma.Shape.ElementCount != channels)
            throw new ArgumentException(
                $"RmsNorm: gamma must have {channels} elements, got {gamma.Shape.ElementCount}.", nameof(gamma));

        Tensor inputF32 = input.DType == DType.F32 ? input : input.CastTo(DType.F32);
        Tensor gammaF32 = gamma.DType == DType.F32 ? gamma : gamma.CastTo(DType.F32);

        float* inPtr = (float*)inputF32.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        float* gPtr = (float*)gammaF32.DataPointer;

        long spatial = (long)height * width;
        long stride_C = spatial;
        long stride_B = (long)channels * spatial;
        float sqrtC = MathF.Sqrt(channels);

        for (int b = 0; b < batch; b++)
        {
            long bOff = b * stride_B;
            for (long s = 0; s < spatial; s++)
            {
                // Sum of squares across the channel dim at this spatial position.
                float sumSq = 0f;
                for (int c = 0; c < channels; c++)
                {
                    float v = inPtr[bOff + c * stride_C + s];
                    sumSq += v * v;
                }
                float invNorm = 1.0f / MathF.Sqrt(sumSq + eps);
                float scale = invNorm * sqrtC;
                for (int c = 0; c < channels; c++)
                {
                    float v = inPtr[bOff + c * stride_C + s];
                    outPtr[bOff + c * stride_C + s] = v * scale * gPtr[c];
                }
            }
        }

        if (!ReferenceEquals(inputF32, input)) inputF32.Dispose();
        if (!ReferenceEquals(gammaF32, gamma)) gammaF32.Dispose();
    }

}
