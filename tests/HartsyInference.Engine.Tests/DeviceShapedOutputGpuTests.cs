using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Tests.Shared;
using HartsyInference.Vision.CrossFrame;
using HartsyInference.Vision.DepthPro2;
using HartsyInference.Vision.Dinov2;
using HartsyInference.Vision.Dinov3;
using Xunit.Abstractions;

namespace HartsyInference.Engine.Tests;

/// <summary>Model stages that used to read a device result back to the host (a <see cref="Tensor.Reshape"/>, a <see cref="Tensor.SliceRows"/> or a
/// host loop): on a GPU they must run without one device-to-host sync and agree with <see cref="CpuBackend"/>. Synthetic tiny weights, no
/// downloads.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class DeviceShapedOutputGpuTests(ITestOutputHelper output)
{
    private const int ViewHeight = 32, ViewWidth = 48, DinoHidden = 128, DinoLayers = 3;
    private static readonly int[] DinoTaps = [0, 1, 2];
    private static readonly int[] DinoCrossFrameBlocks = [1, 2];

    [CudaDeviceFact]
    public void Cuda_DepthPro2Reassemble_StaysOnTheDevice() => DepthPro2Reassemble(GpuDevices.OpenCuda);

    [VulkanDeviceFact]
    public void Vulkan_DepthPro2Reassemble_StaysOnTheDevice() => DepthPro2Reassemble(GpuDevices.OpenVulkan,
        hostConvTransposes: 2);

    [CudaDeviceFact]
    public void Cuda_LongCatAudioProjection_StaysOnTheDevice() => LongCatAudioProjectionForward(GpuDevices.OpenCuda);

    [VulkanDeviceFact]
    public void Vulkan_LongCatAudioProjection_StaysOnTheDevice() =>
        LongCatAudioProjectionForward(GpuDevices.OpenVulkan);

    [CudaDeviceFact]
    public void Cuda_CrossFrameJointViews_StayOnTheDevice() => CrossFrameJoint(GpuDevices.OpenCuda);

    [VulkanDeviceFact]
    public void Vulkan_CrossFrameJointViews_StayOnTheDevice() => CrossFrameJoint(GpuDevices.OpenVulkan);

    [CudaDeviceFact]
    public void Cuda_CrossFrameRing_SurvivesAnActivationSweepBetweenFrames() => CrossFrameRing(GpuDevices.OpenCuda);

    [VulkanDeviceFact]
    public void Vulkan_CrossFrameRing_SurvivesAnActivationSweepBetweenFrames() => CrossFrameRing(GpuDevices.OpenVulkan);

    // Three views attending to each other in the last two of three blocks (Join and Split).
    private void CrossFrameJoint(Func<IBackend> open)
    {
        Random rng = new(13);
        Dictionary<string, Tensor> weights = DinoWeights(rng);
        using Tensor views = Filled(rng, 3, 3, ViewHeight, ViewWidth);
        try
        {
            Dinov3VisionEncoder encoder = new(DinoPreset());
            encoder.LoadWeights(weights);
            float[][] expected = Values(encoder.EncodeFrameIntermediates(new CpuBackend(), views, DinoTaps, DinoCrossFrameBlocks));
            using IBackend device = open();
            using ExactGemmScope exact = new(device);
            using Tensor resident = OnDevice(device, views);
            device.ResetD2hSyncCount();
            Tensor[] taps = encoder.EncodeFrameIntermediates(device, resident, DinoTaps, DinoCrossFrameBlocks);
            long syncs = device.GetD2hSyncCount();
            output.WriteLine($"D2H_SYNCS cross-frame joint views: {syncs}");
            AssertClose(expected, Values(taps), 1e-4f);
            Assert.Equal(0, syncs);
        }
        finally
        {
            DisposeAll(weights.Values);
        }
    }

    // Four frames through a ring of two, so it wraps. After each frame the device frees its unpinned activations, which is what another request on
    // the same backend does between two frames of a stream: a cache that was not pinned comes back stale.
    private void CrossFrameRing(Func<IBackend> open)
    {
        const int frames = 4, window = 2;
        Random rng = new(17);
        Dictionary<string, Tensor> weights = DinoWeights(rng);
        Tensor[] video = [.. Enumerable.Range(0, frames).Select(_ => Filled(rng, 1, 3, ViewHeight, ViewWidth))];
        try
        {
            Dinov3VisionEncoder encoder = new(DinoPreset());
            encoder.LoadWeights(weights);
            CpuBackend cpu = new();
            List<float[][]> expected = [];
            using (CrossFrameKvRing ring = new(window))
            {
                for (int t = 0; t < frames; t++)
                {
                    float[][] cached = Values(encoder.EncodeFrameIntermediates(cpu, video[t], DinoTaps, DinoCrossFrameBlocks, ring));
                    float[][] alone = Values(encoder.EncodeIntermediates(cpu, video[t], DinoTaps, DinoCrossFrameBlocks));
                    // The cached frames must matter, or a lost cache would go unnoticed.
                    if (t > 0) Assert.True(cached[^1].Zip(alone[^1]).Max(pair => MathF.Abs(pair.First - pair.Second)) > 1e-2f);
                    expected.Add(cached);
                }
            }

            using IBackend device = open();
            using ExactGemmScope exact = new(device);
            using CrossFrameKvRing deviceRing = new(window);
            for (int t = 0; t < frames; t++)
            {
                using (Tensor resident = OnDevice(device, video[t]))
                {
                    device.ResetD2hSyncCount();
                    Tensor[] taps = encoder.EncodeFrameIntermediates(device, resident, DinoTaps, DinoCrossFrameBlocks, deviceRing);
                    long syncs = device.GetD2hSyncCount();
                    output.WriteLine($"D2H_SYNCS cross-frame ring frame {t}: {syncs}");
                    AssertClose(expected[t], Values(taps), 1e-4f);
                    Assert.Equal(0, syncs);
                }

                device.FreeActivations();
            }
        }
        finally
        {
            DisposeAll(video);
            DisposeAll(weights.Values);
        }
    }

    // A DINOv3 of three blocks with two 64-wide heads, laid out like Depth Pro 2's backbone.
    private static Dinov2Preset DinoPreset() => new()
    {
        Name = "tiny-dinov3", HiddenSize = DinoHidden, NumLayers = DinoLayers, NumHeads = 2, IntermediateSize = 2 * DinoHidden, ImageSize = ViewWidth,
        PatchSize = 16, NumRegisterTokens = 4, UsesRotaryPositionEmbedding = true, RoundRopePeriodsToBf16 = false, AttentionQkvBias = false,
        HasFinalNorm = false, RopeNormalizesByLongSide = true,
    };

    private static Dictionary<string, Tensor> DinoWeights(Random rng)
    {
        const int h = DinoHidden;
        Dictionary<string, Tensor> w = new()
        {
            ["embeddings.patch_embeddings.weight"] = Scaled(rng, 0.05f, 0, h, 3, 16, 16),
            ["embeddings.patch_embeddings.bias"] = Scaled(rng, 0.05f, 0, h),
            ["embeddings.cls_token"] = Scaled(rng, 0.5f, 0, 1, 1, h),
            ["embeddings.register_tokens"] = Scaled(rng, 0.5f, 0, 1, 4, h),
        };
        for (int i = 0; i < DinoLayers; i++)
        {
            string p = $"layer.{i}.";
            w[p + "norm1.weight"] = Scaled(rng, 0.2f, 1, h);
            w[p + "norm1.bias"] = Scaled(rng, 0.1f, 0, h);
            foreach (string proj in new[] { "q_proj", "k_proj", "v_proj", "o_proj" })
                w[$"{p}attention.{proj}.weight"] = Scaled(rng, 0.2f, 0, h, h);
            w[p + "attention.o_proj.bias"] = Scaled(rng, 0.1f, 0, h);
            w[p + "layer_scale1.lambda1"] = Scaled(rng, 0.2f, 0.5f, h);
            w[p + "norm2.weight"] = Scaled(rng, 0.2f, 1, h);
            w[p + "norm2.bias"] = Scaled(rng, 0.1f, 0, h);
            w[p + "mlp.up_proj.weight"] = Scaled(rng, 0.15f, 0, 2 * h, h);
            w[p + "mlp.up_proj.bias"] = Scaled(rng, 0.1f, 0, 2 * h);
            w[p + "mlp.down_proj.weight"] = Scaled(rng, 0.1f, 0, h, 2 * h);
            w[p + "mlp.down_proj.bias"] = Scaled(rng, 0.1f, 0, h);
            w[p + "layer_scale2.lambda1"] = Scaled(rng, 0.2f, 0.5f, h);
        }

        return w;
    }

    // Uniform in offset +/- scale.
    private static Tensor Scaled(Random rng, float scale, float offset, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>()) v = offset + scale * (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    // On Vulkan ConvTranspose2d has no shader and runs on the host (VULKAN_STATUS.md). Levels 0 and 1 each have one, and each reads its input back.
    private void DepthPro2Reassemble(Func<IBackend> open, int hostConvTransposes = 0)
    {
        const int width = 5, prefix = 2, gridH = 2, gridW = 4, decoder = 4;
        int[] dims = [3, 4, 5, 6];
        Random rng = new(7);
        Dictionary<string, Tensor> weights = DepthPro2Weights(rng, dims, width, decoder);
        Tensor[] taps = [.. Enumerable.Range(0, 4).Select(_ => Filled(rng, 1, prefix + gridH * gridW, width))];
        try
        {
            DepthPro2Head head = new(dims, decoder, width, weights);
            float[][] expected = Values(head.Reassemble(new CpuBackend(), taps, prefix, gridH, gridW));
            using IBackend device = open();
            Tensor[] resident = [.. taps.Select(tap => OnDevice(device, tap))];
            try
            {
                device.ResetD2hSyncCount();
                Tensor[] levels = head.Reassemble(device, resident, prefix, gridH, gridW);
                long syncs = device.GetD2hSyncCount();
                output.WriteLine($"D2H_SYNCS depthpro2 reassemble: {syncs}");
                AssertClose(expected, Values(levels), 1e-4f);
                Assert.Equal(hostConvTransposes, syncs);
            }
            finally
            {
                DisposeAll(resident);
            }
        }
        finally
        {
            DisposeAll(taps);
            DisposeAll(weights.Values);
        }
    }

    private void LongCatAudioProjectionForward(Func<IBackend> open)
    {
        const int seqLen = 5, seqLenVf = 8, blocks = 3, channels = 6, hidden = 16, outputDim = 10, context = 4;
        int inFirst = seqLen * blocks * channels, inLatter = seqLenVf * blocks * channels;
        Random rng = new(11);
        Dictionary<string, Tensor> weights = new()
        {
            ["audio_proj.proj1.weight"] = Filled(rng, hidden, inFirst),
            ["audio_proj.proj1.bias"] = Filled(rng, hidden),
            ["audio_proj.proj1_vf.weight"] = Filled(rng, hidden, inLatter),
            ["audio_proj.proj1_vf.bias"] = Filled(rng, hidden),
            ["audio_proj.proj2.weight"] = Filled(rng, hidden, hidden),
            ["audio_proj.proj2.bias"] = Filled(rng, hidden),
            ["audio_proj.proj3.weight"] = Filled(rng, context * outputDim, hidden),
            ["audio_proj.proj3.bias"] = Filled(rng, context * outputDim),
            ["audio_proj.norm.weight"] = Filled(rng, outputDim),
            ["audio_proj.norm.bias"] = Filled(rng, outputDim),
        };
        Tensor first = Filled(rng, 1, inFirst), latter = Filled(rng, 3, inLatter);
        try
        {
            using LongCatAudioProjection projection = new(seqLen, seqLenVf, blocks, channels, hidden, outputDim, context);
            projection.LoadWeights(weights);
            float[][] expected = Values([projection.Forward(new CpuBackend(), first, latter)]);
            using IBackend device = open();
            // The projection's GEMMs run in TF32 on CUDA by default, which measured 1.4e-4 off the CPU here; this test is about the layout, so it
            // compares exact ones. DepthPro2Head opens the same scope itself.
            using ExactGemmScope exact = new(device);
            using Tensor firstDevice = OnDevice(device, first);
            using Tensor latterDevice = OnDevice(device, latter);
            device.ResetD2hSyncCount();
            Tensor result = projection.Forward(device, firstDevice, latterDevice);
            long syncs = device.GetD2hSyncCount();
            output.WriteLine($"D2H_SYNCS longcat audio projection: {syncs}");
            AssertClose(expected, Values([result]), 1e-4f);
            Assert.Equal(0, syncs);
        }
        finally
        {
            first.Dispose();
            latter.Dispose();
            DisposeAll(weights.Values);
        }
    }

    private static Dictionary<string, Tensor> DepthPro2Weights(Random rng, int[] dims, int width, int decoder)
    {
        Dictionary<string, Tensor> w = [];
        int[] kernels = [4, 2, 0, 3];
        for (int i = 0; i < 4; i++)
        {
            string p = $"encoder.reassemble.{i}.", f = $"decoder.fusions.{i}.";
            w[p + "0.weight"] = Filled(rng, dims[i], width, 1, 1);
            w[p + "0.bias"] = Filled(rng, dims[i]);
            if (kernels[i] > 0)
            {
                w[p + "1.weight"] = Filled(rng, dims[i], dims[i], kernels[i], kernels[i]);
                w[p + "1.bias"] = Filled(rng, dims[i]);
            }

            w[$"decoder.convs.{i}.weight"] = Filled(rng, decoder, dims[i], 3, 3);
            foreach (string block in i < 3 ? new[] { "resnet1.", "resnet2." } : ["resnet2."])
            {
                w[f + block + "residual.1.weight"] = Filled(rng, decoder, decoder, 3, 3);
                w[f + block + "residual.1.bias"] = Filled(rng, decoder);
                w[f + block + "residual.3.weight"] = Filled(rng, decoder, decoder, 3, 3);
                w[f + block + "residual.3.bias"] = Filled(rng, decoder);
            }

            w[f + "deconv.weight"] = Filled(rng, decoder, decoder, 2, 2);
            w[f + "out_conv.weight"] = Filled(rng, decoder, decoder, 1, 1);
            w[f + "out_conv.bias"] = Filled(rng, decoder);
        }

        w["head.0.weight"] = Filled(rng, decoder / 2, decoder, 3, 3);
        w["head.0.bias"] = Filled(rng, decoder / 2);
        w["head.1.weight"] = Filled(rng, decoder / 2, decoder / 2, 2, 2);
        w["head.1.bias"] = Filled(rng, decoder / 2);
        w["head.2.weight"] = Filled(rng, 32, decoder / 2, 3, 3);
        w["head.2.bias"] = Filled(rng, 32);
        w["head.4.weight"] = Filled(rng, 1, 32, 1, 1);
        w["head.4.bias"] = Filled(rng, 1);
        return w;
    }

    private static Tensor Filled(Random rng, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        foreach (ref float v in t.AsSpan<float>()) v = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    // A copy whose only current bytes are on the device, as a preceding op would leave it.
    private static Tensor OnDevice(IBackend device, Tensor host)
    {
        Tensor copy = new(host.Shape, DType.F32);
        device.Scale(copy, host, 1f);
        Assert.False(copy.HostBytesCurrent);
        return copy;
    }

    // Reads and disposes the tensors.
    private static float[][] Values(Tensor[] tensors)
    {
        float[][] values = [.. tensors.Select(t => t.AsReadOnlySpan<float>().ToArray())];
        DisposeAll(tensors);
        return values;
    }

    private static void AssertClose(float[][] expected, float[][] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int t = 0; t < expected.Length; t++)
        {
            Assert.Equal(expected[t].Length, actual[t].Length);
            Assert.Contains(actual[t], v => v != 0f);
            for (int i = 0; i < expected[t].Length; i++)
                Assert.True(MathF.Abs(expected[t][i] - actual[t][i]) <= tolerance, $"tensor {t} element {i}: CPU {expected[t][i]} vs device {actual[t][i]}.");
        }
    }

    private static void DisposeAll(IEnumerable<Tensor> tensors)
    {
        foreach (Tensor t in tensors) t.Dispose();
    }
}
