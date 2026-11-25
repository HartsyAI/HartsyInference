using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Vision.DepthPro2;
using Xunit.Abstractions;

namespace HartsyInference.Engine.Tests;

/// <summary>Model stages that used to reshape a device result on the host (<see cref="Tensor.Reshape"/> and <see cref="Tensor.SliceRows"/> read it
/// back): on a GPU they must run without one device-to-host sync and agree with <see cref="CpuBackend"/>. Synthetic tiny weights, no downloads.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class DeviceShapedOutputGpuTests(ITestOutputHelper output)
{
    [CudaDeviceFact]
    public void Cuda_DepthPro2Reassemble_StaysOnTheDevice() => DepthPro2Reassemble(GpuDevices.OpenCuda);

    [VulkanDeviceFact]
    public void Vulkan_DepthPro2Reassemble_StaysOnTheDevice() => DepthPro2Reassemble(static () => new HartsyInference.Vulkan.VulkanBackend(),
        hostOps: true);

    [CudaDeviceFact]
    public void Cuda_LongCatAudioProjection_StaysOnTheDevice() => LongCatAudioProjectionForward(GpuDevices.OpenCuda);

    [VulkanDeviceFact]
    public void Vulkan_LongCatAudioProjection_StaysOnTheDevice() =>
        LongCatAudioProjectionForward(static () => new HartsyInference.Vulkan.VulkanBackend());

    // hostOps: ConvTranspose2d runs on the host on Vulkan (VULKAN_STATUS.md), so the count is printed and only the values are checked there.
    private void DepthPro2Reassemble(Func<IBackend> open, bool hostOps = false)
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
                if (!hostOps) Assert.Equal(0, syncs);
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
