using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Duo;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Shared;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The LongCat, HunyuanVideo-1.5 and Duo transformers on a GPU against <see cref="CpuBackend"/>, on the tiny random models their CPU tests
/// use. A result left on the device, or an op a backend reads differently, shows here and in no CPU test. The device-to-host syncs of the forward
/// are printed, not asserted: each model still has known host steps.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VideoDitGpuParityTests(ITestOutputHelper output)
{
    /// <summary>The arithmetic a CUDA run uses. Vulkan has neither switch.</summary>
    private enum Math32
    {
        /// <summary>F32 GEMMs and F32 attention. Measured on a 4090 with exact GEMMs: LongCat 2.9e-7, Duo 4.0e-7.</summary>
        Exact,

        /// <summary>F32 GEMMs; an attention call that allows F16 takes it. Only HunyuanVideo-1.5 has such calls: its joint attention narrows Q, K
        /// and V to F16 because the QK norm bounds the scores. Measured on a 4090: 5.7e-5. Rounding the same operands and outputs on the CPU
        /// gives 5.0e-5.</summary>
        F16Attention,

        /// <summary>What a generation runs: TF32 GEMMs and the F16 attention. Measured on a 4090: Duo 1.1e-3, the other two about 4.5e-4.</summary>
        Default,
    }

    private static double Tolerance(Math32 math) => math switch { Math32.Exact => 1e-5, Math32.F16Attention => 1e-4, _ => 5e-3 };

    [CudaDeviceFact]
    public void Cuda_LongCat_MatchesCpu() => Compare("longcat", GpuDevices.OpenCuda, LongCat);

    [VulkanDeviceFact]
    public void Vulkan_LongCat_MatchesCpu() => Compare("longcat", GpuDevices.OpenVulkan, LongCat);

    [CudaDeviceFact]
    public void Cuda_HunyuanVideo15_MatchesCpu() => Compare("hunyuanvideo15", GpuDevices.OpenCuda, HunyuanVideo15);

    [VulkanDeviceFact]
    public void Vulkan_HunyuanVideo15_MatchesCpu() => Compare("hunyuanvideo15", GpuDevices.OpenVulkan, HunyuanVideo15);

    [CudaDeviceFact]
    public void Cuda_Duo_MatchesCpu() => Compare("duo", GpuDevices.OpenCuda, Duo);

    [VulkanDeviceFact]
    public void Vulkan_Duo_MatchesCpu() => Compare("duo", GpuDevices.OpenVulkan, Duo);

    [CudaDeviceFact]
    public void Cuda_HunyuanVideo15_WithF16Attention_StaysWithinItsTolerance() =>
        Compare("hunyuanvideo15 f16 attention", GpuDevices.OpenCuda, HunyuanVideo15, Math32.F16Attention);

    // The path a generation takes; only the tolerance differs.
    [CudaDeviceFact]
    public void Cuda_LongCat_WithDefaultGemms_StaysWithinTheTf32Tolerance() => Compare("longcat tf32", GpuDevices.OpenCuda, LongCat, Math32.Default);

    [CudaDeviceFact]
    public void Cuda_HunyuanVideo15_WithDefaultGemms_StaysWithinTheTf32Tolerance() =>
        Compare("hunyuanvideo15 tf32", GpuDevices.OpenCuda, HunyuanVideo15, Math32.Default);

    [CudaDeviceFact]
    public void Cuda_Duo_WithDefaultGemms_StaysWithinTheTf32Tolerance() => Compare("duo tf32", GpuDevices.OpenCuda, Duo, Math32.Default);

    private void Compare(string name, Func<IBackend> open, Func<IBackend, float[]> run, Math32 math = Math32.Exact)
    {
        float[] expected;
        using (CpuBackend cpu = new())
            expected = run(cpu);
        // The backend reads the F16 attention switch when it is built.
        using IDisposable knobs = KnobProfileScope.Push(KnobProfile.Create("video-dit-parity").With(EngineKnobs.SdpaNoF16, math == Math32.Exact));
        using IBackend device = open();
        // Off, CUDA runs F32 GEMMs in TF32.
        device.HighPrecisionGemm = math != Math32.Default;
        device.ResetD2hSyncCount();
        float[] actual = run(device);
        output.WriteLine($"D2H_SYNCS {name}: {device.GetD2hSyncCount()} (the read of the result included)");

        Assert.Equal(expected.Length, actual.Length);
        double error = 0, power = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            error += Math.Pow((double)expected[i] - actual[i], 2);
            power += Math.Pow(expected[i], 2);
        }
        Assert.True(power > 0, $"{name}: the CPU output is all zero, so the comparison would be empty.");
        double relative = Math.Sqrt(error / power);
        output.WriteLine($"{name}: relative RMS error {relative:E3}");
        double tolerance = Tolerance(math);
        Assert.True(relative <= tolerance, $"{name}: relative RMS error {relative:E3} exceeds {tolerance:E0}.");
    }

    private static float[] LongCat(IBackend backend)
    {
        const int frames = 3, height = 4, width = 6;
        LongCatConfig config = LongCatTestData.Tiny;
        Dictionary<string, Tensor> weights = LongCatTestData.Weights(config, 1);
        try
        {
            using LongCatTransformer transformer = new(config);
            transformer.LoadWeights(weights);
            using Tensor latent = LongCatTestData.Latent(config, frames, height, width, 2);
            using Tensor text = LongCatTestData.Encoder(config, 5, 3);
            LongCatConditioning conditioning = LongCatConditioning.None;
            using Tensor velocity = transformer.Forward(backend, latent, text, conditioning.FrameTimesteps(600f, frames), in conditioning, -1);
            return velocity.AsReadOnlySpan<float>().ToArray();
        }
        finally
        {
            LongCatTestData.DisposeAll(weights);
        }
    }

    private static float[] HunyuanVideo15(IBackend backend)
    {
        const int hidden = 32, blocks = 2;
        using SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", "hy15_dit_tiny.safetensors"));
        Dictionary<string, Tensor> weights = loader.GetAllTensors()
            .Where(pair => !new[] { "ref.", "in.", "win.", "rec." }.Any(prefix => pair.Key.StartsWith(prefix, StringComparison.Ordinal)))
            .ToDictionary(pair => pair.Key, pair => pair.Value);
        using HunyuanVideo15Dit dit = new(new HunyuanVideo15Config
        {
            NumBlocks = blocks, HiddenSize = hidden, NumHeads = 2, LatentChannels = 4, TextDim = 24, ByT5Dim = 16, RopeAxesDim = [4, 6, 6],
        });
        dit.LoadWeights(HunyuanVideo15CheckpointConverter.Convert(weights, blocks, hidden));
        using Tensor velocity = dit.Forward(backend, loader.GetTensor("in.latent"), loader.GetTensor("in.timestep").AsReadOnlySpan<float>()[0],
            loader.GetTensor("in.qwen"), loader.GetTensor("in.byt5"), loader.GetTensor("in.conditioning"));
        return velocity.AsReadOnlySpan<float>().ToArray();
    }

    // A cached pair (a committed prefix and one denoise that reads it), then the dense causal forward over the same rows at once.
    private static float[] Duo(IBackend backend)
    {
        using MiniMaxH3Transformer transformer = DuoTransformer(blocks: 2, out List<Tensor> weights);
        try
        {
            const int textRows = 2, refRows = 2, noisyRows = 4, inner = DuoHeads * DuoHeadDim;
            IReadOnlyList<DuoGroup> refiner = [new DuoGroup(0, 0, DuoGroupKind.Text, DuoModality.None, 0, textRows)];
            float[] times = [0f, 0.7f];
            using Tensor text = Wave(textRows, DuoTextDim, 2);
            using Tensor reference = Wave(refRows, DuoVideoIn, 5);
            using Tensor noisy = Wave(noisyRows, DuoVideoIn, 6);
            using Tensor video = new(new TensorShape(refRows + noisyRows, DuoVideoIn), DType.F32);
            reference.AsReadOnlySpan<float>().CopyTo(video.AsSpan<float>());
            noisy.AsReadOnlySpan<float>().CopyTo(video.AsSpan<float>()[(refRows * DuoVideoIn)..]);
            DuoStreamCache cache = new(backend, blocks: 2, width: inner, capacityRows: 16);

            DuoLayout prefix = DuoLayoutOf(
                [new DuoGroup(0, 0, DuoGroupKind.Shared, DuoModality.None, 0, textRows),
                 new DuoGroup(0, -1, DuoGroupKind.Reference, DuoModality.Video, textRows, textRows + refRows)],
                [1, 1, 0, 0], [0, 0, 0, 0], [2, 3], [0, 1], [], 0);
            (Tensor prefixOut, _) = transformer.ForwardDuoCached(backend, prefix, DuoAttentionMode.Causal, refiner, text, reference, null, times, cache,
                commit: true);
            using Tensor prefixResult = prefixOut;
            DuoLayout current = DuoLayoutOf([new DuoGroup(0, 0, DuoGroupKind.Noisy, DuoModality.Video, 0, noisyRows)],
                [0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 2, 3], [], [0], 4);
            (Tensor currentOut, _) = transformer.ForwardDuoCached(backend, current, DuoAttentionMode.Causal, [], null, noisy, null, times, cache,
                commit: false);
            using Tensor currentResult = currentOut;
            const int prefixRows = textRows + refRows;
            DuoLayout whole = DuoLayoutOf(
                [new DuoGroup(0, 0, DuoGroupKind.Shared, DuoModality.None, 0, textRows),
                 new DuoGroup(0, -1, DuoGroupKind.Reference, DuoModality.Video, textRows, prefixRows),
                 new DuoGroup(0, 0, DuoGroupKind.Noisy, DuoModality.Video, prefixRows, prefixRows + noisyRows)],
                [1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1], [2, 3, 4, 5, 6, 7], [0, 1], [2], 0);
            (Tensor dense, _) = transformer.ForwardDuo(backend, whole, DuoAttentionMode.Causal, refiner, text, video, null, times);
            using Tensor denseResult = dense;
            return [.. prefixOut.AsReadOnlySpan<float>(), .. currentOut.AsReadOnlySpan<float>(), .. dense.AsReadOnlySpan<float>()];
        }
        finally
        {
            foreach (Tensor weight in weights) weight.Dispose();
        }
    }

    private const int DuoHidden = 8, DuoHeads = 2, DuoHeadDim = 96, DuoFfn = 16, DuoTextDim = 5, DuoVideoIn = 96;

    private static DuoLayout DuoLayoutOf(IReadOnlyList<DuoGroup> groups, int[] tags, int[] timeIds, int[] videoRows, int[] textRows,
        IReadOnlyList<int> noisy, int rowOffset)
    {
        double[] positions = new double[tags.Length * 3];
        for (int r = 0; r < tags.Length; r++)
        {
            int row = rowOffset + r;
            positions[r * 3] = row * 0.5;
            positions[r * 3 + 1] = row % 2;
            positions[r * 3 + 2] = row / 2;
        }
        int[] textSource = [.. Enumerable.Range(0, textRows.Length)];
        return new DuoLayout(groups, tags.Length, tags.Length, positions, tags, timeIds, videoRows, [], textRows, textSource, noisy);
    }

    private static MiniMaxH3Transformer DuoTransformer(int blocks, out List<Tensor> owned)
    {
        const int timeHidden = 8, timeDim = 6, inner = DuoHeads * DuoHeadDim;
        Dictionary<string, Tensor> w = new()
        {
            ["video_patch_proj.weight"] = Wave(DuoHidden, DuoVideoIn, 10, 0.2f),
            ["condition_proj.weight"] = Wave(DuoHidden, DuoTextDim, 12, 0.2f),
            ["time_embedder.proj_in.weight"] = Wave(timeHidden, 256, 13, 0.2f),
            ["time_embedder.proj_out.weight"] = Wave(timeDim, timeHidden, 14, 0.2f),
            ["token_refiner.final_norm.weight"] = Ones(DuoHidden),
            ["final_layer.norm.weight"] = Ones(DuoHidden),
            ["final_layer.adaln_proj.linear.weight"] = Wave(2 * DuoHidden, timeDim, 24, 0.2f),
            ["final_layer.video_out.weight"] = Wave(DuoVideoIn, DuoHidden, 25, 0.2f),
        };
        void Block(string prefix, int seed)
        {
            w[prefix + "norm1.weight"] = Ones(DuoHidden);
            w[prefix + "norm2.weight"] = Ones(DuoHidden);
            w[prefix + "attn.qkv_proj.weight"] = Wave(3 * inner, DuoHidden, seed, 0.2f);
            w[prefix + "attn.q_norm.weight"] = Ones(DuoHeadDim);
            w[prefix + "attn.k_norm.weight"] = Ones(DuoHeadDim);
            w[prefix + "attn.out_proj.weight"] = Wave(DuoHidden, inner, seed + 1, 0.2f);
            w[prefix + "mlp.fc1.weight"] = Wave(2 * DuoFfn, DuoHidden, seed + 2, 0.2f);
            w[prefix + "mlp.fc2.weight"] = Wave(DuoHidden, DuoFfn, seed + 3, 0.2f);
        }

        Block("token_refiner.blocks.0.", 15);
        for (int b = 0; b < blocks; b++)
        {
            Block($"blocks.{b}.", 30 + 10 * b);
            w[$"blocks.{b}.adaln_proj.linear.weight"] = Wave(6 * DuoHidden * 3, timeDim, 19 + b, 0.2f);
        }
        owned = [.. w.Values];
        MiniMaxH3Transformer transformer = new(new MiniMaxH3Config
        {
            HiddenSize = DuoHidden, NumLayers = blocks, TokenRefinerNumLayers = 1, NumAttentionHeads = DuoHeads, AttentionHeadDim = DuoHeadDim,
            FfnHiddenSize = DuoFfn, LatentsDim = 24, HasAudio = false, TextDim = DuoTextDim, OutputHeadCount = 4, TimeEmbedHiddenSize = timeHidden,
            TimeEmbedDim = timeDim,
        });
        transformer.LoadWeights(w);
        return transformer;
    }

    private static Tensor Wave(int rows, int cols, int seed, float scale = 1f)
    {
        Tensor t = new(new TensorShape(rows, cols), DType.F32);
        Span<float> values = t.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = scale * MathF.Sin(seed * 1.7f + i * 0.37f);
        return t;
    }

    private static Tensor Ones(int count)
    {
        Tensor t = new(new TensorShape(count), DType.F32);
        t.AsSpan<float>().Fill(1f);
        return t;
    }
}
