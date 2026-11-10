using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers.SanaWm;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The SANA-WM layers on the CUDA backend against the same layers on the CPU backend, on the small committed fixtures
/// and the synthetic two-block layout: a bisection ladder below the full-DiT parity test. The device rounds GEMM inputs and the
/// FFN to BF16, so the bar is a cosine, not equality. GPU lane: <c>Category=GpuIntegration&amp;FullyQualifiedName~SanaWmDeviceParity</c>.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class SanaWmDeviceParityGpuTests : IDisposable
{
    private const double CosineTarget = 0.999;
    private const int Frames = 2;
    private const int Height = 2;
    private const int Width = 2;
    private const int TextTokens = 4;
    private const int CameraChannels = 20;
    private readonly ITestOutputHelper _output;
    private readonly string _path = Path.Combine(Path.GetTempPath(), $"sanawm-device-{Guid.NewGuid():N}.safetensors");

    public SanaWmDeviceParityGpuTests(ITestOutputHelper output) => _output = output;

    public void Dispose()
    {
        if (File.Exists(_path))
            File.Delete(_path);
    }

    [Fact]
    public void GdnBidi_TinyFixture_CudaMatchesCpu()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using SafeTensorsLoader loader = Fixture("sana_wm_s1_gdn_bidi_tiny.safetensors");
        IReadOnlyDictionary<string, string> meta = loader.Metadata!;
        SanaWmGdnWeights weights = new()
        {
            Qkv = loader.GetTensor("qkv_weight"),
            ConvK = loader.GetTensor("conv_k_weight"),
            BetaWeight = loader.GetTensor("beta_proj_weight"),
            BetaBias = loader.GetTensor("beta_proj_bias"),
            GateWeight = loader.GetTensor("gate_proj_weight"),
            GateBias = loader.GetTensor("gate_proj_bias"),
            ALog = loader.GetTensor("a_log"),
            DtBias = loader.GetTensor("dt_bias"),
            QNorm = loader.GetTensor("q_norm_weight"),
            KNorm = loader.GetTensor("k_norm_weight"),
        };
        int heads = Int(meta, "heads"), headDim = Int(meta, "head_dim");
        int frames = Int(meta, "frames"), height = Int(meta, "latent_height"), width = Int(meta, "latent_width");

        float[] Run(IBackend backend)
        {
            using SanaWmGdnBidi layer = new(backend, weights, heads, headDim);
            (Tensor output, Tensor beta, Tensor decay) = layer.ForwardWithGates(loader.GetTensor("x"), frames, height, width,
                loader.GetTensor("rope_cos"), loader.GetTensor("rope_sin"));
            using (output)
            using (beta)
            using (decay)
                return [.. Read(backend, output), .. Read(backend, beta), .. Read(backend, decay)];
        }

        Compare("GDN main branch and gates", RunOnBoth(Run));
    }

    [Fact]
    public void CamBranch_Fixture_CudaMatchesCpu()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using SafeTensorsLoader loader = Fixture("sana_wm_s2_cam_branch.safetensors");
        IReadOnlyDictionary<string, Tensor> fixture = loader.GetAllTensors();
        float[] conditions = fixture["camera_conditions"].AsReadOnlySpan<float>().ToArray();

        float[] Run(IBackend backend)
        {
            using SanaWmCamBranch branch = new(1, 112);
            branch.LoadWeights(fixture);
            using Tensor cam = branch.Forward(backend, fixture["x"], conditions, 4, 2, 3, fixture["rope_cos"], fixture["rope_sin"],
                fixture["beta"], fixture["decay"]);
            return Read(backend, cam);
        }

        Compare("camera branch", RunOnBoth(Run));
    }

    [Theory]
    [InlineData(SanaWmGlumbTempFfn.DefaultHiddenChunkElements)]
    [InlineData(1L)]
    public void Ffn_Bf16OnCuda_StaysWithinBf16ToleranceOfUpstream(long hiddenChunkElements)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend cuda = new(0, PtxDirectory());
        using SanaWmGlumbTempCase c = SanaWmGlumbTempCase.Load("sana_wm_s4_glumb_temp_ffn_t3.safetensors", SanaWmOperandMode.Bf16Storage,
            hiddenChunkElements);
        using Tensor output = c.Ffn.Forward(cuda, c.Input, c.Frames, c.Height, c.Width);
        using Tensor widened = SanaWmTensors.ToF32(cuda, output);
        float[] actual = Read(cuda, widened);
        ReadOnlySpan<float> expected = c.Expected.AsReadOnlySpan<float>();
        float scale = c.ExpectedScale();
        float maxError = 0f;
        for (int i = 0; i < expected.Length; i++)
            maxError = MathF.Max(maxError, MathF.Abs(actual[i] - expected[i]));
        _output.WriteLine($"BF16 FFN on CUDA (chunk budget {hiddenChunkElements}): max abs {maxError:E3}, scale {scale:E3}");
        Assert.True(maxError <= scale / 32f, $"BF16 FFN max abs error {maxError:E3} exceeds 1/32 of {scale:E3}.");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void Dit_SyntheticLayout_CudaMatchesCpu(int batch)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        SanaWmSyntheticLayout.WriteFile(_path);
        using SanaWmCheckpoint checkpoint = SanaWmCheckpointConverter.Load(_path);
        SanaWmConfig config = checkpoint.Config;
        float[] Run(IBackend backend)
        {
            List<Tensor> inputs = [];
            try
            {
                Tensor x = Pattern(inputs, [batch, config.InChannels, Frames, Height, Width], 0.03f);
                Tensor text = Pattern(inputs, [batch, 1, TextTokens, config.TextWidth], 0.02f);
                Tensor mask = Pattern(inputs, [batch, TextTokens], 0f);
                mask.AsSpan<float>().Fill(1f);
                mask.AsSpan<float>()[^1] = 0f;
                Tensor chunk = Pattern(inputs, [batch, config.PluckerChannels, Frames, Height, Width], 0.05f);
                float[] timesteps = [.. Enumerable.Range(0, batch * Frames).Select(i => 100f + 250f * i)];
                using SanaWmDit dit = new(backend, checkpoint);
                using Tensor velocity = dit.Forward(x, timesteps, text, mask, Cameras(batch), chunk);
                return Read(backend, velocity);
            }
            finally
            {
                SanaWmTensors.DisposeAll(inputs);
            }
        }

        Compare($"synthetic DiT, batch {batch}", RunOnBoth(Run));
    }

    /// <summary>The CUDA <see cref="IBackend.FrameDeltaScan"/> (batched Gram, injection and readout GEMMs around the recurrence
    /// kernel) against the host reference at the real head width, for a 112-wide value and for the unit value of a
    /// denominator, with operand scales like the model's (ReLU'd keys scaled by D^-1/2·S^-1/2, beta and decay in (0, 1)).</summary>
    [Theory]
    [InlineData(112)]
    [InlineData(1)]
    public void FrameDeltaScan_RealHeadWidth_CudaMatchesHostReference(int valueDim)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        const int frames = 5, spatial = 40, heads = 3, keyDim = 112;
        long tokens = (long)frames * spatial;
        Random random = new(1234);
        float keyScale = (float)(Math.Pow(keyDim, -0.5) * Math.Pow(spatial, -0.5));
        List<Tensor> owned = [];
        Tensor Fill(long[] dims, Func<float> sample)
        {
            Tensor tensor = new(new TensorShape(dims), DType.F32);
            Span<float> values = tensor.AsSpan<float>();
            for (int i = 0; i < values.Length; i++)
                values[i] = sample();
            owned.Add(tensor);
            return tensor;
        }

        Tensor query = Fill([tokens, heads * keyDim], () => MathF.Max(0f, (float)random.NextDouble() * 2f - 0.5f));
        Tensor key = Fill([tokens, heads * keyDim], () => MathF.Max(0f, (float)random.NextDouble() * 2f - 0.5f) * keyScale * 8f);
        Tensor? value = valueDim == 1 ? null : Fill([tokens, heads * valueDim], () => (float)random.NextDouble() * 2f - 1f);
        Tensor beta = Fill([heads, frames, spatial], () => (float)random.NextDouble());
        Tensor decay = Fill([heads, frames], () => 0.5f + 0.5f * (float)random.NextDouble());
        try
        {
            float[] Run(IBackend backend)
            {
                using Tensor output = new(new TensorShape(tokens, heads * valueDim), DType.F32);
                backend.FrameDeltaScan(output, query, key, value, beta, decay, frames, spatial, heads, keyDim, valueDim);
                return Read(backend, output);
            }

            (float[] cpu, float[] cuda) = RunOnBoth(Run);
            double error = 0, norm = 0;
            for (int i = 0; i < cpu.Length; i++)
            {
                error += (double)(cpu[i] - cuda[i]) * (cpu[i] - cuda[i]);
                norm += (double)cpu[i] * cpu[i];
            }

            double relative = Math.Sqrt(error / Math.Max(norm, double.Epsilon));
            _output.WriteLine($"FrameDeltaScan valueDim {valueDim}: relative L2 {relative:E3} (TF32 GEMMs)");
            Compare($"FrameDeltaScan valueDim {valueDim}", (cpu, cuda));
            Assert.True(relative < 5e-3, $"FrameDeltaScan relative L2 {relative:E3} against the host reference.");

            // Streaming: the same scan started from a carried forward state, also writing the state it ends in.
            Tensor initial = Fill([heads, keyDim, valueDim], () => ((float)random.NextDouble() * 2f - 1f) * 0.05f);
            float[] RunCarried(IBackend backend)
            {
                using Tensor output = new(new TensorShape(tokens, heads * valueDim), DType.F32);
                using Tensor final = new(new TensorShape(heads, keyDim, valueDim), DType.F32);
                backend.FrameDeltaScan(output, query, key, value, beta, decay, frames, spatial, heads, keyDim, valueDim, initial, final);
                return [.. Read(backend, output), .. Read(backend, final)];
            }

            (float[] cpuCarried, float[] cudaCarried) = RunOnBoth(RunCarried);
            double carriedError = 0, carriedNorm = 0;
            for (int i = 0; i < cpuCarried.Length; i++)
            {
                carriedError += (double)(cpuCarried[i] - cudaCarried[i]) * (cpuCarried[i] - cudaCarried[i]);
                carriedNorm += (double)cpuCarried[i] * cpuCarried[i];
            }

            double carriedRelative = Math.Sqrt(carriedError / Math.Max(carriedNorm, double.Epsilon));
            _output.WriteLine($"FrameDeltaScan valueDim {valueDim}, carried state: relative L2 {carriedRelative:E3}");
            Assert.False(cpuCarried.AsSpan(0, cpu.Length).SequenceEqual(cpu), "The carried state did not change the scan.");
            Assert.True(carriedRelative < 5e-3, $"Carried-state FrameDeltaScan relative L2 {carriedRelative:E3} against the host reference.");
        }
        finally
        {
            SanaWmTensors.DisposeAll(owned);
        }
    }

    /// <summary>The fused CUDA kernels behind <see cref="IBackend.UcpeTransform"/>, <see cref="IBackend.DepthwiseConvFrames"/> and
    /// <see cref="IBackend.DepthwiseConvGlu"/> (F32 and BF16) against their host references, at the real head and channel widths.</summary>
    [Theory]
    [InlineData("ucpe")]
    [InlineData("frames")]
    [InlineData("rmsrelu")]
    [InlineData("glu-f32")]
    [InlineData("glu-bf16")]
    [InlineData("glu-bf16-vec")]
    public void FusedOps_CudaMatchHostReference(string op)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        Random random = new(4321);
        List<Tensor> owned = [];
        Tensor Fill(long[] dims, DType dtype = default)
        {
            Tensor f32 = new(new TensorShape(dims), DType.F32);
            Span<float> values = f32.AsSpan<float>();
            for (int i = 0; i < values.Length; i++)
                values[i] = (float)random.NextDouble() * 2f - 1f;
            owned.Add(f32);
            if (dtype != DType.BF16)
                return f32;
            Tensor bf16 = f32.CastTo(DType.BF16);
            owned.Add(bf16);
            return bf16;
        }

        try
        {
            Func<IBackend, float[]> run;
            switch (op)
            {
                case "ucpe":
                {
                    const int tokens = 37, heads = 3, headDim = 112;
                    Tensor input = Fill([tokens, heads * headDim]), matrices = Fill([tokens, 4, 4]);
                    Tensor cos = Fill([tokens, headDim / 2]), sin = Fill([tokens, headDim / 2]);
                    run = backend =>
                    {
                        using Tensor output = new(new TensorShape(tokens, heads * headDim), DType.F32);
                        backend.UcpeTransform(output, input, matrices, cos, sin, tokens, heads, headDim);
                        return Read(backend, output);
                    };
                    break;
                }
                case "rmsrelu":
                {
                    const int rows = 29, width = 2240;
                    Tensor input = Fill([rows, width]), weight = Fill([width]);
                    run = backend =>
                    {
                        using Tensor output = new(new TensorShape(rows * 20, width / 20), DType.F32);
                        backend.RmsNormRelu(output, input, weight, 1e-5f, 0.0213f);
                        return Read(backend, output);
                    };
                    break;
                }
                case "frames":
                {
                    const int frames = 6, spatial = 11, channels = 96, taps = 7;
                    Tensor input = Fill([frames * spatial, channels]), weight = Fill([channels, taps]);
                    run = backend =>
                    {
                        using Tensor output = new(new TensorShape(frames * spatial, channels), DType.F32);
                        backend.DepthwiseConvFrames(output, input, weight, frames, spatial, channels, taps, 3);
                        return Read(backend, output);
                    };
                    break;
                }
                default:
                {
                    // glu-bf16-vec takes the vectorized kernel (whole 8-channel groups, whole 4-pixel runs); the others the
                    // per-element one. The BF16 cases read SiLU'd inputs, as the FFN does.
                    DType dtype = op.StartsWith("glu-bf16", StringComparison.Ordinal) ? DType.BF16 : DType.F32;
                    bool inputSilu = dtype == DType.BF16;
                    int frames = 2, height = 5, width = op == "glu-bf16-vec" ? 8 : 7, hidden = 40;
                    Tensor input = Fill([frames * height * width, 2 * hidden], dtype), taps = Fill([3, 3, 2 * hidden]);
                    Tensor bias = Fill([2 * hidden]);
                    run = backend =>
                    {
                        using Tensor output = new(new TensorShape(frames * height * width, hidden), dtype);
                        backend.DepthwiseConvGlu(output, input, taps, bias, frames, height, width, 3, inputSilu);
                        backend.OffloadActivation(output);
                        using Tensor f32 = output.DType == DType.F32 ? new Tensor(output.Shape, DType.F32) : output.CastTo(DType.F32);
                        return output.DType == DType.F32 ? output.AsReadOnlySpan<float>().ToArray() : f32.AsReadOnlySpan<float>().ToArray();
                    };
                    break;
                }
            }

            (float[] cpu, float[] cuda) = RunOnBoth(run);
            double maxAbs = 0;
            for (int i = 0; i < cpu.Length; i++)
                maxAbs = Math.Max(maxAbs, Math.Abs(cpu[i] - cuda[i]));
            _output.WriteLine($"{op}: max abs {maxAbs:E3} over {cpu.Length} values");
            // F32 kernels sum in the reference's tap order up to FMA contraction; BF16 output rounds once (2^-8 relative).
            Assert.True(maxAbs < (op.StartsWith("glu-bf16", StringComparison.Ordinal) ? 2e-2 : 1e-4), $"{op}: CUDA against host max abs {maxAbs:E3}.");
        }
        finally
        {
            SanaWmTensors.DisposeAll(owned);
        }
    }

    private (float[] Cpu, float[] Cuda) RunOnBoth(Func<IBackend, float[]> run)
    {
        float[] cpu;
        using (CpuBackend host = new())
            cpu = run(host);
        using CudaBackend cuda = new(0, PtxDirectory());
        return (cpu, run(cuda));
    }

    private void Compare(string label, (float[] Cpu, float[] Cuda) results)
    {
        Assert.Equal(results.Cpu.Length, results.Cuda.Length);
        double dot = 0, normCpu = 0, normCuda = 0, maxAbs = 0;
        for (int i = 0; i < results.Cpu.Length; i++)
        {
            Assert.True(float.IsFinite(results.Cuda[i]), $"{label}: CUDA element {i} is {results.Cuda[i]}.");
            dot += (double)results.Cpu[i] * results.Cuda[i];
            normCpu += (double)results.Cpu[i] * results.Cpu[i];
            normCuda += (double)results.Cuda[i] * results.Cuda[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(results.Cpu[i] - results.Cuda[i]));
        }
        double cosine = dot / Math.Sqrt(Math.Max(normCpu * normCuda, double.Epsilon));
        _output.WriteLine($"{label}: cosine {cosine:F6}, max abs {maxAbs:E3}");
        Assert.True(cosine >= CosineTarget, $"{label}: CUDA against CPU cosine {cosine:F6} is below {CosineTarget}.");
    }

    // Reads a tensor's values on the host; a device result is synced back first.
    private static float[] Read(IBackend backend, Tensor tensor)
    {
        backend.OffloadActivation(tensor);
        return tensor.AsReadOnlySpan<float>().ToArray();
    }

    private static Tensor Pattern(List<Tensor> owned, long[] dims, float scale)
    {
        Tensor tensor = new(new TensorShape(dims), DType.F32);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
            values[i] = ((i % 97) - 48) * scale;
        owned.Add(tensor);
        return tensor;
    }

    private static float[] Cameras(int batch)
    {
        float[] conditions = new float[batch * Frames * CameraChannels];
        for (int i = 0; i < batch * Frames; i++)
        {
            float[] pose = [1, 0, 0, 0.1f * i, 0, 1, 0, 0, 0, 0, 1, 0.05f * i, 0, 0, 0, 1];
            pose.CopyTo(conditions, i * CameraChannels);
            conditions[i * CameraChannels + 16] = 2f;
            conditions[i * CameraChannels + 17] = 2f;
            conditions[i * CameraChannels + 18] = 1f;
            conditions[i * CameraChannels + 19] = 1f;
        }
        return conditions;
    }

    private static int Int(IReadOnlyDictionary<string, string> meta, string key) => int.Parse(meta[key], CultureInfo.InvariantCulture);

    private static SafeTensorsLoader Fixture(string name)
    {
        SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(AppContext.BaseDirectory, "Fixtures", name));
        return loader;
    }

    private static string PtxDirectory()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate))
                return candidate;
        }

        throw new DirectoryNotFoundException("HartsyInference.Cuda/Ptx was not found above the test binary.");
    }
}
