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
