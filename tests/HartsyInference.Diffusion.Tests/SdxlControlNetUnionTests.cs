using System.Text;
using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Adapters;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Union SDXL ControlNet (xinsir/controlnet-union-sdxl-1.0) tests: loader union detection (Unit),
/// union/single-mode forward guards (Unit + Integration), and numerical parity vs diffusers
/// <c>ControlNetUnionModel</c> on the REAL ProMax checkpoint (Integration, gated on
/// <c>SDXL_CN_UNION_REF_DIR</c> pointing at the output of <c>tests/python-reference/dump_sdxl_controlnet_union.py</c>
/// plus the checkpoint under <c>HARTSY_CONTROLNET_DIR</c>).</summary>
public sealed class SdxlControlNetUnionTests : IDisposable
{
    private const string CheckpointName = "controlnet-union-sdxl-promax.safetensors";

    private readonly string _tempDir;

    public SdxlControlNetUnionTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"sharpinf-sdxluncn-tests-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    private static string CheckpointDir => Environment.GetEnvironmentVariable("HARTSY_CONTROLNET_DIR")
        ?? "/home/hartsy/Desktop/Swarm/SwarmUI.not too old/Models/controlnet";

    // ── Loader detection ─────────────────────────────────────────────

    [Fact]
    public void ControlNetLoader_UnionSdxlKeys_DetectsUnionAndTypeCount()
    {
        string path = CreateSafeTensorsFile("controlnet-union-sdxl-mini.safetensors", new()
        {
            ["add_embedding.linear_1.weight"] = (DType.F32, [8, 16], new float[8 * 16]),
            ["task_embedding"] = (DType.F32, [8, 320], new float[8 * 320]),
            ["control_add_embedding.linear_1.weight"] = (DType.F32, [8, 2048], new float[8 * 2048]),
            ["spatial_ch_projs.weight"] = (DType.F32, [320, 320], new float[320 * 320]),
            ["controlnet_down_blocks.0.weight"] = (DType.F32, [8, 8, 1, 1], new float[8 * 8]),
        });

        using ControlNetFile file = ControlNetLoader.Load(path);
        Assert.Equal(ControlNetBaseModel.Sdxl, file.BaseModel);
        Assert.Equal(8, file.Config.UnionControlTypeCount);
    }

    [Fact]
    public void ControlNetLoader_PlainSdxlKeys_ReportsNoUnion()
    {
        string path = CreateSafeTensorsFile("controlnet-sdxl-plain-canny.safetensors", new()
        {
            ["add_embedding.linear_1.weight"] = (DType.F32, [8, 16], new float[8 * 16]),
            ["controlnet_down_blocks.0.weight"] = (DType.F32, [8, 8, 1, 1], new float[8 * 8]),
        });

        using ControlNetFile file = ControlNetLoader.Load(path);
        Assert.Equal(ControlNetBaseModel.Sdxl, file.BaseModel);
        Assert.Equal(0, file.Config.UnionControlTypeCount);
    }

    // ── Real checkpoint (gated) ──────────────────────────────────────

    [Fact]
    [Trait("Category", "Integration")]
    public void RealCheckpoint_UnionPromax_LoadsAndConstructs()
    {
        string path = Path.Combine(CheckpointDir, CheckpointName);
        if (!File.Exists(path)) return;

        using ControlNetFile file = ControlNetLoader.Load(path);
        Assert.Equal(ControlNetBaseModel.Sdxl, file.BaseModel);
        Assert.Equal(8, file.Config.UnionControlTypeCount);
        Assert.Contains("task_embedding", file.Weights);
        Assert.Contains("transformer_layes.0.attn.in_proj_weight", file.Weights);
        Assert.Contains("spatial_ch_projs.weight", file.Weights);
        Assert.Contains("control_add_embedding.linear_1.weight", file.Weights);

        using ControlNet adapter = new ControlNet(file.Config, UNetConfig.SdxlBase);
        adapter.LoadWeights(file.Weights);
        Assert.Equal(9, adapter.DownResidualCount);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void RealCheckpoint_Union_ForwardWithoutTypeIndex_Throws()
    {
        string path = Path.Combine(CheckpointDir, CheckpointName);
        if (!File.Exists(path)) return;

        using ControlNetFile file = ControlNetLoader.Load(path);
        using ControlNet adapter = new ControlNet(file.Config, UNetConfig.SdxlBase);
        adapter.LoadWeights(file.Weights);

        using CpuBackend backend = new CpuBackend();
        using Tensor latent = new Tensor(new TensorShape(1, 4, 8, 8), DType.F32);
        using Tensor cond = new Tensor(new TensorShape(1, 3, 64, 64), DType.F32);
        using Tensor text = new Tensor(new TensorShape(1, 4, 2048), DType.F32);
        using Tensor pooled = new Tensor(new TensorShape(1, 1280), DType.F32);
        Assert.Throws<HartsyInference.Core.Exceptions.HartsyInferenceException>(() =>
            adapter.Forward(backend, latent, cond, 500f, text, pooled, [1024f, 1024f, 0f, 0f, 1024f, 1024f]));
    }

    // ── Parity vs diffusers ControlNetUnionModel (real weights, CPU) ─

    [Fact]
    [Trait("Category", "Integration")]
    public void Parity_UnionPromax_Canny_MatchesDiffusers()
    {
        RunParity("sdxl_cn_union_ref_canny.safetensors");
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Parity_UnionPromax_DepthBatch2_MatchesDiffusers()
    {
        RunParity("sdxl_cn_union_ref_depth_b2.safetensors");
    }

    private static void RunParity(string refFileName)
    {
        string? refDir = Environment.GetEnvironmentVariable("SDXL_CN_UNION_REF_DIR");
        if (string.IsNullOrEmpty(refDir))
            return; // reference dump not present — see tests/python-reference/dump_sdxl_controlnet_union.py
        string refPath = Path.Combine(refDir, refFileName);
        string ckptPath = Path.Combine(CheckpointDir, CheckpointName);
        if (!File.Exists(refPath) || !File.Exists(ckptPath))
            return;

        using SafeTensorsLoader refLoader = new SafeTensorsLoader();
        refLoader.Load(refPath);
        Dictionary<string, Tensor> io = refLoader.GetAllTensors();

        using ControlNetFile file = ControlNetLoader.Load(ckptPath);
        // The checkpoint ships F16; the reference model runs float32 (state dict upcast). Cast the
        // weights up so the CPU forward is an apples-to-apples float32 comparison.
        Dictionary<string, Tensor> weightsF32 = new Dictionary<string, Tensor>(file.Weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in file.Weights)
        {
            weightsF32[kvp.Key] = kvp.Value.DType == DType.F32 ? kvp.Value : kvp.Value.CastTo(DType.F32);
        }
        try
        {
            using ControlNet adapter = new ControlNet(file.Config, UNetConfig.SdxlBase);
            adapter.LoadWeights(weightsF32);

            float timestep = io["io.timestep"].AsReadOnlySpan<float>()[0];
            float condScale = io["io.conditioning_scale"].AsReadOnlySpan<float>()[0];
            int controlIdx = (int)io["io.control_type_idx"].AsReadOnlySpan<float>()[0];
            ReadOnlySpan<float> timeIds = io["io.time_ids"].AsReadOnlySpan<float>()[..6];

            using CpuBackend backend = new CpuBackend();
            (Tensor[] down, Tensor mid) = adapter.Forward(
                backend,
                io["io.sample"],
                io["io.controlnet_cond"],
                timestep,
                io["io.encoder_hidden_states"],
                io["io.text_embeds"],
                timeIds,
                condScale,
                controlIdx);

            for (int i = 0; i < down.Length; i++)
            {
                AssertParity(io[$"io.down.{i}"], down[i], $"down.{i}");
            }
            AssertParity(io["io.mid"], mid, "mid");

            foreach (Tensor d in down) d.Dispose();
            mid.Dispose();
        }
        finally
        {
            foreach (KeyValuePair<string, Tensor> kvp in weightsF32)
            {
                if (!ReferenceEquals(kvp.Value, file.Weights[kvp.Key])) kvp.Value.Dispose();
            }
        }
    }

    /// <summary>Correlation ≥ 0.9999 (primary gate) plus a max-abs bound RELATIVE to the residual's own
    /// peak magnitude (2e-3 — the deep 1280-channel residuals reach |values| ≈ 50-100, so an absolute
    /// bound punishes them for scale). Both sides are float32 with identical weights, so the deviation is
    /// pure op-ordering accumulation: observed corr ≥ 0.9999999 with relative maxAbs ≤ 4.5e-4 across all
    /// 10 residuals on both cases. Metrics printed per residual for the parity ledger.</summary>
    private static void AssertParity(Tensor expected, Tensor actual, string name)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        ReadOnlySpan<float> e = expected.AsReadOnlySpan<float>();
        ReadOnlySpan<float> a = actual.AsReadOnlySpan<float>();
        double sumE = 0, sumA = 0, sumEE = 0, sumAA = 0, sumEA = 0;
        float maxAbs = 0f, maxMag = 0f;
        for (int i = 0; i < e.Length; i++)
        {
            double ev = e[i], av = a[i];
            sumE += ev; sumA += av; sumEE += ev * ev; sumAA += av * av; sumEA += ev * av;
            float diff = MathF.Abs(e[i] - a[i]);
            if (diff > maxAbs) maxAbs = diff;
            float mag = MathF.Abs(e[i]);
            if (mag > maxMag) maxMag = mag;
        }
        int n = e.Length;
        double cov = sumEA / n - (sumE / n) * (sumA / n);
        double varE = sumEE / n - (sumE / n) * (sumE / n);
        double varA = sumAA / n - (sumA / n) * (sumA / n);
        double corr = cov / Math.Sqrt(Math.Max(varE * varA, 1e-30));
        float relMaxAbs = maxAbs / MathF.Max(maxMag, 1e-6f);
        Console.WriteLine($"[union-cn parity] {name}: corr {corr:F7}, maxAbs {maxAbs:E3} (peak |ref| {maxMag:F2}, rel {relMaxAbs:E2})");
        Assert.True(corr >= 0.9999 && relMaxAbs < 2e-3f,
            $"{name}: corr {corr:F6}, maxAbs {maxAbs:E3} vs peak |ref| {maxMag:F2} (rel {relMaxAbs:E2}; need corr ≥ 0.9999, rel < 2e-3)");
    }

    // ── Helpers ──────────────────────────────────────────────────────

    private string CreateSafeTensorsFile(string name, Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors)
    {
        using MemoryStream dataStream = new();
        Dictionary<string, (long start, long end)> offsets = [];
        foreach (KeyValuePair<string, (DType dtype, long[] shape, float[] data)> kvp in tensors)
        {
            long start = dataStream.Position;
            foreach (float val in kvp.Value.data)
            {
                dataStream.Write(BitConverter.GetBytes(val), 0, 4);
            }
            long end = dataStream.Position;
            offsets[kvp.Key] = (start, end);
        }
        byte[] dataBlob = dataStream.ToArray();

        Dictionary<string, object> header = [];
        foreach (KeyValuePair<string, (DType dtype, long[] shape, float[] data)> kvp in tensors)
        {
            (long start, long end) = offsets[kvp.Key];
            header[kvp.Key] = new Dictionary<string, object>
            {
                ["dtype"] = kvp.Value.dtype.Name,
                ["shape"] = kvp.Value.shape,
                ["data_offsets"] = new long[] { start, end },
            };
        }

        byte[] headerBytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        string filePath = Path.Combine(_tempDir, name);
        using FileStream fs = new(filePath, FileMode.Create, FileAccess.Write);
        using BinaryWriter writer = new(fs);
        writer.Write((long)headerBytes.Length);
        writer.Write(headerBytes);
        writer.Write(dataBlob);
        return filePath;
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, true);
        }
        catch (Exception)
        {
        }
    }
}
