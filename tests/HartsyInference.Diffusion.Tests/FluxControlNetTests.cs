using System.Text;
using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Adapters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Flux DiT ControlNet adapter tests: loader detection + config derivation (Unit), residual
/// shape/zero-init behavior on tiny synthetic weights (Unit, CPU), and numerical parity vs diffusers
/// <c>FluxControlNetModel</c> (Integration, gated on <c>FLUX_CONTROLNET_REF_DIR</c> pointing at the output of
/// <c>tests/python-reference/dump_flux_controlnet.py</c>).</summary>
public sealed class FluxControlNetTests : IDisposable
{
    private const int Hidden = 16;
    private const int Heads = 2;
    private const int InChannels = 8;
    private const int ContextDim = 32;
    private const int PooledDim = 24;
    private const int HPacked = 4;
    private const int WPacked = 4;
    private const int ImgSeq = HPacked * WPacked;
    private const int TxtSeq = 6;

    private readonly string _tempDir;

    public FluxControlNetTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"sharpinf-fluxcn-tests-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    private static FluxControlNetConfig TinyConfig(int depth, int depthSingle, int? numMode) => new()
    {
        HiddenSize = Hidden,
        NumHeads = Heads,
        Depth = depth,
        DepthSingleBlocks = depthSingle,
        InChannels = InChannels,
        ContextInDim = ContextDim,
        PooledProjectionDim = PooledDim,
        GuidanceEmbed = true,
        NumMode = numMode,
        AxesDim = [2, 4, 2],
    };

    // ── Loader detection ─────────────────────────────────────────────

    [Fact]
    public void ControlNetLoader_FluxUnionPro2_DerivesDepthsAndNoMode()
    {
        // Miniature of the Shakker Union-Pro-2.0 header: 2 double blocks, 0 single, guidance, NO mode embedder.
        Dictionary<string, (DType, long[], float[])> tensors = MiniFluxCheckpointTensors(depth: 2, depthSingle: 0, numMode: null);
        string path = CreateSafeTensorsFile("flux-union-pro2-instantx.safetensors", tensors);

        using ControlNetFile file = ControlNetLoader.Load(path);
        Assert.Equal(ControlNetBaseModel.Flux, file.BaseModel);
        Assert.NotNull(file.FluxConfig);
        Assert.Equal(2, file.FluxConfig!.Depth);
        Assert.Equal(0, file.FluxConfig.DepthSingleBlocks);
        Assert.True(file.FluxConfig.GuidanceEmbed);
        Assert.False(file.FluxConfig.IsUnion);
        Assert.Equal(Hidden, file.FluxConfig.HiddenSize);
        Assert.Equal(InChannels, file.FluxConfig.InChannels);
        Assert.Equal(ContextDim, file.FluxConfig.ContextInDim);
    }

    [Fact]
    public void ControlNetLoader_FluxUnion_DerivesModeCountAndSingleBlocks()
    {
        // Miniature of the InstantX Union header: mode embedder present + single blocks.
        Dictionary<string, (DType, long[], float[])> tensors = MiniFluxCheckpointTensors(depth: 2, depthSingle: 2, numMode: 7);
        string path = CreateSafeTensorsFile("flux-union-instantx.safetensors", tensors);

        using ControlNetFile file = ControlNetLoader.Load(path);
        Assert.Equal(ControlNetBaseModel.Flux, file.BaseModel);
        Assert.True(file.FluxConfig!.IsUnion);
        Assert.Equal(7, file.FluxConfig.NumMode);
        Assert.Equal(2, file.FluxConfig.Depth);
        Assert.Equal(2, file.FluxConfig.DepthSingleBlocks);
    }

    [Fact]
    public void FluxControlNet_LoadsFromControlNetFile_AndRuns()
    {
        Dictionary<string, (DType, long[], float[])> tensors = MiniFluxCheckpointTensors(depth: 1, depthSingle: 1, numMode: 7, randomize: true);
        string path = CreateSafeTensorsFile("flux-union-e2e.safetensors", tensors);

        using ControlNetFile file = ControlNetLoader.Load(path);
        // Real Flux checkpoints keep the default [16, 56, 56] RoPE split; the tiny 8-dim heads here need a
        // matching split (the axes split is a convention, not derivable from tensor shapes).
        using FluxControlNet adapter = new FluxControlNet(file.FluxConfig! with { AxesDim = [2, 4, 2] });
        adapter.LoadWeights(file.Weights);

        using CpuBackend backend = new CpuBackend();
        (Tensor latent, Tensor control, Tensor t5, Tensor pooled) = MakeInputs(seed: 7);
        FluxControlNetResiduals residuals = adapter.Forward(
            backend, latent, control, sigma: 0.5f, t5, pooled, guidanceScale: 3.5f,
            HPacked, WPacked, unionModeId: (int)FluxUnionControlMode.Depth, conditioningScale: 1.0f);

        Assert.Single(residuals.DoubleBlockResiduals);
        Assert.Single(residuals.SingleBlockResiduals);
        residuals.DisposeAll();
        latent.Dispose(); control.Dispose(); t5.Dispose(); pooled.Dispose();
    }

    // ── Residual shapes + zero-init behavior ─────────────────────────

    [Fact]
    public void Forward_ResidualShapes_MatchImageStream()
    {
        FluxControlNetConfig config = TinyConfig(depth: 2, depthSingle: 2, numMode: 7);
        Dictionary<string, Tensor> weights = RandomWeights(config, seed: 42);
        using FluxControlNet adapter = new FluxControlNet(config);
        adapter.LoadWeights(weights);

        using CpuBackend backend = new CpuBackend();
        (Tensor latent, Tensor control, Tensor t5, Tensor pooled) = MakeInputs(seed: 11);
        FluxControlNetResiduals residuals = adapter.Forward(
            backend, latent, control, sigma: 0.7f, t5, pooled, guidanceScale: 3.5f,
            HPacked, WPacked, unionModeId: (int)FluxUnionControlMode.Canny, conditioningScale: 0.8f);

        Assert.Equal(2, residuals.DoubleBlockResiduals.Length);
        Assert.Equal(2, residuals.SingleBlockResiduals.Length);
        foreach (Tensor r in residuals.DoubleBlockResiduals)
        {
            Assert.Equal(new TensorShape(1, ImgSeq, Hidden), r.Shape);
            Assert.False(AllZero(r));
        }
        foreach (Tensor r in residuals.SingleBlockResiduals)
        {
            Assert.Equal(new TensorShape(1, ImgSeq, Hidden), r.Shape);
            Assert.False(AllZero(r));
        }

        residuals.DisposeAll();
        DisposeAll(weights);
        latent.Dispose(); control.Dispose(); t5.Dispose(); pooled.Dispose();
    }

    [Fact]
    public void Forward_ZeroInitControlnetBlocks_ProduceZeroResiduals()
    {
        FluxControlNetConfig config = TinyConfig(depth: 2, depthSingle: 1, numMode: null);
        Dictionary<string, Tensor> weights = RandomWeights(config, seed: 42);
        // Fresh-from-training-init checkpoint: the controlnet output Linears are all-zero → the adapter must
        // contribute exactly nothing (the zero-module guarantee the architecture is built on).
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            if (kvp.Key.StartsWith("controlnet_blocks.", StringComparison.Ordinal)
                || kvp.Key.StartsWith("controlnet_single_blocks.", StringComparison.Ordinal))
            {
                kvp.Value.AsSpan<float>().Clear();
            }
        }
        using FluxControlNet adapter = new FluxControlNet(config);
        adapter.LoadWeights(weights);

        using CpuBackend backend = new CpuBackend();
        (Tensor latent, Tensor control, Tensor t5, Tensor pooled) = MakeInputs(seed: 13);
        FluxControlNetResiduals residuals = adapter.Forward(
            backend, latent, control, sigma: 0.3f, t5, pooled, guidanceScale: 3.5f, HPacked, WPacked);

        foreach (Tensor r in residuals.DoubleBlockResiduals) Assert.True(AllZero(r));
        foreach (Tensor r in residuals.SingleBlockResiduals) Assert.True(AllZero(r));

        residuals.DisposeAll();
        DisposeAll(weights);
        latent.Dispose(); control.Dispose(); t5.Dispose(); pooled.Dispose();
    }

    [Fact]
    public void Forward_ConditioningScale_ScalesResidualsLinearly()
    {
        FluxControlNetConfig config = TinyConfig(depth: 1, depthSingle: 0, numMode: null);
        Dictionary<string, Tensor> weights = RandomWeights(config, seed: 42);
        using FluxControlNet adapter = new FluxControlNet(config);
        adapter.LoadWeights(weights);

        using CpuBackend backend = new CpuBackend();
        (Tensor latent, Tensor control, Tensor t5, Tensor pooled) = MakeInputs(seed: 17);
        FluxControlNetResiduals full = adapter.Forward(
            backend, latent, control, 0.5f, t5, pooled, 3.5f, HPacked, WPacked, null, conditioningScale: 1.0f);
        FluxControlNetResiduals half = adapter.Forward(
            backend, latent, control, 0.5f, t5, pooled, 3.5f, HPacked, WPacked, null, conditioningScale: 0.5f);

        ReadOnlySpan<float> f = full.DoubleBlockResiduals[0].AsReadOnlySpan<float>();
        ReadOnlySpan<float> h = half.DoubleBlockResiduals[0].AsReadOnlySpan<float>();
        for (int i = 0; i < f.Length; i++)
        {
            Assert.True(MathF.Abs(h[i] - 0.5f * f[i]) < 1e-5f, $"scale mismatch at {i}: {h[i]} vs {0.5f * f[i]}");
        }

        full.DisposeAll();
        half.DisposeAll();
        DisposeAll(weights);
        latent.Dispose(); control.Dispose(); t5.Dispose(); pooled.Dispose();
    }

    [Fact]
    public void Forward_UnionCheckpointWithoutModeId_Throws()
    {
        FluxControlNetConfig config = TinyConfig(depth: 1, depthSingle: 0, numMode: 7);
        Dictionary<string, Tensor> weights = RandomWeights(config, seed: 42);
        using FluxControlNet adapter = new FluxControlNet(config);
        adapter.LoadWeights(weights);

        using CpuBackend backend = new CpuBackend();
        (Tensor latent, Tensor control, Tensor t5, Tensor pooled) = MakeInputs(seed: 19);
        Assert.Throws<HartsyInference.Core.Exceptions.HartsyInferenceException>(() =>
            adapter.Forward(backend, latent, control, 0.5f, t5, pooled, 3.5f, HPacked, WPacked, unionModeId: null));

        DisposeAll(weights);
        latent.Dispose(); control.Dispose(); t5.Dispose(); pooled.Dispose();
    }

    // ── Real checkpoint (gated) ──────────────────────────────────────

    [Fact]
    [Trait("Category", "Integration")]
    public void RealCheckpoint_UnionPro2_LoadsAndConstructs()
    {
        // Shakker-Labs FLUX.1-dev-ControlNet-Union-Pro-2.0: 6 double blocks, 0 single, guidance embedder,
        // mode embedder REMOVED. Point FLUX_CN_CHECKPOINT at the safetensors to enable.
        string? path = Environment.GetEnvironmentVariable("FLUX_CN_CHECKPOINT");
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
            return;

        using ControlNetFile file = ControlNetLoader.Load(path);
        Assert.Equal(ControlNetBaseModel.Flux, file.BaseModel);
        FluxControlNetConfig config = file.FluxConfig!;
        Assert.Equal(3072, config.HiddenSize);
        Assert.Equal(24, config.NumHeads);
        Assert.Equal(6, config.Depth);
        Assert.Equal(0, config.DepthSingleBlocks);
        Assert.Equal(64, config.InChannels);
        Assert.Equal(4096, config.ContextInDim);
        Assert.Equal(768, config.PooledProjectionDim);
        Assert.True(config.GuidanceEmbed);
        Assert.False(config.IsUnion);

        using FluxControlNet adapter = new FluxControlNet(config);
        adapter.LoadWeights(file.Weights);
        int weightCount = 0;
        foreach (Tensor _ in adapter.EnumerateWeights()) weightCount++;
        Assert.Equal(222, weightCount);
    }

    // ── Parity vs diffusers FluxControlNetModel ──────────────────────

    [Fact]
    [Trait("Category", "Integration")]
    public void Parity_UnionVariant_MatchesDiffusers()
    {
        RunParity("flux_controlnet_union_ref.safetensors", TinyConfig(depth: 2, depthSingle: 2, numMode: 7),
            unionModeId: (int)FluxUnionControlMode.Depth);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void Parity_PlainVariant_MatchesDiffusers()
    {
        RunParity("flux_controlnet_plain_ref.safetensors", TinyConfig(depth: 2, depthSingle: 0, numMode: null),
            unionModeId: null);
    }

    private static void RunParity(string fileName, FluxControlNetConfig config, int? unionModeId)
    {
        string? refDir = Environment.GetEnvironmentVariable("FLUX_CONTROLNET_REF_DIR");
        if (string.IsNullOrEmpty(refDir))
            return; // reference dump not present — see tests/python-reference/dump_flux_controlnet.py
        string path = Path.Combine(refDir, fileName);
        if (!File.Exists(path))
            return;

        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(path);
        Dictionary<string, Tensor> all = loader.GetAllTensors();
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>();
        foreach (KeyValuePair<string, Tensor> kvp in all)
        {
            if (kvp.Key.StartsWith("weights.", StringComparison.Ordinal))
                weights[kvp.Key["weights.".Length..]] = kvp.Value;
        }

        using FluxControlNet adapter = new FluxControlNet(config);
        adapter.LoadWeights(weights);

        float sigma = all["io.timestep"].AsReadOnlySpan<float>()[0];
        float guidance = all["io.guidance"].AsReadOnlySpan<float>()[0];

        using CpuBackend backend = new CpuBackend();
        FluxControlNetResiduals residuals = adapter.Forward(
            backend,
            all["io.hidden_states"],
            all["io.controlnet_cond"],
            sigma,
            all["io.encoder_hidden_states"],
            all["io.pooled_projections"],
            guidance,
            HPacked, WPacked,
            unionModeId,
            conditioningScale: 0.8f); // must match COND_SCALE in the dump script

        for (int i = 0; i < residuals.DoubleBlockResiduals.Length; i++)
            AssertClose(all[$"io.block_sample.{i}"], residuals.DoubleBlockResiduals[i], 1e-4f, $"block_sample.{i}");
        for (int i = 0; i < residuals.SingleBlockResiduals.Length; i++)
            AssertClose(all[$"io.single_block_sample.{i}"], residuals.SingleBlockResiduals[i], 1e-4f, $"single_block_sample.{i}");

        residuals.DisposeAll();
    }

    private static void AssertClose(Tensor expected, Tensor actual, float tolerance, string name)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        ReadOnlySpan<float> e = expected.AsReadOnlySpan<float>();
        ReadOnlySpan<float> a = actual.AsReadOnlySpan<float>();
        float maxAbs = 0f;
        for (int i = 0; i < e.Length; i++)
        {
            float diff = MathF.Abs(e[i] - a[i]);
            if (diff > maxAbs) maxAbs = diff;
        }
        Assert.True(maxAbs < tolerance, $"{name}: maxAbs {maxAbs:E3} exceeds {tolerance:E1}");
    }

    // ── Helpers ──────────────────────────────────────────────────────

    private static (Tensor latent, Tensor control, Tensor t5, Tensor pooled) MakeInputs(int seed)
    {
        Random rng = new Random(seed);
        return (
            RandomTensor(rng, 1, ImgSeq, InChannels),
            RandomTensor(rng, 1, ImgSeq, InChannels),
            RandomTensor(rng, 1, TxtSeq, ContextDim),
            RandomTensor(rng, 1, PooledDim));
    }

    private static Tensor RandomTensor(Random rng, params long[] dims)
    {
        Tensor t = new Tensor(new TensorShape(dims), DType.F32);
        Span<float> span = t.AsSpan<float>();
        for (int i = 0; i < span.Length; i++)
            span[i] = (float)(rng.NextDouble() * 2.0 - 1.0) * 0.5f;
        return t;
    }

    private static bool AllZero(Tensor t)
    {
        foreach (float v in t.AsReadOnlySpan<float>())
        {
            if (v != 0f) return false;
        }
        return true;
    }

    private static void DisposeAll(Dictionary<string, Tensor> weights)
    {
        foreach (Tensor t in weights.Values) t.Dispose();
    }

    /// <summary>Builds the full random weight dictionary for a tiny <see cref="FluxControlNet"/> (diffusers key
    /// layout), sized from the config.</summary>
    private static Dictionary<string, Tensor> RandomWeights(FluxControlNetConfig config, int seed)
    {
        Random rng = new Random(seed);
        int h = config.HiddenSize;
        int mlp = (int)(h * config.MlpRatio);
        int headDim = h / config.NumHeads;
        Dictionary<string, Tensor> w = new Dictionary<string, Tensor>();

        void Lin(string name, int outDim, int inDim)
        {
            w[$"{name}.weight"] = RandomTensor(rng, outDim, inDim);
            w[$"{name}.bias"] = RandomTensor(rng, outDim);
        }

        Lin("x_embedder", h, config.InChannels);
        Lin("controlnet_x_embedder", h, config.InChannels);
        Lin("context_embedder", h, config.ContextInDim);
        Lin("time_text_embed.timestep_embedder.linear_1", h, 256);
        Lin("time_text_embed.timestep_embedder.linear_2", h, h);
        Lin("time_text_embed.text_embedder.linear_1", h, config.PooledProjectionDim);
        Lin("time_text_embed.text_embedder.linear_2", h, h);
        if (config.GuidanceEmbed)
        {
            Lin("time_text_embed.guidance_embedder.linear_1", h, 256);
            Lin("time_text_embed.guidance_embedder.linear_2", h, h);
        }
        if (config.NumMode is int numMode)
            w["controlnet_mode_embedder.weight"] = RandomTensor(rng, numMode, h);

        for (int i = 0; i < config.Depth; i++)
        {
            string p = $"transformer_blocks.{i}";
            Lin($"{p}.norm1.linear", 6 * h, h);
            Lin($"{p}.norm1_context.linear", 6 * h, h);
            Lin($"{p}.attn.to_q", h, h);
            Lin($"{p}.attn.to_k", h, h);
            Lin($"{p}.attn.to_v", h, h);
            Lin($"{p}.attn.add_q_proj", h, h);
            Lin($"{p}.attn.add_k_proj", h, h);
            Lin($"{p}.attn.add_v_proj", h, h);
            Lin($"{p}.attn.to_out.0", h, h);
            Lin($"{p}.attn.to_add_out", h, h);
            w[$"{p}.attn.norm_q.weight"] = RandomTensor(rng, headDim);
            w[$"{p}.attn.norm_k.weight"] = RandomTensor(rng, headDim);
            w[$"{p}.attn.norm_added_q.weight"] = RandomTensor(rng, headDim);
            w[$"{p}.attn.norm_added_k.weight"] = RandomTensor(rng, headDim);
            Lin($"{p}.ff.net.0.proj", mlp, h);
            Lin($"{p}.ff.net.2", h, mlp);
            Lin($"{p}.ff_context.net.0.proj", mlp, h);
            Lin($"{p}.ff_context.net.2", h, mlp);
            Lin($"controlnet_blocks.{i}", h, h);
        }

        for (int i = 0; i < config.DepthSingleBlocks; i++)
        {
            string p = $"single_transformer_blocks.{i}";
            Lin($"{p}.norm.linear", 3 * h, h);
            Lin($"{p}.attn.to_q", h, h);
            Lin($"{p}.attn.to_k", h, h);
            Lin($"{p}.attn.to_v", h, h);
            w[$"{p}.attn.norm_q.weight"] = RandomTensor(rng, headDim);
            w[$"{p}.attn.norm_k.weight"] = RandomTensor(rng, headDim);
            Lin($"{p}.proj_mlp", mlp, h);
            Lin($"{p}.proj_out", h, h + mlp);
            Lin($"controlnet_single_blocks.{i}", h, h);
        }

        return w;
    }

    /// <summary>Miniature Flux ControlNet checkpoint tensors for loader tests (a representative subset with the
    /// architecture-defining keys; block internals included so <c>FluxControlNet.LoadWeights</c> succeeds when
    /// <paramref name="randomize"/> is true).</summary>
    private static Dictionary<string, (DType, long[], float[])> MiniFluxCheckpointTensors(
        int depth, int depthSingle, int? numMode, bool randomize = false)
    {
        FluxControlNetConfig config = TinyConfig(depth, depthSingle, numMode);
        Dictionary<string, Tensor> weights = RandomWeights(config, seed: randomize ? 99 : 1);
        Dictionary<string, (DType, long[], float[])> tensors = new Dictionary<string, (DType, long[], float[])>();
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            long[] shape = new long[kvp.Value.Shape.Rank];
            for (int d = 0; d < shape.Length; d++) shape[d] = kvp.Value.Shape[d];
            tensors[kvp.Key] = (DType.F32, shape, kvp.Value.AsReadOnlySpan<float>().ToArray());
        }
        DisposeAll(weights);
        return tensors;
    }

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
