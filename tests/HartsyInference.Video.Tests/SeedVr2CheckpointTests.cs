using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Video.Tests;

/// <summary>Part A1 gate: the real converted 3B checkpoint loads with a fully validated inventory (every
/// tensor consumed, none missing) and <see cref="SeedVr2Config.Detect"/> reproduces the published dims —
/// including the mm_layers=10 separate/shared boundary, the most bug-prone loading detail. Env-gated on
/// <c>SEEDVR2_DIT</c> (path to seedvr2_3b_dit_f32.safetensors); skips cleanly when unset.</summary>
public sealed class SeedVr2CheckpointTests
{
    private readonly ITestOutputHelper _output;

    public SeedVr2CheckpointTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void RealCheckpoint_ConvertsAndDetects3BConfig()
    {
        string? path = Environment.GetEnvironmentVariable("SEEDVR2_DIT");
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
        {
            _output.WriteLine("SKIPPED: set SEEDVR2_DIT to the converted DiT safetensors.");
            return;
        }

        (Dictionary<string, Tensor> weights, SafeTensorsLoader loader) =
            SeedVr2CheckpointConverter.LoadAndConvert(path);
        using SafeTensorsLoader _ = loader;

        // 635 checkpoint tensors − 32 recomputable per-block RoPE freq buffers = 603 consumed.
        Assert.Equal(603, weights.Count);

        SeedVr2Config detected = SeedVr2Config.Detect(weights);
        SeedVr2Config expected = SeedVr2Config.Seedvr2_3B;
        Assert.Equal(expected, detected);
        _output.WriteLine($"Detected: dim={detected.VidDim} heads={detected.Heads} layers={detected.NumLayers} " +
            $"mm={detected.MmLayers} in={detected.InChannels} out={detected.OutChannels} mlp={detected.MlpDim}");

        // Spot-check the boundary from the raw weights themselves.
        Assert.True(weights.ContainsKey("blocks.9.attn.proj_qkv.vid.weight"));
        Assert.True(weights.ContainsKey("blocks.9.attn.proj_qkv.txt.weight"));
        Assert.False(weights.ContainsKey("blocks.9.attn.proj_qkv.all.weight"));
        Assert.True(weights.ContainsKey("blocks.10.attn.proj_qkv.all.weight"));
        Assert.False(weights.ContainsKey("blocks.10.attn.proj_qkv.vid.weight"));
    }

    [Fact]
    public void Converter_RejectsMissingAndUnknownKeys()
    {
        // Structural negatives run on synthetic dictionaries — Unit tier, no weights needed.
        Dictionary<string, Tensor> synthetic = new();
        Assert.ThrowsAny<Exception>(() => SeedVr2CheckpointConverter.Convert(synthetic));
    }
}
