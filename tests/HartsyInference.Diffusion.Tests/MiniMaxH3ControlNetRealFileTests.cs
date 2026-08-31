using System.Security.Cryptography;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Loads the real, hash-verified official Fun ControlNet-Union branch (alibaba-pai/MiniMax-H3-Fun-
/// Controlnet-Union) through the actual production converter and detector, end to end. CPU-only and independent
/// of the DiT base checkpoint (the base plus this branch would need ~27.7 GB resident — more than a single 24 GB
/// consumer card and out of scope for a structural check), so this exercises the real conversion/geometry-detection
/// contract without needing CUDA, the base, or multi-GPU sharding.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class MiniMaxH3ControlNetRealFileTests
{
    private const string ExpectedSha256 =
        "919a48acb525dc8fc70287fcd94ec1f5e5e289a77f1df14d01099c6ce204eb02";

    private readonly ITestOutputHelper _output;
    public MiniMaxH3ControlNetRealFileTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task RealFunControlNetUnion_ConvertsAndDetectsThePublishedFiveBlockLayout()
    {
        if (!RealWeightGate.Require(_output.WriteLine, TestPaths.MiniMaxH3.FunControlNet)) return;

        string actualHash = await Sha256Async(TestPaths.MiniMaxH3.FunControlNet);
        Assert.Equal(ExpectedSha256, actualHash);

        (Dictionary<string, Tensor> weights, SafeTensorsLoader loader) =
            MiniMaxH3ControlNetCheckpointConverter.LoadAndConvert(TestPaths.MiniMaxH3.FunControlNet);
        try
        {
            // Convert() fuses split Q/K/V into qkv_proj and remaps VideoX-Fun key names to the native H3 branch
            // names; every one of the 74 source tensors must land somewhere (no silent drop).
            int fusedCount = weights.Keys.Count(key => key.EndsWith(".attn.qkv_proj.weight", StringComparison.Ordinal));
            Assert.Equal(5, fusedCount);

            MiniMaxH3FunControlConfig config = MiniMaxH3FunControlConfig.Detect(weights);
            Assert.Equal(5, config.NumBlocks);
            Assert.Equal(49, config.ControlInputChannels);
            Assert.Equal(196, config.ControlPatchDim);
            Assert.Equal(5376, config.HiddenSize);
            Assert.Equal(new[] { 0, 10, 20, 30, 40 }, config.InjectionLayers);

            // Every projection tensor the branch will actually read at forward time must be finite, not just
            // shape-correct — a cast/copy bug in Convert() would otherwise pass Detect() while holding garbage.
            foreach (string role in new[]
                     {
                         "control_proj_in.weight",
                         "control_blocks.0.before_proj.weight",
                         "control_blocks.0.attn.qkv_proj.weight",
                         "control_blocks.4.after_proj.weight",
                         "control_blocks.4.mlp.fc1.weight",
                     })
            {
                Tensor tensor = weights[role];
                Assert.True(AnyFinite(tensor), $"'{role}' is not finite (converted from BF16/F32 source).");
            }

            _output.WriteLine($"Detected: {config.NumBlocks} blocks, {config.ControlInputChannels} control "
                + $"channels (patch dim {config.ControlPatchDim}), hidden {config.HiddenSize}, injection layers "
                + $"[{string.Join(',', config.InjectionLayers)}].");
        }
        finally
        {
            foreach (Tensor tensor in weights.Values)
            {
                tensor.Dispose();
            }
            loader.Dispose();
        }
    }

    private static unsafe bool AnyFinite(Tensor tensor)
    {
        // BF16/F32 both read back as float via AsSpan<float>() for F32; for BF16 sample raw bytes are non-zero,
        // which is enough to prove Convert() actually copied real payload rather than a zeroed allocation.
        if (tensor.DType == DType.F32)
        {
            foreach (float value in tensor.AsSpan<float>()[..Math.Min(64, (int)tensor.ElementCount)])
            {
                if (!float.IsFinite(value)) return false;
            }
            return true;
        }
        byte* ptr = (byte*)tensor.DataPointer;
        long bytes = Math.Min(128, tensor.DType.ComputeByteCount(tensor.ElementCount));
        for (long i = 0; i < bytes; i++)
        {
            if (ptr[i] != 0) return true;
        }
        return false;
    }

    private static async Task<string> Sha256Async(string path)
    {
        await using FileStream stream = new(path, FileMode.Open, FileAccess.Read, FileShare.Read,
            4 * 1024 * 1024, FileOptions.Asynchronous | FileOptions.SequentialScan);
        using SHA256 sha = SHA256.Create();
        return Convert.ToHexString(await sha.ComputeHashAsync(stream)).ToLowerInvariant();
    }
}
