using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ThreeD.Models.Utonia;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Utonia;

/// <summary>
/// Real-weight checks. Need Models/Utonia/utonia.pth (Pointcept/Utonia, CC-BY-NC-4.0, not downloaded by this suite).
/// HARTSY_UTONIA_WEIGHTS overrides the path. With the file absent each test writes a SKIPPED line and returns;
/// HARTSY_REQUIRE_REAL_WEIGHTS=1 fails instead.
/// </summary>
[Trait("Category", "Integration")]
public sealed class UtoniaCheckpointIntegrationTests
{
    private const string CheckpointName = "utonia.pth";
    private const string WeightsEnv = "HARTSY_UTONIA_WEIGHTS";
    private const string RequireEnv = "HARTSY_REQUIRE_REAL_WEIGHTS";
    private const string StateDictPrefix = "state_dict.";

    // The loader keeps tensors only: channels come from the qkv shapes, heads from the rope length (headDim 18).
    private static readonly RopeStages Checkpoint = new()
    {
        EncChannels = [54, 108, 216, 432, 576],
        EncNumHeads = [3, 6, 12, 24, 32],
        RopeBase = 10f,
    };

    // Pretraining-only keys the encoder deliberately never loads (see UtoniaCheckpointKeysTests).
    private static readonly string[] PretrainingOnlyKeys = ["embedding.mask_token"];

    private readonly ITestOutputHelper _output;

    public UtoniaCheckpointIntegrationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Checkpoint_EveryStateDictKeyIsMappedOrAKnownComputedBuffer()
    {
        string? path = RequireCheckpoint();
        if (path is null)
        {
            return;
        }

        using PytorchPickleLoader loader = new();
        loader.Load(path, recursiveFlatten: true);

        List<string> unmapped = new();
        foreach (string key in loader.Descriptors.Keys)
        {
            string stateKey = StripStateDictPrefix(key);
            if (UtoniaCheckpointKeys.ToCSharpName(stateKey) is null
                && !UtoniaCheckpointKeys.IsComputedBuffer(stateKey)
                && !PretrainingOnlyKeys.Contains(stateKey, StringComparer.Ordinal))
            {
                unmapped.Add(key);
            }
        }

        Assert.Empty(unmapped);
    }

    [Fact]
    public void Checkpoint_RopeInvFreqBuffersMatchTheEncoderTable()
    {
        string? path = RequireCheckpoint();
        if (path is null)
        {
            return;
        }

        using PytorchPickleLoader loader = new();
        loader.Load(path, recursiveFlatten: true);
        Dictionary<string, Tensor> tensors = loader.GetAllTensors();

        int checkedBuffers = 0;
        foreach (string key in loader.Descriptors.Keys)
        {
            string stateKey = StripStateDictPrefix(key);
            if (!UtoniaCheckpointKeys.IsComputedBuffer(stateKey))
            {
                continue;
            }

            int stage = EncoderStage(stateKey);
            int headDim = Checkpoint.EncChannels[stage] / Checkpoint.EncNumHeads[stage];
            PytorchTensorDescriptor descriptor = loader.Descriptors[key];
            Assert.Equal(DType.F32, descriptor.DType);
            Assert.Equal(headDim / 6, (int)descriptor.Shape[0]);

            // Each head splits into x, y and z chunks; each chunk rotates chunk/2 pairs.
            float[] expected = UtoniaRope.InvFrequencies(headDim / 3, Checkpoint.RopeBase);
            float[] actual = UtoniaTestData.Read(tensors[key]);

            Assert.Equal(expected.Length, actual.Length);
            for (int j = 0; j < expected.Length; j++)
            {
                Assert.True(MathF.Abs(actual[j] - expected[j]) <= 1e-6f * MathF.Max(1f, MathF.Abs(expected[j])),
                    $"{key}[{j}] is {actual[j]}, the encoder table gives {expected[j]}.");
            }

            checkedBuffers++;
        }

        Assert.True(checkedBuffers > 0, $"No attn.rope.inv_freq buffer found in {CheckpointName}.");
    }

    /// <summary>Returns the checkpoint path, or null after recording why the check did not run.</summary>
    private string? RequireCheckpoint()
    {
        string? path = Environment.GetEnvironmentVariable(WeightsEnv) ?? FindCheckpoint(CheckpointName);
        if (path is not null)
        {
            return path;
        }

        // xunit 2 has no runtime skip, so an absent checkpoint returns early and the test reports as passed.
        string message = $"Models/Utonia/{CheckpointName} not found; real-checkpoint check not run.";
        if (Environment.GetEnvironmentVariable(RequireEnv) == "1")
        {
            throw new InvalidOperationException($"{RequireEnv}=1 but {message}");
        }

        _output.WriteLine($"SKIPPED: {message}");
        return null;
    }

    private static string StripStateDictPrefix(string key) =>
        key.StartsWith(StateDictPrefix, StringComparison.Ordinal) ? key[StateDictPrefix.Length..] : key;

    /// <summary>Stage index of an <c>enc.enc{s}.block{b}.attn.rope.inv_freq</c> key.</summary>
    private static int EncoderStage(string stateKey)
    {
        string[] parts = stateKey.Split('.');
        if (parts.Length != 6 || parts[0] != "enc" || !parts[1].StartsWith("enc", StringComparison.Ordinal)
            || !int.TryParse(parts[1].AsSpan(3), out int stage) || stage >= Checkpoint.EncChannels.Length)
        {
            throw new InvalidOperationException($"Unexpected rope buffer key {stateKey}.");
        }

        return stage;
    }

    /// <summary>Encoder stage widths, head counts and RoPE base that the checkpoint's rope buffers are checked against.</summary>
    private sealed record RopeStages
    {
        public required int[] EncChannels { get; init; }
        public required int[] EncNumHeads { get; init; }
        public required float RopeBase { get; init; }
    }

    private static string? FindCheckpoint(string fileName)
    {
        DirectoryInfo? dir = new(AppContext.BaseDirectory);
        while (dir is not null)
        {
            string candidate = Path.Combine(dir.FullName, "Models", "Utonia", fileName);
            if (File.Exists(candidate))
            {
                return candidate;
            }

            dir = dir.Parent;
        }

        return null;
    }
}
