using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ThreeD.Models.Utonia;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests.Utonia;

/// <summary>
/// Real-weight checks. Need Models/Utonia/utonia.pth (Pointcept/Utonia, CC-BY-NC-4.0, not downloaded by this suite).
/// With the file absent each test writes a SKIPPED line and returns; HARTSY_REQUIRE_REAL_WEIGHTS=1 fails instead.
/// </summary>
[Trait("Category", "Integration")]
public sealed class UtoniaCheckpointIntegrationTests
{
    private const string CheckpointName = "utonia.pth";
    private const string RequireEnv = "HARTSY_REQUIRE_REAL_WEIGHTS";
    private const string StateDictPrefix = "state_dict.";

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
        float ropeBase = UtoniaTestData.TinyConfig().RopeBase;

        int checkedBuffers = 0;
        foreach (string key in loader.Descriptors.Keys)
        {
            if (!UtoniaCheckpointKeys.IsComputedBuffer(StripStateDictPrefix(key)))
            {
                continue;
            }

            PytorchTensorDescriptor descriptor = loader.Descriptors[key];
            Assert.Equal(DType.F32, descriptor.DType);
            float[] expected = UtoniaRope.InvFrequencies(2 * (int)descriptor.Shape[0], ropeBase);
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
        string? path = FindCheckpoint(CheckpointName);
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
