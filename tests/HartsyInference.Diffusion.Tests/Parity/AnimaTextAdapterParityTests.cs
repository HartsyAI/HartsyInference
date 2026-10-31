using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The Anima text adapter against ComfyUI's LLMAdapter, on the same weights and inputs.</summary>
/// <remarks>The dump comes from tests/python-reference/dump_anima_text_adapter.py.</remarks>
[Trait("Category", "Integration")]
public sealed unsafe class AnimaTextAdapterParityTests
{
    private const string Prefix = "net.llm_adapter.";
    private const float Tolerance = 1e-4f;
    private readonly ITestOutputHelper _output;

    public AnimaTextAdapterParityTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void TextAdapter_MatchesComfyUi_UnderExactGelu()
    {
        string dumpPath = Path.Combine(RepoRoot.Path, "tests", "python-reference", "anima_text_adapter.safetensors");
        string checkpointPath = Path.Combine(
            TestPaths.ModelsDir, "Stable-Diffusion", "Anima", "anima-base-v1.0.safetensors");
        if (!RealWeightGate.Require(_output.WriteLine, dumpPath, checkpointPath)) return;

        using SafeTensorsLoader dump = new();
        dump.Load(dumpPath);
        Tensor qwen = dump.GetTensor("in.qwen");
        Tensor expected = dump.GetTensor("out.refined");
        Tensor idTensor = dump.GetTensor("in.t5ids");
        int[] ids = new int[idTensor.ElementCount];
        float* idValues = (float*)idTensor.DataPointer;
        for (int i = 0; i < ids.Length; i++) ids[i] = (int)idValues[i];

        using SafeTensorsLoader checkpoint = new();
        checkpoint.Load(checkpointPath);
        Dictionary<string, Tensor> weights = new();
        foreach (string key in checkpoint.Descriptors.Keys)
        {
            if (!key.StartsWith(Prefix, StringComparison.Ordinal)) continue;
            Tensor weight = checkpoint.GetTensor(key);
            weights[key[Prefix.Length..]] = weight.DType == DType.F32 ? weight : weight.CastTo(DType.F32);
        }

        CpuBackend backend = new();
        AnimaLlmAdapter adapter = new(new AnimaLlmAdapterConfig());
        adapter.LoadWeights(weights);

        Tensor exact = adapter.Forward(backend, qwen, ids);
        (float exactRelative, float exactAbsolute) = Compare(exact, expected);
        _output.WriteLine($"default: max |diff| {exactAbsolute:G6}, worst relative {exactRelative:G6}");
        Assert.True(exactRelative <= Tolerance, $"default: {exactRelative} exceeds {Tolerance}");

        using (KnobProfiles.Reference.Push())
        {
            Tensor pinned = adapter.Forward(backend, qwen, ids);
            (float pinnedRelative, float pinnedAbsolute) = Compare(pinned, expected);
            _output.WriteLine($"reference profile: max |diff| {pinnedAbsolute:G6}, worst relative {pinnedRelative:G6}");
            Assert.True(pinnedRelative <= Tolerance, $"reference profile: {pinnedRelative} exceeds {Tolerance}");
        }
    }

    /// <summary>The per-element error the parity tests use: |diff| / max(1, |expected|), worst over the tensor.</summary>
    private static (float Relative, float Absolute) Compare(Tensor actual, Tensor expected)
    {
        int count = (int)expected.ElementCount;
        Assert.Equal(count, (int)actual.ElementCount);
        float* e = (float*)expected.DataPointer;
        float* a = (float*)actual.DataPointer;
        float worstRelative = 0f, worstAbsolute = 0f;
        for (int i = 0; i < count; i++)
        {
            float error = MathF.Abs(e[i] - a[i]);
            worstAbsolute = MathF.Max(worstAbsolute, error);
            worstRelative = MathF.Max(worstRelative, error / MathF.Max(1f, MathF.Abs(e[i])));
        }
        return (worstRelative, worstAbsolute);
    }
}
