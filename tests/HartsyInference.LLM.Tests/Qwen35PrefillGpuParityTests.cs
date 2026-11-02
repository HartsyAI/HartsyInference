using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Ssm;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests;

/// <summary>The tiny Qwen3.5 trunk of <see cref="Qwen35HuggingFaceParityTests"/> on every GPU backend: its gated-DeltaNet
/// layers prefill through the device conv and delta-rule kernels, so this ties those kernels to transformers' own
/// hidden states, once in a single pass and once fed in chunks that carry the state between calls, and again with
/// <c>numerics.ssmDevicePrefill</c> off, where the same layers run the host reference between the device projections.</summary>
[Trait("Category", "GpuIntegration")]
public sealed unsafe class Qwen35PrefillGpuParityTests(ITestOutputHelper output)
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Clef");

    public static IEnumerable<object[]> Cases()
    {
        foreach (object[] kind in BackendGate.GpuKinds)
        {
            yield return [kind[0], 0, true];
            yield return [kind[0], 3, true];
            yield return [kind[0], 0, false];
            yield return [kind[0], 3, false];
        }
    }

    /// <param name="chunk">Tokens per prefill call; 0 runs the whole prompt in one call.</param>
    /// <param name="devicePrefill">The <c>numerics.ssmDevicePrefill</c> setting.</param>
    [Theory]
    [MemberData(nameof(Cases))]
    public void HiddenStates_MatchTransformers(string kind, int chunk, bool devicePrefill)
    {
        // The reference is plain F32, and the tolerance below is an F32 one, so this pins plain F32 GEMMs. The engine's
        // default is TF32 for F32 GEMMs on Ampere and newer. On an RTX 4090 that default moves this model's hidden states
        // by a relative L2 of about 4.5e-3 against plain F32, so the default is not checked here.
        // The pins are a profile rather than KnobStore.Set, which is process-wide: a CUDA backend built by another test
        // class running in parallel would otherwise run F32 GEMMs for as long as this test does.
        using IDisposable knobs = KnobProfile.Create("qwen35-f32-reference")
            .With(EngineKnobs.NoTf32, true)
            .With(EngineKnobs.SdpaNoTf32, true)
            .With(EngineKnobs.SsmDevicePrefill, devicePrefill)
            .Push();
        CheckHiddenStates(kind, chunk, devicePrefill);
    }

    private void CheckHiddenStates(string kind, int chunk, bool devicePrefill)
    {
        if (!BackendGate.TryOpen(kind, output.WriteLine, out IBackend? gpu)) return;
        using IBackend backend = gpu!;
        using SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(Dir, "qwen35_tiny.safetensors"));
        Dictionary<string, Tensor> weights = new();
        foreach (string name in loader.Descriptors.Keys) weights[name] = loader.GetTensor(name);
        using SafeTensorsLoader expectedLoader = new();
        expectedLoader.Load(Path.Combine(Dir, "qwen35_tiny_expected.safetensors"));
        Tensor want = expectedLoader.GetTensor("hidden");
        Tensor idTensor = expectedLoader.GetTensor("input_ids");
        int[] ids = new ReadOnlySpan<int>((void*)idTensor.DataPointer, (int)idTensor.ElementCount).ToArray();
        Qwen35HfConfig cfg = Qwen35HfConfig.FromJson(File.ReadAllText(Path.Combine(Dir, "qwen35_tiny_config.json")));

        using Qwen35Model model = Qwen35Model.FromHuggingFace(weights, cfg, maxSequenceLength: 64);
        float[] got = chunk == 0 ? model.ForwardHiddenStates(backend, ids) : Chunked(model, backend, ids, chunk);

        ReadOnlySpan<float> expected = new((void*)want.DataPointer, (int)want.ElementCount);
        Assert.Equal(expected.Length, got.Length);
        float worst = 0;
        for (int i = 0; i < got.Length; i++) worst = MathF.Max(worst, MathF.Abs(expected[i] - got[i]));
        output.WriteLine($"{kind}, chunk {chunk}, device prefill {devicePrefill}: max |Δ| = {worst:E3}");
        Assert.True(worst < 2e-3f, $"max |Δ| = {worst} on {kind}");
    }

    /// <summary>Feeds the prompt <paramref name="chunk"/> tokens at a time through the embedding path, as Drama 3 does.</summary>
    private static float[] Chunked(Qwen35Model model, IBackend backend, int[] ids, int chunk)
    {
        model.ResetState();
        float[] all = new float[ids.Length * model.DModel];
        for (int start = 0; start < ids.Length; start += chunk)
        {
            int count = Math.Min(chunk, ids.Length - start);
            using Tensor embeds = new(new TensorShape(1, count, model.DModel), DType.F32);
            model.EmbedLookup(embeds, ids[start..(start + count)]);
            using Tensor hidden = model.ForwardEmbedsHidden(backend, embeds, count);
            new ReadOnlySpan<float>((void*)hidden.DataPointer, count * model.DModel).CopyTo(all.AsSpan(start * model.DModel));
        }
        return all;
    }
}
