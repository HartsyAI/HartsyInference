using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>The tiny checkpoint and reference velocities come from ComfyUI's own <c>comfy/ldm/flux3/model.py</c>
/// (<c>tests/python-reference/dump_flux3_reference.py</c>; F16 weights, F32 compute): the port, fed through
/// <see cref="Flux3CheckpointConverter"/> (fused qkv split), must reproduce the guided, reference-edit and flow-map
/// velocities and the precomputed text stream.</summary>
public sealed unsafe class Flux3ParityTests(ITestOutputHelper output)
{
    private const int Context = 64;
    private const float Tolerance = 2e-4f;
    private static readonly string FixturePath = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Flux3", "flux3_tiny.safetensors");

    [Fact]
    public void Guided_TextOnly_MatchesComfy() =>
        Run((transformer, backend, f) =>
        {
            Flux3ConditionInput input = Flux3ConditionInput.TextOnly(f["inputs.context"]);
            using Flux3ConditionStates states = transformer.PrecomputeConditioning(backend, input);
            AssertClose(f["guided.txt_states"], states.Text, "txt_states");
            using Tensor velocity = transformer.Forward(backend, f["inputs.latent"], 2, 3, 0.7f, input, states);
            AssertClose(f["guided.velocity"], velocity, "guided velocity");
        }, flowMap: false);

    [Fact]
    public void Edit_TwoReferences_MatchesComfy() =>
        Run((transformer, backend, f) =>
        {
            using Tensor references = new Tensor(new TensorShape(1, 4 + 2, 128), DType.F32);
            backend.Concat(references, [f["inputs.ref0"], f["inputs.ref1"]], 1);
            Flux3ConditionInput input = new Flux3ConditionInput(f["inputs.context"], references, [(2, 2), (1, 2)]);
            using Tensor velocity = transformer.Forward(backend, f["inputs.latent"], 2, 3, 0.7f, input, null);
            AssertClose(f["edit.velocity"], velocity, "edit velocity");
        }, flowMap: false);

    [Fact]
    public void FlowMap_NextSigmaTarget_MatchesComfy() =>
        Run((transformer, backend, f) =>
        {
            Flux3ConditionInput input = Flux3ConditionInput.TextOnly(f["inputs.context"]);
            using Tensor velocity = transformer.Forward(backend, f["inputs.latent"], 2, 3, 0.7f, input, null, targetSigma: 0.3f);
            AssertClose(f["flowmap.velocity"], velocity, "flow-map velocity");
        }, flowMap: true);

    private void Run(Action<Flux3Transformer, IBackend, IReadOnlyDictionary<string, Tensor>> body, bool flowMap)
    {
        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(FixturePath);
        Dictionary<string, Tensor> all = loader.GetAllTensors();
        Dictionary<string, Tensor> raw = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach ((string key, Tensor value) in all)
        {
            if (!key.StartsWith("weights.", StringComparison.Ordinal))
                continue;
            string name = key["weights.".Length..];
            if (flowMap || !name.StartsWith("ref_time_in.", StringComparison.Ordinal))
                raw[name] = value;
        }
        Dictionary<string, Tensor> converted = Flux3CheckpointConverter.ConvertTransformer(raw);
        Flux3Config config = Flux3Config.FromWeights(converted) with { ContextInDim = Context };
        Assert.Equal(flowMap, config.FlowMap);
        using Flux3Transformer transformer = new Flux3Transformer(config);
        // The CPU reference backend has no F16 kernels; the Comfy parity runs the F32 stream (the F16 path is gated on GPU).
        using (KnobProfile.Create("flux3-f32-parity").With(EngineKnobs.DitF16, false).Push())
            transformer.LoadWeights(converted);
        try
        {
            body(transformer, new CpuBackend(), all);
        }
        finally
        {
            foreach ((string key, Tensor value) in converted)
            {
                if (!raw.ContainsKey(key)) value.Dispose();
            }
        }
    }

    private void AssertClose(Tensor expected, Tensor actual, string what)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        ReadOnlySpan<float> e = new ReadOnlySpan<float>((void*)expected.DataPointer, (int)expected.ElementCount);
        ReadOnlySpan<float> a = new ReadOnlySpan<float>((void*)actual.DataPointer, (int)actual.ElementCount);
        float worst = 0f;
        for (int i = 0; i < e.Length; i++) worst = MathF.Max(worst, MathF.Abs(e[i] - a[i]));
        output.WriteLine($"{what}: max |Δ| = {worst:E2}");
        Assert.True(worst <= Tolerance, $"{what}: max |Δ| = {worst}");
    }
}
