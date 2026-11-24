using System.Text.Json;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Parity of the FLUX.2 reference-edit forward, the diffusers-layout reader and the pipeline loop against
/// diffusers (see <c>Fixtures/flux2_edit_tiny/meta.json</c> for the versions that wrote the fixture; regenerate with
/// <c>tests/python-reference/dump_flux2_edit_tiny.py</c>). Random tiny weights in diffusers key names, F32 CPU on both sides.
/// This covers the DiT, the position ids, the shared timestep, CFG, the Euler step and the sigma schedule; the VAE and the
/// text encoder are separate and are covered by the real-weight layer-diff.</summary>
public sealed class Flux2EditParityTests
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "flux2_edit_tiny");

    // Measured, not derived: the C# F32 forward sits 8e-6 (relative RMS) from diffusers here, and from float64, while diffusers' own
    // F32 sits 4.5e-7 from float64 (`velocityF32VsF64RelRms` in meta.json), so the gap is ours, about 18x the rounding floor. It is the
    // existing Flux2Transformer F32 path (the target-only call is as far off as the call with references) and is NOT bisected: RoPE
    // precision (checked: 2e-7), SiLU (4.5e-8) and attention (9e-8) are not the cause. 8e-6 cannot hide a layout error such as a wrong
    // T coordinate or timestep, which move the output by 1e-2 or more. Loop error grows with CFG 3.5 (5.6e-5 after two steps).
    private const double ForwardTolerance = 2e-5;
    private const double LoopTolerance = 2e-4;

    private static Flux2Config Tiny => IrisZTestData.Config(modalities: 0);

    [Fact]
    public void TheDiffusersLayoutIsRecognisedAndSplitIntoTheCanonicalNames()
    {
        using SafeTensorsLoader weights = Weights();
        Dictionary<string, Tensor> raw = weights.GetAllTensors();
        Assert.True(Flux2CheckpointConverter.IsDiffusersLayout(raw.Keys));

        Dictionary<string, Tensor> converted = new Flux2CheckpointConverter(IrisZTestData.Hidden, IrisZTestData.Mlp).ConvertTransformer(raw);

        Assert.True(converted.ContainsKey("single_transformer_blocks.0.attn.linear_in_gate.weight"));
        Assert.True(converted.ContainsKey("single_transformer_blocks.1.attn.to_v.weight"));
        Assert.True(converted.ContainsKey("transformer_blocks.0.ff.linear_in_up.weight"));
        Assert.True(converted.ContainsKey("transformer_blocks.0.ff_context.linear_in_gate.weight"));
        Assert.False(converted.ContainsKey("single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight"));
        Assert.Equal(new TensorShape(IrisZTestData.Mlp, IrisZTestData.Hidden), converted["transformer_blocks.0.ff.linear_in_gate.weight"].Shape);
    }

    [Fact]
    public void TheReferenceForwardMatchesDiffusers()
    {
        using IDisposable f32 = KnobProfile.Create("ovb-parity").With(EngineKnobs.DitF16, false).Push();
        using CpuBackend backend = new();
        using SafeTensorsLoader weights = Weights();
        using Flux2Transformer transformer = Transformer(weights);
        using SafeTensorsLoader reference = Reference();
        using Tensor noise = reference.GetTensor("noise");
        using Tensor refTokens = reference.GetTensor("ref_tokens");
        using Tensor cond = reference.GetTensor("cond");
        using Tensor packed = Flux2LatentOps.Pack(noise);
        int grid = packed.Shape[1] > 0 ? (int)packed.Shape[1] : 0;

        using Tensor both = Concat(backend, packed, refTokens);
        using Tensor velocity = transformer.ForwardWithReferences(backend, both, cond, 0.6f, 2, 3, [0f, 10f]);
        using Tensor expected = reference.GetTensor("velocity_ref_sigma0p6");
        using Tensor expected64 = reference.GetTensor("velocity_ref_sigma0p6_f64");
        AssertRelRms("target + reference velocity vs float64", velocity, expected64, ForwardTolerance);
        AssertRelRms("target + reference velocity", velocity, expected, ForwardTolerance);

        using Tensor aloneVelocity = transformer.ForwardWithReferences(backend, packed, cond, 0.6f, 2, 3, [0f]);
        using Tensor aloneExpected = reference.GetTensor("velocity_noref_sigma0p6");
        AssertRelRms("target-only velocity", aloneVelocity, aloneExpected, ForwardTolerance);
        Assert.Equal(6, grid);
    }

    [Fact]
    public void ThePipelineLoopMatchesDiffusersStepByStep()
    {
        using IDisposable f32 = KnobProfile.Create("ovb-parity").With(EngineKnobs.DitF16, false).Push();
        using CpuBackend backend = new();
        using SafeTensorsLoader weights = Weights();
        using Flux2Transformer transformer = Transformer(weights);
        using SafeTensorsLoader reference = Reference();
        using Tensor noise = reference.GetTensor("noise");
        using Tensor refTokens = reference.GetTensor("ref_tokens");
        using Tensor cond = reference.GetTensor("cond");
        using Tensor uncond = reference.GetTensor("uncond");
        JsonElement meta = Meta();
        int steps = meta.GetProperty("steps").GetInt32();
        float guidance = (float)meta.GetProperty("guidance").GetDouble();

        IrisZTestData.VaeWeights vaeWeights = new();
        VaeDecoder decoder = new(IrisZTestData.Vae);
        decoder.LoadWeights(vaeWeights);
        VaeEncoder encoder = new(IrisZTestData.Vae);
        encoder.LoadWeights(vaeWeights);
        using Tensor bnMean = IrisZTestData.Filled(0f, IrisZTestData.Channels);
        using Tensor bnVar = IrisZTestData.Filled(1f, IrisZTestData.Channels);
        using Flux2EditPipeline pipeline = new(backend, new LlamaStyleEncoder(LlamaStyleEncoderConfig.Qwen3_4B), transformer, decoder,
            encoder, bnMean, bnVar, Tiny, [9, 18, 27]);

        List<Tensor> latents = new();
        TextToImageRequest request = new()
        {
            Prompt = "",
            Width = 3 * 16,
            Height = 2 * 16,
            Steps = steps,
            CfgScale = guidance,
            Seed = 1,
            InitialNoise = noise,
        };
        // The preview is the packed latent un-packed and BatchNorm-denormalised; the same call turns the diffusers step into it.
        pipeline.EditFromTokens(cond, uncond, [new ReferenceLatentTokens(refTokens, 2, 3)], request,
            progress => latents.Add(Copy(progress.Latent!)));
        try
        {
            Assert.Equal(steps, latents.Count);
            for (int i = 0; i < steps; i++)
            {
                using Tensor diffusersPacked = reference.GetTensor($"latents_step_{i}");
                using Tensor diffusers = Flux2LatentOps.FromPackedTokens(diffusersPacked, 2, 3, bnMean, bnVar, 1e-4f, IrisZTestData.LatentChannels, 2);
                AssertRelRms($"latents after step {i}", latents[i], diffusers, LoopTolerance);
            }
        }
        finally
        {
            foreach (Tensor latent in latents) latent.Dispose();
        }
    }

    [Fact]
    public void TheSigmaScheduleMatchesDiffusers()
    {
        JsonElement schedules = Meta().GetProperty("schedules");
        foreach (JsonProperty entry in schedules.EnumerateObject())
        {
            string[] parts = entry.Name.Split('_');
            int tokens = int.Parse(parts[0]), steps = int.Parse(parts[1]);
            float mu = FlowMatchEulerDiscreteScheduler.EmpiricalMu(tokens, steps);
            FlowMatchEulerDiscreteScheduler scheduler = new(MathF.Exp(mu));
            scheduler.SetTimesteps(steps);

            Assert.True(Math.Abs(mu - entry.Value.GetProperty("mu").GetDouble()) < 1e-5, $"{entry.Name}: mu {mu}");
            float[] expectedSigmas = entry.Value.GetProperty("sigmas").EnumerateArray().Select(e => (float)e.GetDouble()).ToArray();
            float[] sigmas = scheduler.Sigmas();
            Assert.Equal(expectedSigmas.Length, sigmas.Length);
            for (int i = 0; i < sigmas.Length; i++)
                Assert.True(Math.Abs(sigmas[i] - expectedSigmas[i]) < 2e-6f, $"{entry.Name}: sigma {i} is {sigmas[i]}, diffusers {expectedSigmas[i]}");
        }
    }

    private static SafeTensorsLoader Weights()
    {
        SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(Dir, "weights.safetensors"));
        return loader;
    }

    private static SafeTensorsLoader Reference()
    {
        SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(Dir, "reference.safetensors"));
        return loader;
    }

    private static JsonElement Meta() => JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "meta.json"))).RootElement;

    /// <summary>The tiny diffusers checkpoint read through the engine's own converter, as the recipe does. The loader backs the
    /// converted tensors, so the caller keeps it alive as long as the transformer.</summary>
    private static Flux2Transformer Transformer(SafeTensorsLoader weights)
    {
        Dictionary<string, Tensor> converted = new Flux2CheckpointConverter(IrisZTestData.Hidden, IrisZTestData.Mlp)
            .ConvertTransformer(weights.GetAllTensors());
        Flux2Transformer transformer = new(Tiny);
        transformer.LoadWeights(converted);
        return transformer;
    }

    private static Tensor Copy(Tensor source)
    {
        Tensor copy = new(source.Shape, DType.F32);
        source.AsReadOnlySpan<float>().CopyTo(copy.AsSpan<float>());
        return copy;
    }

    private static Tensor Concat(CpuBackend backend, Tensor a, Tensor b)
    {
        Tensor joined = new(new TensorShape(1, a.Shape[1] + b.Shape[1], a.Shape[2]), DType.F32);
        backend.Concat(joined, [a, b], 1);
        return joined;
    }

    private static void AssertRelRms(string what, Tensor actual, Tensor expected, double tolerance)
    {
        Assert.Equal(expected.Shape, actual.Shape);
        ReadOnlySpan<float> a = actual.AsReadOnlySpan<float>(), e = expected.AsReadOnlySpan<float>();
        double sum = 0, err = 0;
        for (int i = 0; i < e.Length; i++)
        {
            sum += (double)e[i] * e[i];
            err += (double)(a[i] - e[i]) * (a[i] - e[i]);
        }
        double rel = Math.Sqrt(err / Math.Max(sum, 1e-30));
        Assert.True(rel <= tolerance, $"{what}: relative RMS {rel:E2} against diffusers exceeds {tolerance:E0}");
    }
}
