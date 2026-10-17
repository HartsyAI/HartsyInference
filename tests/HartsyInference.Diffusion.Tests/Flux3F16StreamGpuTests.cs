using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>FLUX.3 with the damped F16 activation stream (<c>numerics.ditF16</c>) against the F32 stream on one CUDA device,
/// over the one-head synthetic config of <see cref="Flux3TransformerSmokeTests"/>. Each run loads fresh weights, since the
/// residual damp scales weights in place. Gates that the F16 path runs (the outputs must differ) and tracks F32 within F16
/// rounding on the joint, reference and final-layer path. Real-weight F16 headroom is checked by the Swarm generation run.</summary>
public sealed unsafe class Flux3F16StreamGpuTests(ITestOutputHelper output)
{
    private const int Hidden = 128;
    private const int Mlp = 64;
    private const int Context = 32;

    private readonly ITestOutputHelper _output = output;

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void F16Stream_TracksF32Stream_OnCuda()
    {
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: PtxDir());
        float[] f32 = Velocity(backend, f16: false);
        float[] f16 = Velocity(backend, f16: true);

        float maxRef = 0f, maxDiff = 0f;
        for (int i = 0; i < f32.Length; i++)
        {
            maxRef = MathF.Max(maxRef, MathF.Abs(f32[i]));
            maxDiff = MathF.Max(maxDiff, MathF.Abs(f32[i] - f16[i]));
        }
        _output.WriteLine($"velocity: n={f32.Length} maxAbs(F32)={maxRef:E3} maxDiff(F16-F32)={maxDiff:E3} rel={maxDiff / maxRef:E3}");

        // Identical outputs would mean the F16 path silently fell back to F32.
        Assert.True(maxDiff > 0f, "F16 stream produced bit-identical velocity to F32; the F16 path did not run.");
        Assert.True(maxDiff <= 0.03f * maxRef, $"F16 velocity drifted {maxDiff:E3} from F32 (max |v| {maxRef:E3}).");
    }

    private static float[] Velocity(CudaBackend backend, bool f16)
    {
        using IDisposable knob = KnobProfile.Create("flux3-f16-stream-gpu").With(EngineKnobs.DitF16, f16).Push();
        int seed = 11;
        Dictionary<string, Tensor> weights = Weights(ref seed);
        Flux3Config config = Flux3Config.FromWeights(weights) with { ContextInDim = Context };
        using Flux3Transformer transformer = new Flux3Transformer(config);
        transformer.LoadWeights(weights);

        using Tensor text = Random(ref seed, 1, 4, Context);
        using Tensor latent = Random(ref seed, 1, 4, 128);
        using Tensor references = Random(ref seed, 1, 6, 128);
        Flux3ConditionInput input = new Flux3ConditionInput(text, references, [(2, 2), (1, 2)]);
        using Tensor velocity = transformer.Forward(backend, latent, 2, 2, 0.5f, input, null);
        backend.Sync();
        return new ReadOnlySpan<float>((float*)velocity.DataPointer, (int)velocity.ElementCount).ToArray();
    }

    private static Dictionary<string, Tensor> Weights(ref int seed)
    {
        Dictionary<string, Tensor> w = new Dictionary<string, Tensor>(StringComparer.Ordinal)
        {
            ["emb_in.image.weight"] = Random(ref seed, Hidden, 128),
            ["emb_in.image_cond.weight"] = Random(ref seed, Hidden, 128),
            ["txt_in.weight"] = Random(ref seed, Hidden, Context),
            ["time_in.in_layer.weight"] = Random(ref seed, Hidden, 256),
            ["time_in.out_layer.weight"] = Random(ref seed, Hidden, Hidden),
            ["final_layer.image.adaLN_modulation.1.weight"] = Random(ref seed, 2 * Hidden, Hidden),
            ["final_layer.image.linear.weight"] = Random(ref seed, 128, Hidden),
        };
        foreach (string mode in new[] { "image", "image_cond", "txt" })
        {
            w[$"early_stream_modulations.{mode}.lin.weight"] = Random(ref seed, 3 * Hidden, Hidden);
            w[$"single_stream_modulations.{mode}.lin.weight"] = Random(ref seed, 3 * Hidden, Hidden);
        }
        foreach (string prefix in new[] { "content_mode_blocks.image.0", "content_mode_blocks.image_cond.0", "txt_mode_blocks.0", "single_blocks.0", "single_blocks.1" })
        {
            w[$"{prefix}.q_proj.weight"] = Random(ref seed, Hidden, Hidden);
            w[$"{prefix}.k_proj.weight"] = Random(ref seed, Hidden, Hidden);
            w[$"{prefix}.v_proj.weight"] = Random(ref seed, Hidden, Hidden);
            w[$"{prefix}.mlp_in.weight"] = Random(ref seed, 2 * Mlp, Hidden);
            w[$"{prefix}.attn_out.weight"] = Random(ref seed, Hidden, Hidden);
            w[$"{prefix}.mlp_out.weight"] = Random(ref seed, Hidden, Mlp);
            w[$"{prefix}.norm.query_norm.weight"] = Ones(Hidden);
            w[$"{prefix}.norm.key_norm.weight"] = Ones(Hidden);
        }
        return w;
    }

    private static Tensor Random(ref int seed, params long[] shape)
    {
        Tensor t = new Tensor(new TensorShape(shape), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++)
        {
            seed = seed * 1103515245 + 12345;
            p[i] = ((seed >> 16) & 0x7fff) / 32768f * 0.2f - 0.1f;
        }
        return t;
    }

    private static Tensor Ones(long width)
    {
        Tensor t = new Tensor(new TensorShape(width), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < width; i++) p[i] = 1f;
        return t;
    }

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }
}
