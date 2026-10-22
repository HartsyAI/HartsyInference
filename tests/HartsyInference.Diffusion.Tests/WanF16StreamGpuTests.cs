using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Wan's F16 attention and FFN branches over the F32 residual stream (<c>numerics.wanF16</c>) on CUDA. One block at
/// TI2V-5B geometry, fed a stream with the outlier channels the real residual carries, tracks its F32 twin, and its
/// self-attention stays on cuDNN past the SageAttention knee, which an unrefused call at the same geometry reaches. With the
/// switch off, or with <c>numerics.ditF16</c> off, the transformer is the F32 path assembled from its parts, bit for bit; the
/// forwards that stay F32 ignore the switch; a text K/V scope serves only forwards of its own dtype; and the debug dump records
/// the F16 branches as F32 without moving the output.</summary>
public sealed unsafe class WanF16StreamGpuTests(ITestOutputHelper output)
{
    private const int GridH = 15, GridW = 26;   // 832×480 after the VAE and the patch
    private const int TextTokens = 512;
    private const int SageKnee = 8192;

    private readonly ITestOutputHelper _output = output;

    /// <summary>The bar is on the block's update (output minus input), which the residual's outlier channels would hide in the
    /// output. For scale, the report also gives what storing the F32 block's own output in F16 would cost.</summary>
    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData(4)]
    [InlineData(22)]
    [InlineData(31)]
    public void F16BranchesTrackTheF32BlockAtTi2v5bGeometry(int frames)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using IDisposable knobs = KnobProfile.Create("wan-f16-block").With(EngineKnobs.SageAttn, true)
            .With(EngineKnobs.SageF16MinSkv, SageKnee).Push();
        WanVideoConfig cfg = WanVideoConfig.Ti2V5B;
        int dim = cfg.InnerDim, s = frames * GridH * GridW;
        int seed = 17;
        Dictionary<string, Tensor> weights = BlockWeights(cfg, ref seed);
        WanVideoBlock block = new(cfg, crossAttnNorm: true);
        block.LoadWeights(weights, "b");
        using Tensor hidden = ResidualStream(ref seed, s, dim);
        float[] x = Read(hidden);
        using Tensor context = Uniform(ref seed, MathF.Sqrt(3f), TextTokens, dim);
        using Tensor temb = Uniform(ref seed, 0.5f, 1, 6, dim);
        WanRope rope = new(cfg.HeadDim, cfg.RopeTheta, cfg.RopeMaxSeqLen);
        (Tensor cos, Tensor sin) = rope.BuildCosSin(frames, GridH, GridW);

        float[] f32, f16;
        using (CudaBackend backend = new(0, PtxDir()))
        {
            using (Tensor out32 = block.Forward(backend, hidden, context, temb, rope, cos, sin, s))
                f32 = Read(out32);
            Assert.Equal(0, backend.SageAttentionExecutionCount);
            using Tensor out16 = block.Forward(backend, hidden, context, temb, rope, cos, sin, s, branchDType: DType.F16);
            Assert.Equal(DType.F32, out16.DType);
            Assert.Equal(0, backend.SageAttentionExecutionCount);
            f16 = Read(out16);
            SelfAttentionRows(backend, cfg, s, ref seed);
        }
        foreach (Tensor t in weights.Values.Concat([cos, sin])) t.Dispose();

        double diff = 0, update = 0, outNorm = 0, dot = 0, update16 = 0, rounding = 0;
        float maxIn = 0, maxOut = 0, maxDiff = 0;
        for (int i = 0; i < f32.Length; i++)
        {
            Assert.True(float.IsFinite(f16[i]), $"F16 block output {i} is {f16[i]}.");
            double d32 = f32[i] - x[i], d16 = f16[i] - x[i], e = f16[i] - f32[i], r = (float)(Half)f32[i] - f32[i];
            diff += e * e;
            rounding += r * r;
            update += d32 * d32;
            update16 += d16 * d16;
            dot += d32 * d16;
            outNorm += (double)f32[i] * f32[i];
            maxIn = MathF.Max(maxIn, MathF.Abs(x[i]));
            maxOut = MathF.Max(maxOut, MathF.Abs(f32[i]));
            maxDiff = MathF.Max(maxDiff, MathF.Abs(f16[i] - f32[i]));
        }
        double relOut = Math.Sqrt(diff / outNorm), relUpdate = Math.Sqrt(diff / update), floor = Math.Sqrt(rounding / update);
        double cosine = dot / Math.Sqrt(update * update16);
        _output.WriteLine($"S={s}: max|in|={maxIn:F1} max|out|={maxOut:F1} max|diff|={maxDiff:E3} relL2(out)={relOut:E3} " +
            $"relL2(out-in)={relUpdate:E3} cos(out-in)={cosine:F6}; an F16 stream's output rounding alone: relL2(out-in)={floor:E3}");

        Assert.True(diff > 0, "The F16 branches wrote the F32 block's bits; the F16 path did not run.");
        Assert.True(relUpdate <= 2e-3, $"The F16 branches miss the F32 block's update by {relUpdate:E3}.");
    }

    /// <summary>Self-attention's F16 operands at <paramref name="s"/> tokens: refusing INT8 as the F16 twins do, the call plans
    /// to cuDNN's F16 row; unrefused, past the knee it plans to and runs on SageAttention.</summary>
    private static void SelfAttentionRows(CudaBackend backend, WanVideoConfig cfg, int s, ref int seed)
    {
        long[] shape = [1, cfg.NumHeads, s, cfg.HeadDim];
        using Tensor q = Half(Uniform(ref seed, 1f, shape)), k = Half(Uniform(ref seed, 1f, shape)), v = Half(Uniform(ref seed, 1f, shape));
        using Tensor o = new(new TensorShape(shape), DType.F16);
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(o, q, k, v, null, 1f / MathF.Sqrt(cfg.HeadDim), allowF16: true);
        AttentionCall refusing = call with { Numerics = call.Numerics with { Refused = AttentionFormats.Int8 } };

        Assert.Equal("cudnn.f16", backend.PlanAttention(in refusing).RowId);
        Assert.Equal(s >= SageKnee ? "sage.int8.f16" : "cudnn.f16", backend.PlanAttention(in call).RowId);
        backend.Attention(in call);
        Assert.Equal(s >= SageKnee ? 1 : 0, backend.SageAttentionExecutionCount);
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void WithTheSwitchOffTheTransformerIsTheF32PathBitForBit()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        WanVideoConfig cfg = Tiny(image: false);
        Dictionary<string, Tensor> w = WanSyntheticWeights.BuildTransformer(cfg);
        WanVideoTransformer transformer = new(cfg);
        transformer.LoadWeights(w);
        int seed = 3;
        using Tensor latent = Uniform(ref seed, 1f, 1, cfg.InChannels, 3, 8, 8);
        using Tensor encoder = Uniform(ref seed, 1f, 5, cfg.TextDim);
        using CudaBackend backend = new(0, PtxDir());

        float[] expected = F32Path(backend, cfg, w, latent, encoder, 500f);
        float[] off = Forward(false, () => transformer.Forward(backend, latent, encoder, 500f));
        float[] ditOff = Forward(true, () => transformer.Forward(backend, latent, encoder, 500f), ditF16: false);
        float[] on = Forward(true, () => transformer.Forward(backend, latent, encoder, 500f));

        Assert.Equal(expected, off);
        Assert.Equal(expected, ditOff);
        double rel = RelL2(on, expected);
        _output.WriteLine($"switch on vs the F32 path: relL2 {rel:E3}");
        Assert.True(rel > 0 && rel < 1e-2, $"With the switch on the forward is {rel:E3} from the F32 path; expected F16 rounding.");
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [InlineData("per-frame timesteps")]
    [InlineData("image context")]
    [InlineData("step cache")]
    public void AForwardThatStaysF32IgnoresTheSwitch(string path)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        WanVideoConfig cfg = Tiny(image: path == "image context");
        WanVideoTransformer transformer = new(cfg);
        transformer.LoadWeights(WanSyntheticWeights.BuildTransformer(cfg));
        int seed = 5;
        using Tensor latent = Uniform(ref seed, 1f, 1, cfg.InChannels, 3, 8, 8);
        using Tensor encoder = Uniform(ref seed, 1f, 5, cfg.TextDim);
        using Tensor image = Uniform(ref seed, 1f, 4, Math.Max(1, cfg.ImageDim));
        using CudaBackend backend = new(0, PtxDir());

        Tensor Run() => path switch
        {
            "per-frame timesteps" => transformer.Forward(backend, latent, encoder, [0f, 500f, 500f]),
            "image context" => transformer.Forward(backend, latent, encoder, [500f], image),
            _ => RunCached(),
        };
        Tensor RunCached()
        {
            using DeviceFeatureCache cache = new();
            return transformer.Forward(backend, latent, encoder, 500f, cache);
        }

        Assert.Equal(Forward(false, Run), Forward(true, Run));
    }

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void ATextKvScopeServesOnlyForwardsOfItsDtypeAndHitsBitForBit()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        WanVideoConfig cfg = Tiny(image: false);
        WanVideoTransformer transformer = new(cfg);
        transformer.LoadWeights(WanSyntheticWeights.BuildTransformer(cfg));
        int seed = 9;
        using Tensor latent = Uniform(ref seed, 1f, 1, cfg.InChannels, 3, 8, 8);
        using Tensor cond = Uniform(ref seed, 1f, 5, cfg.TextDim), uncond = Uniform(ref seed, 1f, 5, cfg.TextDim);
        using CudaBackend backend = new(0, PtxDir());

        List<float[]> Steps()
        {
            List<float[]> outputs = [];
            foreach (float t in new[] { 900f, 500f, 100f })
                foreach (bool f16 in new[] { true, false })
                    foreach (Tensor encoder in new[] { cond, uncond })
                        outputs.Add(Forward(f16, () => transformer.Forward(backend, latent, encoder, t)));
            return outputs;
        }
        List<float[]> expected = Steps(), cached;
        using (transformer.CacheTextCrossAttentionKv(DType.F16))
            cached = Steps();

        Assert.Equal(expected, cached);
    }

    /// <summary>The dump keeps the raw scale rows, which the F16 branches' fused norm takes anyway.</summary>
    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void TheDebugDumpRecordsTheF16BranchesAsF32WithoutMovingTheOutput()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        WanVideoConfig cfg = Tiny(image: false);
        WanVideoTransformer transformer = new(cfg);
        transformer.LoadWeights(WanSyntheticWeights.BuildTransformer(cfg));
        int seed = 13;
        using Tensor latent = Uniform(ref seed, 1f, 1, cfg.InChannels, 3, 8, 8);
        using Tensor encoder = Uniform(ref seed, 1f, 5, cfg.TextDim);
        using CudaBackend backend = new(0, PtxDir());
        string dir = Path.Combine(Path.GetTempPath(), $"wan-f16-dump-{Guid.NewGuid():N}");
        try
        {
            float[] plain = Forward(true, () => transformer.Forward(backend, latent, encoder, 500f));
            float[] dumped;
            using (KnobProfile.Create("wan-f16-dump").With(EngineKnobs.DitF16, true).With(EngineKnobs.WanF16, true)
                .With(EngineKnobs.WanDebugDir, dir).With(EngineKnobs.WanDebugTag, "").Push())
            using (Tensor velocity = transformer.Forward(backend, latent, encoder, 500f))
                dumped = Read(velocity);

            Assert.Equal(plain, dumped);
            const int Tokens = 3 * 4 * 4;
            foreach (string name in new[] { "b0_n1", "blocks_0", "blocks_1" })
            {
                float[] values = ReadDump(Path.Combine(dir, "layers", name + ".bin"));
                Assert.Equal(Tokens * cfg.InnerDim, values.Length);
                Assert.All(values, v => Assert.True(float.IsFinite(v), $"{name} holds {v}."));
            }
            // The branch input is F16; the stream between blocks stays F32.
            Assert.All(ReadDump(Path.Combine(dir, "layers", "b0_n1.bin")), v => Assert.True((float)(Half)v == v, $"b0_n1 holds {v}, not an F16 value."));
            Assert.Contains(ReadDump(Path.Combine(dir, "layers", "blocks_1.bin")), v => (float)(Half)v != v);
        }
        finally
        {
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>Patch embed, F32 blocks, final layer and unpatchify, each called directly with the transformer's weights.</summary>
    private static float[] F32Path(CudaBackend backend, WanVideoConfig cfg, Dictionary<string, Tensor> w, Tensor latent, Tensor encoder,
        float timestep)
    {
        int dim = cfg.InnerDim;
        (int pt, int ph, int pw) = cfg.PatchSize;
        int gt = (int)latent.Shape[2] / pt, gh = (int)latent.Shape[3] / ph, gw = (int)latent.Shape[4] / pw, s = gt * gh * gw;
        using Tensor patchW = WanDitOps.Reshape2d(w["patch_embedding.weight"], dim, cfg.InChannels * pt * ph * pw);
        Tensor cur = WanDitOps.Patchify(backend, latent, cfg.InChannels, dim, cfg.PatchSize, patchW, w["patch_embedding.bias"]);
        const string Cond = "condition_embedder.";
        (Tensor temb, Tensor proj) = WanDitOps.ConditionTimeGroups(backend, [timestep], cfg.FreqDim, dim,
            w[Cond + "time_embedder.linear_1.weight"], w[Cond + "time_embedder.linear_1.bias"], w[Cond + "time_embedder.linear_2.weight"],
            w[Cond + "time_embedder.linear_2.bias"], w[Cond + "time_proj.weight"], w[Cond + "time_proj.bias"]);
        using Tensor context = WanDitOps.TextEmbed(backend, encoder, dim, w[Cond + "text_embedder.linear_1.weight"],
            w[Cond + "text_embedder.linear_1.bias"], w[Cond + "text_embedder.linear_2.weight"], w[Cond + "text_embedder.linear_2.bias"]);
        WanRope rope = new(cfg.HeadDim, cfg.RopeTheta, cfg.RopeMaxSeqLen);
        (Tensor cos, Tensor sin) = rope.BuildCosSin(gt, gh, gw);
        for (int i = 0; i < cfg.NumLayers; i++)
        {
            WanVideoBlock block = new(cfg, crossAttnNorm: true);
            block.LoadWeights(w, $"blocks.{i}");
            Tensor next = block.Forward(backend, cur, context, proj, rope, cos, sin, s);
            cur.Dispose();
            cur = next;
        }
        using Tensor projected = WanDitOps.FinalLayer(backend, cur, temb, w["scale_shift_table"], w["proj_out.weight"], w["proj_out.bias"],
            s, dim, cfg.Eps, s);
        using Tensor velocity = WanDitOps.Unpatchify(projected, cfg.OutChannels, gt, gh, gw, cfg.PatchSize);
        foreach (Tensor t in new[] { cur, temb, proj, cos, sin }) t.Dispose();
        return Read(velocity);
    }

    /// <summary>One forward with <c>numerics.wanF16</c> set to <paramref name="f16"/> and <c>numerics.ditF16</c> at its default, on.</summary>
    private static float[] Forward(bool f16, Func<Tensor> run, bool ditF16 = true)
    {
        using IDisposable knobs = KnobProfile.Create(f16 ? "wan-f16-on" : "wan-f16-off").With(EngineKnobs.DitF16, ditF16)
            .With(EngineKnobs.WanF16, f16).Push();
        using Tensor velocity = run();
        return Read(velocity);
    }

    /// <summary>Two heads of 64 so the F16 branches take the cuDNN row, two blocks, a 3×4×4 token grid.</summary>
    private static WanVideoConfig Tiny(bool image) => new()
    {
        NumHeads = 2, HeadDim = 64, InChannels = 48, OutChannels = 48, TextDim = 32, FreqDim = 32, FfnDim = 256, NumLayers = 2,
        PatchSize = (1, 2, 2), ImageDim = image ? 32 : 0, AddedKvProjDim = image ? 32 : 0, F16Branches = true,
    };

    /// <summary>One block's weights in the checkpoint's layout: F16 projections at unit-gain scale, F32 norms and modulation table.</summary>
    private static Dictionary<string, Tensor> BlockWeights(WanVideoConfig cfg, ref int seed)
    {
        int dim = cfg.InnerDim, ffn = cfg.FfnDim;
        float gain = MathF.Sqrt(3f / dim);
        Dictionary<string, Tensor> w = new(StringComparer.Ordinal)
        {
            ["b.scale_shift_table"] = Uniform(ref seed, 0.1f, 1, 6, dim),
            ["b.norm2.weight"] = Around(ref seed, 1f, 0.1f, dim),
            ["b.norm2.bias"] = Uniform(ref seed, 0.05f, dim),
            ["b.ffn.net.0.proj.weight"] = Half(Uniform(ref seed, gain, ffn, dim)),
            ["b.ffn.net.0.proj.bias"] = Half(Uniform(ref seed, 0.02f, ffn)),
            ["b.ffn.net.2.weight"] = Half(Uniform(ref seed, MathF.Sqrt(3f / ffn), dim, ffn)),
            ["b.ffn.net.2.bias"] = Half(Uniform(ref seed, 0.02f, dim)),
        };
        foreach (string a in new[] { "attn1", "attn2" })
        {
            foreach (string p in new[] { "to_q", "to_k", "to_v", "to_out.0" })
            {
                w[$"b.{a}.{p}.weight"] = Half(Uniform(ref seed, gain, dim, dim));
                w[$"b.{a}.{p}.bias"] = Half(Uniform(ref seed, 0.02f, dim));
            }
            w[$"b.{a}.norm_q.weight"] = Around(ref seed, 1f, 0.1f, dim);
            w[$"b.{a}.norm_k.weight"] = Around(ref seed, 1f, 0.1f, dim);
        }
        return w;
    }

    /// <summary>A residual stream of unit-scale tokens with a few outlier channels in the hundreds, the shape real Wan's carries.</summary>
    private static Tensor ResidualStream(ref int seed, int tokens, int dim)
    {
        Tensor t = Uniform(ref seed, 4f * MathF.Sqrt(3f), tokens, dim);
        float* p = (float*)t.DataPointer;
        for (int c = 0; c < 12; c++)
        {
            int channel = (c * 257 + 11) % dim;
            float sign = (c & 1) == 0 ? 1f : -1f;
            for (int r = 0; r < tokens; r++)
                p[(long)r * dim + channel] = sign * (450f + 350f * MathF.Abs(MathF.Sin(r * 0.37f + c)));
        }
        return t;
    }

    private static Tensor Uniform(ref int seed, float amplitude, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++)
        {
            seed = seed * 1103515245 + 12345;
            p[i] = ((seed >> 16) & 0x7fff) / 32768f * 2f * amplitude - amplitude;
        }
        return t;
    }

    private static Tensor Around(ref int seed, float center, float amplitude, long width)
    {
        Tensor t = Uniform(ref seed, amplitude, width);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < width; i++) p[i] += center;
        return t;
    }

    private static Tensor Half(Tensor f32)
    {
        Tensor h = f32.CastTo(DType.F16);
        f32.Dispose();
        return h;
    }

    private static double RelL2(float[] a, float[] reference)
    {
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a[i] - reference[i];
            num += d * d;
            den += (double)reference[i] * reference[i];
        }
        return Math.Sqrt(num / den);
    }

    private static float[] Read(Tensor t) => new ReadOnlySpan<float>(t.DataPointer, checked((int)t.ElementCount)).ToArray();

    private static float[] ReadDump(string path) => System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(path)).ToArray();

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }
}
