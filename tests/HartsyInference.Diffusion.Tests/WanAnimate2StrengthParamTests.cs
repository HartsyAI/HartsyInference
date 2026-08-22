using Xunit;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The Animate-2 strength knobs (ComfyUI <c>pose_strength</c> / <c>reference_image_strength</c>): the 1.0
/// default must be an exact no-op (the skip path is bit-identical), non-default values must actually reach the
/// output, the reference scale must touch ONLY the reference-image slot's rows, and Wan-Animate V1 — which has no
/// such pathway — must refuse a non-default value by name rather than silently ignore it.</summary>
public unsafe class WanAnimate2StrengthParamTests
{
    private const int Frames = 3;
    private const int GridH = 3;
    private const int GridW = 2;

    private static WanVideoConfig Config(int layers) => new WanVideoConfig
    {
        NumHeads = 2,
        HeadDim = 16,
        InChannels = 36,
        OutChannels = 16,
        VaeLatentChannels = 16,
        FfnDim = 64,
        NumLayers = layers,
        VaeSpatialCompression = 8,
        ImageDim = 8,
        AddedKvProjDim = 8,
        IsAnimate2 = true,
    };

    private static Tensor Random(TensorShape shape, int seed)
    {
        Tensor t = new Tensor(shape, DType.F32);
        Random rng = new Random(seed);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return t;
    }

    private static double MeanAbsDiff(Tensor a, Tensor b)
    {
        float* pa = (float*)a.DataPointer, pb = (float*)b.DataPointer;
        double s = 0;
        for (long i = 0; i < a.ElementCount; i++) s += Math.Abs(pa[i] - pb[i]);
        return s / a.ElementCount;
    }

    private static long FirstBitDifference(Tensor a, Tensor b)
    {
        int* pa = (int*)a.DataPointer, pb = (int*)b.DataPointer;
        for (long i = 0; i < a.ElementCount; i++)
        {
            if (pa[i] != pb[i]) return i;
        }
        return -1;
    }

    private static (WanAnimate2Transformer Dit, CpuBackend Backend, Tensor Latents, Tensor Encoder, Tensor Clip,
        WanAnimate2DrivingCache Cache) Build()
    {
        WanVideoConfig c = Config(layers: 2);
        Dictionary<string, Tensor> weights = WanSyntheticWeights.BuildTransformer(c);
        CpuBackend backend = new CpuBackend();
        WanAnimate2Transformer dit = new WanAnimate2Transformer(c);
        dit.LoadWeights(weights);
        (int T, int H, int W) genGrid = (Frames + 1, GridH, GridW);
        Tensor latents = Random(new TensorShape([1L, c.InChannels, Frames + 1, GridH * 2, GridW * 2]), 11);
        Tensor encoder = Random(new TensorShape(6, c.TextDim), 12);
        Tensor clip = Random(new TensorShape(5, c.ImageDim), 13);
        using Tensor driving = Random(new TensorShape([1L, c.VaeLatentChannels, Frames, GridH * 2, GridW * 2]), 101);
        WanAnimate2DrivingCache cache = dit.EncodeDriving(backend, driving, encoder, clip, genGrid);
        return (dit, backend, latents, encoder, clip, cache);
    }

    /// <summary>Explicit 1.0 must take the identical code path as the defaults — the skip conditions in
    /// <see cref="WanVideoBlock"/> guarantee a generation with the params untouched cannot drift by a single bit.</summary>
    [Fact]
    public void DefaultStrengths_BitIdenticalToExplicitOne()
    {
        (WanAnimate2Transformer dit, CpuBackend backend, Tensor latents, Tensor encoder, Tensor clip,
            WanAnimate2DrivingCache cache) = Build();
        using (dit)
        using (backend)
        using (latents)
        using (encoder)
        using (clip)
        using (cache)
        {
            using Tensor byDefault = dit.Forward(backend, latents, encoder, 1000f, cache, clip);
            using Tensor explicitOne = dit.Forward(backend, latents, encoder, 1000f, cache, clip,
                poseStrength: 1.0f, referenceImageStrength: 1.0f);
            Assert.Equal(-1, FirstBitDifference(byDefault, explicitOne));
        }
    }

    [Fact]
    public void PoseStrength_ChangesOutput()
    {
        (WanAnimate2Transformer dit, CpuBackend backend, Tensor latents, Tensor encoder, Tensor clip,
            WanAnimate2DrivingCache cache) = Build();
        using (dit)
        using (backend)
        using (latents)
        using (encoder)
        using (clip)
        using (cache)
        {
            using Tensor baseline = dit.Forward(backend, latents, encoder, 1000f, cache, clip);
            using Tensor scaled = dit.Forward(backend, latents, encoder, 1000f, cache, clip, poseStrength: 1.3f);
            Assert.True(MeanAbsDiff(baseline, scaled) > 1e-6,
                "poseStrength=1.3 is INERT: output matches the 1.0 baseline.");
        }
    }

    [Fact]
    public void ReferenceImageStrength_ChangesOutput()
    {
        (WanAnimate2Transformer dit, CpuBackend backend, Tensor latents, Tensor encoder, Tensor clip,
            WanAnimate2DrivingCache cache) = Build();
        using (dit)
        using (backend)
        using (latents)
        using (encoder)
        using (clip)
        using (cache)
        {
            using Tensor baseline = dit.Forward(backend, latents, encoder, 1000f, cache, clip);
            using Tensor scaled = dit.Forward(backend, latents, encoder, 1000f, cache, clip, referenceImageStrength: 1.3f);
            Assert.True(MeanAbsDiff(baseline, scaled) > 1e-6,
                "referenceImageStrength=1.3 is INERT: output matches the 1.0 baseline.");
        }
    }

    /// <summary>The mechanism itself: only the reference-image slot — rows <c>[0, hw)</c> — is scaled, every other
    /// row is bitwise untouched (ComfyUI's <c>v[:, :hw] *= ref_strength</c>).</summary>
    [Fact]
    public void ScaleReferenceRows_ScalesOnlyRowsBelowHw()
    {
        const int hw = 6, frames = 3, dim = 32;
        using CpuBackend backend = new CpuBackend();
        using Tensor v = Random(new TensorShape(hw * frames, dim), 42);
        using Tensor original = Random(new TensorShape(hw * frames, dim), 42);
        WanVideoBlock.ScaleReferenceRows(backend, v, hw, dim, 1.5f);
        float* pv = (float*)v.DataPointer;
        float* po = (float*)original.DataPointer;
        for (long i = 0; i < hw * dim; i++)
        {
            Assert.Equal(po[i] * 1.5f, pv[i]);
        }
        for (long i = hw * dim; i < v.ElementCount; i++)
        {
            Assert.Equal(BitConverter.SingleToInt32Bits(po[i]), BitConverter.SingleToInt32Bits(pv[i]));
        }
    }

    [Fact]
    public void ScaleReferenceRows_AtOne_IsExactNoOp()
    {
        const int hw = 6, frames = 3, dim = 32;
        using CpuBackend backend = new CpuBackend();
        using Tensor v = Random(new TensorShape(hw * frames, dim), 42);
        using Tensor original = Random(new TensorShape(hw * frames, dim), 42);
        WanVideoBlock.ScaleReferenceRows(backend, v, hw, dim, 1.0f);
        Assert.Equal(-1, FirstBitDifference(original, v));
    }

    /// <summary>V1 has no strength pathway: a non-default value must refuse by name, and a toggled-but-1.0 value
    /// must pass through (it is a no-op on both models). The 1.0 case is proven to get PAST the strength check by
    /// hitting the next validation in line (the missing reference image).</summary>
    [Fact]
    public void WanAnimateV1_RefusesNonDefaultStrengths()
    {
        // Not disposed: every component is a null stand-in (validation fires before any is touched), and Dispose
        // would dereference them.
        WanAnimateRecipePipeline v1 = new WanAnimateRecipePipeline(
            null!, null!, null!, null!, null!, null!, null!, null, []);
        ImageData still = new ImageData { Rgb = new byte[3], Width = 1, Height = 1 };
        VideoRequest bad = new VideoRequest { Prompt = "x", InitImage = still, AnimatePoseStrength = 1.5, AnimateReferenceImageStrength = 0.5 };
        NotSupportedException ex = Assert.Throws<NotSupportedException>(() => v1.Generate(bad, null, CancellationToken.None));
        Assert.Contains(nameof(VideoRequest.AnimatePoseStrength), ex.Message);
        Assert.Contains(nameof(VideoRequest.AnimateReferenceImageStrength), ex.Message);

        VideoRequest toggledDefault = new VideoRequest { Prompt = "x", InitImage = still, AnimatePoseStrength = 1.0, AnimateReferenceImageStrength = 1.0 };
        InvalidOperationException past = Assert.Throws<InvalidOperationException>(() => v1.Generate(toggledDefault, null, CancellationToken.None));
        Assert.Contains("character identity image", past.Message);
    }
}
