using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>CPU structural tests for <see cref="WanAnimate2Transformer"/> on tiny synthetic weights. Each covers one
/// of the four reference behaviours that are silent when wrong: which driving frame a generation frame may attend,
/// the driving RoPE offset being derived per call, the unconditional block-9 skip, and the <c>log_scale</c> band.
/// ComfyUI implements none of the last three, so it is not a parity reference for any of them.</summary>
public unsafe class WanAnimate2TransformerTests
{
    private const int GenFrames = 4;
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

    private static Tensor GenLatent(WanVideoConfig c, int seed) =>
        Random(new TensorShape([1L, c.InChannels, GenFrames, GridH * 2, GridW * 2]), seed);

    private static Tensor DrivingLatent(WanVideoConfig c, int frames, int seed) =>
        Random(new TensorShape([1L, c.VaeLatentChannels, frames, GridH * 2, GridW * 2]), seed);

    /// <summary>One forward over the tiny model, returning the velocity <c>[1, 16, T, H, W]</c>.</summary>
    private static Tensor Run(CpuBackend backend, WanAnimate2Transformer dit, Tensor latent, Tensor drivingLatent,
        Tensor encoder, Tensor clip, (int T, int H, int W) grid, bool unconditional = false)
    {
        using WanAnimate2DrivingCache cache = dit.EncodeDriving(backend, drivingLatent, encoder, clip, grid);
        return dit.Forward(backend, latent, encoder, 500f, cache, clip, unconditional);
    }

    private static WanAnimate2Transformer Load(WanVideoConfig c, Dictionary<string, Tensor> weights)
    {
        WanAnimate2Transformer dit = new WanAnimate2Transformer(c);
        dit.LoadWeights(weights);
        return dit;
    }

    /// <summary>Maximum absolute difference of one latent frame across two velocity tensors.</summary>
    private static float FrameDelta(Tensor a, Tensor b, int frame)
    {
        int channels = (int)a.Shape[1], frames = (int)a.Shape[2];
        long plane = a.Shape[3] * a.Shape[4];
        float* pa = (float*)a.DataPointer, pb = (float*)b.DataPointer;
        float worst = 0f;
        for (int ch = 0; ch < channels; ch++)
            for (long i = 0; i < plane; i++)
            {
                long idx = ((long)ch * frames + frame) * plane + i;
                worst = MathF.Max(worst, MathF.Abs(pa[idx] - pb[idx]));
            }
        return worst;
    }

    private static void AssertFinite(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"non-finite output at {i}");
    }

    /// <summary>Generation frame <c>j</c> may see driving frame <c>j-1</c> and nothing else; frame 0 (the
    /// reference-image slot) sees no driving frame at all. Proven on a ONE-block model, where the driving stream
    /// influences the output only through that block's spliced K/V, so the locality is exact rather than diffused.</summary>
    [Fact]
    public void FrameLocalAttention_GenFrameJSeesOnlyDrivingFrameJMinusOne()
    {
        WanVideoConfig c = Config(layers: 1);
        Dictionary<string, Tensor> weights = WanSyntheticWeights.BuildTransformer(c);
        using CpuBackend backend = new CpuBackend();
        using WanAnimate2Transformer dit = Load(c, weights);
        using Tensor latent = GenLatent(c, 11);
        using Tensor encoder = Random(new TensorShape(6, c.TextDim), 12);
        using Tensor clip = Random(new TensorShape(5, c.ImageDim), 13);
        (int T, int H, int W) grid = (GenFrames, GridH, GridW);

        using Tensor driving = DrivingLatent(c, GenFrames - 1, 14);
        using Tensor baseline = Run(backend, dit, latent, driving, encoder, clip, grid);
        AssertFinite(baseline);

        for (int m = 0; m < GenFrames - 1; m++)
        {
            using Tensor perturbed = DrivingLatent(c, GenFrames - 1, 14);
            PerturbFrame(perturbed, m);
            using Tensor got = Run(backend, dit, latent, perturbed, encoder, clip, grid);
            for (int j = 0; j < GenFrames; j++)
            {
                float delta = FrameDelta(baseline, got, j);
                if (j == m + 1) Assert.True(delta > 1e-4f, $"driving frame {m} did not reach generation frame {j} (delta {delta}).");
                else Assert.True(delta == 0f, $"driving frame {m} leaked into generation frame {j} (delta {delta}).");
            }
        }
    }

    private static void PerturbFrame(Tensor drivingLatent, int frame)
    {
        int channels = (int)drivingLatent.Shape[1], frames = (int)drivingLatent.Shape[2];
        long plane = drivingLatent.Shape[3] * drivingLatent.Shape[4];
        float* p = (float*)drivingLatent.DataPointer;
        for (int ch = 0; ch < channels; ch++)
            for (long i = 0; i < plane; i++) p[((long)ch * frames + frame) * plane + i] += 0.75f;
    }

    /// <summary>The driving RoPE offsets come from the CURRENT generation grid on every call. The reference stores
    /// <c>refer_offset_w = -1</c> and overwrites the sentinel on its first forward, so a second resolution keeps the
    /// first one's horizontal offset — this runs both resolutions through ONE transformer instance and requires the
    /// second resolution's output to equal a fresh instance's.</summary>
    [Fact]
    public void RefRopeOffsets_AreDerivedPerCall_NotResolvedOnceAndCached()
    {
        Assert.Equal((1, 0, GridW), WanAnimate2Transformer.RefRopeOffsets(GridW));
        Assert.Equal((1, 0, GridW * 2), WanAnimate2Transformer.RefRopeOffsets(GridW * 2));

        WanVideoConfig c = Config(layers: 2);
        Dictionary<string, Tensor> weights = WanSyntheticWeights.BuildTransformer(c);
        using CpuBackend backend = new CpuBackend();
        using Tensor encoder = Random(new TensorShape(6, c.TextDim), 22);
        using Tensor clip = Random(new TensorShape(5, c.ImageDim), 23);

        // A wider grid first, then the narrow one — the order that traps a sentinel resolved on the first forward.
        const int wideW = GridW * 2;
        using Tensor wideLatent = Random(new TensorShape([1L, c.InChannels, GenFrames, GridH * 2, wideW * 2]), 24);
        using Tensor wideDriving = Random(new TensorShape([1L, c.VaeLatentChannels, GenFrames - 1, GridH * 2, wideW * 2]), 25);
        using Tensor narrowLatent = GenLatent(c, 26);
        using Tensor narrowDriving = DrivingLatent(c, GenFrames - 1, 27);

        using WanAnimate2Transformer fresh = Load(c, weights);
        using Tensor expected = Run(backend, fresh, narrowLatent, narrowDriving, encoder, clip, (GenFrames, GridH, GridW));

        using WanAnimate2Transformer reused = Load(c, weights);
        using Tensor wide = Run(backend, reused, wideLatent, wideDriving, encoder, clip, (GenFrames, GridH, wideW));
        AssertFinite(wide);
        using Tensor got = Run(backend, reused, narrowLatent, narrowDriving, encoder, clip, (GenFrames, GridH, GridW));

        float worst = 0f;
        for (int j = 0; j < GenFrames; j++) worst = MathF.Max(worst, FrameDelta(expected, got, j));
        Assert.True(worst == 0f, $"the narrow-grid output changed after a wide-grid forward (max delta {worst}) — the driving RoPE offset outlived its resolution.");
    }

    /// <summary>Block 9 is skipped entirely on the unconditional pass, so the negative branch runs one block fewer
    /// and its output differs from the conditional branch's beyond the timestep/prompt.</summary>
    [Fact]
    public void UnconditionalPass_SkipsBlockNineEntirely()
    {
        WanVideoConfig c = Config(layers: 12);
        Dictionary<string, Tensor> weights = WanSyntheticWeights.BuildTransformer(c);
        using CpuBackend backend = new CpuBackend();
        using WanAnimate2Transformer dit = Load(c, weights);
        using Tensor latent = GenLatent(c, 31);
        using Tensor encoder = Random(new TensorShape(6, c.TextDim), 32);
        using Tensor clip = Random(new TensorShape(5, c.ImageDim), 33);
        using Tensor driving = DrivingLatent(c, GenFrames - 1, 34);
        (int T, int H, int W) grid = (GenFrames, GridH, GridW);
        using WanAnimate2DrivingCache cache = dit.EncodeDriving(backend, driving, encoder, clip, grid);

        List<int> visited = [];
        dit.BeforeBlockForward = visited.Add;

        visited.Clear();
        using Tensor cond = dit.Forward(backend, latent, encoder, 500f, cache, clip, unconditional: false);
        Assert.Equal(Enumerable.Range(0, c.NumLayers), visited);

        visited.Clear();
        using Tensor uncond = dit.Forward(backend, latent, encoder, 500f, cache, clip, unconditional: true);
        Assert.Equal(Enumerable.Range(0, c.NumLayers).Where(i => i != WanAnimate2Transformer.UncondSkipBlockIndex), visited);
        Assert.DoesNotContain(WanAnimate2Transformer.UncondSkipBlockIndex, visited);

        AssertFinite(cond);
        AssertFinite(uncond);
        Assert.True(FrameDelta(cond, uncond, 1) > 1e-5f, "the uncond pass produced the same output as the cond pass.");
    }

    /// <summary>The bias covers exactly the <c>[hw, 2hw)</c> key band — the keys of generation latent frame 1 — and
    /// is stored as ONE row, because the reference <c>score_mod</c> has no query-side condition. The <c>[hw, keys]</c>
    /// duplicate of it is what used to force every biased call onto the materialized score matrix.</summary>
    [Theory]
    [InlineData(6, 24)]
    [InlineData(6, 30)]
    public void LogScaleBias_CoversOnlyTheKeysOfGenerationFrameOne(int hw, int keys)
    {
        const float logScale = -1.3f;
        using Tensor bias = WanAnimate2Transformer.BuildLogScaleBias(hw, keys, logScale);
        Assert.Equal(new TensorShape(1, keys), bias.Shape);
        float* p = (float*)bias.DataPointer;
        for (int k = 0; k < keys; k++)
            Assert.Equal(k >= hw && k < 2 * hw ? logScale : 0f, p[k]);
    }

    /// <summary>The bias must reproduce upstream's <c>_score_mod_impl</c> — <c>score + log_scale</c> exactly on the
    /// key band <c>[hw, 2hw)</c>, the score untouched everywhere else, with no query-side condition — for the tiny
    /// T=4 grid and a real T=21 480x800-sized grid, at both the gen (<c>hw*T</c>) and spliced (<c>hw*(T+1)</c>) key
    /// lengths the transformer builds.</summary>
    [Theory]
    [InlineData(6, 4)]
    [InlineData(1500, 21)]
    public void BuildLogScaleBias_MatchesUpstreamScoreMod(int hw, int frames)
    {
        const float logScale = -1.3f;
        foreach (int keys in new[] { hw * frames, hw * (frames + 1) })
        {
            using Tensor bias = WanAnimate2Transformer.BuildLogScaleBias(hw, keys, logScale);
            Assert.Equal(new TensorShape(1, keys), bias.Shape);
            float* p = (float*)bias.DataPointer;
            for (int k = 0; k < keys; k++)
            {
                float score = 0.25f * (k % 7) - 0.5f;
                float upstream = k >= hw && k < 2 * hw ? score + logScale : score;
                Assert.Equal(upstream, score + p[k]);
            }
        }
    }

    /// <summary>The base build's <c>log_scale = 0</c> must take the unmasked path, and the distillation build's
    /// <c>-1.3</c> must actually change the output — this is the ONLY difference between the two checkpoints.</summary>
    [Fact]
    public void LogScale_ChangesTheOutput_AndIsInertAtZero()
    {
        WanVideoConfig baseConfig = Config(layers: 2);
        WanVideoConfig distill = baseConfig with { Animate2LogScale = -1.3f };
        Dictionary<string, Tensor> weights = WanSyntheticWeights.BuildTransformer(baseConfig);
        using CpuBackend backend = new CpuBackend();
        using Tensor latent = GenLatent(baseConfig, 41);
        using Tensor encoder = Random(new TensorShape(6, baseConfig.TextDim), 42);
        using Tensor clip = Random(new TensorShape(5, baseConfig.ImageDim), 43);
        using Tensor driving = DrivingLatent(baseConfig, GenFrames - 1, 44);
        (int T, int H, int W) grid = (GenFrames, GridH, GridW);

        using WanAnimate2Transformer baseDit = Load(baseConfig, weights);
        using Tensor withoutBias = Run(backend, baseDit, latent, driving, encoder, clip, grid);
        using WanAnimate2Transformer distillDit = Load(distill, weights);
        using Tensor withBias = Run(backend, distillDit, latent, driving, encoder, clip, grid);

        AssertFinite(withoutBias);
        AssertFinite(withBias);
        // The band is the keys of generation frame 1, which every frame's queries attend — so every frame moves.
        for (int j = 0; j < GenFrames; j++)
            Assert.True(FrameDelta(withoutBias, withBias, j) > 1e-5f, $"log_scale left generation frame {j} unchanged.");
    }

    /// <summary>The driving stream must carry exactly one fewer latent frame than the generation stream, and a cache
    /// built for one grid must never be spliced into another.</summary>
    [Fact]
    public void DrivingCache_RefusesAMismatchedFrameCountOrGrid()
    {
        WanVideoConfig c = Config(layers: 1);
        Dictionary<string, Tensor> weights = WanSyntheticWeights.BuildTransformer(c);
        using CpuBackend backend = new CpuBackend();
        using WanAnimate2Transformer dit = Load(c, weights);
        using Tensor encoder = Random(new TensorShape(6, c.TextDim), 52);
        using Tensor clip = Random(new TensorShape(5, c.ImageDim), 53);
        (int T, int H, int W) grid = (GenFrames, GridH, GridW);

        using Tensor tooMany = DrivingLatent(c, GenFrames, 54);
        Assert.Throws<ArgumentException>(() => dit.EncodeDriving(backend, tooMany, encoder, clip, grid));

        using Tensor driving = DrivingLatent(c, GenFrames - 1, 55);
        using WanAnimate2DrivingCache cache = dit.EncodeDriving(backend, driving, encoder, clip, grid);
        using Tensor wrongGrid = Random(new TensorShape([1L, c.InChannels, GenFrames, GridH * 2, GridW * 4]), 56);
        Assert.Throws<ArgumentException>(() => dit.Forward(backend, wrongGrid, encoder, 500f, cache, clip));
    }

    /// <summary>The two Animate-2 builds are key-for-key identical and both declare only
    /// <c>model_type: "animate2"</c>, so the distillation build's <c>log_scale = -1.3</c> can only come from the file
    /// name. It never reached the config at all before: <c>WanConfigDetector</c> set <c>IsAnimate2</c> and left the
    /// score bias at the base build's 0, so a distillation checkpoint silently ran without the bias it was trained
    /// with, and <c>LogScaleBias</c> took the null/unmasked path.</summary>
    [Theory]
    [InlineData("wan_animate_2_bf16_distillation.safetensors", -1.3f)]
    [InlineData("/models/Wan/Animate2/WAN_ANIMATE_2_DISTILL.safetensors", -1.3f)]
    [InlineData("wan_animate_2_bf16.safetensors", 0f)]
    [InlineData("wan_animate_2_int8_convrot.safetensors", 0f)]
    public void ResolveLogScale_RoutesTheDistillationBuildByName(string path, float expected)
    {
        Assert.Equal(expected, WanAnimate2Transformer.ResolveLogScale(path));
    }
}
