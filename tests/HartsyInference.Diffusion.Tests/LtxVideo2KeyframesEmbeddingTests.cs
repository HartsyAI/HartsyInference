using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Covers the LTX-2.5 keyframe marker and the variant cross-check. The marker lands on the tokens at
/// temporal position 0; getting that row range wrong changes output without any load-time signal, and the
/// cross-check is what stops a mis-detected checkpoint from generating quietly wrong video.</summary>
public sealed unsafe class LtxVideo2KeyframesEmbeddingTests
{
    private static LtxVideo2Config Tiny(bool keyframes, bool ffBias = false) => LtxVideo2Config.V23 with
    {
        NumLayers = 1,
        NumHeads = 2,
        HeadDim = 8,
        CrossAttentionDim = 16,
        InChannels = 4,
        OutChannels = 4,
        AudioNumHeads = 2,
        AudioHeadDim = 4,
        AudioInChannels = 4,
        AudioOutChannels = 4,
        AudioCrossAttentionDim = 8,
        CaptionChannels = 16,
        FfBias = ffBias,
        UseKeyframesAbsPosEmbedding = keyframes,
    };

    private static (int Frames, int Height, int Width) Grid => (3, 2, 2);

    private static Tensor Filled(int rows, int cols, float value)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = value;
        return t;
    }

    /// <summary>Runs one forward and returns the final video velocity, plus the post-<c>proj_in</c> hidden state
    /// captured before any block. The marker has to be read off that pre-block state: self-attention mixes every
    /// token, so by the first block's output a frame-0-only change has already reached every row.</summary>
    private static (float[] Output, float[] PostProjIn) RunVideo(LtxVideo2Config c, Dictionary<string, Tensor> weights)
    {
        IBackend backend = new CpuBackend();
        using LtxVideo2Transformer dit = new LtxVideo2Transformer(c);
        dit.LoadWeights(weights);

        float[]? postProjIn = null;
        dit.OnBlockOutput = (index, hidden, _) =>
        {
            if (index != -1 || postProjIn is not null) return;
            postProjIn = new float[hidden.ElementCount];
            new Span<float>((void*)hidden.DataPointer, postProjIn.Length).CopyTo(postProjIn);
        };

        int videoTokens = Grid.Frames * Grid.Height * Grid.Width;
        const int audioFrames = 4;
        using Tensor video = Filled(videoTokens, c.InChannels, 0.05f);
        using Tensor audio = Filled(audioFrames, c.AudioInChannels, 0.05f);
        using Tensor encVideo = Filled(6, c.CrossAttentionDim, 0.02f);
        using Tensor encAudio = Filled(6, c.AudioCrossAttentionDim, 0.02f);

        (Tensor outVideo, Tensor outAudio) = dit.Forward(backend, video, audio, encVideo, encAudio,
            timestep: 500f, Grid, audioFrames, fps: 24.0, null, null);
        using (outVideo)
        using (outAudio)
        {
            float[] copy = new float[outVideo.ElementCount];
            new Span<float>((void*)outVideo.DataPointer, copy.Length).CopyTo(copy);
            return (copy, postProjIn!);
        }
    }

    [Fact]
    public void MarkerLandsExactlyOnTheFirstLatentFrame()
    {
        // Same weights either way apart from the marker, so any delta is attributable to it. The rope clamps the
        // causal temporal start at 0 for frame 0 only, so exactly the leading Height*Width token rows may move —
        // and each moves by the marker's value, since the marker is added straight onto the projected tokens.
        const float marker = 0.75f;
        LtxVideo2Config off = Tiny(keyframes: false);
        Dictionary<string, Tensor> baseline = LtxSyntheticWeights.BuildTransformer2(off);

        LtxVideo2Config on = Tiny(keyframes: true);
        Dictionary<string, Tensor> marked = new(baseline)
        {
            ["keyframes_abs_pos_embedding"] = ConstRow(on.InnerDim, marker),
        };

        float[] without = RunVideo(off, baseline).PostProjIn;
        float[] with = RunVideo(on, marked).PostProjIn;

        int rowsPerFrame = Grid.Height * Grid.Width;
        int inner = on.InnerDim;
        Assert.Equal(Grid.Frames * rowsPerFrame * inner, with.Length);

        for (int row = 0; row < Grid.Frames * rowsPerFrame; row++)
        {
            float expected = row < rowsPerFrame ? marker : 0f;
            for (int col = 0; col < inner; col++)
            {
                int i = row * inner + col;
                Assert.Equal(expected, with[i] - without[i], 5);
            }
        }
    }

    [Fact]
    public void MarkerReachesEveryTokenThroughAttention()
    {
        // The mask is narrow but its effect is not: guards against the marker being applied somewhere that the
        // blocks never see, which the pre-block assertion above cannot detect on its own.
        LtxVideo2Config off = Tiny(keyframes: false);
        Dictionary<string, Tensor> baseline = LtxSyntheticWeights.BuildTransformer2(off);
        LtxVideo2Config on = Tiny(keyframes: true);
        Dictionary<string, Tensor> marked = new(baseline)
        {
            ["keyframes_abs_pos_embedding"] = ConstRow(on.InnerDim, 0.75f),
        };

        float[] without = RunVideo(off, baseline).Output;
        float[] with = RunVideo(on, marked).Output;

        int differing = 0;
        for (int i = 0; i < with.Length; i++)
            if (MathF.Abs(with[i] - without[i]) > 1e-6f) differing++;

        Assert.True(differing > 0, "the keyframe marker made no difference to the velocity");
    }

    [Fact]
    public void MarkerIsInertWhenAbsent()
    {
        LtxVideo2Config c = Tiny(keyframes: false);
        Dictionary<string, Tensor> w = LtxSyntheticWeights.BuildTransformer2(c);

        float[] first = RunVideo(c, w).Output;
        float[] second = RunVideo(c, w).Output;

        Assert.Equal(first.Length, second.Length);
        for (int i = 0; i < first.Length; i++) Assert.Equal(first[i], second[i], 6);
        Assert.All(first, f => Assert.True(float.IsFinite(f)));
    }

    [Fact]
    public void BiasFreeVideoFfnMatchesAnExplicitZeroBias()
    {
        // 2.5 removes the video FFN bias entirely. A bias-free Linear must compute the same thing an all-zero bias
        // would — "it loaded without throwing" is not evidence of that.
        LtxVideo2Config c = Tiny(keyframes: false, ffBias: false);
        Dictionary<string, Tensor> biasFree = LtxSyntheticWeights.BuildTransformer2(c);

        LtxVideo2Config withBias = c with { FfBias = true };
        Dictionary<string, Tensor> zeroBias = new(biasFree)
        {
            ["transformer_blocks.0.ff.net.0.proj.bias"] = ConstRow(c.FfnMultiplier * c.InnerDim, 0f),
            ["transformer_blocks.0.ff.net.2.bias"] = ConstRow(c.InnerDim, 0f),
        };

        float[] a = RunVideo(c, biasFree).Output;
        float[] b = RunVideo(withBias, zeroBias).Output;

        Assert.Equal(a.Length, b.Length);
        for (int i = 0; i < a.Length; i++) Assert.Equal(a[i], b[i], 6);
    }

    [Fact]
    public void CheckpointWithoutTheMarkerRejectsA25Config()
    {
        LtxVideo2Config claims25 = Tiny(keyframes: true);
        Dictionary<string, Tensor> without = LtxSyntheticWeights.BuildTransformer2(claims25 with { UseKeyframesAbsPosEmbedding = false });

        using LtxVideo2Transformer dit = new LtxVideo2Transformer(claims25);
        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(() => dit.LoadWeights(without));
        Assert.Contains("keyframes_abs_pos_embedding", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void DetectedRepackConfigLoadsTheRepack()
    {
        // The detector and the load-time cross-check have to agree on a metadata-stripped 2.5 repack: run the real
        // detector over the real key set, then load with what it returned.
        LtxVideo2Config shape = Tiny(keyframes: true, ffBias: false);
        Dictionary<string, Tensor> weights = LtxSyntheticWeights.BuildTransformer2(shape);

        LtxVideo2Config detected = LtxVideo2VariantDetector.Detect(
            new Dictionary<string, string> { ["format"] = "pt" },
            weights.ContainsKey);

        Assert.False(detected.FfBias);
        Assert.True(detected.UseKeyframesAbsPosEmbedding);

        // Carry the detected generation flags onto the tiny geometry this fixture uses.
        LtxVideo2Config runnable = shape with
        {
            FfBias = detected.FfBias,
            UseKeyframesAbsPosEmbedding = detected.UseKeyframesAbsPosEmbedding,
        };
        using LtxVideo2Transformer dit = new LtxVideo2Transformer(runnable);
        dit.LoadWeights(weights);
    }

    [Fact]
    public void FfnBiasMismatchIsRejected()
    {
        LtxVideo2Config claimsNoBias = Tiny(keyframes: false, ffBias: false);
        Dictionary<string, Tensor> withBias = LtxSyntheticWeights.BuildTransformer2(claimsNoBias with { FfBias = true });

        using LtxVideo2Transformer dit = new LtxVideo2Transformer(claimsNoBias);
        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(() => dit.LoadWeights(withBias));
        Assert.Contains("FfBias", ex.Message, StringComparison.Ordinal);
    }

    private static Tensor ConstRow(int n, float value)
    {
        Tensor t = new Tensor(new TensorShape(1, n), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < n; i++) p[i] = value;
        return t;
    }
}
