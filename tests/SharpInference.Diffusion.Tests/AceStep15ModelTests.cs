using Xunit;
using SharpInference.Core.Backends;
using SharpInference.Core.Models;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.Music;
using SharpInference.Diffusion.Pipelines;
using SharpInference.Tests.Common;

namespace SharpInference.Diffusion.Tests;

/// <summary>Tiny-config CPU tests for the ACE-Step v1.5 turbo stack: the fixed shift timestep tables, condition
/// packing ([lyric ‖ timbre ‖ text] with optional segments), the DiT forward (shape, finiteness, condition/timestep
/// sensitivity), and the end-to-end 8-step pipeline against a fake latent codec (the silence-encode path included).
/// Numerics validation-pending vs the Python reference.</summary>
public unsafe class AceStep15ModelTests
{
    [Fact]
    public void TimestepTables_MatchReferenceConstantsAndSnap()
    {
        float[] shift3 = AceStep15Config.GetTimesteps(3f);
        Assert.Equal(8, shift3.Length);
        Assert.Equal(1f, shift3[0], 6);
        Assert.Equal(0.9545454545f, shift3[1], 6);
        Assert.Equal(0.6428571428f, shift3[5], 6);
        Assert.Equal(0.3f, shift3[7], 6);

        float[] shift2 = AceStep15Config.GetTimesteps(2f);
        Assert.Equal(0.9333333333f, shift2[1], 6);
        Assert.Equal(0.2222222222f, shift2[7], 6);

        float[] shift1 = AceStep15Config.GetTimesteps(1f);
        for (int i = 0; i < 8; i++) Assert.Equal(1f - i * 0.125f, shift1[i], 6);

        // Shift snaps to the nearest of VALID_SHIFTS [1, 2, 3].
        Assert.Equal(shift1, AceStep15Config.GetTimesteps(1.4f));
        Assert.Equal(shift2, AceStep15Config.GetTimesteps(2.2f));
        Assert.Equal(shift3, AceStep15Config.GetTimesteps(9f));
    }

    [Fact]
    public void ConditionEncoder_PacksLyricTimbreTextWithOptionalSegments()
    {
        CpuBackend backend = new();
        AceStep15Config cfg = AceStep15SyntheticWeights.TinyConfig;
        AceStep15ConditionEncoder encoder = new(cfg);
        encoder.LoadWeights(AceStep15SyntheticWeights.BuildModel(cfg));

        Tensor text = Rand([3, cfg.TextHiddenDim], seed: 61);
        Tensor lyric = Rand([4, cfg.TextHiddenDim], seed: 62);
        Tensor timbre = Rand([5, cfg.TimbreHiddenDim], seed: 63);

        Tensor full = encoder.EncodeConditions(backend, text, lyric, timbre);
        Assert.Equal(new TensorShape(1, 4 + 1 + 3, cfg.HiddenSize), full.Shape);
        AssertFinite(full);

        Tensor noTimbre = encoder.EncodeConditions(backend, text, lyric, null);
        Assert.Equal(7, (int)noTimbre.Shape[1]);
        Tensor textOnly = encoder.EncodeConditions(backend, text, null, null);
        Assert.Equal(3, (int)textOnly.Shape[1]);

        // The text segment sits last and is identical whether or not lyric/timbre segments are present.
        float* fp = (float*)full.DataPointer;
        float* tp = (float*)textOnly.DataPointer;
        int dim = cfg.HiddenSize;
        for (int i = 0; i < 3 * dim; i++)
            Assert.Equal(tp[i], fp[(4 + 1) * dim + i], 5);

        full.Dispose(); noTimbre.Dispose(); textOnly.Dispose();
        text.Dispose(); lyric.Dispose(); timbre.Dispose();
    }

    [Fact]
    public void Dit_Forward_ShapeFinitenessAndSensitivity()
    {
        CpuBackend backend = new();
        AceStep15Config cfg = AceStep15SyntheticWeights.TinyConfig;
        using AceStep15Dit dit = new(cfg);
        dit.LoadWeights(AceStep15SyntheticWeights.BuildModel(cfg));

        const int frames = 12;   // > 2·(window+1) so the alternating sliding mask is live
        Tensor noisy = Rand([1, frames, cfg.LatentChannels], seed: 71);
        Tensor context = Rand([1, frames, 2 * cfg.LatentChannels], seed: 72);
        Tensor cond = Rand([1, 5, cfg.HiddenSize], seed: 73);

        Tensor v1 = dit.Forward(backend, noisy, context, cond, timestep: 0.9f, timestepR: 0.9f);
        Assert.Equal(new TensorShape(1, frames, cfg.LatentChannels), v1.Shape);
        AssertFinite(v1);

        // Conditions feed cross-attention in every layer.
        Tensor cond2 = Rand([1, 5, cfg.HiddenSize], seed: 74);
        Tensor v2 = dit.Forward(backend, noisy, context, cond2, 0.9f, 0.9f);
        Assert.True(MaxDiff(v1, v2) > 1e-7f, "condition change must alter the velocity");

        // Timestep feeds the AdaLN-6 modulation.
        Tensor v3 = dit.Forward(backend, noisy, context, cond, 0.3f, 0.3f);
        Assert.True(MaxDiff(v1, v3) > 1e-7f, "timestep change must alter the velocity");

        // Context latents are concatenated into proj_in.
        Tensor context2 = Rand([1, frames, 2 * cfg.LatentChannels], seed: 75);
        Tensor v4 = dit.Forward(backend, noisy, context2, cond, 0.9f, 0.9f);
        Assert.True(MaxDiff(v1, v4) > 1e-7f, "context latents must alter the velocity");

        noisy.Dispose(); context.Dispose(); context2.Dispose(); cond.Dispose(); cond2.Dispose();
        v1.Dispose(); v2.Dispose(); v3.Dispose(); v4.Dispose();
    }

    [Fact]
    public void Pipeline_Generate_EndToEndTurbo()
    {
        CpuBackend backend = new();
        AceStep15Config cfg = AceStep15SyntheticWeights.TinyConfig;
        Dictionary<string, Tensor> weights = AceStep15SyntheticWeights.BuildModel(cfg);
        AceStep15Dit dit = new(cfg);
        dit.LoadWeights(weights);
        AceStep15ConditionEncoder encoder = new(cfg);
        encoder.LoadWeights(weights);
        FakeLatentCodec vae = new(cfg.LatentChannels, cfg.SamplesPerLatent);

        using AceStepPipeline15 pipeline = new(backend, dit, encoder, vae, cfg);
        Tensor text = Rand([3, cfg.TextHiddenDim], seed: 81);
        Tensor lyric = Rand([2, cfg.TextHiddenDim], seed: 82);

        int progressCalls = 0;
        (float[] left, float[] right, int rate, int seed) = pipeline.Generate(
            text, lyric, durationSeconds: 1, seed: 7, onProgress: _ => progressCalls++);

        // 1 s → 25 latent frames, rounded up to 26 (patch 2) → ×hop samples per channel.
        int frames = 26;
        Assert.Equal(frames * cfg.SamplesPerLatent, left.Length);
        Assert.Equal(left.Length, right.Length);
        Assert.Equal(cfg.SampleRate, rate);
        Assert.Equal(7, seed);
        Assert.Equal(8, progressCalls);
        foreach (float s in left) Assert.True(float.IsFinite(s));

        // Silence src latents came from the codec's encoder (1 s of zeros → 25 latent frames).
        Assert.True(vae.EncodeCalls == 1, "pipeline must VAE-encode silence exactly once");
        Assert.Equal(25, vae.LastEncodedLatentFrames);

        // Same seed reproduces; different seed diverges.
        (float[] again, _, _, _) = pipeline.Generate(text, lyric, 1, seed: 7);
        Assert.Equal(1, vae.EncodeCalls);   // silence latent cached
        Assert.Equal(left, again);
        (float[] other, _, _, _) = pipeline.Generate(text, lyric, 1, seed: 8);
        Assert.NotEqual(left, other);

        text.Dispose(); lyric.Dispose();
    }

    /// <summary>Tiny stand-in for the Oobleck VAE: deterministic linear "decode" (channel mix + nearest upsample)
    /// and an all-constant encode, implementing both latent-codec interfaces like <c>OobleckVae</c> does.</summary>
    private sealed class FakeLatentCodec : IAudioLatentDecoder, IAudioLatentEncoder
    {
        private readonly int _latentDim;
        private readonly int _hop;

        public FakeLatentCodec(int latentDim, int hop)
        {
            _latentDim = latentDim;
            _hop = hop;
        }

        public int EncodeCalls { get; private set; }

        public int LastEncodedLatentFrames { get; private set; }

        public bool CanEncode => true;

        public int AudioChannels => 2;

        public Tensor Decode(IBackend backend, Tensor latent)
        {
            int t = (int)latent.Shape[2];
            Tensor wav = new Tensor(new TensorShape(1, 2, (long)t * _hop), DType.F32);
            float* lp = (float*)latent.DataPointer;
            float* wp = (float*)wav.DataPointer;
            for (int i = 0; i < t; i++)
            {
                float sum = 0f;
                for (int c = 0; c < _latentDim; c++) sum += lp[(long)c * t + i];
                for (int s = 0; s < _hop; s++)
                {
                    wp[(long)i * _hop + s] = 0.1f * sum;
                    wp[(long)t * _hop + i * _hop + s] = -0.1f * sum;
                }
            }
            return wav;
        }

        public Tensor EncodeMode(IBackend backend, Tensor pcm)
        {
            EncodeCalls++;
            int tLat = (int)(pcm.Shape[2] / _hop);
            LastEncodedLatentFrames = tLat;
            Tensor latent = new Tensor(new TensorShape(1, _latentDim, tLat), DType.F32);
            float* lp = (float*)latent.DataPointer;
            for (long i = 0; i < latent.Shape.ElementCount; i++) lp[i] = 0.25f;
            return latent;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            yield break;
        }
    }

    private static void AssertFinite(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]));
    }

    private static float MaxDiff(Tensor a, Tensor b)
    {
        float* ap = (float*)a.DataPointer;
        float* bp = (float*)b.DataPointer;
        float max = 0;
        for (long i = 0; i < a.Shape.ElementCount; i++) max = MathF.Max(max, MathF.Abs(ap[i] - bp[i]));
        return max;
    }

    private static Tensor Rand(int[] dims, int seed)
    {
        long[] d = Array.ConvertAll(dims, x => (long)x);
        Tensor t = new Tensor(new TensorShape(d), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }
}
