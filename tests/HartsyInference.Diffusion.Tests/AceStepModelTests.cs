using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tiny-config CPU tests for the ACE-Step v1 stack: guidance math (CFG/APG/CFG-Zero★), the ping-pong
/// sampler, the Music-DCAE decoder (8× both axes), the ADaMoS vocoder upsample chain, the DiT forward
/// (LiteLA + tri-source context sensitivity), and the end-to-end pipeline. Numerics validation-pending.</summary>
public unsafe class AceStepModelTests
{
    [Fact]
    public void Guidance_CfgApgAndZeroStar_MatchTheirFormulas()
    {
        // CFG: uncond + g·(cond − uncond), exact.
        Tensor cond = Fill([1f, 2f, 3f, 4f]);
        Tensor uncond = Fill([0.5f, 1f, 1.5f, 2f]);
        AceStepGuidance.Cfg(cond, uncond, 2f);
        float* cp = (float*)cond.DataPointer;
        Assert.Equal(1.5f, cp[0], 5);
        Assert.Equal(6f, cp[3], 5);
        cond.Dispose(); uncond.Dispose();

        // APG with η=0: the applied update is orthogonal to cond ⇒ ⟨result − cond, cond⟩ = 0.
        Tensor cond2 = Fill([1f, 0f, 1f, 0f]);
        Tensor original = Fill([1f, 0f, 1f, 0f]);
        Tensor uncond2 = Fill([0.2f, -0.4f, 0.1f, 0.3f]);
        AceStepGuidance.Apg(cond2, uncond2, 3f, momentum: null, normThreshold: 0f);
        float* c2 = (float*)cond2.DataPointer;
        float* o2 = (float*)original.DataPointer;
        double dot = 0;
        for (int i = 0; i < 4; i++) dot += (c2[i] - o2[i]) * o2[i];
        Assert.Equal(0.0, dot, 4);
        cond2.Dispose(); uncond2.Dispose(); original.Dispose();

        // CFG-Zero★ zero-init clears the prediction.
        Tensor cond3 = Fill([5f, -5f, 5f, -5f]);
        Tensor uncond3 = Fill([1f, 1f, 1f, 1f]);
        AceStepGuidance.CfgZeroStar(cond3, uncond3, 7f, zeroInit: true);
        float* c3 = (float*)cond3.DataPointer;
        for (int i = 0; i < 4; i++) Assert.Equal(0f, c3[i]);
        cond3.Dispose(); uncond3.Dispose();
    }

    [Fact]
    public void PingPongScheduler_TerminalStepLandsOnCleanEstimate()
    {
        FlowMatchPingPongScheduler sched = new(shift: 3f);
        sched.SetTimesteps(4);
        Assert.Equal(5, sched.Sigmas.Length);
        Assert.Equal(1f, sched.Sigmas[0], 5);       // σ(t=1) = 1 under the shift map
        Assert.Equal(0f, sched.Sigmas[^1], 5);

        // Final step: x ← x̂₀ exactly (σ_next = 0), regardless of the noise.
        const float x0 = 0.75f, eps = -1.25f;
        float sigma = sched.Sigmas[3];
        Tensor sample = Fill([(1 - sigma) * x0 + sigma * eps]);
        Tensor velocity = Fill([eps - x0]);
        Tensor noise = Fill([9.9f]);
        sched.Step(sample, velocity, noise, 3);
        Assert.Equal(x0, *(float*)sample.DataPointer, 4);
        sample.Dispose(); velocity.Dispose(); noise.Dispose();
    }

    [Fact]
    public void MusicDcae_Decode_Upsamples8xBothAxes()
    {
        CpuBackend backend = new();
        MusicDcaeDecoder dcae = new(latentChannels: 8, outChannels: 2, blockOutChannels: [4, 8, 16, 32], layersPerBlock: [1, 1, 1, 1]);
        dcae.LoadWeights(AceStepSyntheticWeights.BuildDcaeDecoder());

        Tensor latent = Rand([1, 8, 4, 6], seed: 31);
        Tensor mel = dcae.Decode(backend, latent);
        Assert.Equal(2, (int)mel.Shape[1]);
        Assert.Equal(32, (int)mel.Shape[2]);    // 4 · 8
        Assert.Equal(48, (int)mel.Shape[3]);    // 6 · 8
        float* p = (float*)mel.DataPointer;
        for (long i = 0; i < mel.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]));
        latent.Dispose(); mel.Dispose();
    }

    [Fact]
    public void AdaMosVocoder_Decode_UpsamplesByRateProduct()
    {
        CpuBackend backend = new();
        AdaMosHiFiGanV1 vocoder = new(depths: [1, 1, 1, 1], dims: [4, 6, 8, 8],
            upsampleRates: [2, 2], upsampleKernels: [4, 4], resblockKernels: [3]);
        vocoder.LoadWeights(AceStepSyntheticWeights.BuildVocoder(melBins: 32));
        Assert.Equal(4, vocoder.TotalUpsample);

        Tensor mel = Rand([1, 32, 10], seed: 32);
        Tensor wav = vocoder.Decode(backend, mel);
        Assert.Equal(40, (int)wav.Shape[2]);    // 10 · 2 · 2
        float* p = (float*)wav.DataPointer;
        for (long i = 0; i < wav.Shape.ElementCount; i++)
        {
            Assert.True(float.IsFinite(p[i]));
            Assert.InRange(p[i], -1f, 1f);       // tanh output head
        }
        mel.Dispose(); wav.Dispose();
    }

    [Fact]
    public void Dit_Forward_RespondsToTextLyricsAndSpeaker()
    {
        CpuBackend backend = new();
        AceStepConfig cfg = AceStepSyntheticWeights.TinyConfig;
        using AceStepDit dit = new(cfg);
        dit.LoadWeights(AceStepSyntheticWeights.BuildDit(cfg));

        Tensor text = Rand([3, cfg.TextDim], seed: 41);
        int[] lyrics = [40, 5, 7, 2];
        Tensor ctx = dit.BuildContext(backend, text, lyrics, speakerVec: null);
        Assert.Equal(1 + 3 + 4, (int)ctx.Shape[0]);

        Tensor latent = Rand([1, cfg.InChannels, cfg.LatentHeight, 12], seed: 42);
        Tensor v1 = dit.Forward(backend, latent, ctx, timestep: 800f);
        Assert.Equal(latent.Shape, v1.Shape);
        float* p = (float*)v1.DataPointer;
        for (long i = 0; i < v1.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]));

        // Different lyrics must change the velocity (Conformer + cross-attn live).
        Tensor ctx2 = dit.BuildContext(backend, text, [40, 9, 11, 13, 2], speakerVec: null);
        Tensor v2 = dit.Forward(backend, latent, ctx2, 800f);
        Assert.True(MaxDiff(v1, v2) > 1e-7f, "lyric change must alter the prediction");

        // Speaker vector must change the context.
        float[] spk = new float[cfg.SpeakerDim];
        spk[0] = 1f;
        Tensor ctx3 = dit.BuildContext(backend, text, lyrics, spk);
        Tensor v3 = dit.Forward(backend, latent, ctx3, 800f);
        Assert.True(MaxDiff(v1, v3) > 1e-7f, "speaker vector must alter the prediction");
    }

    [Fact]
    public void Pipeline_Generate_ProducesStereoAudio()
    {
        CpuBackend backend = new();
        AceStepConfig cfg = AceStepSyntheticWeights.TinyConfig;
        AceStepDit dit = new(cfg);
        dit.LoadWeights(AceStepSyntheticWeights.BuildDit(cfg));
        MusicDcaeDecoder dcae = new(latentChannels: 8, outChannels: 2, blockOutChannels: [4, 8, 16, 32], layersPerBlock: [1, 1, 1, 1]);
        dcae.LoadWeights(AceStepSyntheticWeights.BuildDcaeDecoder());
        AdaMosHiFiGanV1 vocoder = new(depths: [1, 1, 1, 1], dims: [4, 6, 8, 8],
            upsampleRates: [2, 2], upsampleKernels: [4, 4], resblockKernels: [3]);
        vocoder.LoadWeights(AceStepSyntheticWeights.BuildVocoder(melBins: 32));
        AceStepPipeline pipeline = new(backend, dit, dcae, vocoder, cfg);

        Tensor text = Rand([3, cfg.TextDim], seed: 51);
        (float[] left, float[] right, int rate, int seed) = pipeline.Generate(
            text, [40, 4, 2], durationSeconds: 1, steps: 2, guidance: 3f,
            guidanceMode: AceStepPipeline.GuidanceMode.Cfg, seed: 7);

        Assert.Equal(44100, rate);
        Assert.Equal(7, seed);
        Assert.Equal(left.Length, right.Length);
        // 1 s → ~10 latent frames (≥8 floor) → mel frames ·8 → samples ·4 (tiny vocoder).
        int fLat = Math.Max(8, cfg.LatentFrames(1));
        Assert.Equal(fLat * 8 * 4, left.Length);
        foreach (float s in left) Assert.True(float.IsFinite(s));

        // PingPong + APG path also runs clean.
        (float[] left2, _, _, _) = pipeline.Generate(text, [], durationSeconds: 1, steps: 2,
            guidanceMode: AceStepPipeline.GuidanceMode.Apg, sampler: AceStepPipeline.SamplerMode.PingPong, seed: 8);
        Assert.Equal(left.Length, left2.Length);
    }

    private static float MaxDiff(Tensor a, Tensor b)
    {
        float* ap = (float*)a.DataPointer;
        float* bp = (float*)b.DataPointer;
        float max = 0;
        for (long i = 0; i < a.Shape.ElementCount; i++) max = MathF.Max(max, MathF.Abs(ap[i] - bp[i]));
        return max;
    }

    private static Tensor Fill(float[] values)
    {
        Tensor t = new Tensor(new TensorShape(values.Length), DType.F32);
        values.CopyTo(new Span<float>((float*)t.DataPointer, values.Length));
        return t;
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
