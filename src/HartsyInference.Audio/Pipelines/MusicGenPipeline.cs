using System.Diagnostics;
using HartsyInference.Audio.Models.Codecs.EnCodec;
using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Audio.Sampling;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>MusicGen / AudioGen text-to-audio pipeline. Takes <b>precomputed T5-base cross-attn states</b>
/// (the Audio package carries no T5 dependency), AR-generates the delayed EnCodec token grid with
/// classifier-free guidance, reverts the <see cref="MusicGenDelay">delay pattern</see>, and decodes via the
/// built <see cref="EnCodec"/>. Reuses <see cref="NucleusSampler"/> (top-k/top-p draw) and the EnCodec
/// decoder. The special token is masked out of every active codebook's sampling so only real codes are drawn.</summary>
public sealed unsafe class MusicGenPipeline : IDisposable
{
    private readonly MusicGenConfig _cfg;
    private readonly MusicGenDecoder _decoder;
    private readonly EnCodec _codec;
    private int _disposed;

    public MusicGenPipeline(MusicGenConfig cfg, MusicGenDecoder decoder, EnCodec codec)
    {
        _cfg = cfg;
        _decoder = decoder;
        _codec = codec;
    }

    /// <summary>Generates audio from precomputed T5 states <c>[1, T_text, textDim]</c>. <paramref name="seconds"/>
    /// sets the real frame count (<c>frameRate × seconds</c>). Returns mono PCM at the codec sample rate.
    /// <para>The AudioCraft generation params (<c>set_generation_params</c>) are exposed per-call, defaulting to the
    /// config: <paramref name="guidance"/> = <c>cfg_coef</c> (3.0), <paramref name="temperature"/> (1.0),
    /// <paramref name="topK"/> (250), <paramref name="topP"/> (0.0 = off), <paramref name="useSampling"/> (true;
    /// false = greedy argmax).</para></summary>
    public float[] Synthesize(IBackend backend, Tensor t5States, float seconds = 8f, int seed = 0,
        float? guidance = null, float? temperature = null, int? topK = null, float? topP = null, bool useSampling = true)
    {
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();
        int k = _cfg.NumCodebooks;
        float temp = temperature ?? _cfg.Temperature;
        int tk = topK ?? _cfg.TopK;
        float tp = topP ?? _cfg.TopP;
        int tReal = Math.Max(1, (int)MathF.Round(_cfg.CodecFrameRate * seconds));
        int maxDelay = MusicGenDelay.Max(_cfg.DelayPattern);
        int tTotal = tReal + maxDelay;

        // Project T5 once for the conditional stream; the unconditional stream cross-attends to zeros.
        Tensor condCross = _decoder.ProjectText(backend, t5States);
        Tensor nullCross = new(new TensorShape(1, 1, _cfg.Hidden), DType.F32);   // zeroed → CFG null branch

        uint rng = DeterministicRng.Seed(seed);
        // Delayed grid filled autoregressively; each codebook's lead-in and tail hold the special token.
        int[,] delayed = new int[tTotal, k];

        float g = guidance ?? _cfg.GuidanceScale;
        // KV caches with once-projected cross K/V; the CFG null branch decodes against its own cache.
        MusicGenKvCache condCache = _decoder.CreateCache(backend, condCross, tTotal);
        MusicGenKvCache? uncondCache = g != 1f ? _decoder.CreateCache(backend, nullCross, tTotal) : null;
        condCross.Dispose();
        nullCross.Dispose();

        // Standard causal-LM AR loop (HF `_sample`): feed the previous delay-masked frame — starting with the
        // all-special BOS/decoder_start frame — and read the next-frame logits it produces to sample row `step`.
        // The prediction of row `step` conditions on rows [0..step-1] only; the current row is never pre-fed.
        int[] prev = new int[k];
        for (int c = 0; c < k; c++) prev[c] = _cfg.SpecialToken;   // BOS frame (decoder_start_token)
        for (int step = 0; step < tTotal; step++)
        {
            float[][] condLogits = _decoder.ForwardStep(backend, prev, condCache);
            float[][] uncondLogits = uncondCache is not null
                ? _decoder.ForwardStep(backend, prev, uncondCache)
                : condLogits;

            for (int c = 0; c < k; c++)
            {
                int j = step - _cfg.DelayPattern[c];   // this codebook's real-frame index at this step
                if (j < 0 || j >= tReal)
                {
                    delayed[step, c] = _cfg.SpecialToken;   // lead-in (j<0) or tail (j>=tReal) → pad, per delay mask
                    continue;
                }
                float[] logits = condLogits[c];
                if (g != 1f)
                {
                    float[] u = uncondLogits[c];
                    for (int v = 0; v < logits.Length; v++) logits[v] = u[v] + g * (logits[v] - u[v]);
                }
                delayed[step, c] = useSampling
                    ? NucleusSampler.Draw(logits, _cfg.CodebookSize, temp, tk, tp, ref rng, maskToken: _cfg.SpecialToken)
                    : Argmax(logits, _cfg.CodebookSize, _cfg.SpecialToken);
            }

            // The delay-masked frame just produced becomes the next step's input.
            for (int c = 0; c < k; c++) prev[c] = delayed[step, c];
        }

        int[,] real = MusicGenDelay.Revert(delayed, _cfg.DelayPattern, tReal);

        // Pack into EnCodec's [nQ, batch=1, T] Int32 grid and decode.
        Tensor codes = new(new TensorShape(k, 1, tReal), DType.I32);
        int* cp = (int*)codes.DataPointer;
        for (int c = 0; c < k; c++)
            for (int j = 0; j < tReal; j++)
                cp[(long)c * tReal + j] = Math.Clamp(real[j, c], 0, _cfg.CodebookSize - 1);

        Tensor audioT = _codec.Decode(backend, codes, batch: 1, tFrames: tReal);
        codes.Dispose();

        int n = (int)audioT.Shape[audioT.Shape.Rank - 1];
        float[] audio = new float[n];
        Buffer.MemoryCopy((void*)audioT.DataPointer, System.Runtime.CompilerServices.Unsafe.AsPointer(ref audio[0]), n * 4, n * 4);
        audioT.Dispose();
        sw.Stop();
        Logs.Info($"MusicGen: {tReal} frames → {audio.Length} samples ({audio.Length / (double)_cfg.CodecSampleRate:F1}s) in {sw.ElapsedMilliseconds}ms.");
        return audio;
    }

    /// <summary>Greedy (use_sampling=false): argmax over the valid codebook range, skipping the special token.</summary>
    private static int Argmax(float[] logits, int vocab, int maskToken)
    {
        int best = 0; float bv = float.NegativeInfinity;
        for (int v = 0; v < vocab; v++)
        {
            if (v == maskToken) continue;
            if (logits[v] > bv) { bv = logits[v]; best = v; }
        }
        return best;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _decoder.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(MusicGenPipeline));
    }
}
