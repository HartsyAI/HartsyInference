using System.Diagnostics;
using SharpInference.Audio.Models.Codecs.EnCodec;
using SharpInference.Audio.Dsp;
using SharpInference.Audio.Models.Music;
using SharpInference.Audio.Sampling;
using SharpInference.Core.Backends;
using SharpInference.Core.Logging;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Pipelines;

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
    /// sets the real frame count (<c>frameRate × seconds</c>). Returns mono PCM at the codec sample rate.</summary>
    public float[] Synthesize(IBackend backend, Tensor t5States, float seconds = 8f, int seed = 0)
    {
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();
        int k = _cfg.NumCodebooks;
        int tReal = Math.Max(1, (int)MathF.Round(_cfg.CodecFrameRate * seconds));
        int maxDelay = MusicGenDelay.Max(_cfg.DelayPattern);
        int tTotal = tReal + maxDelay;

        // Project T5 once for the conditional stream; the unconditional stream cross-attends to zeros.
        Tensor condCross = _decoder.ProjectText(backend, t5States);
        Tensor nullCross = new(new TensorShape(1, 1, _cfg.Hidden), DType.F32);   // zeroed → CFG null branch

        uint rng = DeterministicRng.Seed(seed);
        // Delayed grid we fill autoregressively, prefixed with one all-special BOS frame.
        int[,] delayed = new int[tTotal, k];
        for (int c = 0; c < k; c++) delayed[0, c] = _cfg.SpecialToken;   // step 0 lead-in is special anyway

        float g = _cfg.GuidanceScale;
        for (int step = 0; step < tTotal; step++)
        {
            // Feed frames [0..step] and predict frame `step` (BOS at index 0 shifts prediction by one).
            int len = step + 1;
            int[,] prefix = new int[len, k];
            for (int s = 0; s < len; s++)
                for (int c = 0; c < k; c++) prefix[s, c] = delayed[s, c];

            Tensor embeds = _decoder.EmbedFrames(prefix);
            float[][] condLogits = _decoder.Forward(backend, embeds, condCross);
            Tensor embeds2 = _decoder.EmbedFrames(prefix);
            float[][] uncondLogits = g != 1f ? _decoder.Forward(backend, embeds2, nullCross) : condLogits;
            embeds.Dispose(); embeds2.Dispose();

            for (int c = 0; c < k; c++)
            {
                if (!MusicGenDelay.IsActive(step, c, _cfg.DelayPattern))
                {
                    if (step < tTotal) SetIfInRange(delayed, step, c, _cfg.SpecialToken);
                    continue;
                }
                float[] logits = condLogits[c];
                if (g != 1f)
                {
                    float[] u = uncondLogits[c];
                    for (int v = 0; v < logits.Length; v++) logits[v] = u[v] + g * (logits[v] - u[v]);
                }
                int tok = NucleusSampler.Draw(logits, _cfg.CodebookSize, _cfg.Temperature, _cfg.TopK, _cfg.TopP,
                    ref rng, maskToken: _cfg.SpecialToken);
                SetIfInRange(delayed, step, c, tok);
            }
        }

        condCross.Dispose();
        nullCross.Dispose();

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

    private static void SetIfInRange(int[,] grid, int step, int c, int value)
    {
        if (step < grid.GetLength(0)) grid[step, c] = value;
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
