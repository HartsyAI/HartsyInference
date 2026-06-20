using System.Diagnostics;
using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Audio.Models.Zonos;
using HartsyInference.Audio.Sampling;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using DacModel = HartsyInference.Audio.Models.Codecs.Dac.Dac;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Zonos-v0.1 (transformer) pipeline: prefills the backbone with a conditioning prefix (cond +
/// unconditional, for CFG), then runs a delayed-AR loop over the 9 DAC codebooks — summed-embedding input,
/// per-codebook heads, CFG-combined logits, min-p sampling — until codebook 0 emits EOS, flushes the 9-step
/// delay, reverts, and DAC-decodes to 44.1 kHz. Takes precomputed conditioning prefixes (the espeak +
/// conditioner front-end is caller-side). Reuses <see cref="ZonosBackbone"/>, <see cref="ZonosCodebooks"/>,
/// <see cref="MusicGenDelay"/>, <see cref="NucleusSampler"/>, and the built DAC.</summary>
public sealed unsafe class ZonosPipeline : IDisposable
{
    private readonly ZonosConfig _cfg;
    private readonly ZonosBackbone _bbCond;
    private readonly ZonosBackbone _bbUncond;
    private readonly ZonosCodebooks _codebooks;
    private readonly DacModel _dac;
    private int _disposed;

    public ZonosPipeline(ZonosConfig cfg)
    {
        _cfg = cfg;
        _bbCond = new ZonosBackbone(cfg);
        _bbUncond = new ZonosBackbone(cfg);
        _codebooks = new ZonosCodebooks(cfg);
        _dac = new DacModel(cfg.Codec);
    }

    public int SampleRate => _cfg.Codec.SampleRate;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> model, IReadOnlyDictionary<string, Tensor> dac)
    {
        _bbCond.LoadWeights(model);
        _bbUncond.LoadWeights(model);
        _codebooks.LoadWeights(model);
        _dac.LoadWeights(dac);
    }

    /// <summary>Generates 44.1 kHz mono PCM. <paramref name="condPrefix"/> / <paramref name="uncondPrefix"/>
    /// are the precomputed conditioning prefixes <c>[1, P, hidden]</c> (phonemes + speaker + controls).</summary>
    public float[] Generate(IBackend backend, Tensor condPrefix, Tensor uncondPrefix, int maxTokens = 0,
        int seed = 0, Action<GenerationProgress>? progress = null)
    {
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();
        int ch = _cfg.Channels;
        int[] delay = _cfg.BuildDelays();
        int maxDelay = ch;                 // delays are [1..9] → max 9
        int max = maxTokens > 0 ? maxTokens : _cfg.MaxNewTokens;
        int cap = max + maxDelay + 2;

        int pCond = (int)condPrefix.Shape[1], pUncond = (int)uncondPrefix.Shape[1];
        using StreamingKvCache cacheC = new(_cfg.NumLayers, 1, _cfg.NumKvHeads, cap + pCond, _cfg.HeadDim);
        using StreamingKvCache cacheU = new(_cfg.NumLayers, 1, _cfg.NumKvHeads, cap + pUncond, _cfg.HeadDim);

        // Prefill both branches with their conditioning prefix.
        _bbCond.Forward(backend, condPrefix, pCond, 0, cacheC).Dispose();
        _bbUncond.Forward(backend, uncondPrefix, pUncond, 0, cacheU).Dispose();

        uint rng = DeterministicRng.Seed(seed);
        int[,] grid = new int[cap, ch];
        for (int c = 0; c < ch; c++) grid[0, c] = _cfg.MaskedToken;   // delayed-grid lead-in
        int eosStep = -1, lastStep = cap - 1;
        Span<int> frame = stackalloc int[ch];

        for (int s = 0; s < cap - 1; s++)
        {
            for (int c = 0; c < ch; c++) frame[c] = grid[s, c];
            float[][] logits = StepCfg(backend, frame, pCond + s, pUncond + s, cacheC, cacheU);

            int target = s + 1;
            for (int c = 0; c < ch; c++)
            {
                if (target < delay[c]) { grid[target, c] = _cfg.MaskedToken; continue; }
                if (eosStep >= 0)
                {
                    int idx = ch - (eosStep + maxDelay - s);
                    grid[target, c] = c == idx ? _cfg.EosToken : c < idx ? _cfg.MaskedToken : Sample(logits[c], c, ref rng);
                    continue;
                }
                int tok = Sample(logits[c], c, ref rng);
                grid[target, c] = tok;
                if (c == 0 && tok == _cfg.EosToken && eosStep < 0) eosStep = s;
            }
            if (eosStep >= 0 && s >= eosStep + maxDelay) { lastStep = target; break; }
            if (progress != null && (s & 63) == 0) progress(new(s, cap, sw.Elapsed.TotalMilliseconds));
        }

        int tReal = Math.Max(0, lastStep - maxDelay);
        if (tReal <= 0) { Logs.Warning("Zonos: no audio frames generated."); return []; }
        int[,] delayedReal = new int[lastStep, ch];
        for (int s = 0; s < lastStep; s++)
            for (int c = 0; c < ch; c++) delayedReal[s, c] = grid[s + 1, c];
        int[,] real = MusicGenDelay.Revert(delayedReal, delay, tReal);

        Tensor codes = new(new TensorShape(ch, 1, tReal), DType.I32);
        int* cp = (int*)codes.DataPointer;
        for (int c = 0; c < ch; c++)
            for (int j = 0; j < tReal; j++)
            {
                int v = real[j, c];
                cp[(long)c * tReal + j] = v >= 1024 ? 0 : v;     // zero EOS/masked, keep valid codes
            }

        Tensor audioT = _dac.Decode(backend, codes, batch: 1, tFrames: tReal);
        codes.Dispose();
        int n = (int)audioT.Shape[audioT.Shape.Rank - 1];
        float[] audio = new float[n];
        Buffer.MemoryCopy((void*)audioT.DataPointer,
            System.Runtime.CompilerServices.Unsafe.AsPointer(ref audio[0]), n * 4, n * 4);
        audioT.Dispose();
        sw.Stop();
        Logs.Info($"Zonos: {tReal} frames → {audio.Length} samples ({audio.Length / (double)SampleRate:F1}s) in {sw.ElapsedMilliseconds}ms.");
        return audio;
    }

    private float[][] StepCfg(IBackend backend, ReadOnlySpan<int> frame, int posC, int posU,
        StreamingKvCache cacheC, StreamingKvCache cacheU)
    {
        Tensor embC = _codebooks.EmbedFrame(frame);
        Tensor hidC = _bbCond.Forward(backend, embC, 1, posC, cacheC); embC.Dispose();
        float[][] lc = _codebooks.Heads(backend, hidC); hidC.Dispose();

        Tensor embU = _codebooks.EmbedFrame(frame);
        Tensor hidU = _bbUncond.Forward(backend, embU, 1, posU, cacheU); embU.Dispose();
        float[][] lu = _codebooks.Heads(backend, hidU); hidU.Dispose();

        float g = _cfg.CfgScale;
        for (int c = 0; c < _cfg.Channels; c++)
            for (int i = 0; i < _cfg.OutputVocab; i++)
                lc[c][i] = lu[c][i] + (lc[c][i] - lu[c][i]) * g;     // uncond + (cond-uncond)*cfg
        return lc;
    }

    private int Sample(float[] logits, int channel, ref uint rng)
    {
        if (channel != 0) logits[_cfg.EosToken] = float.NegativeInfinity;   // only codebook 0 may EOS
        return NucleusSampler.Draw(logits, _cfg.OutputVocab, _cfg.Temperature, 0, 0f, ref rng, -1, _cfg.MinP);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _bbCond.EnumerateWeights()) yield return t;
        foreach (Tensor t in _codebooks.EnumerateWeights()) yield return t;
        foreach (Tensor t in _dac.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _bbCond.Dispose(); _bbUncond.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(ZonosPipeline));
    }
}
