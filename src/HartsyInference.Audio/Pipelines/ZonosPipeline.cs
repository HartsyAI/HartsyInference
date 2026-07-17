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
    private readonly ZonosBackbone _bb;
    private readonly ZonosCodebooks _codebooks;
    private readonly DacModel _dac;
    private int _disposed;

    public ZonosPipeline(ZonosConfig cfg)
    {
        _cfg = cfg;
        // A single backbone with two KV caches serves both CFG branches — the reference batches cond+uncond
        // through one weight set; duplicating the ~2B-param backbone would double VRAM (16 GB F32) for nothing.
        _bb = new ZonosBackbone(cfg);
        _codebooks = new ZonosCodebooks(cfg);
        _dac = new DacModel(cfg.Codec);
    }

    public int SampleRate => _cfg.Codec.SampleRate;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> model, IReadOnlyDictionary<string, Tensor> dac)
    {
        LoadBackboneWeights(model);
        _dac.LoadWeights(dac);
    }

    /// <summary>Loads only the transformer backbone and the codebook embed/head weights, skipping the DAC.
    /// Used by deterministic code-level parity tests that never decode to audio.</summary>
    public void LoadBackboneWeights(IReadOnlyDictionary<string, Tensor> model)
    {
        _bb.LoadWeights(model);
        _codebooks.LoadWeights(model);
    }

    /// <summary>Generates 44.1 kHz mono PCM. <paramref name="condPrefix"/> / <paramref name="uncondPrefix"/>
    /// are the precomputed conditioning prefixes <c>[1, P, hidden]</c> (phonemes + speaker + controls).
    /// Sampling knobs default (&lt; 0 or NaN) to the config; <paramref name="temperature"/> == 0 forces greedy
    /// argmax (used by the deterministic parity test).</summary>
    public float[] Generate(IBackend backend, Tensor condPrefix, Tensor uncondPrefix, int maxTokens = 0,
        int seed = 0, float cfgScale = float.NaN, float temperature = float.NaN, float minP = float.NaN,
        float repetitionPenalty = float.NaN, Action<GenerationProgress>? progress = null)
    {
        (int[,] real, int tReal) = GenerateCodes(backend, condPrefix, uncondPrefix, maxTokens, seed,
            cfgScale, temperature, minP, repetitionPenalty, progress);
        if (tReal <= 0) { Logs.Warning("Zonos: no audio frames generated."); return []; }

        int ch = _cfg.Channels;
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
        Logs.Info($"Zonos: {tReal} frames → {audio.Length} samples ({audio.Length / (double)SampleRate:F1}s).");
        return audio;
    }

    /// <summary>Runs the delayed-AR loop and returns the reverted DAC codes <c>[tReal, channels]</c> (before the
    /// EOS/masked → 0 clean-up and DAC decode). Exposed for deterministic code-level parity tests.</summary>
    public (int[,] codes, int tReal) GenerateCodes(IBackend backend, Tensor condPrefix, Tensor uncondPrefix,
        int maxTokens = 0, int seed = 0, float cfgScale = float.NaN, float temperature = float.NaN,
        float minP = float.NaN, float repetitionPenalty = float.NaN, Action<GenerationProgress>? progress = null)
    {
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();
        int ch = _cfg.Channels;
        int[] delay = _cfg.BuildDelays();
        int maxDelay = ch;                 // delays are [1..9] → max 9
        int max = maxTokens > 0 ? maxTokens : _cfg.MaxNewTokens;
        int cap = max + maxDelay + 2;

        float cfg = float.IsNaN(cfgScale) ? _cfg.CfgScale : cfgScale;
        float temp = float.IsNaN(temperature) ? _cfg.Temperature : temperature;
        float mp = float.IsNaN(minP) ? _cfg.MinP : minP;
        float rp = float.IsNaN(repetitionPenalty) ? _cfg.RepetitionPenalty : repetitionPenalty;
        int rw = _cfg.RepetitionWindow;

        int pCond = (int)condPrefix.Shape[1], pUncond = (int)uncondPrefix.Shape[1];
        using StreamingKvCache cacheC = new(_cfg.NumLayers, 1, _cfg.NumKvHeads, cap + pCond, _cfg.HeadDim);
        using StreamingKvCache cacheU = new(_cfg.NumLayers, 1, _cfg.NumKvHeads, cap + pUncond, _cfg.HeadDim);

        // Full-F32 GEMM: TF32 (the GPU default) accumulates ~1e-3 error per forward which, over the AR loop,
        // flips a sampled argmax and degenerates generation (verified: TF32 → babble/early-EOS, F32 → reference
        // parity). No-op on CPU/Vulkan. Restored in the finally so other models keep TF32.
        bool prevHighPrec = backend.HighPrecisionGemm;
        backend.HighPrecisionGemm = true;

        // GPU-resident the backbone + codebook weights so per-step decode ops don't pay a cache-miss H2D each.
        backend.PreloadWeights(_bb.EnumerateWeights());
        backend.PreloadWeights(_codebooks.EnumerateWeights());

        // Prefill both branches with their conditioning prefix.
        _bb.Forward(backend, condPrefix, pCond, 0, cacheC).Dispose();
        _bb.Forward(backend, uncondPrefix, pUncond, 0, cacheU).Dispose();

        uint rng = DeterministicRng.Seed(seed);
        int[,] grid = new int[cap, ch];
        for (int c = 0; c < ch; c++) grid[0, c] = _cfg.MaskedToken;   // delayed-grid lead-in
        int eosStep = -1, lastStep = cap - 1;
        Span<int> frame = stackalloc int[ch];

        string? logitDump = Environment.GetEnvironmentVariable("ZONOS_LOGIT_DUMP");
        int logitDumpSteps = logitDump != null ? 33 : 0;
        List<float>? logitBuf = logitDump != null ? new List<float>() : null;

        for (int s = 0; s < cap - 1; s++)
        {
            for (int c = 0; c < ch; c++) frame[c] = grid[s, c];
            float[][] logits = StepCfg(backend, frame, pCond + s, pUncond + s, cacheC, cacheU, cfg);
            if (logitBuf != null && s < logitDumpSteps)
                for (int c = 0; c < ch; c++) logitBuf.AddRange(logits[c]);
            if (logitBuf != null && s == logitDumpSteps - 1)
            {
                byte[] bytes = new byte[logitBuf.Count * 4];
                Buffer.BlockCopy(logitBuf.ToArray(), 0, bytes, 0, bytes.Length);
                File.WriteAllBytes(logitDump!, bytes);
                Logs.Info($"[Zonos] dumped {logitDumpSteps} steps of logits ({ch}x{_cfg.OutputVocab}) to {logitDump}");
            }

            int target = s + 1;
            for (int c = 0; c < ch; c++)
            {
                if (target < delay[c]) { grid[target, c] = _cfg.MaskedToken; continue; }
                if (eosStep >= 0)
                {
                    int idx = ch - (eosStep + maxDelay - s);
                    grid[target, c] = c == idx ? _cfg.EosToken
                        : c < idx ? _cfg.MaskedToken
                        : Sample(logits[c], c, grid, s, temp, mp, rp, rw, ref rng);
                    continue;
                }
                int tok = Sample(logits[c], c, grid, s, temp, mp, rp, rw, ref rng);
                grid[target, c] = tok;
                if (c == 0 && tok == _cfg.EosToken && eosStep < 0) eosStep = s;
            }
            if (eosStep >= 0 && s >= eosStep + maxDelay) { lastStep = target; break; }
            if (progress != null && (s & 63) == 0) progress(new(s, cap, sw.Elapsed.TotalMilliseconds));
        }

        backend.HighPrecisionGemm = prevHighPrec;
        int tReal = Math.Max(0, lastStep - maxDelay);
        sw.Stop();
        if (tReal <= 0) return (new int[0, ch], 0);
        // Revert reads delayed[j + delay[c]] with delay[c] = c+1, so it must see the full delayed grid starting at
        // the all-masked lead-in frame grid[0] (delayed position 0). Feeding grid[s+1] would shift output by 1.
        int[,] delayedReal = new int[lastStep, ch];
        for (int s = 0; s < lastStep; s++)
            for (int c = 0; c < ch; c++) delayedReal[s, c] = grid[s, c];
        int[,] real = MusicGenDelay.Revert(delayedReal, delay, tReal);
        Logs.Info($"Zonos: {tReal} frames in {sw.ElapsedMilliseconds}ms (cfg={cfg}, temp={temp}, minP={mp}, rep={rp}).");
        return (real, tReal);
    }

    /// <summary>Prefills both branches with their conditioning prefix, then returns the CFG-combined logits
    /// <c>[channels][OutputVocab]</c> for the first (all-masked) audio frame. Mirrors the reference
    /// <c>_prefill</c> first-step logits; used for deterministic backbone/codebook logit parity.</summary>
    public float[][] PrefillFirstLogits(IBackend backend, Tensor condPrefix, Tensor uncondPrefix, float cfgScale = float.NaN)
    {
        ThrowIfDisposed();
        int ch = _cfg.Channels;
        float cfg = float.IsNaN(cfgScale) ? _cfg.CfgScale : cfgScale;
        int pCond = (int)condPrefix.Shape[1], pUncond = (int)uncondPrefix.Shape[1];
        using StreamingKvCache cacheC = new(_cfg.NumLayers, 1, _cfg.NumKvHeads, pCond + 1, _cfg.HeadDim);
        using StreamingKvCache cacheU = new(_cfg.NumLayers, 1, _cfg.NumKvHeads, pUncond + 1, _cfg.HeadDim);
        _bb.Forward(backend, condPrefix, pCond, 0, cacheC).Dispose();
        _bb.Forward(backend, uncondPrefix, pUncond, 0, cacheU).Dispose();
        Span<int> frame = stackalloc int[ch];
        for (int c = 0; c < ch; c++) frame[c] = _cfg.MaskedToken;
        return StepCfg(backend, frame, pCond, pUncond, cacheC, cacheU, cfg);
    }

    private float[][] StepCfg(IBackend backend, ReadOnlySpan<int> frame, int posC, int posU,
        StreamingKvCache cacheC, StreamingKvCache cacheU, float cfg)
    {
        Tensor embC = _codebooks.EmbedFrame(frame);
        Tensor hidC = _bb.Forward(backend, embC, 1, posC, cacheC); embC.Dispose();
        float[][] lc = _codebooks.Heads(backend, hidC); hidC.Dispose();

        Tensor embU = _codebooks.EmbedFrame(frame);
        Tensor hidU = _bb.Forward(backend, embU, 1, posU, cacheU); embU.Dispose();
        float[][] lu = _codebooks.Heads(backend, hidU); hidU.Dispose();

        for (int c = 0; c < _cfg.Channels; c++)
            for (int i = 0; i < _cfg.OutputVocab; i++)
                lc[c][i] = lu[c][i] + (lc[c][i] - lu[c][i]) * cfg;   // uncond + (cond-uncond)*cfg
        return lc;
    }

    /// <summary>Applies the (per-codebook, sliding-window) repetition penalty, forbids EOS on codebooks &gt; 0,
    /// then draws — greedy argmax when <paramref name="temp"/> ≤ 0, else nucleus/min-p.</summary>
    private int Sample(float[] logits, int channel, int[,] grid, int s, float temp, float minP, float rep, int window,
        ref uint rng)
    {
        if (rep != 1f && window > 0) ApplyRepetitionPenalty(logits, grid, channel, s, rep, window);
        if (channel != 0) logits[_cfg.EosToken] = float.NegativeInfinity;   // only codebook 0 may EOS
        if (temp <= 0f)
            return NucleusSampler.Draw(logits, _cfg.OutputVocab, 1f, 1, 0f, ref rng);   // top-k=1 → argmax
        return NucleusSampler.Draw(logits, _cfg.OutputVocab, temp, 0, 0f, ref rng, -1, minP);
    }

    /// <summary>Divides (logit &gt; 0) / multiplies (logit ≤ 0) by <paramref name="rep"/>^count for each token id
    /// that codebook <paramref name="channel"/> emitted in the last <paramref name="window"/> delayed steps —
    /// matching <c>modify_logit_for_repetition_penalty</c> (generated positions strictly before the new frame).</summary>
    private void ApplyRepetitionPenalty(float[] logits, int[,] grid, int channel, int s, float rep, int window)
    {
        for (int j = 0; j < window; j++)
        {
            int pos = s - j;                 // generated positions 0..s (before target = s+1)
            if (pos < 0) break;
            int tok = grid[pos, channel];
            // The masked token (1025) sits in the reference's padded -inf region, so its penalty is a no-op there;
            // clamping it into range would wrongly penalize EOS (1024). Skip it; penalize real codes + EOS only.
            if (tok < 0 || tok >= _cfg.OutputVocab) continue;
            float v = logits[tok];
            logits[tok] = v <= 0f ? v * rep : v / rep;
        }
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _bb.EnumerateWeights()) yield return t;
        foreach (Tensor t in _codebooks.EnumerateWeights()) yield return t;
        foreach (Tensor t in _dac.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _bb.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(ZonosPipeline));
    }
}
