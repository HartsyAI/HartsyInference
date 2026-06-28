using System.Diagnostics;
using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Dia;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Audio.Sampling;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using DacModel = HartsyInference.Audio.Models.Codecs.Dac.Dac;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Dia text-to-dialogue pipeline: encode byte text (conditional + unconditional for CFG), then
/// run the cross-attending decoder one frame at a time over the 9-channel delayed code grid, CFG-combine the
/// per-channel logits, sample, and once channel 0 emits EOS flush the remaining delay before reverting the
/// delay and DAC-decoding to 44.1 kHz audio. Reuses <see cref="MusicGenDelay"/>, <see cref="NucleusSampler"/>,
/// and the built DAC. CFG uses two decoder instances sharing weights (separate cross-KV + self-cache).</summary>
public sealed unsafe class DiaPipeline : IDisposable
{
    private readonly DiaConfig _cfg;
    private readonly DiaEncoder _encoder;
    private readonly DiaDecoder _decCond;
    private readonly DiaDecoder _decUncond;
    private readonly DacModel _dac;
    private IDisposable[] _retain = [];   // weight mmaps held for the pipeline's lifetime
    private int _disposed;

    public DiaPipeline(DiaConfig cfg)
    {
        _cfg = cfg;
        _encoder = new DiaEncoder(cfg);
        _decCond = new DiaDecoder(cfg);
        _decUncond = new DiaDecoder(cfg);
        _dac = new DacModel(cfg.Codec);
    }

    public int SampleRate => _cfg.Codec.SampleRate;

    /// <summary>Loads the encoder + decoder (both CFG branches share the same weight tensors) and DAC. The Dia
    /// checkpoint is the nari-native DenseGeneral layout — it is run through <see cref="DiaWeights.Adapt"/> here
    /// (transposes the projections to <c>[out,in]</c>, renames the fused MLP / logits head) and the real key
    /// prefixes are <c>encoder.</c> / <c>decoder.</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> model, IReadOnlyDictionary<string, Tensor> dac)
    {
        Dictionary<string, Tensor> m = DiaWeights.Adapt(model);
        _encoder.LoadWeights(m, "encoder");
        _decCond.LoadWeights(m, "decoder", "decoder.logits_dense.weight");
        _decUncond.LoadWeights(m, "decoder", "decoder.logits_dense.weight");
        _dac.LoadWeights(dac);
    }

    /// <summary>Builds a pipeline from local files: the Dia <c>model.safetensors</c> and the descript DAC-44kHz
    /// checkpoint (<c>.pth</c>/<c>.safetensors</c>).</summary>
    public static DiaPipeline LoadFromFiles(string diaSafetensors, string dacPath, DiaConfig? config = null)
    {
        DiaConfig cfg = config ?? DiaConfig.Dia1_6B;
        DiaPipeline p = new(cfg);
        List<IDisposable> retain = new();
        IReadOnlyDictionary<string, Tensor> diaW = LoadAny(diaSafetensors, retain);
        IReadOnlyDictionary<string, Tensor> dac = LoadAny(dacPath, retain);
        p.LoadWeights(diaW, dac);
        // The pass-through weights (embeddings / norms) borrow the loaders' mmaps — keep them alive.
        p._retain = retain.ToArray();
        return p;
    }

    private static Dictionary<string, Tensor> LoadAny(string path, List<IDisposable> retain)
    {
        if (path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            HartsyInference.ModelHandler.SafeTensors.SafeTensorsLoader l = new();
            l.Load(path);
            retain.Add(l);
            return l.GetAllTensors();
        }
        HartsyInference.ModelHandler.PyTorch.PytorchPickleLoader pk = new();
        pk.Load(path, recursiveFlatten: true);
        retain.Add(pk);
        return pk.GetAllTensors();
    }

    /// <summary>Generates 44.1 kHz mono PCM from byte-level text token ids (UTF-8 bytes, speaker tags inline).</summary>
    public float[] Generate(IBackend backend, ReadOnlySpan<int> textBytes, int maxTokens = 1720, int seed = 0,
        Action<GenerationProgress>? progress = null)
    {
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();
        int ch = _cfg.Channels;
        int[] delay = [.. _cfg.DelayPattern];
        int maxDelay = _cfg.MaxDelay;

        // Encode conditional + unconditional (all-pad) text and cache cross-KV per branch.
        int[] cond = textBytes.ToArray();
        int[] uncond = new int[Math.Max(1, cond.Length)];
        Array.Fill(uncond, _cfg.TextPad);
        Tensor encCond = _encoder.Forward(backend, cond);
        Tensor encUncond = _encoder.Forward(backend, uncond);
        _decCond.PrecomputeCrossKv(backend, encCond, cond.Length);
        _decUncond.PrecomputeCrossKv(backend, encUncond, uncond.Length);
        encCond.Dispose(); encUncond.Dispose();

        int cap = Math.Min(_cfg.MaxAudio, maxTokens + maxDelay + 2);
        using StreamingKvCache cacheC = new(_decCond.NumLayers, 1, _decCond.KvHeads, cap, _decCond.HeadDim);
        using StreamingKvCache cacheU = new(_decUncond.NumLayers, 1, _decUncond.KvHeads, cap, _decUncond.HeadDim);

        uint rng = DeterministicRng.Seed(seed);
        // Delayed grid; row 0 is the all-BOS prefill.
        int[,] grid = new int[cap, ch];
        for (int c = 0; c < ch; c++) grid[0, c] = _cfg.AudioBos;

        int eosStep = -1, lastStep = cap - 1;
        Span<int> frame = stackalloc int[ch];
        for (int s = 0; s < cap - 1; s++)
        {
            for (int c = 0; c < ch; c++) frame[c] = grid[s, c];
            float[][] logits = StepCfg(backend, frame, s, cacheC, cacheU);

            int target = s + 1;
            for (int c = 0; c < ch; c++)
            {
                if (target < delay[c]) { grid[target, c] = _cfg.AudioBos; continue; }
                if (eosStep >= 0) { grid[target, c] = target >= delay[c] + eosStep ? _cfg.AudioPad : SampleChannel(logits[c], c, ref rng); continue; }
                int tok = SampleChannel(logits[c], c, ref rng);
                grid[target, c] = tok;
                if (c == 0 && tok == _cfg.AudioEos && eosStep < 0) eosStep = target;
            }
            if (eosStep >= 0 && s >= eosStep + maxDelay) { lastStep = target; break; }
            if (progress != null && (s & 63) == 0) progress(new(s, cap, sw.Elapsed.TotalMilliseconds));
        }

        // Revert delay to real codes, strip the BOS prefill row, keep only valid DAC codes.
        int tReal = Math.Max(0, lastStep - maxDelay);
        if (tReal <= 0) { Logs.Warning("Dia: no audio frames generated."); return []; }
        int[,] delayedReal = new int[lastStep, ch];
        for (int s = 0; s < lastStep; s++)
            for (int c = 0; c < ch; c++) delayedReal[s, c] = grid[s + 1, c];   // drop BOS prefill row
        int[,] real = MusicGenDelay.Revert(delayedReal, delay, tReal);

        Tensor codes = new(new TensorShape(ch, 1, tReal), DType.I32);
        int* cp = (int*)codes.DataPointer;
        for (int c = 0; c < ch; c++)
            for (int j = 0; j < tReal; j++)
                cp[(long)c * tReal + j] = Math.Clamp(real[j, c], 0, 1023);

        Tensor audioT = _dac.Decode(backend, codes, batch: 1, tFrames: tReal);
        codes.Dispose();
        int n = (int)audioT.Shape[audioT.Shape.Rank - 1];
        float[] audio = new float[n];
        Buffer.MemoryCopy((void*)audioT.DataPointer,
            System.Runtime.CompilerServices.Unsafe.AsPointer(ref audio[0]), n * 4, n * 4);
        audioT.Dispose();
        sw.Stop();
        Logs.Info($"Dia: {tReal} frames → {audio.Length} samples ({audio.Length / (double)SampleRate:F1}s) in {sw.ElapsedMilliseconds}ms.");
        return audio;
    }

    /// <summary>Steps both CFG branches and returns per-channel CFG-combined logits.</summary>
    private float[][] StepCfg(IBackend backend, ReadOnlySpan<int> frame, int posStart, StreamingKvCache cacheC, StreamingKvCache cacheU)
    {
        Tensor lc = _decCond.StepLogits(backend, frame, posStart, cacheC);
        Tensor lu = _decUncond.StepLogits(backend, frame, posStart, cacheU);
        float* pc = (float*)lc.DataPointer;
        float* pu = (float*)lu.DataPointer;
        int v = _cfg.AudioVocab;
        float g = _cfg.CfgScale;
        float[][] outL = new float[_cfg.Channels][];
        for (int c = 0; c < _cfg.Channels; c++)
        {
            float[] arr = new float[v];
            long baseOff = (long)c * v;
            for (int i = 0; i < v; i++)
            {
                float cond = pc[baseOff + i], uncond = pu[baseOff + i];
                arr[i] = cond + g * (cond - uncond);
            }
            outL[c] = arr;
        }
        lc.Dispose(); lu.Dispose();
        return outL;
    }

    /// <summary>Samples one channel's token; masks PAD/BOS always and EOS for channels &gt; 0.</summary>
    private int SampleChannel(float[] logits, int channel, ref uint rng)
    {
        logits[_cfg.AudioPad] = float.NegativeInfinity;
        logits[_cfg.AudioBos] = float.NegativeInfinity;
        if (channel != 0) logits[_cfg.AudioEos] = float.NegativeInfinity;
        return NucleusSampler.Draw(logits, _cfg.AudioVocab, _cfg.Temperature, _cfg.TopK, _cfg.TopP, ref rng);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _encoder.EnumerateWeights()) yield return t;
        foreach (Tensor t in _decCond.EnumerateWeights()) yield return t;
        foreach (Tensor t in _dac.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _encoder.Dispose();
        _decCond.Dispose();
        _decUncond.Dispose();
        foreach (IDisposable d in _retain) d.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(DiaPipeline));
    }
}
