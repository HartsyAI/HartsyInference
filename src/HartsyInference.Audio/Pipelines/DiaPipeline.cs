using System.Diagnostics;
using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Dia;
using HartsyInference.Audio.Models.Music;
using HartsyInference.Audio.Sampling;
using HartsyInference.Audio.Streaming;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using DacModel = HartsyInference.Audio.Models.Codecs.Dac.Dac;

namespace HartsyInference.Audio.Pipelines;

/// <summary>Dia text-to-dialogue pipeline: encode byte text (conditional + unconditional for CFG), then
/// run the cross-attending decoder one frame at a time over the 9-channel delayed code grid, pick candidates
/// from the top-K of the CFG-combined logits but sample the CONDITIONAL distribution over them (upstream
/// cfg_filter_top_k), and once channel 0 emits EOS flush each channel's delayed EOS/PAD tail before reverting
/// the delay and DAC-decoding to 44.1 kHz audio. Reuses <see cref="MusicGenDelay"/>, <see cref="NucleusSampler"/>,
/// and the built DAC. CFG uses two decoder instances sharing weights (separate cross-KV + self-cache).</summary>
public sealed unsafe class DiaPipeline : IDisposable
{
    private readonly DiaConfig _cfg;
    // Per-call sampler overrides, set at the top of Generate. The pipeline instance is cached and reused,
    // so these must be assigned every run rather than baked into the config.
    private float _runCfgScale;
    private int _runTopK;
    private float _runTemperature;
    private float _runTopP;
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
        // Non-recursive: the descript DAC .pth is a flat state_dict under a {state_dict, metadata} envelope, and
        // recursive flatten KEEPS the `state_dict.` wrapper — keys came out as `state_dict.encoder.block.0.weight_g`
        // and the weight-norm fuse missed them → KeyNotFound on `.weight`.
        IReadOnlyDictionary<string, Tensor> diaW = CheckpointLoader.Load(diaSafetensors, retain, recursiveFlatten: false);
        IReadOnlyDictionary<string, Tensor> dac = CheckpointLoader.Load(dacPath, retain, recursiveFlatten: false);
        p.LoadWeights(diaW, dac);
        // The pass-through weights (embeddings / norms) borrow the loaders' mmaps — keep them alive.
        p._retain = retain.ToArray();
        return p;
    }

    /// <summary>Generates 44.1 kHz mono PCM from byte-level text token ids (UTF-8 bytes, speaker tags inline —
    /// literal <c>[S1]</c>/<c>[S2]</c> byte runs are folded to 0x01/0x02 like upstream <c>_encode_text</c>).</summary>
    public float[] Generate(IBackend backend, ReadOnlySpan<int> textBytes, int maxTokens = 1720, int seed = 0,
        Action<GenerationProgress>? progress = null)
        => Generate(backend, textBytes, maxTokens, seed, progress, null, null, null, null);

    /// <summary>As above, overriding the sampler settings. Upstream <c>Dia.generate</c> defaults are
    /// <c>cfg_scale=3.0</c>, <c>temperature=1.2</c>, <c>top_p=0.95</c>, <c>cfg_filter_top_k=45</c>; null keeps
    /// the checkpoint config's value.</summary>
    public float[] Generate(IBackend backend, ReadOnlySpan<int> textBytes, int maxTokens, int seed,
        Action<GenerationProgress>? progress, double? cfgScale, int? cfgFilterTopK, double? temperature, double? topP)
        => Generate(backend, textBytes, maxTokens, seed, progress, cfgScale, cfgFilterTopK, temperature, topP, null);

    /// <param name="onSettledFrame">Invoked once per REAL (already delay-reverted) audio frame's channel array,
    /// as soon as it becomes fully determined — i.e. every channel's delayed write for that frame position has
    /// happened. A channel's delay means frame <c>j</c> isn't settled until AR step <c>j + maxDelay</c>, so this
    /// fires with a lag behind the raw per-step loop, not per-step itself; nothing fires for frames the EOS flush
    /// ultimately truncates away (mirrors <c>tReal</c>'s own EOS-aware truncation exactly, computed incrementally
    /// instead of after the fact). Passing null (the default) changes nothing about this method's behavior.</param>
    public float[] Generate(IBackend backend, ReadOnlySpan<int> textBytes, int maxTokens, int seed,
        Action<GenerationProgress>? progress, double? cfgScale, int? cfgFilterTopK, double? temperature, double? topP,
        Action<int[]>? onSettledFrame)
    {
        _runCfgScale = cfgScale is > 0 ? (float)cfgScale.Value : _cfg.CfgScale;
        _runTopK = cfgFilterTopK is > 0 ? cfgFilterTopK.Value : _cfg.TopK;
        _runTemperature = temperature is > 0 ? (float)temperature.Value : _cfg.Temperature;
        _runTopP = topP is > 0 ? (float)topP.Value : _cfg.TopP;
        ThrowIfDisposed();
        Stopwatch sw = Stopwatch.StartNew();
        int ch = _cfg.Channels;
        int[] delay = [.. _cfg.DelayPattern];
        int maxDelay = _cfg.MaxDelay;

        // Pin the F32 weights VRAM-resident up front (no-op on CPU). Without this the CUDA path leaves weights
        // host-side and re-copies each one to the GPU per op — so a 6.4 GB model has to stay in system RAM and
        // never actually lives on the card. One-shot preload (6.4 GB fits a 12 GB 3060) makes both CFG streams
        // read resident weights; freed in the finally so a repeated call / other model reclaims the VRAM.
        backend.PreloadWeights(EnumerateWeights());
        try
        {
        // Encode conditional + unconditional (all-pad) text and cache cross-KV per branch.
        int[] cond = FoldSpeakerTags(textBytes, _cfg.MaxText);
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
        int nextSettled = 0; // next real (delay-reverted) frame index not yet reported via onSettledFrame.
        // DEBUG: dump per-step channel-0 tokens to localize the repetition loop (DIA_DEBUG_TOKENS=path).
        string? dbgPath = EngineKnobs.DiaDebugTokens.Value;
        System.Text.StringBuilder? dbg = dbgPath is null ? null : new();
        Span<int> frame = stackalloc int[ch];
        for (int s = 0; s < cap - 1; s++)
        {
            for (int c = 0; c < ch; c++) frame[c] = grid[s, c];
            (float[][] lc, float[][] lg) = StepCfg(backend, frame, s, cacheC, cacheU);

            int target = s + 1;
            for (int c = 0; c < ch; c++)
            {
                // BOS lead-in runs through t == delay[c] (upstream apply_audio_delay: BOS where t - delay <= 0).
                if (target <= delay[c]) { grid[target, c] = _cfg.AudioBos; continue; }
                if (eosStep >= 0)
                {
                    // EOS flush: EOS lands at exactly eosStep + delay[c], PAD after, sampled before (model.py:729-741).
                    int off = target - eosStep;
                    grid[target, c] = off == delay[c] ? _cfg.AudioEos
                        : off > delay[c] ? _cfg.AudioPad
                        : SampleChannel(lc[c], lg[c], c, ref rng);
                    continue;
                }
                grid[target, c] = SampleChannel(lc[c], lg[c], c, ref rng);
            }
            dbg?.Append(grid[target, 0]).Append(' ');
            if (eosStep < 0 && grid[target, 0] == _cfg.AudioEos) eosStep = target;
            // Near the cap, force the flush so every channel closes cleanly (model.py:721 is_max_len).
            if (eosStep < 0 && target >= cap - 1 - maxDelay) { eosStep = target; grid[target, 0] = _cfg.AudioEos; }

            // Real frame j is fully determined once every channel's delayed write for it has happened, i.e. once
            // `target` reaches j + maxDelay. Stop at tRealSoFar (= eosStep - 1 once eosStep is known, matching
            // the final truncation below exactly) so frames the EOS flush will discard are never reported.
            if (onSettledFrame is not null)
            {
                int tRealSoFar = eosStep >= 0 ? eosStep - 1 : int.MaxValue;
                // grid[0] is the BOS prefill row the non-streaming path drops (real[j,c] = delayed[j+delay[c],c] =
                // grid[j+delay[c]+1,c]); frame j is fully settled once target reaches j+maxDelay+1.
                while (nextSettled <= target - maxDelay - 1 && nextSettled < tRealSoFar)
                {
                    int[] realFrame = new int[ch];
                    for (int c = 0; c < ch; c++) realFrame[c] = grid[nextSettled + delay[c] + 1, c];
                    onSettledFrame(realFrame);
                    nextSettled++;
                }
            }

            if (eosStep >= 0 && target >= eosStep + maxDelay) { lastStep = target; break; }
            if (progress != null && (s & 63) == 0) progress(new(s, cap, sw.Elapsed.TotalMilliseconds));
            // Two CFG streams double per-step device pressure; trim the pool periodically on long runs.
            if ((s & 255) == 255) backend.TrimMemoryPool();
        }

        if (dbg is not null) { try { File.WriteAllText(dbgPath!, dbg.ToString()); Logs.Info($"Dia: dumped ch0 tokens → {dbgPath}"); } catch { } }
        // Codes are host-side ints now; reclaim the AR loop's device activations + pool before the DAC decode
        // (the CFG double-stream loop otherwise leaves enough resident to OOM 12GB cards at the vocoder).
        backend.FreeActivations();
        backend.TrimMemoryPool();

        // Revert delay to real codes, strip the BOS prefill row and the EOS frame (upstream length = eosStep - 1).
        int tReal = Math.Max(0, (eosStep >= 0 ? eosStep - 1 : lastStep - maxDelay));
        if (tReal <= 0) { Logs.Warning("Dia: no audio frames generated."); return []; }
        int[,] delayedReal = new int[lastStep, ch];
        for (int s = 0; s < lastStep; s++)
            for (int c = 0; c < ch; c++) delayedReal[s, c] = grid[s + 1, c];   // drop BOS prefill row
        int[,] real = MusicGenDelay.Revert(delayedReal, delay, tReal);

        Tensor codes = new(new TensorShape(ch, 1, tReal), DType.I32);
        int* cp = (int*)codes.DataPointer;
        for (int c = 0; c < ch; c++)
            for (int j = 0; j < tReal; j++)
                cp[(long)c * tReal + j] = (uint)real[j, c] > 1023u ? 0 : real[j, c];   // invalid codes → 0 (model.py:509-512)

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
        finally { backend.FreeWeights(EnumerateWeights()); }
    }

    /// <summary>Decodes ALL settled real frames accumulated so far (from <paramref name="onSettledFrame"/>) and
    /// returns only the NEW trailing samples beyond <paramref name="alreadyEmittedSamples"/> — i.e. a full
    /// re-decode of the whole utterance-so-far every call, not an incremental append. This mirrors nari-labs'
    /// OWN streaming attempt for Dia (<c>generate_streaming()</c>, upstream PR #262): their author found that
    /// feeding only the new tokens to DAC's vocoder produces audible boundary artifacts (DAC's decoder uses
    /// symmetric/non-causal padding throughout — verified against the real weights, no causal variant exists —
    /// so it genuinely needs both past AND future code-frame context, unlike Mimi/Firefly), and worked around it
    /// with exactly this full-recompute approach rather than a proper windowed fix. Their successor model
    /// (Dia2) replaced DAC with Kyutai's Mimi codec specifically to get real streaming instead of solving this.
    /// Cost grows O(n²) in the number of chunks for a given utterance — acceptable for realistic TTS utterance
    /// lengths (a few hundred frames), the same tradeoff nari-labs' own reference client accepted.</summary>
    public float[] DecodeSettledFramesTail(IBackend backend, IReadOnlyList<int[]> allSettledFrames, int alreadyEmittedSamples)
    {
        int ch = _cfg.Channels;
        int tReal = allSettledFrames.Count;
        Tensor codes = new(new TensorShape(ch, 1, tReal), DType.I32);
        int* cp = (int*)codes.DataPointer;
        for (int c = 0; c < ch; c++)
            for (int j = 0; j < tReal; j++)
                cp[(long)c * tReal + j] = (uint)allSettledFrames[j][c] > 1023u ? 0 : allSettledFrames[j][c];
        Tensor audioT = _dac.Decode(backend, codes, batch: 1, tFrames: tReal);
        codes.Dispose();

        int n = (int)audioT.Shape[audioT.Shape.Rank - 1];
        int tailLen = Math.Max(0, n - alreadyEmittedSamples);
        float[] tail = new float[tailLen];
        if (tailLen > 0)
        {
            float* ap = (float*)audioT.DataPointer;
            Buffer.MemoryCopy(ap + alreadyEmittedSamples, System.Runtime.CompilerServices.Unsafe.AsPointer(ref tail[0]),
                tailLen * 4L, tailLen * 4L);
        }
        audioT.Dispose();
        return tail;
    }

    /// <summary>Steps both CFG branches and returns per-channel (conditional, CFG-combined) logits.</summary>
    private (float[][] Cond, float[][] Guided) StepCfg(IBackend backend, ReadOnlySpan<int> frame, int posStart,
        StreamingKvCache cacheC, StreamingKvCache cacheU)
    {
        Tensor lc = _decCond.StepLogits(backend, frame, posStart, cacheC);
        Tensor lu = _decUncond.StepLogits(backend, frame, posStart, cacheU);
        float* pc = (float*)lc.DataPointer;
        float* pu = (float*)lu.DataPointer;
        int v = _cfg.AudioVocab;
        float g = _runCfgScale;
        float[][] condL = new float[_cfg.Channels][];
        float[][] guidedL = new float[_cfg.Channels][];
        for (int c = 0; c < _cfg.Channels; c++)
        {
            float[] cArr = new float[v];
            float[] gArr = new float[v];
            long baseOff = (long)c * v;
            for (int i = 0; i < v; i++)
            {
                float cond = pc[baseOff + i], uncond = pu[baseOff + i];
                cArr[i] = cond;
                gArr[i] = cond + g * (cond - uncond);
            }
            condL[c] = cArr;
            guidedL[c] = gArr;
        }
        lc.Dispose(); lu.Dispose();
        return (condL, guidedL);
    }

    private int SampleChannel(float[] cond, float[] guided, int channel, ref uint rng)
        => SampleDiaChannel(cond, guided, channel, _cfg, ref rng, _runTopK, _runTemperature, _runTopP);

    /// <summary>Faithful Dia sampling (model.py:440-464 + _sample_next_token): the candidate set is the
    /// top-<c>TopK</c> of the CFG-combined logits, but the distribution sampled is the CONDITIONAL logits
    /// restricted to it; PAD/BOS/1027 are masked for all channels and EOS for channels &gt; 0; EOS is masked
    /// unless it is the argmax, in which case it is forced.</summary>
    internal static int SampleDiaChannel(float[] cond, float[] guided, int channel, DiaConfig cfg, ref uint rng)
        => SampleDiaChannel(cond, guided, channel, cfg, ref rng, cfg.TopK, cfg.Temperature, cfg.TopP);

    /// <inheritdoc cref="SampleDiaChannel(float[], float[], int, DiaConfig, ref uint)"/>
    internal static int SampleDiaChannel(float[] cond, float[] guided, int channel, DiaConfig cfg, ref uint rng,
        int topK, float temperature, float topP)
    {
        int v = cfg.AudioVocab;
        int k = Math.Min(topK, v);
        int[] order = new int[v];
        for (int i = 0; i < v; i++) order[i] = i;
        Array.Sort(order, (a, b) => guided[b].CompareTo(guided[a]));
        float[] arr = new float[v];
        Array.Fill(arr, float.NegativeInfinity);
        // model.py:442-445: the candidate window is the top-K of the CFG-COMBINED logits, but the sampled
        // distribution (EOS-argmax + top_p + multinomial in _sample_next_token) is the CONDITIONAL logits
        // restricted to it (`cond_logits.masked_fill(mask, -inf)`). Filling `guided` here is WRONG: at an
        // utterance's end the all-pad UNCOND branch also wants EOS, so cfg*(cond-uncond) shrinks (or flips)
        // the EOS margin — EOS then never becomes the argmax and generation runs to the cap (never-EOS +
        // off-distribution garble → near-silent decode). Sample the conditional, exactly like upstream.
        for (int r = 0; r < k; r++) arr[order[r]] = cond[order[r]];
        // The window is the top-K of GUIDED, but at an utterance's end the all-pad uncond branch also wants
        // EOS, so guided = cond+g*(cond-uncond) shrinks/flips the EOS score and EOS drops out of the top-K —
        // even though the reference (and we) sample the CONDITIONAL. Once EOS leaves the window it can never
        // be the argmax, so channel 0 never terminates (never-EOS → the model babbles/repeats to the cap).
        // Rescue: if the CONDITIONAL itself ranks EOS within its own top-K, re-admit it to the window for
        // channel 0. It still only fires through the EOS-is-argmax path below, so mid-utterance (low cond EOS)
        // is unchanged; it fires exactly when the conditional most wants to stop.
        if (channel == 0 && float.IsNegativeInfinity(arr[cfg.AudioEos]))
        {
            float eosCond = cond[cfg.AudioEos];
            int rank = 0;
            for (int i = 0; i <= cfg.AudioEos && rank < k; i++) if (cond[i] > eosCond) rank++;
            if (rank < k) arr[cfg.AudioEos] = eosCond;
        }
        for (int i = cfg.AudioEos + 1; i < v; i++) arr[i] = float.NegativeInfinity;
        if (channel != 0) arr[cfg.AudioEos] = float.NegativeInfinity;
        int top = 0;
        for (int i = 1; i < v; i++) if (arr[i] > arr[top]) top = i;
        if (top == cfg.AudioEos)
        {
            for (int i = 0; i < cfg.AudioEos; i++) arr[i] = float.NegativeInfinity;   // force EOS
        }
        else
        {
            arr[cfg.AudioEos] = float.NegativeInfinity;   // EOS only when it is the argmax
        }
        return NucleusSampler.Draw(arr, v, temperature, 0, topP, ref rng);
    }

    /// <summary>Replaces literal <c>[S1]</c>/<c>[S2]</c> byte runs with 0x01/0x02 (upstream _encode_text)
    /// and truncates to <paramref name="maxLen"/>. Idempotent for callers that already folded the tags.</summary>
    internal static int[] FoldSpeakerTags(ReadOnlySpan<int> textBytes, int maxLen)
    {
        List<int> ids = new(textBytes.Length);
        for (int i = 0; i < textBytes.Length; i++)
        {
            if (i + 3 < textBytes.Length && textBytes[i] == '[' && textBytes[i + 1] == 'S'
                && (textBytes[i + 2] == '1' || textBytes[i + 2] == '2') && textBytes[i + 3] == ']')
            {
                ids.Add(textBytes[i + 2] - '0');
                i += 3;
                continue;
            }
            ids.Add(textBytes[i]);
        }
        if (ids.Count > maxLen) ids.RemoveRange(maxLen, ids.Count - maxLen);
        return [.. ids];
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
