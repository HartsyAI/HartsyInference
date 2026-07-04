using HartsyInference.Audio.Models.LanguageModels.Qwen2;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.Music;

/// <summary>YuE Stage-2 residual upsampler — a ~1.5B LLaMA decoder (reuses <see cref="Qwen2Model"/>, same
/// body as Stage-1) that takes one track's codebook-0 stream and autoregressively predicts residual
/// codebooks 1..7, yielding the full 8-codebook grid X-Codec decodes to waveform. Faithful port of
/// upstream <c>infer.py:stage2_generate</c> (m-a-p/YuE, <c>YuE-s2-1B-general</c>).
///
/// <para><b>xcodec token layout</b> (mm_tokenizer_v0.2): codebook k's index i maps to absolute id
/// <c>i + 45334 + k·1024</c>. Stage-2 sees cb0 tokens (k=0, range [45334,46358)) and emits residuals in
/// [46358,53526) (cb1..cb7). Upstream masks everything outside [46358,53526) and greedily decodes 7
/// tokens/frame; the token's slot (1..7) determines which codebook it belongs to (so a token whose value
/// falls in the wrong codebook's sub-range decodes out of [0,1023] and is patched afterward).</para>
///
/// <para><b>Windowing</b>: upstream processes independent 6-second (300-frame) windows, each with a fresh
/// prompt <c>[SOA][&lt;stage_1&gt;] + cb0(window) + [&lt;stage_2&gt;]</c> then per-frame teacher forcing.
/// The trailing partial window (&lt;300 frames) is handled the same way. We reproduce that per-window.</para>
///
/// <para><b>Sampling</b>: greedy argmax (upstream <c>generate</c> passes no temperature/top_p and forces
/// exactly 7 new tokens → HF greedy). No repetition penalty / CFG in Stage-2.</para></summary>
public sealed unsafe class YueStage2Lm : IDisposable
{
    // xcodec offset math (CodecManipulator("xcodec"): global_offset=45334, codebook_size=1024).
    private const int GlobalOffset = 45_334;
    private const int CodebookSize = 1_024;
    private const int Cb0Lo = GlobalOffset;                          // 45334
    private const int ResidualLo = GlobalOffset + 1 * CodebookSize;  // 46358 (cb1 start)
    private const int ResidualHi = GlobalOffset + 8 * CodebookSize;  // 53526 (cb7 end, exclusive)
    private const int WindowFrames = 300;                           // upstream 6s @ 50 fps

    private readonly YueConfig _cfg;
    private readonly Qwen2Model _lm;
    private readonly int _soa, _stage1Tok, _stage2Tok;
    private int _disposed;

    /// <param name="soaId">mm tokenizer &lt;SOA&gt; (32001).</param>
    /// <param name="stage1Id">mm tokenizer &lt;stage_1&gt; (32013).</param>
    /// <param name="stage2Id">mm tokenizer &lt;stage_2&gt; (32017).</param>
    public YueStage2Lm(YueConfig cfg, int soaId, int stage1Id, int stage2Id)
    {
        _cfg = cfg;
        _lm = new Qwen2Model(cfg.Stage2);
        _soa = soaId;
        _stage1Tok = stage1Id;
        _stage2Tok = stage2Id;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "model")
        => _lm.LoadWeights(w, prefix);

    /// <summary>Upsamples one track's codebook-0 stream (<paramref name="cb0Indices"/>, raw indices in
    /// [0,1023]) to a full <c>[8][T]</c> codebook grid (row 0 = the input cb0). Runs the LM per 300-frame
    /// window; invalid residuals are patched to the per-row mode (upstream's "fix invalid codes").</summary>
    public int[][] Upsample(IBackend backend, ReadOnlySpan<int> cb0Indices)
    {
        ThrowIfDisposed();
        int t = cb0Indices.Length;
        int[][] codes = new int[_cfg.NumCodebooks][];
        for (int k = 0; k < _cfg.NumCodebooks; k++) codes[k] = new int[t];
        for (int i = 0; i < t; i++) codes[0][i] = cb0Indices[i];   // cb0 row is the given stream
        if (t == 0) return codes;

        for (int w0 = 0; w0 < t; w0 += WindowFrames)
        {
            int wlen = Math.Min(WindowFrames, t - w0);
            UpsampleWindow(backend, cb0Indices.Slice(w0, wlen), codes, w0);
        }
        FixInvalidCodes(codes);
        return codes;
    }

    /// <summary>One independent window: prime <c>[SOA][stage_1] + cb0(window) + [stage_2]</c>, then for each
    /// frame feed its cb0 token and greedily emit the 7 residual tokens (cb1..cb7). Uses an incremental KV
    /// cache — numerically identical to upstream's re-prefill-per-frame loop, O(T) instead of O(T²).</summary>
    private void UpsampleWindow(IBackend backend, ReadOnlySpan<int> cb0, int[][] codes, int frameOffset)
    {
        int wlen = cb0.Length;
        int promptLen = 2 + wlen + 1;                                   // [soa,stage_1] + cb0 outline + [stage_2]
        int seqMax = promptLen + wlen * (_cfg.NumCodebooks - 1) + wlen + 8; // + per-frame (cb0 + 7 residuals)
        int cacheCap = Math.Min(_cfg.Stage2.MaxPositionEmbeddings, seqMax);
        using IKvCache cache = _lm.CreateDecodeCache(cacheCap);

        int[] prompt = new int[promptLen];
        prompt[0] = _soa;
        prompt[1] = _stage1Tok;
        for (int i = 0; i < wlen; i++) prompt[2 + i] = cb0[i] + GlobalOffset;  // cb0 outline (k=0 offset)
        prompt[promptLen - 1] = _stage2Tok;

        int h = _cfg.Stage2.HiddenSize;
        int nResidual = _cfg.NumCodebooks - 1;                          // 7
        Tensor hidden = _lm.Forward(backend, prompt, 1, 0, cache);
        try
        {
            for (int t = 0; t < wlen; t++)
            {
                // Feed this frame's cb0 token; the resulting hidden predicts residual codebook 1.
                int[] cb0Step = [cb0[t] + GlobalOffset];
                hidden.Dispose();
                hidden = _lm.Forward(backend, cb0Step, 1, cache.CurrentLength, cache);
                if (cache.CurrentLength >= cacheCap - (nResidual + 2)) break;

                for (int j = 1; j <= nResidual; j++)                    // codebooks 1..7
                {
                    Tensor last = SliceLast(hidden, h);
                    hidden.Dispose();
                    Tensor logitsT = _lm.ProjectLogits(backend, last, 1, 1);
                    last.Dispose();
                    float* lp = (float*)logitsT.DataPointer;

                    // Greedy argmax over the residual range [46358,53526) (== upstream block_list mask).
                    int best = ResidualLo;
                    float bestv = float.NegativeInfinity;
                    for (int v = ResidualLo; v < ResidualHi; v++)
                        if (lp[v] > bestv) { bestv = lp[v]; best = v; }
                    logitsT.Dispose();

                    // Decode assuming this slot is codebook j; out-of-range -> -1 (patched later).
                    int decoded = best - (GlobalOffset + j * CodebookSize);
                    codes[j][frameOffset + t] = (uint)decoded < (uint)CodebookSize ? decoded : -1;

                    // Autoregress: the emitted token must enter the cache for subsequent positions.
                    int[] resStep = [best];
                    hidden = _lm.Forward(backend, resStep, 1, cache.CurrentLength, cache);
                }
            }
        }
        finally
        {
            hidden.Dispose();
        }
    }

    /// <summary>Upstream "fix invalid codes": any residual marked invalid (-1) is replaced with the most
    /// frequent valid value in its codebook row (falling back to 0 for an all-invalid row).</summary>
    private void FixInvalidCodes(int[][] codes)
    {
        Span<int> counts = stackalloc int[CodebookSize];
        for (int k = 1; k < codes.Length; k++)
        {
            int[] row = codes[k];
            bool anyInvalid = false;
            counts.Clear();
            foreach (int v in row)
            {
                if ((uint)v < (uint)CodebookSize) counts[v]++;
                else anyInvalid = true;
            }
            if (!anyInvalid) continue;
            int mode = 0, modeCount = -1;
            for (int v = 0; v < CodebookSize; v++)
                if (counts[v] > modeCount) { modeCount = counts[v]; mode = v; }
            for (int i = 0; i < row.Length; i++)
                if ((uint)row[i] >= (uint)CodebookSize) row[i] = mode;
        }
    }

    private static Tensor SliceLast(Tensor hidden, int h)
    {
        int t = (int)hidden.Shape[1];
        Tensor last = new(new TensorShape(1, 1, h), DType.F32);
        Buffer.MemoryCopy((float*)hidden.DataPointer + (long)(t - 1) * h, (void*)last.DataPointer, h * 4, h * 4);
        return last;
    }

    public IEnumerable<Tensor> EnumerateWeights() => _lm.EnumerateWeights();

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _lm.Dispose();
        GC.SuppressFinalize(this);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(YueStage2Lm));
    }
}
