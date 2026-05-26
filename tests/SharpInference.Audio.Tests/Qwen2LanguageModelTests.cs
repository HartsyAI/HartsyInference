using SharpInference.Audio.Cache;
using SharpInference.Audio.Models.LanguageModels.Qwen2;
using SharpInference.Audio.Streaming;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.ModelHandler.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace SharpInference.Audio.Tests;

/// <summary>Smoke tests for the native Qwen2.5 autoregressive LM (used by VibeVoice).
/// Loads the LM half of the cached <c>VibeVoice-1.5B</c> checkpoint (338 tensors) and
/// runs a prefill + a single-token decode, verifying:
/// <list type="bullet">
///   <item>All LM weights bind (no missing keys).</item>
///   <item>Prefill forward on a small prompt produces a <c>[1, T, 1536]</c> hidden
///         state, all finite.</item>
///   <item>One incremental decode step produces a <c>[1, 1, 1536]</c> hidden, with the
///         KV cache advanced by exactly one position.</item>
///   <item>Logits projection (tied embedding) produces <c>[1, 1, 151936]</c> with finite
///         values, and the argmax yields a token ID in the valid range.</item>
/// </list>
///
/// <para>The tests skip when the VibeVoice cache is missing. They are <i>structural</i> —
/// without a Python-side reference dump we can't verify the actual hidden values; that's
/// the next-session validation step.</para></summary>
public sealed class Qwen2LanguageModelTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly SafeTensorsShardLoader? _loader;
    private readonly bool _cacheReady;

    public Qwen2LanguageModelTests(ITestOutputHelper output)
    {
        _out = output;
        string repoDir = AudioModelCache.GetRepoDirectory("vibevoice/VibeVoice-1.5B");
        _cacheReady = File.Exists(Path.Combine(repoDir, "model-00001-of-00003.safetensors"))
            && File.Exists(Path.Combine(repoDir, "model-00002-of-00003.safetensors"))
            && File.Exists(Path.Combine(repoDir, "model-00003-of-00003.safetensors"));
        if (_cacheReady)
        {
            _loader = new SafeTensorsShardLoader();
            _loader.LoadDirectory(repoDir);
        }
    }

    public void Dispose() => _loader?.Dispose();

    private bool Skip()
    {
        if (!_cacheReady) _out.WriteLine("VibeVoice-1.5B cache missing — skipping.");
        return !_cacheReady;
    }

    [Fact]
    public void Config_Qwen25_1_5B_MatchesVibeVoiceDefaults()
    {
        Qwen2Config cfg = Qwen2Config.Qwen25_1_5B;
        Assert.Equal(1_536, cfg.HiddenSize);
        Assert.Equal(28, cfg.NumHiddenLayers);
        Assert.Equal(12, cfg.NumAttentionHeads);
        Assert.Equal(2, cfg.NumKeyValueHeads);
        Assert.Equal(8_960, cfg.IntermediateSize);
        Assert.Equal(151_936, cfg.VocabSize);
        Assert.Equal(65_536, cfg.MaxPositionEmbeddings);
        Assert.Equal(128, cfg.HeadDim);
        Assert.Equal(256, cfg.KvHiddenSize);
        Assert.Equal(6, cfg.KvGroupSize);
        Assert.Equal(1_000_000f, cfg.RopeTheta);
        Assert.True(cfg.TieWordEmbeddings);
    }

    [Fact]
    public void LoadsAll338LmWeights_NoMissingKeys()
    {
        if (Skip()) return;
        Qwen2Config cfg = Qwen2Config.Qwen25_1_5B;
        using Qwen2Model lm = new(cfg);
        lm.LoadWeights(_loader!.GetAllTensors(), prefix: "model.language_model");
        int bound = lm.EnumerateWeights().Count();
        _out.WriteLine($"Qwen2Model bound {bound} tensors.");
        // 28 layers × 12 tensors per layer (q+bias, k+bias, v+bias, o, input_norm,
        // post_attn_norm, gate, up, down) = 336. Plus embed + final norm = 338. With
        // tied embeddings _lmHead is null so EnumerateWeights returns 338.
        Assert.Equal(338, bound);
    }

    [Fact]
    public unsafe void Prefill_FiveTokenPrompt_ProducesFiniteHidden()
    {
        if (Skip()) return;
        Qwen2Config cfg = Qwen2Config.Qwen25_1_5B;
        using Qwen2Model lm = new(cfg);
        lm.LoadWeights(_loader!.GetAllTensors(), prefix: "model.language_model");

        using CpuBackend backend = new();
        // Small KV cache budget — prefill 5 tokens then decode a few.
        int maxSeq = 16;
        using StreamingKvCache cache = new(numLayers: cfg.NumHiddenLayers, batch: 1,
            numKvHeads: cfg.NumKeyValueHeads, maxSequenceLength: maxSeq, headDim: cfg.HeadDim);

        // Use known Qwen2 tokens: BOS = 151643, then a few arbitrary IDs.
        int[] prompt = [cfg.BosTokenId, 100, 200, 300, 400];
        _out.WriteLine($"Prefill prompt: [{string.Join(", ", prompt)}]");

        DateTime t0 = DateTime.Now;
        using Tensor hidden = lm.Forward(backend, prompt, batch: 1, posStart: 0, cache);
        TimeSpan dt = DateTime.Now - t0;
        _out.WriteLine($"Prefill 5 tokens: {dt.TotalSeconds:F2} s; hidden shape = {hidden.Shape}.");

        Assert.Equal(new TensorShape(1, prompt.Length, cfg.HiddenSize), hidden.Shape);
        Assert.Equal(prompt.Length, cache.CurrentLength);
        AssertAllFinite(hidden, "prefill.hidden");
    }

    [Fact]
    public unsafe void Prefill_Then_SingleTokenDecode_AdvancesCacheByOne()
    {
        if (Skip()) return;
        Qwen2Config cfg = Qwen2Config.Qwen25_1_5B;
        using Qwen2Model lm = new(cfg);
        lm.LoadWeights(_loader!.GetAllTensors(), prefix: "model.language_model");

        using CpuBackend backend = new();
        int maxSeq = 16;
        using StreamingKvCache cache = new(numLayers: cfg.NumHiddenLayers, batch: 1,
            numKvHeads: cfg.NumKeyValueHeads, maxSequenceLength: maxSeq, headDim: cfg.HeadDim);

        int[] prompt = [cfg.BosTokenId, 100, 200, 300];
        using (Tensor h = lm.Forward(backend, prompt, batch: 1, posStart: 0, cache)) { }
        Assert.Equal(prompt.Length, cache.CurrentLength);

        // Decode one new token at the next position.
        int[] next = [500];
        DateTime t0 = DateTime.Now;
        using Tensor stepHidden = lm.Forward(backend, next, batch: 1, posStart: cache.CurrentLength, cache);
        TimeSpan dt = DateTime.Now - t0;
        _out.WriteLine($"Decode-1 step: {dt.TotalSeconds * 1000:F0} ms; hidden shape = {stepHidden.Shape}.");

        Assert.Equal(new TensorShape(1, 1, cfg.HiddenSize), stepHidden.Shape);
        Assert.Equal(prompt.Length + 1, cache.CurrentLength);
        AssertAllFinite(stepHidden, "decode1.hidden");
    }

    [Fact]
    public unsafe void Logits_FromDecodeStep_IsSaneVocabDistribution()
    {
        if (Skip()) return;
        Qwen2Config cfg = Qwen2Config.Qwen25_1_5B;
        using Qwen2Model lm = new(cfg);
        lm.LoadWeights(_loader!.GetAllTensors(), prefix: "model.language_model");

        using CpuBackend backend = new();
        int maxSeq = 8;
        using StreamingKvCache cache = new(cfg.NumHiddenLayers, 1, cfg.NumKeyValueHeads, maxSeq, cfg.HeadDim);

        int[] prompt = [cfg.BosTokenId, 100, 200];
        using Tensor hidden = lm.Forward(backend, prompt, 1, 0, cache);

        // Project the LAST position to vocab logits (tied embedding).
        using Tensor lastHidden = SliceLastT(hidden, batch: 1, seq: prompt.Length, hidden: cfg.HiddenSize);
        using Tensor logits = lm.ProjectLogits(backend, lastHidden, batch: 1, t: 1);

        Assert.Equal(new TensorShape(1, 1, cfg.VocabSize), logits.Shape);
        AssertAllFinite(logits, "logits");

        // Argmax should be a valid token ID and the distribution shouldn't be degenerate
        // (all the same value — that would mean the head produced zero logits).
        float* lp = (float*)logits.DataPointer;
        int argmax = 0;
        float maxV = lp[0];
        float minV = lp[0];
        for (int i = 1; i < cfg.VocabSize; i++)
        {
            float v = lp[i];
            if (v > maxV) { maxV = v; argmax = i; }
            if (v < minV) minV = v;
        }
        _out.WriteLine($"Logits: argmax={argmax}, max={maxV:F3}, min={minV:F3}, spread={maxV - minV:F3}.");
        Assert.InRange(argmax, 0, cfg.VocabSize - 1);
        Assert.True(maxV - minV > 1f, $"Logit spread is degenerate ({maxV - minV:F3}).");
    }

    [Fact]
    public unsafe void PrefillThenTwoDecodeSteps_AllShapesAndCacheConsistent()
    {
        if (Skip()) return;
        Qwen2Config cfg = Qwen2Config.Qwen25_1_5B;
        using Qwen2Model lm = new(cfg);
        lm.LoadWeights(_loader!.GetAllTensors(), prefix: "model.language_model");

        using CpuBackend backend = new();
        int maxSeq = 16;
        using StreamingKvCache cache = new(cfg.NumHiddenLayers, 1, cfg.NumKeyValueHeads, maxSeq, cfg.HeadDim);

        // Prefill.
        int[] prompt = [cfg.BosTokenId, 100, 200];
        using (Tensor h = lm.Forward(backend, prompt, 1, 0, cache)) { }
        Assert.Equal(3, cache.CurrentLength);

        // Decode step 1.
        using (Tensor h1 = lm.Forward(backend, [400], 1, cache.CurrentLength, cache))
        {
            Assert.Equal(new TensorShape(1, 1, cfg.HiddenSize), h1.Shape);
        }
        Assert.Equal(4, cache.CurrentLength);

        // Decode step 2.
        using (Tensor h2 = lm.Forward(backend, [500], 1, cache.CurrentLength, cache))
        {
            Assert.Equal(new TensorShape(1, 1, cfg.HiddenSize), h2.Shape);
            AssertAllFinite(h2, "decode2.hidden");
        }
        Assert.Equal(5, cache.CurrentLength);
        _out.WriteLine($"3 prefill + 2 decode → cache length {cache.CurrentLength}/16.");
    }

    // ── helpers ───────────────────────────────────────────────────────────

    private static unsafe Tensor SliceLastT(Tensor src, int batch, int seq, int hidden)
    {
        Tensor dst = new(new TensorShape(batch, 1, hidden), DType.F32);
        float* sp = (float*)src.DataPointer;
        float* dp = (float*)dst.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            long srcOff = ((long)b * seq + (seq - 1)) * hidden;
            long dstOff = (long)b * hidden;
            Buffer.MemoryCopy(sp + srcOff, dp + dstOff, hidden * 4, hidden * 4);
        }
        return dst;
    }

    private static unsafe void AssertAllFinite(Tensor t, string ctx)
    {
        long n = t.ElementCount;
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < n; i++)
        {
            float v = p[i];
            if (float.IsNaN(v) || float.IsInfinity(v))
                Assert.Fail($"[{ctx}] non-finite value {v} at index {i}.");
        }
    }
}
