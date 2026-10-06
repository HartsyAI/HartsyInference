using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Sampling;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;

namespace HartsyInference.Audio.Models.BreezeTts;

/// <summary>The audio-token half of Breeze TTS 2: a Qwen3 backbone that reads the prompt embeddings and then the sum of
/// each generated frame's 16 codebook embeddings, an <c>lm_head</c> over the first codebook plus a backbone-EOS class,
/// and a CSM-style depth decoder that fills in the other 15 codebooks from the backbone's last hidden state.
/// (Port of <c>BreezeForConditionalGeneration</c> / <c>BreezeDepthDecoderForCausalLM</c> and the sampling in
/// <c>fast_streaming.py</c>.) Text conditioning comes from <see cref="T5Gemma2TextEncoder"/>.</summary>
public sealed unsafe class BreezeTts2Model : IDisposable
{
    private readonly BreezeTts2Config _cfg;
    private readonly GenericTransformer _backbone;
    private readonly GenericTransformer _depth;
    private Tensor? _audioEmbed, _depthProj, _lmHead;
    private Tensor[] _codebookHeads = [];
    private int _disposed;

    public BreezeTts2Model(BreezeTts2Config cfg)
    {
        _cfg = cfg;
        _backbone = new GenericTransformer(new TransformerConfig
        {
            HiddenSize = cfg.HiddenSize, NumLayers = cfg.NumBackboneLayers, NumHeads = cfg.NumBackboneHeads,
            NumKvHeads = cfg.NumBackboneKeyValueHeads, HeadDim = cfg.BackboneHeadDim,
            IntermediateSize = cfg.BackboneIntermediateSize, VocabSize = 1, MaxPositionEmbeddings = cfg.MaxSequenceLength,
            RopeTheta = cfg.BackboneRopeTheta, RmsNormEps = cfg.BackboneRmsNormEps, AttentionBias = false, QkNorm = true,
            TieWordEmbeddings = false,
        });
        _depth = new GenericTransformer(new TransformerConfig
        {
            HiddenSize = cfg.DepthDecoderHiddenSize, NumLayers = cfg.NumDepthDecoderLayers,
            NumHeads = cfg.DepthDecoderHeads, NumKvHeads = cfg.DepthDecoderKeyValueHeads, HeadDim = cfg.DepthDecoderHeadDim,
            IntermediateSize = cfg.DepthDecoderIntermediateSize, VocabSize = 1,
            MaxPositionEmbeddings = cfg.DepthDecoderMaxPositions, RopeTheta = cfg.DepthDecoderRopeTheta,
            RmsNormEps = cfg.DepthDecoderRmsNormEps, AttentionBias = false, QkNorm = false, TieWordEmbeddings = false,
            RopeScaling = new RopeScaling
            {
                Type = RopeScalingType.Llama3, Factor = 32.0, OriginalContextLength = 16,
                LowFreqFactor = 0.001953125, HighFreqFactor = 0.0078125,
            },
        });
    }

    public int HiddenSize => _cfg.HiddenSize;

    /// <summary>Loads <c>backbone_model.*</c>, <c>depth_decoder.*</c> and <c>lm_head.weight</c>. The released checkpoint
    /// ties the backbone's audio embedding to <c>depth_decoder.model.embed_tokens</c>, so only the latter is stored.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _backbone.LoadWeightsHeadless(w, "backbone_model");
        _depth.LoadWeightsHeadless(w, "depth_decoder.model");
        _audioEmbed = WhisperOps.EnsureF32(w["depth_decoder.model.embed_tokens.weight"]);
        _depthProj = w["depth_decoder.model.inputs_embeds_projector.weight"];
        _lmHead = w["lm_head.weight"];

        // codebooks_head.weight is [numCodebooks-1, hidden, vocab]; Linear wants [vocab, hidden] per codebook.
        Tensor heads = WhisperOps.EnsureF32(w["depth_decoder.codebooks_head.weight"]);
        int n = (int)heads.Shape[0], hid = (int)heads.Shape[1], vocab = (int)heads.Shape[2];
        _codebookHeads = new Tensor[n];
        float* src = (float*)heads.DataPointer;
        for (int k = 0; k < n; k++)
        {
            Tensor t = new(new TensorShape(vocab, hid), DType.F32);
            float* dst = (float*)t.DataPointer;
            for (int d = 0; d < hid; d++)
                for (int v = 0; v < vocab; v++) dst[(long)v * hid + d] = src[((long)k * hid + d) * vocab + v];
            _codebookHeads[k] = t;
        }
    }

    /// <summary>Embedding of a generated/prompt audio frame: the sum over codebooks of
    /// <c>embed[codebook · vocab + code]</c>. Returns <c>[hidden]</c>.</summary>
    public float[] EmbedFrame(ReadOnlySpan<int> codes)
    {
        int h = _cfg.HiddenSize, v = _cfg.AudioVocabSize;
        float[] o = new float[h];
        float* e = (float*)_audioEmbed!.DataPointer;
        for (int i = 0; i < _cfg.NumCodebooks; i++)
        {
            float* row = e + ((long)i * v + codes[i]) * h;
            for (int c = 0; c < h; c++) o[c] += row[c];
        }
        return o;
    }

    public IKvCache CreateBackboneCache(int maxSeqLen) => KvCaches.ForDecode(
        _cfg.NumBackboneLayers, _cfg.NumBackboneKeyValueHeads, _cfg.BackboneHeadDim, maxSeqLen);

    /// <summary>Runs the backbone over <paramref name="embeds"/> (<c>[t, hidden]</c>, host) and returns the final-norm
    /// hidden state of the last position (<c>[hidden]</c>).</summary>
    public float[] BackboneStep(IBackend backend, float[] embeds, int t, int posStart, IKvCache cache)
    {
        int h = _cfg.HiddenSize;
        using Tensor input = new(new TensorShape(1, t, h), DType.F32);
        embeds.AsSpan(0, t * h).CopyTo(new Span<float>((void*)input.DataPointer, t * h));
        using Tensor hidden = _backbone.ForwardEmbeds(backend, input, t, posStart, cache, applyFinalNorm: true);
        return new ReadOnlySpan<float>((float*)hidden.DataPointer + (long)(t - 1) * h, h).ToArray();
    }

    /// <summary><c>lm_head</c> over a backbone hidden state: <c>[audioVocab + 1]</c> logits (the last class is the
    /// backbone's end-of-speech token).</summary>
    public float[] BackboneLogits(IBackend backend, float[] hidden)
    {
        int h = _cfg.HiddenSize, n = (int)_lmHead!.Shape[0];
        using Tensor inp = new(new TensorShape(1, 1, h), DType.F32);
        hidden.AsSpan().CopyTo(new Span<float>((void*)inp.DataPointer, h));
        using Tensor o = new(new TensorShape(1, 1, n), DType.F32);
        backend.Linear(o, inp, _lmHead, null);
        return new ReadOnlySpan<float>((void*)o.DataPointer, n).ToArray();
    }

    /// <summary>Samples the 16-codebook frame: <paramref name="firstCode"/> from the backbone, the other 15 from the depth
    /// decoder (incremental, cached). With <paramref name="uncondHidden"/> the depth decoder runs a second, unconditional
    /// stream and each codebook's logits are <c>uncond + cfgScale · (cond − uncond)</c>. Reserved ids (the 2048+ specials)
    /// are never drawn.</summary>
    public int[] SampleFrame(IBackend backend, float[] backboneHidden, int firstCode, ref uint rng,
        float[]? uncondHidden = null, float cfgScale = 1f, float? temperature = null, int? topK = null, float? topP = null)
    {
        int n = _cfg.NumCodebooks, v = _cfg.AudioVocabSize, h = _cfg.HiddenSize;
        int[] codes = new int[n];
        codes[0] = firstCode;
        using IKvCache condCache = KvCaches.ForDecode(_cfg.NumDepthDecoderLayers, _cfg.DepthDecoderKeyValueHeads,
            _cfg.DepthDecoderHeadDim, n + 1);
        using IKvCache? uncondCache = uncondHidden is null ? null : KvCaches.ForDecode(_cfg.NumDepthDecoderLayers,
            _cfg.DepthDecoderKeyValueHeads, _cfg.DepthDecoderHeadDim, n + 1);

        // positions 0 (backbone hidden) and 1 (first code) go in together, then one position per further code
        float[] first = EmbedDepthToken(firstCode, 0);
        float[] condInputs = [.. backboneHidden, .. first];
        float[]? uncondInputs = uncondHidden is null ? null : [.. uncondHidden, .. first];
        int pos = 0, count = 2;
        for (int k = 1; k < n; k++)
        {
            float[] logits = CodebookLogits(backend, DepthStep(backend, condInputs, count, pos, condCache), k - 1);
            if (uncondInputs is not null)
            {
                float[] u = CodebookLogits(backend, DepthStep(backend, uncondInputs, count, pos, uncondCache!), k - 1);
                for (int i = 0; i < logits.Length; i++) logits[i] = u[i] + cfgScale * (logits[i] - u[i]);
            }
            pos += count;
            for (int r = _cfg.CodebookSize; r < v; r++) logits[r] = float.NegativeInfinity;
            codes[k] = NucleusSampler.Draw(logits, v, temperature ?? _cfg.Temperature, topK ?? _cfg.TopK, topP ?? _cfg.TopP, ref rng);
            if (k < n - 1)
            {
                condInputs = EmbedDepthToken(codes[k], k);
                uncondInputs = uncondInputs is null ? null : condInputs;
                count = 1;
            }
        }
        return codes;
    }

    private float[] EmbedDepthToken(int code, int codebook)
    {
        int h = _cfg.HiddenSize, v = _cfg.AudioVocabSize;
        float[] o = new float[h];
        new ReadOnlySpan<float>((float*)_audioEmbed!.DataPointer + ((long)codebook * v + code) * h, h).CopyTo(o);
        return o;
    }

    // inputs: [count, audioEmbedSize]; projected to the depth width, run through the depth stack; returns the
    // final-norm hidden state of the last position.
    private float[] DepthStep(IBackend backend, float[] inputs, int count, int posStart, IKvCache cache)
    {
        int h = _cfg.HiddenSize, dd = _cfg.DepthDecoderHiddenSize;
        using Tensor raw = new(new TensorShape(1, count, h), DType.F32);
        inputs.AsSpan(0, count * h).CopyTo(new Span<float>((void*)raw.DataPointer, count * h));
        using Tensor projected = new(new TensorShape(1, count, dd), DType.F32);
        backend.Linear(projected, raw, _depthProj!, null);
        using Tensor hidden = _depth.ForwardEmbeds(backend, projected, count, posStart, cache, applyFinalNorm: true);
        return new ReadOnlySpan<float>((float*)hidden.DataPointer + (long)(count - 1) * dd, dd).ToArray();
    }

    private float[] CodebookLogits(IBackend backend, float[] hidden, int head)
    {
        int dd = _cfg.DepthDecoderHiddenSize, v = _cfg.AudioVocabSize;
        using Tensor inp = new(new TensorShape(1, 1, dd), DType.F32);
        hidden.AsSpan().CopyTo(new Span<float>((void*)inp.DataPointer, dd));
        using Tensor o = new(new TensorShape(1, 1, v), DType.F32);
        backend.Linear(o, inp, _codebookHeads[head], null);
        return new ReadOnlySpan<float>((void*)o.DataPointer, v).ToArray();
    }

    /// <summary>Diagnostics: logits <c>[len-1, vocab]</c> of the depth decoder for <c>[placeholder, t0, t1, …]</c> with
    /// <paramref name="backboneHidden"/> at position 0 (the official full-sequence forward, positions 1.. use head p-1).</summary>
    public float[] DebugDepthLogits(IBackend backend, float[] backboneHidden, int[] tokens)
    {
        int h = _cfg.HiddenSize, v = _cfg.AudioVocabSize, len = tokens.Length;
        float[] inputs = new float[len * h];
        backboneHidden.CopyTo(inputs, 0);
        for (int p = 1; p < len; p++) EmbedDepthToken(tokens[p], p - 1).CopyTo(inputs, p * h);
        using IKvCache cache = KvCaches.ForDecode(_cfg.NumDepthDecoderLayers, _cfg.DepthDecoderKeyValueHeads,
            _cfg.DepthDecoderHeadDim, len + 1);
        float[] result = new float[(len - 1) * v];
        // run position-by-position through the cache (equivalent to the full causal forward)
        for (int p = 0; p < len; p++)
        {
            float[] hidden = DepthStep(backend, inputs.AsSpan(p * h, h).ToArray(), 1, p, cache);
            if (p >= 1) CodebookLogits(backend, hidden, p - 1).CopyTo(result, (p - 1) * v);
        }
        return result;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _audioEmbed, _depthProj, _lmHead }) if (t is not null) yield return t;
        foreach (Tensor t in _codebookHeads) yield return t;
        foreach (Tensor t in _backbone.EnumerateWeights()) yield return t;
        foreach (Tensor t in _depth.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _backbone.Dispose(); _depth.Dispose();
        GC.SuppressFinalize(this);
    }
}
