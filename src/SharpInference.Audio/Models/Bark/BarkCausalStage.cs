using SharpInference.Audio.Dsp;
using SharpInference.Audio.Models.LanguageModels.Gpt;
using SharpInference.Audio.Models.Whisper;
using SharpInference.Audio.Sampling;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.Bark;

/// <summary>A causal Bark stage (semantic or coarse) — a single token embedding table + the shared
/// <see cref="GptBackbone"/> + an output head, generating tokens autoregressively with the shared
/// <see cref="NucleusSampler"/>. Runs full-sequence per step (no KV cache); fine for the scaffold.</summary>
public sealed unsafe class BarkCausalStage : IDisposable
{
    private readonly GptConfig _gpt;
    private readonly int _inVocab;
    private readonly int _outVocab;
    private readonly GptBackbone _backbone;
    private int _disposed;

    private Tensor? _inputEmbed;   // [inVocab, hidden]
    private Tensor? _lmHead;       // [outVocab, hidden]

    public BarkCausalStage(GptConfig gpt, int inVocab, int outVocab)
    {
        _gpt = gpt;
        _inVocab = inVocab;
        _outVocab = outVocab;
        _backbone = new GptBackbone(gpt);
    }

    /// <summary>Loads under the HF Bark stage prefix (e.g. <c>semantic</c> / <c>coarse_acoustics</c>):
    /// <c>{p}.input_embeds_layer.weight</c>, <c>{p}.position_embeds_layer.weight</c>, <c>{p}.layers.{i}.*</c>,
    /// <c>{p}.layernorm_final.*</c>, <c>{p}.lm_head.weight</c>.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _inputEmbed = WhisperOps.EnsureF32(w[$"{p}input_embeds_layer.weight"]);
        _backbone.LoadWeights(w, $"{p}position_embeds_layer.weight", $"{p}layers",
            $"{p}layernorm_final.weight", $"{p}layernorm_final.bias");
        _lmHead = WhisperOps.EnsureF32(w[$"{p}lm_head.weight"]);
    }

    /// <summary>Autoregressively generates up to <paramref name="maxTokens"/> tokens after the prompt,
    /// stopping at <paramref name="eosToken"/>. Returns the generated token IDs (EOS excluded).</summary>
    public List<int> Generate(IBackend backend, IReadOnlyList<int> promptTokenIds, int maxTokens,
        float temperature, int topK, float topP, int eosToken, int seed)
    {
        if (_inputEmbed is null) throw new InvalidOperationException("BarkCausalStage weights not loaded.");
        int h = _gpt.Hidden;
        List<int> seq = new(promptTokenIds);
        List<int> generated = new(Math.Min(maxTokens, 256));
        uint rng = DeterministicRng.Seed(seed);

        for (int step = 0; step < maxTokens; step++)
        {
            int t = Math.Min(seq.Count, _gpt.BlockSize);
            Tensor embeds = EmbedTail(seq, t, h);
            Tensor hidden = _backbone.Forward(backend, embeds, nonCausal: false);
            embeds.Dispose();

            Tensor last = SliceLast(hidden, h);
            hidden.Dispose();
            Tensor logits = WhisperOps.ProjectLinear(backend, last, _lmHead!, bias: null, 1, 1, h, _outVocab);
            last.Dispose();

            int next = NucleusSampler.Draw(new Span<float>((void*)logits.DataPointer, _outVocab),
                _outVocab, temperature, topK, topP, ref rng);
            logits.Dispose();
            if (next == eosToken) break;
            generated.Add(next);
            seq.Add(next);
            if (seq.Count >= _gpt.BlockSize) break;
        }
        return generated;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_inputEmbed is not null) yield return _inputEmbed;
        foreach (Tensor t in _backbone.EnumerateWeights()) yield return t;
        if (_lmHead is not null) yield return _lmHead;
    }

    private Tensor EmbedTail(List<int> seq, int t, int h)
    {
        int start = seq.Count - t;
        Tensor embeds = new(new TensorShape(1, t, h), DType.F32);
        float* ep = (float*)embeds.DataPointer;
        float* tab = (float*)_inputEmbed!.DataPointer;
        for (int s = 0; s < t; s++)
        {
            int id = seq[start + s];
            if ((uint)id >= (uint)_inVocab) id = 0;
            Buffer.MemoryCopy(tab + (long)id * h, ep + (long)s * h, h * 4, h * 4);
        }
        return embeds;
    }

    private static Tensor SliceLast(Tensor hidden, int h)
    {
        int t = (int)hidden.Shape[1];
        Tensor last = new(new TensorShape(1, 1, h), DType.F32);
        Buffer.MemoryCopy((float*)hidden.DataPointer + (long)(t - 1) * h, (void*)last.DataPointer, h * 4, h * 4);
        return last;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _backbone.Dispose();
        GC.SuppressFinalize(this);
    }
}
