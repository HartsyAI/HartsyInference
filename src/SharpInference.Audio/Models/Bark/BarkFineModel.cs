using SharpInference.Audio.Models.LanguageModels.Gpt;
using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.Bark;

/// <summary>Bark fine acoustics — a non-causal (bidirectional) GPT that fills EnCodec codebooks 2..7
/// given the 2 coarse codebooks, in six refinement passes. Eight input codebook embeddings are summed
/// per timestep; seven output heads predict codebooks 1..7. Reuses the shared <see cref="GptBackbone"/>
/// in non-causal mode. Deterministic argmax fill (Bark uses low-temp / argmax for fine).</summary>
public sealed unsafe class BarkFineModel : IDisposable
{
    private readonly BarkConfig _cfg;
    private readonly GptBackbone _backbone;
    private readonly Tensor?[] _inputEmbeds;   // 8 × [fineVocab, hidden]
    private readonly Tensor?[] _outHeads;      // 7 × [fineVocab, hidden]
    private int _disposed;

    public BarkFineModel(BarkConfig cfg)
    {
        _cfg = cfg;
        _backbone = new GptBackbone(cfg.Stage);
        _inputEmbeds = new Tensor?[cfg.NumCodebooks];
        _outHeads = new Tensor?[cfg.NumCodebooks - 1];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "fine_acoustics")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        for (int i = 0; i < _cfg.NumCodebooks; i++)
            _inputEmbeds[i] = WhisperOps.EnsureF32(w[$"{p}input_embeds_layer.{i}.weight"]);
        _backbone.LoadWeights(w, $"{p}position_embeds_layer.weight", $"{p}layers",
            $"{p}layernorm_final.weight", $"{p}layernorm_final.bias");
        for (int i = 0; i < _cfg.NumCodebooks - 1; i++)
            _outHeads[i] = WhisperOps.EnsureF32(w[$"{p}lm_heads.{i}.weight"]);
    }

    /// <summary>Fills codebooks 2..7 from the 2 coarse codebooks. <paramref name="coarse"/> is
    /// <c>[numCoarse, T]</c>; returns all 8 codebooks <c>[8, T]</c>.</summary>
    public int[,] Refine(IBackend backend, int[,] coarse)
    {
        if (_inputEmbeds[0] is null) throw new InvalidOperationException("BarkFineModel weights not loaded.");
        int t = coarse.GetLength(1);
        int h = _cfg.Stage.Hidden;
        int[,] codes = new int[_cfg.NumCodebooks, t];
        for (int cb = 0; cb < _cfg.NumCoarseCodebooks; cb++)
            for (int j = 0; j < t; j++) codes[cb, j] = coarse[cb, j];

        // Predict codebooks numCoarse..7 in order.
        for (int pred = _cfg.NumCoarseCodebooks; pred < _cfg.NumCodebooks; pred++)
        {
            Tensor input = SumEmbeds(codes, pred, t, h);
            Tensor hidden = _backbone.Forward(backend, input, nonCausal: true);
            input.Dispose();
            Tensor logits = WhisperOps.ProjectLinear(backend, hidden, _outHeads[pred - 1]!, bias: null, 1, t, h, _cfg.FineVocab);
            hidden.Dispose();
            float* lp = (float*)logits.DataPointer;
            for (int j = 0; j < t; j++)
            {
                int best = 0; float bestV = float.NegativeInfinity;
                long off = (long)j * _cfg.FineVocab;
                for (int v = 0; v < _cfg.CodebookSize; v++)   // only the real codebook range is valid
                    if (lp[off + v] > bestV) { bestV = lp[off + v]; best = v; }
                codes[pred, j] = best;
            }
            logits.Dispose();
        }
        return codes;
    }

    private Tensor SumEmbeds(int[,] codes, int upTo, int t, int h)
    {
        // Sum the embeddings of all known codebooks [0, upTo).
        Tensor outT = new(new TensorShape(1, t, h), DType.F32);
        float* op = (float*)outT.DataPointer;
        for (int cb = 0; cb < upTo; cb++)
        {
            float* tab = (float*)_inputEmbeds[cb]!.DataPointer;
            for (int j = 0; j < t; j++)
            {
                int id = codes[cb, j];
                if ((uint)id >= (uint)_cfg.FineVocab) id = 0;
                float* row = tab + (long)id * h;
                long dst = (long)j * h;
                for (int c = 0; c < h; c++) op[dst + c] += row[c];
            }
        }
        return outT;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? e in _inputEmbeds) if (e is not null) yield return e;
        foreach (Tensor t in _backbone.EnumerateWeights()) yield return t;
        foreach (Tensor? hd in _outHeads) if (hd is not null) yield return hd;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _backbone.Dispose();
        GC.SuppressFinalize(this);
    }
}
