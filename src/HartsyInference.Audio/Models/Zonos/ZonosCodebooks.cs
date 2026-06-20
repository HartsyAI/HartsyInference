using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Zonos;

/// <summary>Zonos's 9 per-codebook embedding tables (summed to form the decoder input) and 9 independent
/// output heads (stacked → <c>[9, OutputVocab]</c> logits). Input vocab 1026 (codes + EOS + masked), output
/// vocab 1025 (codes + EOS).</summary>
public sealed unsafe class ZonosCodebooks
{
    private readonly ZonosConfig _cfg;
    private readonly Tensor?[] _embed;
    private readonly Tensor?[] _head;

    public ZonosCodebooks(ZonosConfig cfg)
    {
        _cfg = cfg;
        _embed = new Tensor?[cfg.Channels];
        _head = new Tensor?[cfg.Channels];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        for (int k = 0; k < _cfg.Channels; k++)
        {
            _embed[k] = WhisperOps.EnsureF32(w[$"embeddings.{k}.weight"]);
            _head[k] = WhisperOps.EnsureF32(w[$"heads.{k}.weight"]);
        }
    }

    /// <summary>Sum of the 9 codebooks' embeddings for one frame → <c>[1,1,hidden]</c>.</summary>
    public Tensor EmbedFrame(ReadOnlySpan<int> codes)
    {
        int h = _cfg.Hidden;
        Tensor outT = new(new TensorShape(1, 1, h), DType.F32);
        float* op = (float*)outT.DataPointer;
        for (int k = 0; k < _cfg.Channels; k++)
        {
            float* row = (float*)_embed[k]!.DataPointer + (long)codes[k] * h;
            for (int i = 0; i < h; i++) op[i] += row[i];
        }
        return outT;
    }

    /// <summary>Applies the 9 heads to a final hidden <c>[1,1,hidden]</c> → per-codebook logits arrays.</summary>
    public float[][] Heads(IBackend backend, Tensor hidden)
    {
        float[][] logits = new float[_cfg.Channels][];
        for (int k = 0; k < _cfg.Channels; k++)
        {
            Tensor l = WhisperOps.ProjectLinear(backend, hidden, _head[k]!, null, 1, 1, _cfg.Hidden, _cfg.OutputVocab);
            float[] arr = new float[_cfg.OutputVocab];
            new Span<float>((void*)l.DataPointer, _cfg.OutputVocab).CopyTo(arr);
            l.Dispose();
            logits[k] = arr;
        }
        return logits;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in _embed) if (t is not null) yield return t;
        foreach (Tensor? t in _head) if (t is not null) yield return t;
    }
}
