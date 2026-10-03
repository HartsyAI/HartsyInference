using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.IndexTts;

/// <summary>IndexTTS-1.5's GPT speech-conditioning path (<c>condition_type: conformer_perceiver</c>): Conformer
/// encoder over the reference mel, then a Perceiver-Resampler down to 32 fixed-length continuous vectors that are
/// prepended to the GPT's text/mel token embeddings as a conditioning prefix.</summary>
internal sealed class IndexTtsSpeakerEncoder : IDisposable
{
    private readonly IndexTtsConformerEncoder _conformer;
    private readonly IndexTtsPerceiver _perceiver;
    private int _disposed;

    public IndexTtsSpeakerEncoder(IndexTtsConformerConfig conformerCfg, int gptHidden)
    {
        _conformer = new IndexTtsConformerEncoder(conformerCfg);
        _perceiver = new IndexTtsPerceiver(gptHidden, conformerCfg.OutputSize);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string conformerPrefix, string perceiverPrefix)
    {
        _conformer.LoadWeights(w, conformerPrefix);
        _perceiver.LoadWeights(w, perceiverPrefix);
    }

    /// <summary>Encodes a reference mel <c>[1, T, nMels]</c> into the 32-vector GPT conditioning prefix <c>[1, 32, gptHidden]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor referenceMel, int t)
    {
        Tensor conformerOut = _conformer.Forward(backend, referenceMel, t);
        int tOut = (int)conformerOut.Shape[1];
        Tensor prefix = _perceiver.Forward(backend, conformerOut, tOut);
        conformerOut.Dispose();
        return prefix;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _conformer.EnumerateWeights()) yield return t;
        foreach (Tensor t in _perceiver.EnumerateWeights()) yield return t;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _conformer.Dispose();
        _perceiver.Dispose();
        GC.SuppressFinalize(this);
    }
}
