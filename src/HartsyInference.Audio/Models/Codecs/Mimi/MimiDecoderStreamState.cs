namespace HartsyInference.Audio.Models.Codecs.Mimi;

/// <summary>All carried state for one streamed-decode utterance: the decoder transformer's growing K/V history plus the SEANet decoder's causal-conv/upsample-overlap tails.</summary>
/// <remarks>Create one per utterance, reuse it for every chunk of that utterance in order, dispose it when the utterance is done.</remarks>
public sealed class MimiDecoderStreamState : IDisposable
{
    internal MimiTransformerCache Transformer { get; } = new();
    internal MimiSeanetDecoderStreamState Decoder { get; } = new();
    private int _disposed;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        Transformer.Dispose();
        Decoder.Dispose();
    }
}
