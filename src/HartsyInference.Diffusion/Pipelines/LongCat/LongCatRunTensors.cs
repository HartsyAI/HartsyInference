using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Pipelines.LongCat;

/// <summary>The device tensors a LongCat run feeds to every denoise pass, built once per prompt and once per audio array and kept alive for the
/// run. The transformer's text projection cache is keyed on the encoder tensor's identity, so a tensor built per pass would miss it on every pass.
/// Disposing this releases the tensors; it belongs to the denoise loop that created it.</summary>
internal sealed class LongCatRunTensors(int captionChannels, int audioWidth) : IDisposable
{
    private readonly Dictionary<LongCatPrompt, Tensor> _texts = new(ReferenceEqualityComparer.Instance);
    private readonly Dictionary<float[], Tensor> _audio = new(ReferenceEqualityComparer.Instance);

    /// <summary>The prompt's umT5 embeddings as a <c>[Length, captionChannels]</c> tensor, the same object for every pass of the run.</summary>
    public Tensor Text(LongCatPrompt prompt)
    {
        if (_texts.TryGetValue(prompt, out Tensor? text))
            return text;
        text = new Tensor(new TensorShape(prompt.Length, captionChannels), DType.F32);
        prompt.Embeds.AsSpan().CopyTo(text.AsSpan<float>());
        _texts[prompt] = text;
        return text;
    }

    /// <summary>The audio tokens of <paramref name="rows"/> rows, the same object for every pass that reads <paramref name="audio"/>.</summary>
    public Tensor Audio(float[] audio, int rows)
    {
        if (_audio.TryGetValue(audio, out Tensor? tokens))
            return tokens;
        tokens = new Tensor(new TensorShape(rows, audioWidth), DType.F32);
        audio.AsSpan().CopyTo(tokens.AsSpan<float>());
        _audio[audio] = tokens;
        return tokens;
    }

    public void Dispose()
    {
        foreach (Tensor text in _texts.Values)
            text.Dispose();
        foreach (Tensor tokens in _audio.Values)
            tokens.Dispose();
        _texts.Clear();
        _audio.Clear();
    }
}
