using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.World.Pipelines.SanaWm;

/// <summary><see cref="ISanaWmRefinerTextEncoder"/> over a loaded Gemma-3-12B and LTX-2.0 text connector (upstream
/// <c>DiffusersLTX2Refiner._encode_prompt</c>): the trimmed prompt's tokens, BOS first, run through the tower; all of its
/// hidden states go to the connector, which pads the prompt to <see cref="SequenceLength"/> with its registers. Borrows the
/// tower, the tokenizer and the connector; their owner releases them.</summary>
/// <remarks>Upstream left-pads the tokens to the sequence length and masks the padding inside the tower. The tower's
/// rotation is relative and padding is never attended to, so the prompt's own tokens give the same hidden states, and
/// only those are encoded here.</remarks>
public sealed class SanaWmGemma3RefinerTextEncoder : ISanaWmRefinerTextEncoder
{
    /// <summary>The length upstream pads the refiner prompt to (<c>text_max_sequence_length</c>); longer prompts are cut.</summary>
    public const int SequenceLength = 1024;

    private readonly IBackend _backend;
    private readonly ILtx2TextTower _tower;
    private readonly GemmaTokenizer _tokenizer;
    private readonly LtxVideo2PackedTextConnector _connector;

    /// <summary>Wraps a loaded tower, its tokenizer (which must cut at <see cref="SequenceLength"/>) and the connector.</summary>
    public SanaWmGemma3RefinerTextEncoder(IBackend backend, ILtx2TextTower tower, GemmaTokenizer tokenizer,
        LtxVideo2PackedTextConnector connector)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(tower);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(connector);
        if (tokenizer.MaxLength > SequenceLength)
            throw new ArgumentException($"The refiner prompt is cut at {SequenceLength} tokens; the tokenizer cuts at "
                + $"{tokenizer.MaxLength}.", nameof(tokenizer));
        if (tower.NumLayers + 1 != connector.Config.States)
            throw new ArgumentException($"The connector takes {connector.Config.States} hidden states; the tower gives "
                + $"{tower.NumLayers + 1}.", nameof(connector));
        _backend = backend;
        _tower = tower;
        _tokenizer = tokenizer;
        _connector = connector;
    }

    /// <inheritdoc/>
    public Tensor Encode(string prompt)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        int[] tokens = _tokenizer.Encode(prompt.Trim());
        if (tokens.Length == 0)
            throw new ArgumentException("The refiner prompt has no tokens.", nameof(prompt));
        int[] states = new int[_tower.NumLayers + 1];
        for (int i = 0; i < states.Length; i++)
            states[i] = i;
        using Tensor hidden = _tower.EncodeMultiLayer(_backend, [tokens], states);
        return _connector.Forward(_backend, hidden, SequenceLength);
    }
}
