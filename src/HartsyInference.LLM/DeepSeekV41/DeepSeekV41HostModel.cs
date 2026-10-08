using HartsyInference.LLM.DeepSeekV41.Engram;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>The V4.1 backbone on the host reference path: embedding, hyper-connection streams through every block, final collapse and norm, and the output head.</summary>
/// <remarks>Matches upstream <c>Transformer.forward</c> without vision, draft layers or sampling. Prefill from position 0 runs as one chunk; a later multi-token call is run one
/// token at a time, which is what attention requires.</remarks>
public sealed class DeepSeekV41HostModel
{
    private readonly int _dim;
    private readonly int _hc;
    private readonly int _vocab;
    private readonly float _normEps;
    private readonly DeepSeekV41Weight _embed;
    private readonly DeepSeekV41Block[] _blocks;
    private readonly float[] _finalNorm;
    private readonly DeepSeekV41Weight _head;
    private readonly EngramConstants? _engramConstants;

    /// <summary>Hidden width.</summary>
    public int Dim => _dim;

    /// <summary>Vocabulary size.</summary>
    public int VocabSize => _vocab;

    /// <summary>Number of blocks.</summary>
    public int Layers => _blocks.Length;

    /// <param name="dim">Hidden width.</param>
    /// <param name="hc">Residual copies.</param>
    /// <param name="vocab">Vocabulary size.</param>
    /// <param name="normEps">Config <c>rms_norm_eps</c>.</param>
    /// <param name="embed">Token embedding, <c>[vocab, dim]</c>.</param>
    /// <param name="blocks">The backbone blocks in order.</param>
    /// <param name="finalNorm">Norm weight after the final collapse, <c>[dim]</c>.</param>
    /// <param name="head">Output projection, <c>[vocab, dim]</c>.</param>
    /// <param name="engramConstants">Hash constants when any block has Engram, else null; null selects the shipped defaults for such a model.</param>
    public DeepSeekV41HostModel(int dim, int hc, int vocab, float normEps, DeepSeekV41Weight embed, DeepSeekV41Block[] blocks, float[] finalNorm, DeepSeekV41Weight head,
        EngramConstants? engramConstants = null)
    {
        ArgumentNullException.ThrowIfNull(embed);
        ArgumentNullException.ThrowIfNull(blocks);
        ArgumentNullException.ThrowIfNull(finalNorm);
        ArgumentNullException.ThrowIfNull(head);
        if (blocks.Length == 0) throw new ArgumentException("A model needs at least one block.", nameof(blocks));
        if (embed.Elements != (long)vocab * dim || head.Elements != (long)vocab * dim) throw new ArgumentException("embed and head must each be [vocab, dim].");
        if (finalNorm.Length != dim) throw new ArgumentException("finalNorm must hold dim values.", nameof(finalNorm));
        _dim = dim;
        _hc = hc;
        _vocab = vocab;
        _normEps = normEps;
        _embed = embed;
        _blocks = blocks;
        _finalNorm = finalNorm;
        _head = head;
        _engramConstants = engramConstants;
    }

    /// <summary>Host bytes one sequence of <paramref name="maxTokens"/> positions will hold across all layers.</summary>
    public long EstimateStateBytes(int maxTokens) => _blocks.Sum(b => DeepSeekV41AttentionState.EstimateBytes(b.AttentionSettings, maxTokens));

    /// <summary>Installs a diagnostic tap on every block (block index, stage name, values); null removes it. See <see cref="DeepSeekV41Block.Probe"/>.</summary>
    public void SetProbe(Action<int, string, float[]>? probe)
    {
        for (int i = 0; i < _blocks.Length; i++)
        {
            int layer = i;
            _blocks[i].Probe = probe is null ? null : (stage, values) => probe(layer, stage, values);
        }
    }

    /// <summary>Creates empty per-sequence state for up to <paramref name="maxTokens"/> positions.</summary>
    public DeepSeekV41SequenceState CreateState(int maxTokens)
    {
        DeepSeekV41AttentionState[] layers = new DeepSeekV41AttentionState[_blocks.Length];
        for (int i = 0; i < layers.Length; i++) layers[i] = new DeepSeekV41AttentionState(_blocks[i].AttentionSettings, maxTokens);
        EngramHasher? hasher = _blocks.Any(b => b.HasEngram) ? new EngramHasher(_engramConstants) : null;
        return new DeepSeekV41SequenceState(layers, hasher, maxTokens);
    }

    /// <summary>Runs <paramref name="ids"/> through the model and writes every position's final normed hidden state to <paramref name="hidden"/>.</summary>
    /// <param name="ids">Token ids; the first sits at <c>state.Length</c>.</param>
    /// <param name="state">The sequence's state; advanced by <c>ids.Length</c>.</param>
    /// <param name="hidden">Receives <c>[ids.Length, Dim]</c>.</param>
    public void Forward(ReadOnlySpan<int> ids, DeepSeekV41SequenceState state, Span<float> hidden)
    {
        ArgumentNullException.ThrowIfNull(state);
        if (ids.IsEmpty) throw new ArgumentException("Pass at least one token.", nameof(ids));
        if (hidden.Length != (long)ids.Length * _dim) throw new ArgumentException("hidden must hold ids x dim values.", nameof(hidden));
        if (state.Length + ids.Length > state.Capacity) throw new InvalidOperationException("The sequence state is full.");
        // checked before anything runs, so a bad id late in a multi-token call cannot leave the state half-advanced
        for (int i = 0; i < ids.Length; i++)
            if ((uint)ids[i] >= (uint)_vocab) throw new ArgumentOutOfRangeException(nameof(ids), ids[i], "Token id is outside the vocabulary.");
        if (state.Length > 0 && ids.Length > 1)
        {
            for (int i = 0; i < ids.Length; i++) Forward(ids.Slice(i, 1), state, hidden.Slice(i * _dim, _dim));
            return;
        }
        Chunk(ids, state, hidden);
    }

    /// <summary>The output logits of one hidden row, <c>[VocabSize]</c>.</summary>
    public float[] Logits(ReadOnlySpan<float> hiddenRow) => _head.Linear(hiddenRow, 1, _dim, _vocab);

    private void Chunk(ReadOnlySpan<int> ids, DeepSeekV41SequenceState state, Span<float> hidden)
    {
        int tokens = ids.Length, startPos = state.Length;
        float[] stream = new float[checked(tokens * _hc * _dim)];
        for (int t = 0; t < tokens; t++)
        {
            Span<float> first = stream.AsSpan(t * _hc * _dim, _dim);
            _embed.CopyRow(ids[t], first);
            for (int c = 1; c < _hc; c++) first.CopyTo(stream.AsSpan((t * _hc + c) * _dim, _dim));
        }

        long[] hashIds = [];
        int hashLayers = 0;
        if (state.Hasher is { } hasher)
        {
            hashLayers = hasher.ValuesPerPosition / EngramConstants.ColumnsPerLayer;
            hashIds = new long[checked(tokens * hasher.ValuesPerPosition)];
            bool[] live = new bool[tokens];
            Array.Fill(live, true);
            hasher.Hash(ids, live, startPos, hashIds);
        }

        // the first block collapses with a one-hot mix, which is just copy zero
        float[] preMix = new float[tokens * _hc], nextPreMix = new float[tokens * _hc];
        for (int t = 0; t < tokens; t++) preMix[t * _hc] = 1f;
        for (int i = 0; i < _blocks.Length; i++)
        {
            _blocks[i].Forward(stream, tokens, startPos, preMix, nextPreMix, state.Layers[i], state.Shared, hashIds, hashLayers, default, default);
            (preMix, nextPreMix) = (nextPreMix, preMix);
        }

        _blocks[^1].Collapse(stream, preMix, tokens, hidden);
        DeepSeekV41HostMath.RmsNormRows(hidden, _finalNorm, _dim, _normEps);
        state.Length += tokens;
    }
}
