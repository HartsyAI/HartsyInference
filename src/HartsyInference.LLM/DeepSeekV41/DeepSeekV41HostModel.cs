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
    private readonly int[] _mainHiddenLayers;

    /// <summary>Hidden width.</summary>
    public int Dim => _dim;

    /// <summary>Vocabulary size.</summary>
    public int VocabSize => _vocab;

    /// <summary>Number of blocks.</summary>
    public int Layers => _blocks.Length;

    /// <summary>Block indices whose entry stream the DSpark draft reads, ascending in the order upstream concatenates them; empty when none.</summary>
    public IReadOnlyList<int> MainHiddenLayers => _mainHiddenLayers;

    /// <summary>Width of one main-hidden row: one hidden width per tapped layer.</summary>
    public int MainHiddenWidth => _mainHiddenLayers.Length * _dim;

    /// <param name="dim">Hidden width.</param>
    /// <param name="hc">Residual copies.</param>
    /// <param name="vocab">Vocabulary size.</param>
    /// <param name="normEps">Config <c>rms_norm_eps</c>.</param>
    /// <param name="embed">Token embedding, <c>[vocab, dim]</c>.</param>
    /// <param name="blocks">The backbone blocks in order.</param>
    /// <param name="finalNorm">Norm weight after the final collapse, <c>[dim]</c>.</param>
    /// <param name="head">Output projection, <c>[vocab, dim]</c>.</param>
    /// <param name="engramConstants">Hash constants when any block has Engram, else null; null selects the shipped defaults for such a model.</param>
    /// <param name="mainHiddenLayers">Blocks whose entry stream is captured for the DSpark draft, from the config's <c>dspark_target_layer_ids</c>; null or empty captures nothing.</param>
    public DeepSeekV41HostModel(int dim, int hc, int vocab, float normEps, DeepSeekV41Weight embed, DeepSeekV41Block[] blocks, float[] finalNorm, DeepSeekV41Weight head,
        EngramConstants? engramConstants = null, IReadOnlyList<int>? mainHiddenLayers = null)
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
        _mainHiddenLayers = NormalizeTapLayers(mainHiddenLayers, blocks.Length);
    }

    /// <summary>Distinct, ascending block indices; upstream captures each target layer once, in layer order, whatever order the config lists them in.</summary>
    private static int[] NormalizeTapLayers(IReadOnlyList<int>? layers, int blockCount)
    {
        if (layers is null || layers.Count == 0) return [];
        int[] sorted = layers.Distinct().Order().ToArray();
        if (sorted[0] < 0 || sorted[^1] >= blockCount) throw new ArgumentOutOfRangeException(nameof(layers), "A DSpark target layer lies outside the loaded blocks.");
        return sorted;
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
    public void Forward(ReadOnlySpan<int> ids, DeepSeekV41SequenceState state, Span<float> hidden) => Forward(ids, state, hidden, default);

    /// <summary>As <c>Forward(ids, state, hidden)</c>, and also writes the DSpark target rows when <paramref name="mainHidden"/> is given.</summary>
    /// <param name="ids">Token ids; the first sits at <c>state.Length</c>.</param>
    /// <param name="state">The sequence's state; advanced by <c>ids.Length</c>.</param>
    /// <param name="hidden">Receives <c>[ids.Length, Dim]</c>.</param>
    /// <param name="mainHidden">Receives <c>[ids.Length, MainHiddenWidth]</c>: per position, the hc-mean entry stream of each layer in <see cref="MainHiddenLayers"/>, in that order,
    /// as upstream's <c>main_hidden</c>. Empty when not wanted.</param>
    public void Forward(ReadOnlySpan<int> ids, DeepSeekV41SequenceState state, Span<float> hidden, Span<float> mainHidden)
    {
        ArgumentNullException.ThrowIfNull(state);
        if (ids.IsEmpty) throw new ArgumentException("Pass at least one token.", nameof(ids));
        if (hidden.Length != (long)ids.Length * _dim) throw new ArgumentException("hidden must hold ids x dim values.", nameof(hidden));
        if (!mainHidden.IsEmpty)
        {
            if (_mainHiddenLayers.Length == 0) throw new InvalidOperationException("This model has no DSpark target layers, so there are no main hidden rows to write.");
            if (mainHidden.Length != (long)ids.Length * MainHiddenWidth) throw new ArgumentException("mainHidden must hold ids x MainHiddenWidth values.", nameof(mainHidden));
        }
        if (state.Length + ids.Length > state.Capacity) throw new InvalidOperationException("The sequence state is full.");
        // checked before anything runs, so a bad id late in a multi-token call cannot leave the state half-advanced
        for (int i = 0; i < ids.Length; i++)
            if ((uint)ids[i] >= (uint)_vocab) throw new ArgumentOutOfRangeException(nameof(ids), ids[i], "Token id is outside the vocabulary.");
        if (state.Length > 0 && ids.Length > 1)
        {
            for (int i = 0; i < ids.Length; i++)
                Forward(ids.Slice(i, 1), state, hidden.Slice(i * _dim, _dim), mainHidden.IsEmpty ? default : mainHidden.Slice(i * MainHiddenWidth, MainHiddenWidth));
            return;
        }
        Chunk(ids, state, hidden, mainHidden);
    }

    /// <summary>The output logits of one hidden row, <c>[VocabSize]</c>.</summary>
    public float[] Logits(ReadOnlySpan<float> hiddenRow) => _head.Linear(hiddenRow, 1, _dim, _vocab);

    private void Chunk(ReadOnlySpan<int> ids, DeepSeekV41SequenceState state, Span<float> hidden, Span<float> mainHidden)
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
        // the DSpark tap reads each target block's entry stream after its Engram step, so it is taken inside the block
        float[] tapScratch = mainHidden.IsEmpty ? Array.Empty<float>() : new float[checked(tokens * _dim)];
        int tap = 0;
        for (int i = 0; i < _blocks.Length; i++)
        {
            if (!mainHidden.IsEmpty && tap < _mainHiddenLayers.Length && _mainHiddenLayers[tap] == i)
            {
                _blocks[i].Forward(stream, tokens, startPos, preMix, nextPreMix, state.Layers[i], state.Shared, hashIds, hashLayers, default, default, tapScratch);
                for (int t = 0; t < tokens; t++)
                    tapScratch.AsSpan(t * _dim, _dim).CopyTo(mainHidden.Slice(t * MainHiddenWidth + tap * _dim, _dim));
                tap++;
            }
            else
            {
                _blocks[i].Forward(stream, tokens, startPos, preMix, nextPreMix, state.Layers[i], state.Shared, hashIds, hashLayers, default, default, default);
            }
            (preMix, nextPreMix) = (nextPreMix, preMix);
        }

        _blocks[^1].Collapse(stream, preMix, tokens, hidden);
        DeepSeekV41HostMath.RmsNormRows(hidden, _finalNorm, _dim, _normEps);
        state.Length += tokens;
    }
}
