using HartsyInference.LLM.Generation.Speculative;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Scores a speculative draft with the V4.1 host model over a sequence state it owns. The context's new part is appended, then the draft, and the logits are
/// read after the last context token and after each drafted token.</summary>
/// <remarks>The host runs a sequence's first append as one prefill chunk and every later append of more than one token one token at a time, so the draft and an
/// accepted tail are computed exactly as plain decoding computes them. Only the prompt is a chunk.
/// The state keeps the draft after the call, so the next call rolls back only the rejected part, which <see cref="DeepSeekV41GenerationState.SyncTo"/> replays:
/// a prefill of the prompt, then one decode per token after it. A call whose context the state already holds replays nothing and runs only its draft.
/// A failed append resets the state, so the next call prefills the context again.</remarks>
internal sealed class DeepSeekV41SpeculativeScorer : ISpeculativeScorer
{
    private readonly DeepSeekV41HostModel _model;
    private readonly DeepSeekV41GenerationState _state;

    /// <param name="model">The model the state belongs to.</param>
    /// <param name="state">The sequence this scorer owns; it is left holding the context and the draft.</param>
    public DeepSeekV41SpeculativeScorer(DeepSeekV41HostModel model, DeepSeekV41GenerationState state)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(state);
        _model = model;
        _state = state;
    }

    /// <inheritdoc />
    public void Score(ReadOnlySpan<int> context, ReadOnlySpan<int> draft, float[][] rows)
    {
        if (context.IsEmpty) throw new ArgumentException("The context must hold at least one token.", nameof(context));
        if (rows.Length != draft.Length + 1) throw new ArgumentException("rows must hold draft.Length + 1 rows.", nameof(rows));
        // checked before the rollback, so a bad call leaves the state as it was
        if (context.Length + draft.Length > _state.Capacity) throw new InvalidOperationException("The context and the draft do not fit the sequence state.");
        foreach (float[] row in rows)
            if (row.Length != _model.VocabSize) throw new ArgumentException("Every row must hold one logit per vocabulary entry.", nameof(rows));

        _state.SyncTo(context);
        Read(_state.LastHidden, rows[0]);   // read before the draft moves the last row

        int dim = _model.Dim;
        float[] draftHidden = new float[draft.Length * dim];
        if (!draft.IsEmpty) _state.Append(draft, draftHidden);
        for (int j = 0; j < draft.Length; j++) Read(draftHidden.AsSpan(j * dim, dim), rows[j + 1]);
    }

    private void Read(ReadOnlySpan<float> hiddenRow, float[] row) => _model.Logits(hiddenRow).CopyTo(row, 0);
}
