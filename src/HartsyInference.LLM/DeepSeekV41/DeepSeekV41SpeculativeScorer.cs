using HartsyInference.LLM.Generation.Speculative;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Scores a speculative draft with the V4.1 host model over a sequence state it owns. The context's new part is appended as one call (a prefill when the
/// state is empty, one token at a time otherwise, as decoding does), then the draft is appended, and the logits are read after the last context token and after each
/// drafted token.</summary>
/// <remarks>The state keeps the draft after the call, so the next call usually rolls back only the rejected part. That rollback is
/// <see cref="DeepSeekV41GenerationState.Truncate"/>, which replays the kept prefix, so a rejection costs a prefill of the context.</remarks>
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

        // roll back to the longest prefix of the context the state already holds; the last context token is always appended, so its logits can be read
        IReadOnlyList<int> held = _state.Tokens;
        int limit = Math.Min(held.Count, context.Length - 1), common = 0;
        while (common < limit && held[common] == context[common]) common++;
        if (common < _state.Length) _state.Truncate(common);

        int dim = _model.Dim;
        float[] contextHidden = new float[(context.Length - common) * dim];
        _state.Append(context[common..], contextHidden);
        float[] draftHidden = new float[draft.Length * dim];
        if (!draft.IsEmpty) _state.Append(draft, draftHidden);

        Read(contextHidden.AsSpan((context.Length - common - 1) * dim, dim), rows[0]);
        for (int j = 0; j < draft.Length; j++) Read(draftHidden.AsSpan(j * dim, dim), rows[j + 1]);
    }

    private void Read(ReadOnlySpan<float> hiddenRow, float[] row) => _model.Logits(hiddenRow).CopyTo(row, 0);
}
