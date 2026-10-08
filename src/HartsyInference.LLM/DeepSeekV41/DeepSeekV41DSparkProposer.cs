using HartsyInference.LLM.Generation.Speculative;

namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>Drafts with the DSpark head over a sequence whose state records the target's draft rows. Each call syncs the state to the context, rebuilds the head's
/// window from the committed rows before the context's last token, and drafts from that token. The window therefore only ever holds committed positions, so a
/// rejection needs no window restore: the next call rebuilds it.</summary>
/// <remarks>The drafts are greedy, so the block carries no probabilities. The window is rebuilt on every call, one window of latents per stage, which is small
/// beside the target's own forward. The state must record the rows this head reads, and the head's window is only as good as those rows.</remarks>
internal sealed class DeepSeekV41DSparkProposer : IDraftProposer
{
    private readonly DeepSeekV41DSpark _draft;
    private readonly DeepSeekV41GenerationState _state;

    /// <param name="draft">The draft head; its input rows are the target's taps for its target layers.</param>
    /// <param name="state">The sequence the target runs on, created with <c>recordMainRows</c> so it keeps those taps. The scorer must share it.</param>
    public DeepSeekV41DSparkProposer(DeepSeekV41DSpark draft, DeepSeekV41GenerationState state)
    {
        ArgumentNullException.ThrowIfNull(draft);
        ArgumentNullException.ThrowIfNull(state);
        if (state.MainWidth != draft.InputWidth)
            throw new ArgumentException($"The state records {state.MainWidth} target values per position; the draft head reads {draft.InputWidth}.", nameof(state));
        _draft = draft;
        _state = state;
    }

    /// <inheritdoc />
    public DraftBlock Propose(ReadOnlySpan<int> context, int maxTokens)
    {
        if (context.IsEmpty) throw new ArgumentException("The context must hold at least one token.", nameof(context));
        if (maxTokens <= 0) return new DraftBlock([], null);

        _state.SyncTo(context);
        int anchor = context.Length - 1, width = _draft.InputWidth;
        float[] committed = new float[anchor * width];
        for (int p = 0; p < anchor; p++) _state.MainRow(p).CopyTo(committed.AsSpan(p * width, width));
        DeepSeekV41DSparkState window = _draft.CreateState(_state.Capacity);
        _draft.Seed(committed, anchor, window);
        DeepSeekV41DSparkDraft draft = _draft.Draft(context[anchor], _state.MainRow(anchor), anchor, window);

        // Ids[0] is the anchor itself; the rest are the drafted tokens
        int count = Math.Min(maxTokens, draft.Ids.Length - 1);
        return new DraftBlock(draft.Ids.AsSpan(1, count).ToArray(), null);
    }
}
