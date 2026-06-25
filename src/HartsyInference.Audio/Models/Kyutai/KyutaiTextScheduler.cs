using System.Collections.Generic;

namespace HartsyInference.Audio.Models.Kyutai;

/// <summary>The Kyutai/moshi TTS text state machine (Delayed Streams Modeling). Each frame the model predicts a
/// PAD or NEW_WORD special token; this machine decides the actual text token to feed next: it paces words with
/// padding, pops a word's SentencePiece tokens into the stream, and (for <c>second_stream_ahead</c>) multiplexes
/// a lookahead second stream into the fed token as <c>(second+1)·card + main</c> (the demuxing embedding splits
/// it again). Direct port of moshi <c>tts.py</c> StateMachine; pure logic, no tensors.</summary>
public sealed class KyutaiTextScheduler
{
    // TokenIds for tts-1.6b-en_fr.
    public const int Card = 8001, NewWord = 0, Pad = 3, Main = 1, Other = 2, Zero = -1;

    private readonly int _secondStreamAhead, _maxPadding, _initialPadding;

    public KyutaiTextScheduler(int secondStreamAhead = 2, int maxPadding = 6, int initialPadding = 2)
    {
        _secondStreamAhead = secondStreamAhead;
        _maxPadding = maxPadding;
        _initialPadding = initialPadding;
    }

    /// <summary>One word to synthesize: its SentencePiece <paramref name="Tokens"/> (empty = a pure pause), and
    /// <paramref name="Padding"/> forced-pad steps after it.</summary>
    public sealed record Entry(IReadOnlyList<int> Tokens, string Text, int Padding = 0);

    public sealed class State
    {
        public readonly Queue<Entry> Entries;
        public int RemainingPadding, ForcedPadding;
        public readonly Queue<int> Queued = new();
        public readonly Queue<int> LookaheadQueued = new();
        public int? EndStep;
        public State(IEnumerable<Entry> entries, int initialPadding)
        {
            Entries = new Queue<Entry>(entries);
            RemainingPadding = initialPadding;
            ForcedPadding = initialPadding;
        }
        public IReadOnlyList<int> TokensAhead(int lookahead)
        {
            foreach (Entry e in Entries)
                if (e.Tokens.Count > 0 && --lookahead == 0) return e.Tokens;
            return System.Array.Empty<int>();
        }
    }

    public State NewState(IEnumerable<Entry> entries) => new(entries, _initialPadding);

    /// <summary>Given the model's predicted <paramref name="token"/> at <paramref name="step"/>, advances
    /// <paramref name="state"/> and returns the (possibly multiplexed) text token to feed next.</summary>
    public int Process(int step, State state, int token, out bool consumedNewWord)
    {
        consumedNewWord = false;
        if (token != NewWord && token != Pad) token = Pad;

        if (state.Queued.Count > 0) token = Pad;
        else if (state.ForcedPadding > 0) token = Pad;
        else if (state.RemainingPadding <= 0) token = NewWord;

        if (token == NewWord)
        {
            if (state.Entries.Count > 0)
            {
                Entry entry = state.Entries.Dequeue();
                consumedNewWord = true;
                if (entry.Tokens.Count > 0)
                {
                    foreach (int t in entry.Tokens) state.Queued.Enqueue(t);
                    if (_secondStreamAhead > 0)
                        foreach (int t in state.TokensAhead(_secondStreamAhead)) state.LookaheadQueued.Enqueue(t);
                    state.RemainingPadding = _maxPadding;
                }
                else token = Pad;   // break-only entry
                state.ForcedPadding = entry.Padding;
            }
            else
            {
                token = Pad;
                if (_secondStreamAhead > 0 && state.EndStep is null) token = NewWord;
                state.EndStep ??= step;
            }
        }

        int output;
        if (token == Pad)
        {
            if (state.RemainingPadding > 0) state.RemainingPadding--;
            if (state.ForcedPadding > 0) state.ForcedPadding--;
            output = state.Queued.Count > 0 ? state.Queued.Dequeue() : Pad;
        }
        else if (token == NewWord) output = NewWord;
        else if (token == Zero) output = Zero;
        else throw new System.InvalidOperationException($"Invalid token {token}");

        if (_secondStreamAhead > 0)
        {
            int second = -1;
            if (output == NewWord)
            {
                second = NewWord;
                output = state.Queued.Count > 0 ? state.Queued.Dequeue() : Pad;
            }
            else if (state.LookaheadQueued.Count > 0) second = state.LookaheadQueued.Dequeue();
            output = (second + 1) * Card + output;   // multiplex; demuxed by the text embedding
        }
        return output;
    }
}
