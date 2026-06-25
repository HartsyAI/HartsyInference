using System.Collections.Generic;
using HartsyInference.Audio.Models.Kyutai;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Validates the <see cref="KyutaiTextScheduler"/> (the DSM text state machine) against the moshi
/// reference: the same fixed words + model predictions must produce the same multiplexed fed tokens and end
/// step. Pure logic, checkpoint-free.</summary>
public sealed class KyutaiSchedulerTests
{
    [Fact]
    public void Scheduler_MatchesMoshiReference()
    {
        KyutaiTextScheduler sm = new(secondStreamAhead: 2, maxPadding: 6, initialPadding: 2);
        List<KyutaiTextScheduler.Entry> entries = new()
        {
            new(new[] { 101, 102 }, "hello", 0),
            new(new[] { 201 }, "there", 1),
            new(new[] { 301, 302, 303 }, "world", 0),
        };
        KyutaiTextScheduler.State state = sm.NewState(entries);

        int[] preds = { 3, 0, 3, 3, 0, 3, 0, 3, 3, 0, 0, 3, 3, 3, 0, 3, 3, 0, 3, 0, 3, 3, 3, 0, 3 };
        int[] expected = { 3, 3, 8102, 2416404, 8202, 2424306, 8302, 2432606, 303, 8004, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 };

        int[] outputs = new int[preds.Length];
        for (int step = 0; step < preds.Length; step++)
            outputs[step] = sm.Process(step, state, preds[step], out _);

        Assert.Equal(expected, outputs);
        Assert.Equal(9, state.EndStep);
    }
}
