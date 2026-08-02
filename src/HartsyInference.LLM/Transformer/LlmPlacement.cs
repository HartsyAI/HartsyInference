using HartsyInference.Core.Backends;

namespace HartsyInference.LLM.Transformer;

/// <summary>One contiguous decoder-layer range bound to one backend; ranges are [Start, End).</summary>
public sealed record LlmStage(IBackend Backend, int StartLayer, int EndLayer);

/// <summary>A resolved layer-split plan for one model: ordered stages whose ranges tile [0, layerCount).
/// Single-stage = today's single-device behavior. The embedding gather is host-side (stage-agnostic); the final
/// norm, lm_head and logits belong to the LAST stage's backend.</summary>
public sealed record LlmPlacement
{
    /// <summary>Ordered pipeline stages; ranges are contiguous, ascending, and tile the full layer stack.</summary>
    public IReadOnlyList<LlmStage> Stages { get; }

    public LlmPlacement(IReadOnlyList<LlmStage> stages)
    {
        ArgumentNullException.ThrowIfNull(stages);
        if (stages.Count == 0)
        {
            throw new ArgumentException("A placement needs at least one stage.", nameof(stages));
        }
        int expectedStart = 0;
        foreach (LlmStage stage in stages)
        {
            ArgumentNullException.ThrowIfNull(stage.Backend);
            if (stage.StartLayer != expectedStart || stage.EndLayer <= stage.StartLayer)
            {
                throw new ArgumentException(
                    $"Stage ranges must be contiguous ascending; got [{stage.StartLayer}, {stage.EndLayer}) after {expectedStart}.",
                    nameof(stages));
            }
            expectedStart = stage.EndLayer;
        }
        Stages = stages;
    }

    /// <summary>Single-device plan covering the whole stack — behaviorally identical to the unstaged path.</summary>
    public static LlmPlacement Single(IBackend backend, int layerCount) =>
        new LlmPlacement([new LlmStage(backend, 0, layerCount)]);

    /// <summary>True when one stage covers everything (no boundary handoffs).</summary>
    public bool IsSingle => Stages.Count == 1;

    /// <summary>The backend that owns the final norm, lm_head, logits, and sampling inputs.</summary>
    public IBackend LastBackend => Stages[^1].Backend;

    /// <summary>Total layers covered.</summary>
    public int LayerCount => Stages[^1].EndLayer;
}
