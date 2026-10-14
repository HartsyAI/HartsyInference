namespace HartsyInference.Audio.Sampling;

/// <summary>Shared end-of-generation gate for autoregressive audio decoders. A decoder that can sample its stop
/// token on step 0 emits an empty stream, which the downstream decoder cannot turn into audio. The gate keeps the
/// stop token out of the draw until a floor is reached, by writing <see cref="float.NegativeInfinity"/> into its
/// logit. Masking before the draw covers top-k, top-p and greedy argmax alike, and works with any number of stop
/// ids (Spark-TTS has two, and GPT-SoVITS's EOS is not guaranteed to be the last index).</summary>
/// <remarks>
/// The floor is <c>max(MinTokens, (int)(targetTextTokens × MinTokenTextRatio))</c>. A stop is allowed once the
/// generated count reaches the floor. Per-model values are set where the model's config is defined, and each one
/// records whether it comes from the reference implementation or was chosen because the reference has no gate.
/// </remarks>
public readonly record struct StopGate(int MinTokens, float MinTokenTextRatio = 0f)
{
    /// <summary>No floor: the stop token is always drawable. Used where a model's behavior must stay unchanged.</summary>
    public static StopGate Disabled => new(0, 0f);

    /// <summary>Minimum generated count before the stop token may be drawn.</summary>
    public int FloorFor(int targetTextTokens) =>
        Math.Max(MinTokens, (int)(targetTextTokens * MinTokenTextRatio));

    /// <summary>True once the stop token may be drawn after <paramref name="generated"/> tokens.</summary>
    public bool AllowStop(int generated, int targetTextTokens) => generated >= FloorFor(targetTextTokens);

    /// <summary>Writes -inf into every stop id's logit while the gate is closed. No-op once it is open, so the
    /// open path is bit-identical to an ungated draw.</summary>
    public void Apply(Span<float> logits, int generated, int targetTextTokens, ReadOnlySpan<int> stopIds)
    {
        if (AllowStop(generated, targetTextTokens)) return;
        Suppress(logits, stopIds);
    }

    /// <summary>Single-stop convenience for <see cref="Apply(Span{float}, int, int, ReadOnlySpan{int})"/>.</summary>
    public void Apply(Span<float> logits, int generated, int targetTextTokens, int stopId)
    {
        if (AllowStop(generated, targetTextTokens)) return;
        Suppress(logits, [stopId]);
    }

    /// <summary>Writes -inf into each stop id's logit. A stop id outside <paramref name="logits"/> can never be drawn
    /// from that head, so there is nothing to mask and it is skipped.</summary>
    public static void Suppress(Span<float> logits, ReadOnlySpan<int> stopIds)
    {
        foreach (int id in stopIds)
            if ((uint)id < (uint)logits.Length) logits[id] = float.NegativeInfinity;
    }
}
