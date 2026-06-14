namespace HartsyInference.Diffusion.Adapters;

/// <summary>Builds the per-cross-attention-layer scale array for an IP-Adapter conditioning, given the user's base scale and weight-type selection. Pipelines call this once per generation (the schedule is fixed across steps) and multiply by a step-fraction gate to get the final per-step scale array.
///
/// <para>Weight types map to per-layer scaling profiles. The exact profile is a heuristic: Cubiq's IPAdapterPlus distinguishes ~15 named schedules with hand-tuned coefficients for SD1.5 / SDXL block indices, but Swarm's UI surfaces only three options. Our mapping is documented per branch below; values are reasonable approximations of the named intent rather than byte-exact ports of any one reference impl. If a particular weight type produces visibly different output than Cubiq's, that's expected — the math is in the same family but the constants differ.</para></summary>
public static class IpAdapterScaleSchedule
{
    /// <summary>Build the per-cross-attn-layer scale array from a base scale and weight-type string. Returns an array of length <paramref name="layerCount"/> ready to pass into <c>UNet.Forward</c>'s <c>ipaScalePerLayer</c> parameter (after multiplying by any per-step gating factor).</summary>
    /// <param name="weightType">Swarm's <c>IPAdapterWeightType</c> dropdown value: <c>"standard"</c>, <c>"prompt is more important"</c>, or <c>"style transfer"</c>. Unknown values fall back to <c>"standard"</c>.</param>
    /// <param name="baseScale">The user's "IP-Adapter Weight" param. <c>1.0</c> = full strength.</param>
    /// <param name="layerCount">Total cross-attention sub-layer count (the UNet's <c>CrossAttentionLayerCount</c>). Length of the returned array.</param>
    public static float[] Build(string weightType, float baseScale, int layerCount)
    {
        if (layerCount <= 0)
        {
            return [];
        }
        float[] result = new float[layerCount];
        switch ((weightType ?? "standard").Trim().ToLowerInvariant())
        {
            case "standard":
            case "linear":
            default:
                // Uniform scale — every cross-attn layer gets baseScale.
                for (int i = 0; i < layerCount; i++) result[i] = baseScale;
                break;

            case "prompt is more important":
                // The "prompt is more important" intent: reduce IP-Adapter influence on the
                // layers that bind subject / composition (encoder + mid) so the prompt drives
                // those, and keep IPA influence on the layers that contribute style (decoder).
                // We approximate this by applying baseScale uniformly on the decoder half and
                // scaling encoder + mid down to ~0.4 × baseScale. Empirically the boundary is
                // around layerCount/3 — for SDXL's 70 cross-attns that puts 24 layers (down +
                // mid) at the reduced weight and 46 (up blocks) at full.
                {
                    int boundary = (layerCount + 2) / 3;
                    for (int i = 0; i < layerCount; i++)
                    {
                        result[i] = i < boundary ? baseScale * 0.4f : baseScale;
                    }
                }
                break;

            case "style transfer":
                // The "style transfer" intent: apply IPA only at the layers most associated
                // with style (rather than subject / composition). Cubiq's "style transfer
                // (SDXL)" enables block index 6 (the up-block at the deepest level), other
                // blocks at zero. Approximating: apply baseScale on the middle third of the
                // layer list, zero everywhere else. For SDXL's 70 layers that's roughly
                // layers 23..47 — covers mid block + the start of up-block 0 where style
                // features dominate.
                {
                    int third = layerCount / 3;
                    int start = third;
                    int end = layerCount - third;
                    for (int i = 0; i < layerCount; i++)
                    {
                        result[i] = (i >= start && i < end) ? baseScale : 0f;
                    }
                }
                break;
        }
        return result;
    }

    /// <summary>Compute the step-fraction gate for an adapter at the current step. Returns <c>1.0</c> if the current step is within the adapter's active window <c>[start, end]</c>; <c>0.0</c> otherwise. The pipeline multiplies the per-layer scale array by this gate to short-circuit IPA on out-of-window steps.</summary>
    /// <param name="currentStep">Zero-based index of the current step (or its float fraction equivalent).</param>
    /// <param name="totalSteps">Total step count of the schedule. Must be &gt; 0.</param>
    /// <param name="startFraction">Adapter's start fraction (0..1). Step is considered "in window" iff its fraction &gt;= start.</param>
    /// <param name="endFraction">Adapter's end fraction (0..1). Step is in window iff its fraction &lt;= end.</param>
    public static float StepGate(int currentStep, int totalSteps, float startFraction, float endFraction)
    {
        if (totalSteps <= 0) return 0f;
        float fraction = (float)currentStep / totalSteps;
        return (fraction >= startFraction && fraction <= endFraction) ? 1.0f : 0.0f;
    }
}
