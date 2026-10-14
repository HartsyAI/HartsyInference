namespace HartsyInference.Engine.Placement;

/// <summary>How a text model's weights are placed on the devices it runs on.</summary>
public enum TextPlacementMode
{
    /// <summary>Choose the fastest placement that fits: one GPU, then a layer split across GPUs, then expert offload.</summary>
    Auto,

    /// <summary>Everything on the request's GPU; refuse when it does not fit.</summary>
    Gpu,

    /// <summary>Layers split across the GPUs, in proportion to their free memory.</summary>
    Split,

    /// <summary>Dense weights and the hottest routed experts on the GPU; the other experts run on the CPU. MoE models only.</summary>
    Offload,
}

/// <summary>Spelling of <see cref="TextPlacementMode"/> on requests and settings.</summary>
public static class TextPlacementModes
{
    /// <summary>Parses <c>auto</c>, <c>gpu</c>, <c>split</c> or <c>offload</c>, any case; null or blank is <see cref="TextPlacementMode.Auto"/>.</summary>
    /// <exception cref="Core.Exceptions.HartsyInferenceException">Any other value.</exception>
    public static TextPlacementMode Parse(string? value) => value?.Trim().ToLowerInvariant() switch
    {
        null or "" or "auto" => TextPlacementMode.Auto,
        "gpu" => TextPlacementMode.Gpu,
        "split" => TextPlacementMode.Split,
        "offload" => TextPlacementMode.Offload,
        _ => throw new Core.Exceptions.HartsyInferenceException(
            $"Placement '{value}' is not one of auto, gpu, split or offload."),
    };
}

/// <summary>The device memory a text model needs, read from its checkpoint header before anything is mapped.</summary>
/// <param name="DenseBytes">Device bytes of every weight except the routed experts, as the load keeps them (quantized
/// where the device reads the format, widened to F32 where it does not, duplicated where the load keeps a fused copy).</param>
/// <param name="ExpertBytes">Device bytes of the routed experts; zero for a dense model.</param>
/// <param name="LayerCount">Transformer layers, for a layer split.</param>
/// <param name="KvBytesPerToken">KV-cache bytes one token adds across every layer.</param>
/// <param name="ContextTokens">Tokens the KV cache is sized for.</param>
/// <param name="PrefillBytesPerToken">Activation bytes one prompt token holds during prefill, when the whole prompt runs as one pass
/// (the residual, the normalized input, the attention and FFN intermediates). Zero when unknown.</param>
public readonly record struct TextPlacementDemand(long DenseBytes, long ExpertBytes, int LayerCount, long KvBytesPerToken,
    int ContextTokens, long PrefillBytesPerToken = 0)
{
    /// <summary>Whether the model has routed experts to offload.</summary>
    public bool IsMoe => ExpertBytes > 0;

    /// <summary>KV-cache bytes for the whole context.</summary>
    public long KvBytes => KvBytesPerToken * ContextTokens;

    /// <summary>Prefill activation bytes at the full context. A prompt as long as the context runs in one prefill pass, and its
    /// activations are live on the device while the KV cache fills, so they count beside the cache rather than after it.</summary>
    public long PrefillBytes => PrefillBytesPerToken * ContextTokens;

    /// <summary>Everything the model needs on one device.</summary>
    public long TotalBytes => DenseBytes + ExpertBytes + KvBytes + PrefillBytes;

    /// <summary>Whether the CPU can run the routed experts in place: every expert tensor in a format the packed CPU kernels read,
    /// with widths that fit its blocks. Offload needs it; the other placements do not.</summary>
    public bool ExpertsHostRunnable { get; init; } = true;

    /// <summary>Why <see cref="ExpertsHostRunnable"/> is false; null when it is true.</summary>
    public string? ExpertsHostReason { get; init; }
}

/// <summary>One candidate device: its selector (<c>cuda:0</c>) and the bytes it has free now.</summary>
public readonly record struct TextPlacementDevice(string Device, long FreeBytes);

/// <summary>The planner's decision.</summary>
/// <param name="Mode">The placement chosen; never <see cref="TextPlacementMode.Auto"/>.</param>
/// <param name="Feasible">False when nothing the mode allows fits; <paramref name="Reason"/> says which account failed.</param>
/// <param name="Devices">The devices the model runs on: one, or the stages of a split in order.</param>
/// <param name="ExpertBudgetBytes">Device bytes the expert cache may hold under <see cref="TextPlacementMode.Offload"/>;
/// zero otherwise.</param>
/// <param name="RequiredBytes">Bytes the placement needs on its devices, reserve included.</param>
/// <param name="AvailableBytes">Free bytes on those devices when it was planned.</param>
/// <param name="Reason">One user-facing line: why this placement, or why none fits.</param>
public sealed record TextPlacement(TextPlacementMode Mode, bool Feasible, IReadOnlyList<string> Devices, long ExpertBudgetBytes,
    long RequiredBytes, long AvailableBytes, string Reason)
{
    /// <summary>The device key a split loads under (<c>cuda:0+cuda:1</c>); the single device otherwise.</summary>
    public string DeviceKey => string.Join('+', Devices);
}

/// <summary>
/// Decides where a text model runs before any weight is uploaded. In <see cref="TextPlacementMode.Auto"/> it takes the first of:
/// everything on the primary GPU; a layer split across every candidate GPU; expert offload on the primary GPU (MoE models only).
/// Faster placements come first. A forced mode is checked alone and refused when it does not fit, so a request never runs out of
/// memory partway through loading.
/// </summary>
/// <remarks>Each device keeps <see cref="ReserveBytes"/> free for activations, workspaces and dequantized casts, the same order as
/// the layer-split planner's reserve. Expert offload also needs the experts it leaves on the host to stay readable, which a memory
/// map provides from the page cache, so host RAM is checked against the experts the device does not hold.</remarks>
public static class TextPlacementPlanner
{
    /// <summary>Device bytes held back on every device the model uses.</summary>
    public const long ReserveBytes = 1536L << 20;

    /// <summary>Fewest tokens a plan sizes the KV cache for: a loaded model serves later, longer requests than the one that loaded
    /// it.</summary>
    public const int MinContextTokens = 8192;

    /// <summary>Smallest expert cache worth offloading into: below this nearly every expert runs on the CPU.</summary>
    public const long MinExpertBudgetBytes = 512L << 20;

    /// <summary>Plans a placement.</summary>
    /// <param name="demand">What the model needs.</param>
    /// <param name="devices">Candidate GPUs; the first is the request's primary device.</param>
    /// <param name="hostFreeBytes">Free host RAM; null when unknown, which skips the host check.</param>
    /// <param name="mode">The requested mode.</param>
    public static TextPlacement Plan(TextPlacementDemand demand, IReadOnlyList<TextPlacementDevice> devices, long? hostFreeBytes,
        TextPlacementMode mode)
    {
        ArgumentNullException.ThrowIfNull(devices);
        if (devices.Count == 0) throw new ArgumentException("At least one device is required.", nameof(devices));
        TextPlacementDevice primary = devices[0];

        TextPlacement single = Single(demand, primary);
        if (mode == TextPlacementMode.Gpu || (mode == TextPlacementMode.Auto && single.Feasible)) return single;

        TextPlacement split = Split(demand, devices);
        if (mode == TextPlacementMode.Split) return split;
        if (mode == TextPlacementMode.Auto && split.Feasible && devices.Count > 1) return split;

        TextPlacement offload = Offload(demand, primary, hostFreeBytes);
        if (mode == TextPlacementMode.Offload) return offload;
        // Auto fell through to offload: say why the faster placements did not fit, so the reason names what would change it.
        if (offload.Feasible) return offload with { Reason = $"{offload.Reason} One GPU: {single.Reason} Split: {split.Reason}" };

        // Auto and nothing fits: report every account, so the reason names what would have to change.
        return offload with
        {
            Reason = $"No placement fits. One GPU: {single.Reason} Split: {split.Reason} Offload: {offload.Reason}",
        };
    }

    private static TextPlacement Single(TextPlacementDemand demand, TextPlacementDevice device)
    {
        long need = demand.TotalBytes + ReserveBytes;
        bool fits = need <= device.FreeBytes;
        string reason = fits
            ? $"Fits {device.Device}: needs {Gb(need)} of {Gb(device.FreeBytes)} free."
            : $"{device.Device} has {Gb(device.FreeBytes)} free; the model needs {Gb(need)}.";
        return new TextPlacement(TextPlacementMode.Gpu, fits, [device.Device], 0, need, device.FreeBytes, reason);
    }

    /// <summary>Every device keeps its own reserve, and the KV cache splits with the layers, so the sum is what must fit.</summary>
    /// <remarks>Approximate: each device is charged a share of the weights and KV cache proportional to its free memory, not the
    /// layer ranges <c>LlmSplitPlan</c> will give it, so an uneven split can still differ from this estimate by a layer's worth.
    /// Every device is charged the full prefill, which the stage runs on its own GPU.</remarks>
    private static TextPlacement Split(TextPlacementDemand demand, IReadOnlyList<TextPlacementDevice> devices)
    {
        string[] names = [.. devices.Select(static d => d.Device)];
        long free = devices.Sum(static d => d.FreeBytes);
        long need = demand.TotalBytes + ReserveBytes * devices.Count;
        if (devices.Count < 2)
        {
            return new TextPlacement(TextPlacementMode.Split, false, names, 0, need, free, "A split needs at least two GPUs.");
        }
        if (demand.LayerCount < devices.Count)
        {
            return new TextPlacement(TextPlacementMode.Split, false, names, 0, need, free,
                $"The model has {demand.LayerCount} layers, fewer than the {devices.Count} GPUs.");
        }
        // Every stage runs the whole prompt's prefill on its own device, so each device must hold its share of the weights and KV
        // cache plus a full prefill and its reserve. The share is proportional to free memory, as the layer split is. An aggregate
        // check alone would accept two cards whose sum covers the weights and one prefill, then run out of memory in prefill.
        if (free <= 0)
        {
            return new TextPlacement(TextPlacementMode.Split, false, names, 0, need, free, "The GPUs report no free memory.");
        }
        long shared = demand.DenseBytes + demand.ExpertBytes + demand.KvBytes;
        // The stage with the least room left decides the split; its own need and free are what the placement reports.
        TextPlacementDevice worst = devices[0];
        long worstNeed = 0, worstMargin = long.MinValue;
        foreach (TextPlacementDevice d in devices)
        {
            long stage = (long)(shared * (d.FreeBytes / (double)free)) + demand.PrefillBytes + ReserveBytes;
            if (stage - d.FreeBytes <= worstMargin) continue;
            worst = d;
            worstNeed = stage;
            worstMargin = stage - d.FreeBytes;
        }
        bool fits = worstMargin <= 0;
        string reason = fits
            ? $"Split across {string.Join(" and ", names)}: each stage fits its share, prefill included."
            : $"{worst.Device} has {Gb(worst.FreeBytes)} free; its share of the split plus the prefill needs {Gb(worstNeed)}.";
        return new TextPlacement(TextPlacementMode.Split, fits, names, 0, worstNeed, worst.FreeBytes, reason);
    }

    private static TextPlacement Offload(TextPlacementDemand demand, TextPlacementDevice device, long? hostFreeBytes)
    {
        long fixedNeed = demand.DenseBytes + demand.KvBytes + demand.PrefillBytes + ReserveBytes;
        if (!demand.IsMoe)
        {
            return new TextPlacement(TextPlacementMode.Offload, false, [device.Device], 0, fixedNeed, device.FreeBytes,
                "Expert offload needs a mixture-of-experts model; this one is dense.");
        }
        if (!demand.ExpertsHostRunnable)
        {
            return new TextPlacement(TextPlacementMode.Offload, false, [device.Device], 0, fixedNeed, device.FreeBytes,
                $"Expert offload runs missed experts on the CPU, and {demand.ExpertsHostReason}");
        }
        long budget = Math.Min(device.FreeBytes - fixedNeed, demand.ExpertBytes);
        if (budget < MinExpertBudgetBytes)
        {
            return new TextPlacement(TextPlacementMode.Offload, false, [device.Device], 0, fixedNeed + MinExpertBudgetBytes,
                device.FreeBytes,
                $"{device.Device} has {Gb(device.FreeBytes)} free; the dense weights and KV cache alone need {Gb(fixedNeed)}, "
                + $"plus at least {Gb(MinExpertBudgetBytes)} for experts.");
        }
        long hostExperts = demand.ExpertBytes - budget;
        if (hostFreeBytes is long host && hostExperts > host)
        {
            return new TextPlacement(TextPlacementMode.Offload, false, [device.Device], budget, fixedNeed + budget, device.FreeBytes,
                $"The {Gb(hostExperts)} of experts left on the CPU do not fit the {Gb(host)} of free host RAM.");
        }
        string experts = budget >= demand.ExpertBytes
            ? $"every expert ({Gb(budget)}) cached there"
            : $"{(double)budget / demand.ExpertBytes:P0} of the experts ({Gb(budget)}) cached there, the rest run on the CPU";
        return new TextPlacement(TextPlacementMode.Offload, true, [device.Device], budget, fixedNeed + budget, device.FreeBytes,
            $"Offload on {device.Device}: dense weights and KV cache on the GPU, {experts}.");
    }

    private static string Gb(long bytes) => $"{bytes / (double)(1L << 30):0.0} GB";
}
