using System.Text.Json;
using HartsyInference.LLM.Transformer.Mhc;

namespace HartsyInference.LLM.Ssm;

/// <summary>The execution plan of a SenseNova-Loop checkpoint: which physical layers run twice and which of them carry the two-stream
/// manifold-constrained hyper-connection (mHC). Mirrors <c>execution_plan</c> in the checkpoint's <c>modeling_sensenova_loop.py</c>:
/// the same validation, and the same layer order (<c>[0, start)</c>, <c>[start, end)</c> repeated <see cref="Count"/> times, <c>[end, layers)</c>).</summary>
/// <remarks>Every execution slot owns its own attention cache and recurrent state; only the weights are shared between visits.
/// The reference rejects checkpoints it cannot run (a Loop gate, a variable count, an mHC range that crosses a loop boundary);
/// so does this reader.</remarks>
public sealed class SenseNovaLoopConfig
{
    /// <summary>First physical layer of the repeated block.</summary>
    public int Start { get; }

    /// <summary>One past the last physical layer of the repeated block.</summary>
    public int End { get; }

    /// <summary>Visits to the repeated block.</summary>
    public int Count { get; }

    /// <summary>Physical layers in the checkpoint.</summary>
    public int PhysicalLayers { get; }

    /// <summary>First physical layer with hyper-connections.</summary>
    public int MhcStart { get; }

    /// <summary>One past the last physical layer with hyper-connections.</summary>
    public int MhcEnd { get; }

    /// <summary>Residual streams (2 or 4).</summary>
    public int Streams { get; }

    /// <summary>Sinkhorn rounds on the stream-mixing matrix.</summary>
    public int Iterations { get; }

    /// <summary>Scale on the collapsed branch input.</summary>
    public float PreScale { get; }

    /// <summary>Execution slots: the physical layer each slot runs, in order.</summary>
    public IReadOnlyList<int> Plan { get; }

    /// <summary>Plan length (56 for SenseNova-Loop-9B).</summary>
    public int Slots => Plan.Count;

    /// <summary>Validates the settings exactly as the reference does and builds the plan.</summary>
    /// <exception cref="InvalidDataException">The settings describe something the reference does not run.</exception>
    public SenseNovaLoopConfig(int start, int end, int count, int physicalLayers, int mhcStart, int mhcEnd, int streams, int iterations, float preScale = 1f)
    {
        if (!(0 <= start && start < end && end <= physicalLayers && count >= 1))
        {
            throw new InvalidDataException($"Invalid Loop interval [{start}, {end}) x {count} over {physicalLayers} layers.");
        }
        if (streams is not (2 or 4) || iterations < 1 || !(0 <= mhcStart && mhcStart < mhcEnd && mhcEnd <= physicalLayers))
        {
            throw new InvalidDataException("Invalid mHC range, stream count or iteration count.");
        }
        if (!float.IsFinite(preScale) || preScale <= 0f)
        {
            throw new InvalidDataException("Invalid mHC pre_scale.");
        }
        int[] bounds = [0, start, end, physicalLayers];
        bool inOnePhase = false;
        for (int b = 0; b + 1 < bounds.Length; b++)
        {
            inOnePhase |= bounds[b] <= mhcStart && mhcEnd <= bounds[b + 1];
        }
        if (!inOnePhase)
        {
            throw new InvalidDataException("The mHC range cannot cross a Loop phase boundary.");
        }
        Start = start;
        End = end;
        Count = count;
        PhysicalLayers = physicalLayers;
        MhcStart = mhcStart;
        MhcEnd = mhcEnd;
        Streams = streams;
        Iterations = iterations;
        PreScale = preScale;
        List<int> plan = [];
        for (int p = 0; p < start; p++) plan.Add(p);
        for (int c = 0; c < count; c++)
            for (int p = start; p < end; p++) plan.Add(p);
        for (int p = end; p < physicalLayers; p++) plan.Add(p);
        Plan = plan;
    }

    /// <summary>True when execution slot <paramref name="slot"/> runs a layer with hyper-connections.</summary>
    public bool IsMhcSlot(int slot) => Plan[slot] >= MhcStart && Plan[slot] < MhcEnd;

    /// <summary>True when the slot is the first layer of the mHC range (the streams are created from the hidden state).</summary>
    public bool EntersStreams(int slot) => Plan[slot] == MhcStart;

    /// <summary>True when the slot is the last layer of the mHC range (the streams are averaged back).</summary>
    public bool LeavesStreams(int slot) => Plan[slot] == MhcEnd - 1;

    /// <summary>The slot that first runs physical layer <paramref name="physical"/> (where a GGUF stores its tensors).</summary>
    public int FirstSlot(int physical)
    {
        for (int s = 0; s < Plan.Count; s++)
        {
            if (Plan[s] == physical) return s;
        }
        throw new ArgumentOutOfRangeException(nameof(physical), physical, "No slot runs this layer.");
    }

    /// <summary>The hyper-connection formula of the checkpoint: no epsilons and no scales, the RMS epsilon inside the square root, post gates
    /// doubled, and the mix applied as <c>out[i] = sum_j mix[i, j] * stream[j]</c> (the reference's <c>matmul(mix, residual)</c>).</summary>
    public HyperConnectionConfig HyperConnection(int dim, float normEps) => new()
    {
        Hc = Streams,
        Dim = dim,
        SinkhornIters = Iterations,
        SinkhornEps = 0f,
        PreEps = 0f,
        PostMultiplier = 2f,
        NormEps = normEps,
        MixRms = RmsEpsPlacement.InsideSqrt,
        HeadRms = RmsEpsPlacement.InsideSqrt,
        TransposedMix = true,
    };

    /// <summary>Reads <c>loop_settings</c>, <c>mhc</c> and <c>text_config.num_hidden_layers</c> from a checkpoint's <c>config.json</c>; null when the
    /// file declares no Loop (a plain Qwen3.5).</summary>
    /// <exception cref="InvalidDataException">The Loop settings are unsupported.</exception>
    public static SenseNovaLoopConfig? FromJson(string json)
    {
        using JsonDocument doc = JsonDocument.Parse(json);
        JsonElement root = doc.RootElement;
        if (!root.TryGetProperty("loop_settings", out JsonElement loop) || loop.ValueKind != JsonValueKind.Object)
        {
            return null;
        }
        JsonElement text = root.TryGetProperty("text_config", out JsonElement t) ? t : root;
        int layers = text.GetProperty("num_hidden_layers").GetInt32();
        int count = loop.GetProperty("eval_count").GetInt32();
        if (loop.TryGetProperty("use_gate", out JsonElement gate) && gate.ValueKind == JsonValueKind.True)
        {
            throw new InvalidDataException("A Loop checkpoint with a gate is not supported.");
        }
        if (!loop.TryGetProperty("counts", out JsonElement counts) || counts.GetArrayLength() != 1 || counts[0].GetInt32() != count)
        {
            throw new InvalidDataException("Only fixed-count Loop checkpoints are supported.");
        }
        if (!root.TryGetProperty("mhc", out JsonElement mhc) || mhc.ValueKind != JsonValueKind.Object)
        {
            throw new InvalidDataException("This Loop implementation requires an mHC configuration.");
        }
        if (mhc.TryGetProperty("fixed_identity", out JsonElement fixedId) && fixedId.GetBoolean())
        {
            throw new InvalidDataException("mHC with a fixed identity mix is not supported.");
        }
        return new SenseNovaLoopConfig(
            loop.GetProperty("start").GetInt32(), loop.GetProperty("end").GetInt32(), count, layers,
            mhc.GetProperty("start").GetInt32(), mhc.GetProperty("end").GetInt32(),
            mhc.GetProperty("streams").GetInt32(), mhc.GetProperty("iterations").GetInt32(),
            mhc.TryGetProperty("pre_scale", out JsonElement scale) ? (float)scale.GetDouble() : 1f);
    }
}
