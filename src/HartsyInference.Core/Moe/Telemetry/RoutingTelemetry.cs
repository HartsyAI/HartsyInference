using HartsyInference.Core.Backends;

namespace HartsyInference.Core.Moe.Telemetry;

/// <summary>
/// Fixed-capacity routing telemetry. Every array is allocated in the constructor; <see cref="Record"/> and
/// <see cref="BeginStep"/> never allocate. Aggregates are indexed by layer and expert within one bank; the ring keeps the
/// full <see cref="ExpertKey"/>. Single-threaded: one owner records.
/// </summary>
public sealed class RoutingTelemetry
{
    private readonly RoutingStepRecord[] ring;
    private readonly LayerRoutingCounters[] layers;
    private readonly long[] frequency;
    private readonly int layerCount;
    private readonly int expertsPerLayer;
    private long step;
    private int head;
    private int count;

    /// <summary>Creates telemetry for <paramref name="layerCount"/> layers of <paramref name="expertsPerLayer"/> experts.</summary>
    public RoutingTelemetry(int layerCount, int expertsPerLayer, int ringCapacity)
    {
        if (layerCount <= 0) throw new ArgumentOutOfRangeException(nameof(layerCount));
        if (expertsPerLayer <= 0) throw new ArgumentOutOfRangeException(nameof(expertsPerLayer));
        if (ringCapacity <= 0) throw new ArgumentOutOfRangeException(nameof(ringCapacity));

        this.layerCount = layerCount;
        this.expertsPerLayer = expertsPerLayer;
        ring = new RoutingStepRecord[ringCapacity];
        layers = new LayerRoutingCounters[layerCount];
        frequency = new long[checked((long)layerCount * expertsPerLayer)];
    }

    /// <summary>Number of layers this telemetry tracks.</summary>
    public int LayerCount => layerCount;

    /// <summary>Experts per layer this telemetry tracks.</summary>
    public int ExpertsPerLayer => expertsPerLayer;

    /// <summary>Capacity of the step ring.</summary>
    public int RingCapacity => ring.Length;

    /// <summary>Number of records currently retained in the ring.</summary>
    public int RingCount => count;

    /// <summary>The current step number; increments with <see cref="BeginStep"/>.</summary>
    public long CurrentStep => step;

    /// <summary>Marks the start of a new step (for example one decode token).</summary>
    public void BeginStep() => step++;

    /// <summary>
    /// Records one routed access. <paramref name="bytesMoved"/> is what was transferred for this access (zero on a hit).
    /// </summary>
    public void Record(ExpertKey key, bool hit, long bytesMoved, bool ranOnGpu)
    {
        if ((uint)key.Layer >= (uint)layerCount) throw new ArgumentOutOfRangeException(nameof(key), "Layer out of range.");
        if ((uint)key.Expert >= (uint)expertsPerLayer) throw new ArgumentOutOfRangeException(nameof(key), "Expert out of range.");
        if (bytesMoved < 0) throw new ArgumentOutOfRangeException(nameof(bytesMoved));

        RoutingEventFlags flags = RoutingEventFlags.None;
        if (hit) flags |= RoutingEventFlags.Hit;
        if (ranOnGpu) flags |= RoutingEventFlags.GpuRun;

        ring[head] = new RoutingStepRecord(step, key, bytesMoved, flags);
        head++;
        if (head == ring.Length) head = 0;
        if (count < ring.Length) count++;

        LayerRoutingCounters c = layers[key.Layer];
        layers[key.Layer] = c with
        {
            Routed = c.Routed + 1,
            Hits = c.Hits + (hit ? 1 : 0),
            Misses = c.Misses + (hit ? 0 : 1),
            BytesMoved = c.BytesMoved + bytesMoved,
            CpuRuns = c.CpuRuns + (ranOnGpu ? 0 : 1),
            GpuRuns = c.GpuRuns + (ranOnGpu ? 1 : 0),
        };
        frequency[((long)key.Layer * expertsPerLayer) + key.Expert]++;
    }

    /// <summary>Running totals for one layer.</summary>
    public LayerRoutingCounters GetLayer(int layer)
    {
        if ((uint)layer >= (uint)layerCount) throw new ArgumentOutOfRangeException(nameof(layer));
        return layers[layer];
    }

    /// <summary>How many times one expert has been routed since construction or the last <see cref="Reset"/>.</summary>
    public long Frequency(int layer, int expert)
    {
        if ((uint)layer >= (uint)layerCount) throw new ArgumentOutOfRangeException(nameof(layer));
        if ((uint)expert >= (uint)expertsPerLayer) throw new ArgumentOutOfRangeException(nameof(expert));
        return frequency[((long)layer * expertsPerLayer) + expert];
    }

    /// <summary>Reads the retained record at <paramref name="indexFromOldest"/>; zero is the oldest retained record.</summary>
    public bool TryGetRecent(int indexFromOldest, out RoutingStepRecord record)
    {
        if ((uint)indexFromOldest >= (uint)count)
        {
            record = default;
            return false;
        }

        int start = count < ring.Length ? 0 : head;
        int index = start + indexFromOldest;
        if (index >= ring.Length) index -= ring.Length;
        record = ring[index];
        return true;
    }

    /// <summary>Clears the ring, counters and frequencies. The step number restarts at zero.</summary>
    public void Reset()
    {
        Array.Clear(ring);
        Array.Clear(layers);
        Array.Clear(frequency);
        step = 0;
        head = 0;
        count = 0;
    }
}
