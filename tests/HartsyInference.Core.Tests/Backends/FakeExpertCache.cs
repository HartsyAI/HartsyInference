using HartsyInference.Core.Backends;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>Records the transfer hooks so policy tests can assert order without a device.</summary>
internal sealed class FakeExpertCache : ExpertCacheBase
{
    public FakeExpertCache(long budgetBytes) : base(budgetBytes) { }

    public List<string> Events { get; } = [];
    public bool FencesComplete { get; set; } = true;
    public int LiveFences { get; private set; }
    public bool Drained { get; private set; }
    public ExpertKey? FailUploadOf { get; set; }
    public ExpertKey? FailAwaitOf { get; set; }
    public bool FailRecordFence { get; set; }

    /// <summary>Returns the exception an upload of the key should throw, or null to let it succeed.</summary>
    public Func<ExpertKey, Exception?>? UploadFault { get; set; }

    /// <summary>Returns the exception an await of the key should throw, or null to let it succeed.</summary>
    public Func<ExpertKey, Exception?>? AwaitFault { get; set; }

    /// <summary>Returns the exception an abandon of the key should throw, or null to let it succeed.</summary>
    public Func<ExpertKey, Exception?>? AbandonFault { get; set; }

    protected override object? BeginUpload(ExpertWeights weights)
    {
        if (FailUploadOf == weights.Key) throw new InvalidOperationException("upload failed");
        if (UploadFault?.Invoke(weights.Key) is { } fault) throw fault;
        Events.Add("upload " + weights.Key);
        return weights.Key;
    }

    protected override void AwaitUpload(object pending)
    {
        if (FailAwaitOf is { } key && Equals(pending, key)) throw new InvalidOperationException("await failed");
        if (AwaitFault?.Invoke((ExpertKey)pending) is { } awaitFault) throw awaitFault;
        Events.Add("await " + pending);
    }

    protected override void AbandonUpload(object pending)
    {
        Events.Add("abandon " + pending);
        if (AbandonFault?.Invoke((ExpertKey)pending) is { } fault) throw fault;
    }

    protected override void Evict(ExpertWeights weights) => Events.Add("evict " + weights.Key);

    protected override object RecordFence()
    {
        if (FailRecordFence) throw new InvalidOperationException("fence failed");
        LiveFences++;
        return new FenceBox { Done = FencesComplete };
    }

    protected override bool IsFenceDone(object fence) => ((FenceBox)fence).Done;

    protected override void WaitFence(object fence)
    {
        ((FenceBox)fence).Done = true;
        Events.Add("waitfence");
    }

    protected override void DestroyFence(object fence) => LiveFences--;

    protected override void Drain() => Drained = true;

    private sealed class FenceBox
    {
        public bool Done;
    }
}
