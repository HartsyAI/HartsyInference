using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Condition features after the network's input projections (official <c>PreprocessedConditions</c>); they
/// do not depend on the latent or the timestep, so one instance serves every sampling step. All tensors are F32
/// <c>[B, T, hidden]</c> (the pooled conditions have <c>T = 1</c>) and owned by this object.</summary>
public sealed class ControlFoleyConditions : IDisposable
{
    private int _disposed;

    internal ControlFoleyConditions(Tensor clipF, Tensor syncF, Tensor textF, Tensor audioF, Tensor timbreF, Tensor clipFC,
        Tensor textFC)
    {
        ClipF = clipF;
        SyncF = syncF;
        TextF = textF;
        AudioF = audioF;
        TimbreF = timbreF;
        ClipFC = clipFC;
        TextFC = textFC;
    }

    /// <summary>Projected CLIP + visual tokens.</summary>
    public Tensor ClipF { get; }

    /// <summary>Projected Synchformer tokens resampled to the latent length.</summary>
    public Tensor SyncF { get; }

    /// <summary>Projected text tokens.</summary>
    public Tensor TextF { get; }

    /// <summary>Projected reference-audio token.</summary>
    public Tensor AudioF { get; }

    /// <summary>Pooled and projected timbre feature.</summary>
    public Tensor TimbreF { get; }

    /// <summary>Pooled and projected CLIP feature.</summary>
    public Tensor ClipFC { get; }

    /// <summary>Pooled and projected text feature.</summary>
    public Tensor TextFC { get; }

    /// <summary>Number of batch rows.</summary>
    public int Batch => (int)ClipF.Shape[0];

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        foreach (Tensor t in new[] { ClipF, SyncF, TextF, AudioF, TimbreF, ClipFC, TextFC })
        {
            t.Dispose();
        }
    }
}
