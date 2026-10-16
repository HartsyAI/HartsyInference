using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>MOSS-Image's per-step modulation: the six rows every joint block shares
/// (<c>shared_block_modulation(time)</c> as <c>[shift_a, scale_a, gate_a, shift_f, scale_f, gate_f]</c>) and the final
/// head's <c>[shift, scale]</c>. Each row is an F32 <c>[1, hidden]</c> device tensor sliced from its projection.</summary>
public sealed class MossImageModulation : IDisposable
{
    private readonly Tensor[] _rows;
    private int _disposed;

    private MossImageModulation(Tensor[] rows) => _rows = rows;

    /// <summary>Attention-branch shift.</summary>
    public Tensor ShiftAttn => _rows[0];

    /// <summary>Attention-branch raw scale (the <c>1 +</c> is applied by the consuming op).</summary>
    public Tensor ScaleAttn => _rows[1];

    /// <summary>Attention-branch residual gate.</summary>
    public Tensor GateAttn => _rows[2];

    /// <summary>FFN-branch shift.</summary>
    public Tensor ShiftFfn => _rows[3];

    /// <summary>FFN-branch raw scale.</summary>
    public Tensor ScaleFfn => _rows[4];

    /// <summary>FFN-branch residual gate.</summary>
    public Tensor GateFfn => _rows[5];

    /// <summary>Final head shift.</summary>
    public Tensor ShiftHead => _rows[6];

    /// <summary>Final head raw scale.</summary>
    public Tensor ScaleHead => _rows[7];

    /// <summary>Slices <paramref name="blockRows"/> <c>[1, 6·hidden]</c> and <paramref name="headRows"/>
    /// <c>[1, 2·hidden]</c> into their rows on the device. The sources stay owned by the caller.</summary>
    public static MossImageModulation Split(IBackend backend, Tensor blockRows, Tensor headRows, int hidden)
    {
        ArgumentNullException.ThrowIfNull(backend);
        Tensor[] rows = new Tensor[8];
        TensorShape row = new(1, hidden);
        for (int i = 0; i < 8; i++)
        {
            rows[i] = new Tensor(row, DType.F32);
            backend.SliceRows(rows[i], i < 6 ? blockRows : headRows, i < 6 ? i : i - 6);
        }
        return new MossImageModulation(rows);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (Tensor t in _rows) t.Dispose();
    }
}
