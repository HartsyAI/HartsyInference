using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Auk;

/// <summary>AdaLN-Zero chunks (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) from <c>Linear(silu(t))</c>; the two scales already carry the +1.</summary>
public sealed class AukModulation : IDisposable
{
    public Tensor ShiftMsa { get; }
    public Tensor ScaleMsa { get; }
    public Tensor GateMsa { get; }
    public Tensor ShiftMlp { get; }
    public Tensor ScaleMlp { get; }
    public Tensor GateMlp { get; }

    /// <summary><paramref name="siluTime"/> is the <c>[1, 1, dim]</c> SiLU'd timestep embedding.</summary>
    public AukModulation(IBackend backend, Tensor siluTime, Tensor linearW, Tensor linearB, int dim)
    {
        Tensor mods = WhisperOps.ProjectLinear(backend, siluTime, linearW, linearB, 1, 1, dim, 6 * dim);
        ShiftMsa = Slice(backend, mods, 0, dim);
        ScaleMsa = Slice(backend, mods, 1, dim);
        GateMsa = Slice(backend, mods, 2, dim);
        ShiftMlp = Slice(backend, mods, 3, dim);
        ScaleMlp = Slice(backend, mods, 4, dim);
        GateMlp = Slice(backend, mods, 5, dim);
        mods.Dispose();
        backend.AddScalar(ScaleMsa, ScaleMsa, 1f);
        backend.AddScalar(ScaleMlp, ScaleMlp, 1f);
    }

    public void Dispose()
    {
        ShiftMsa.Dispose(); ScaleMsa.Dispose(); GateMsa.Dispose();
        ShiftMlp.Dispose(); ScaleMlp.Dispose(); GateMlp.Dispose();
    }

    private static Tensor Slice(IBackend backend, Tensor mods, int index, int dim)
    {
        Tensor slice = new(new TensorShape(1, 1, dim), DType.F32);
        backend.SliceLastDim(slice, mods, index * dim);
        return slice;
    }
}
