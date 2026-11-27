using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>The intermediates of one SANA-WM step. Each is released as soon as its last reader has run, so the live set stays at
/// the step's working set; whatever is still held when the scope ends (an early exit or a failure) is disposed then.</summary>
internal sealed class SanaWmScratch : IDisposable
{
    private readonly List<Tensor> _tensors = [];

    /// <summary>A new tracked F32 tensor.</summary>
    public Tensor New(params long[] dims) => Track(new Tensor(new TensorShape(dims), DType.F32));

    /// <summary>A new tracked F32 tensor of <paramref name="shape"/>.</summary>
    public Tensor New(TensorShape shape) => Track(new Tensor(shape, DType.F32));

    /// <summary>A new tracked tensor of <paramref name="shape"/> and <paramref name="dtype"/>.</summary>
    public Tensor New(TensorShape shape, DType dtype) => Track(new Tensor(shape, dtype));

    /// <summary>Tracks <paramref name="tensor"/>, which this scope now owns.</summary>
    public Tensor Track(Tensor tensor)
    {
        _tensors.Add(tensor);
        return tensor;
    }

    /// <summary>Disposes a tracked tensor now.</summary>
    public void Release(Tensor tensor)
    {
        int index = _tensors.LastIndexOf(tensor);
        if (index < 0)
            throw new InvalidOperationException("Released a tensor this scope does not own.");
        _tensors.RemoveAt(index);
        tensor.Dispose();
    }

    /// <summary>Hands a tracked tensor to the caller, who disposes it.</summary>
    public Tensor Detach(Tensor tensor)
    {
        int index = _tensors.LastIndexOf(tensor);
        if (index < 0)
            throw new InvalidOperationException("Detached a tensor this scope does not own.");
        _tensors.RemoveAt(index);
        return tensor;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        foreach (Tensor tensor in _tensors)
            tensor.Dispose();
        _tensors.Clear();
    }
}
