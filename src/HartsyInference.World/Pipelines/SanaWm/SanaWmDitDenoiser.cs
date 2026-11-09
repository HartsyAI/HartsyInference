using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.SanaWm;
using HartsyInference.World.Camera;

namespace HartsyInference.World.Pipelines.SanaWm;

/// <summary><see cref="ISanaWmDenoiser"/> over a loaded <see cref="SanaWmDit"/>. The caption, mask, camera conditions and
/// chunk Plücker of each branch layout are uploaded once and reused every step; only the latents move per call. Borrows the
/// DiT and the conditioning; disposing releases the uploaded copies.</summary>
public sealed class SanaWmDitDenoiser : ISanaWmDenoiser, IDisposable
{
    private readonly IBackend _backend;
    private readonly SanaWmDit _dit;
    private readonly SanaWmTextConditioning _text;
    private readonly SanaWmCameraConditioning _camera;
    private readonly Dictionary<int, BranchInputs> _inputs = [];
    private int _disposed;

    /// <summary>Binds <paramref name="dit"/> to one run's conditioning.</summary>
    public SanaWmDitDenoiser(IBackend backend, SanaWmDit dit, SanaWmTextConditioning text, SanaWmCameraConditioning camera)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(dit);
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(camera);
        _backend = backend;
        _dit = dit;
        _text = text;
        _camera = camera;
    }

    /// <inheritdoc/>
    public bool SupportsSingleBranch => true;

    /// <inheritdoc/>
    public Tensor Predict(Tensor latents, ReadOnlySpan<float> timesteps, ReadOnlySpan<SanaWmGuidanceBranch> branches)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(latents);
        if (latents.Shape.Rank != 5 || latents.Shape[0] != branches.Length || branches.IsEmpty)
            throw new ArgumentException($"Latents {latents.Shape} must hold one batch item per branch ({branches.Length}).", nameof(latents));

        BranchInputs inputs = Inputs(branches);
        using Tensor x = Upload(latents);
        using Tensor velocity = _dit.Forward(x, timesteps, inputs.Text, inputs.Mask, inputs.Conditions, inputs.Plucker);
        Tensor host = new(velocity.Shape, DType.F32, DeviceKind.Cpu);
        try
        {
            _backend.CopyTo(host, velocity);
            return host;
        }
        catch
        {
            host.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (BranchInputs inputs in _inputs.Values)
            inputs.Dispose();
        _inputs.Clear();
    }

    private BranchInputs Inputs(ReadOnlySpan<SanaWmGuidanceBranch> branches)
    {
        // A leading 1 bit keeps layouts of different lengths apart.
        int key = 1;
        foreach (SanaWmGuidanceBranch branch in branches)
            key = key * 2 + (branch == SanaWmGuidanceBranch.Conditional ? 1 : 0);
        if (_inputs.TryGetValue(key, out BranchInputs? cached))
            return cached;

        int batch = branches.Length;
        List<Tensor> built = [];
        try
        {
            Tensor text = Stack(branches, static (text, branch) => text.Features(branch), built);
            Tensor mask = Stack(branches, static (text, branch) => text.Mask(branch), built);
            Tensor plucker = Repeat(_camera.ChunkPlucker, batch, built);
            float[] conditions = new float[_camera.Conditions.Length * batch];
            for (int b = 0; b < batch; b++)
                _camera.Conditions.CopyTo(conditions, b * _camera.Conditions.Length);
            BranchInputs inputs = new(text, mask, plucker, conditions);
            _inputs[key] = inputs;
            return inputs;
        }
        catch
        {
            foreach (Tensor tensor in built)
                tensor.Dispose();
            throw;
        }
    }

    // Stacks one host tensor per branch along a new leading batch axis and uploads the result.
    private Tensor Stack(ReadOnlySpan<SanaWmGuidanceBranch> branches, Func<SanaWmTextConditioning, SanaWmGuidanceBranch, Tensor> select,
        List<Tensor> built)
    {
        Tensor first = select(_text, branches[0]);
        long[] dims = new long[first.Shape.Rank];
        for (int i = 0; i < dims.Length; i++)
            dims[i] = first.Shape[i];
        dims[0] = branches.Length;
        using Tensor host = new(new TensorShape(dims), DType.F32, DeviceKind.Cpu);
        Span<float> destination = host.AsSpan<float>();
        int length = (int)first.Shape.ElementCount;
        for (int b = 0; b < branches.Length; b++)
        {
            Tensor source = select(_text, branches[b]);
            if (source.Shape != first.Shape)
                throw new InvalidOperationException($"Caption tensors of the two branches differ in shape: {first.Shape} and {source.Shape}.");
            source.AsReadOnlySpan<float>().CopyTo(destination.Slice(b * length, length));
        }
        Tensor device = Upload(host);
        built.Add(device);
        return device;
    }

    private Tensor Repeat(Tensor single, int batch, List<Tensor> built)
    {
        long[] dims = new long[single.Shape.Rank];
        for (int i = 0; i < dims.Length; i++)
            dims[i] = single.Shape[i];
        dims[0] = batch;
        using Tensor host = new(new TensorShape(dims), DType.F32, DeviceKind.Cpu);
        Span<float> destination = host.AsSpan<float>();
        int length = (int)single.Shape.ElementCount;
        for (int b = 0; b < batch; b++)
            single.AsReadOnlySpan<float>().CopyTo(destination.Slice(b * length, length));
        Tensor device = Upload(host);
        built.Add(device);
        return device;
    }

    private Tensor Upload(Tensor host)
    {
        Tensor device = new(host.Shape, DType.F32, _backend.Device);
        try
        {
            _backend.CopyTo(device, host);
            return device;
        }
        catch
        {
            device.Dispose();
            throw;
        }
    }

    /// <summary>The device inputs of one branch layout.</summary>
    private sealed class BranchInputs(Tensor text, Tensor mask, Tensor plucker, float[] conditions) : IDisposable
    {
        public Tensor Text { get; } = text;

        public Tensor Mask { get; } = mask;

        public Tensor Plucker { get; } = plucker;

        public float[] Conditions { get; } = conditions;

        public void Dispose()
        {
            Text.Dispose();
            Mask.Dispose();
            Plucker.Dispose();
        }
    }
}
