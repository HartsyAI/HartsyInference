using HartsyInference.Core.Backends;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

/// <summary>
/// Runs one planned expert on the CUDA device from its pinned resident copy. Synchronous and per expert: each <see cref="Run"/>
/// uploads the input, runs the expert and downloads the output before it returns.
/// </summary>
/// <remarks>
/// The lease is the planner's: <see cref="ExpertScheduler.Plan"/> pins the resident experts through it, so the weights stay on the
/// device until the lease is disposed. An expert the lease does not hold is never uploaded here; <see cref="Run"/> throws before any
/// device work. The weights must be F32 and unquantized, and must each have a device copy, which is what the planner's pin guarantees.
/// </remarks>
public sealed class CudaExpertDeviceRunner : IExpertDeviceRunner
{
    private readonly CudaBackend _backend;
    private readonly CudaExpertKernels _kernels;
    private readonly ExpertLease _lease;
    private readonly ExpertProgram _program;
    private readonly int _hidden;
    private readonly int _intermediate;

    /// <summary>Creates a runner for one layer's pinned experts.</summary>
    /// <param name="backend">The CUDA backend whose streaming cache holds the resident weights and whose compute stream runs the expert.</param>
    /// <param name="kernels">The expert FFN kernels, loaded into <paramref name="backend"/>'s context.</param>
    /// <param name="lease">The planner's lease, holding every expert this runner may be asked to run.</param>
    /// <param name="program">Activation and clamp bounds of the layer.</param>
    /// <param name="hidden">Model width H.</param>
    /// <param name="intermediate">Expert width I.</param>
    public CudaExpertDeviceRunner(CudaBackend backend, CudaExpertKernels kernels, ExpertLease lease, ExpertProgram program,
        int hidden, int intermediate)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _kernels = kernels ?? throw new ArgumentNullException(nameof(kernels));
        _lease = lease ?? throw new ArgumentNullException(nameof(lease));
        _program = program ?? throw new ArgumentNullException(nameof(program));
        _program.Validated();
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hidden);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(intermediate);
        _hidden = hidden;
        _intermediate = intermediate;
    }

    /// <inheritdoc/>
    /// <exception cref="InvalidOperationException">
    /// <paramref name="key"/> is not pinned by the lease, or a weight has no device copy. Nothing has been uploaded or launched.
    /// </exception>
    /// <exception cref="NotSupportedException">A weight is quantized or not F32.</exception>
    public void Run(ExpertKey key, ReadOnlySpan<float> x, int rows, Span<float> y)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        long elements = (long)rows * _hidden;
        if (x.Length != elements) throw new ArgumentException($"x must hold {rows} rows of {_hidden}.", nameof(x));
        if (y.Length != elements) throw new ArgumentException($"y must hold {rows} rows of {_hidden}.", nameof(y));

        // Every check that can fail runs before the first allocation or launch.
        ExpertWeights weights = Pinned(key);
        CudaExpertKernels.Buffers resident = new(
            Device(key, weights.W1, (long)_intermediate * _hidden),
            Device(key, weights.W3, (long)_intermediate * _hidden),
            Device(key, weights.W2, (long)_hidden * _intermediate),
            0, 0, 0, 0, 0);

        nuint inputBytes = (nuint)(elements * sizeof(float));
        nuint hiddenBytes = (nuint)((long)rows * _intermediate * sizeof(float));
        ulong xd = 0, gd = 0, ud = 0, hd = 0, yd = 0;
        nint stream = _backend.Stream.Handle;
        try
        {
            xd = CudaMemory.Allocate(inputBytes);
            gd = CudaMemory.Allocate(hiddenBytes);
            ud = CudaMemory.Allocate(hiddenBytes);
            hd = CudaMemory.Allocate(hiddenBytes);
            yd = CudaMemory.Allocate(inputBytes);
            unsafe
            {
                fixed (float* px = x) CudaMemory.CopyHostToDeviceAsync(xd, px, inputBytes, stream);
            }
            CudaExpertKernels.Buffers buffers = resident with { X = xd, Gate = gd, Up = ud, Hidden = hd, Y = yd };
            _kernels.RunExpert(buffers, rows, _hidden, _intermediate, _program, stream);
            unsafe
            {
                fixed (float* py = y) CudaMemory.CopyDeviceToHostAsync(py, yd, inputBytes, stream);
            }
            _backend.Sync();
        }
        finally
        {
            foreach (ulong buffer in new[] { xd, gd, ud, hd, yd })
            {
                if (buffer != 0) CudaMemory.Free(buffer);
            }
        }
    }

    private ExpertWeights Pinned(ExpertKey key)
    {
        try
        {
            return _lease.Get(key);
        }
        catch (KeyNotFoundException error)
        {
            throw new InvalidOperationException($"{key} is not pinned by the planner's lease, so it is not resident and was not uploaded.", error);
        }
    }

    private static ulong Device(ExpertKey key, ExpertMatrix matrix, long expectedElements)
    {
        if (matrix.Recipe is not null)
            throw new NotSupportedException($"{key}: quantized expert weights are not supported by the CUDA expert runner yet.");
        Tensor weight = matrix.Weight;
        if (weight.DType != DType.F32) throw new NotSupportedException($"{key}: the CUDA expert runner needs F32 weights, got {weight.DType}.");
        if (weight.ElementCount != expectedElements)
            throw new ArgumentException($"{key}: expected {expectedElements} weight elements, got {weight.ElementCount}.");
        if (!GpuTransferHelper.TryGetCachedDevice(weight, out ulong device))
            throw new InvalidOperationException($"{key} has no device copy; the planner must pin it before the device runs it.");
        return device;
    }
}
