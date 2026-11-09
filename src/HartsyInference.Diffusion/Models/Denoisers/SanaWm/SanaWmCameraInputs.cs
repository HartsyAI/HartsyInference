using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>The camera-branch inputs of one forward, built once and read by every block: the camera RoPE tables cut from the main
/// 3D table and, per batch item, the per-token ray matrices <c>P</c>, <c>Pᵀ</c> and <c>P⁻¹</c>. Owns its tensors.</summary>
internal sealed class SanaWmCameraInputs : IDisposable
{
    private readonly List<Tensor> _owned = [];
    private readonly Tensor[] _rays;
    private readonly Tensor[] _raysTransposed;
    private readonly Tensor[] _raysInverse;
    private int _disposed;

    private SanaWmCameraInputs(int batch)
    {
        _rays = new Tensor[batch];
        _raysTransposed = new Tensor[batch];
        _raysInverse = new Tensor[batch];
    }

    /// <summary>Camera RoPE cosines <c>[N, D/2]</c>, each angle written to both elements of its pair.</summary>
    public Tensor Cos { get; private set; } = null!;

    /// <summary>Camera RoPE sines in the same layout.</summary>
    public Tensor Sin { get; private set; } = null!;

    /// <summary>The negated sines, the inverse rotation the read-out applies.</summary>
    public Tensor NegSin { get; private set; } = null!;

    /// <summary>Batch items the rays cover.</summary>
    public int Batch => _rays.Length;

    /// <summary>Builds the inputs on <paramref name="backend"/>.</summary>
    /// <param name="mainCos">Per-complex main 3D RoPE cosines <c>[N, D/2]</c>; the camera columns are cut from it.</param>
    /// <param name="mainSin">Per-complex main sines in the same layout.</param>
    /// <param name="cameraConditions">Per batch item and frame, 16 camera-to-world floats then fx, fy, cx, cy in latent pixels.</param>
    public static SanaWmCameraInputs Create(IBackend backend, Tensor mainCos, Tensor mainSin, int headDim, ReadOnlySpan<float> cameraConditions,
        int batch, int frames, int height, int width)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (batch <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), batch, "Batch must be positive.");
        int itemConditions = frames * SanaWmCamRays.ConditionWidth;
        if (cameraConditions.Length != batch * itemConditions)
            throw new ArgumentException($"Camera conditions must hold {batch * itemConditions} floats; got {cameraConditions.Length}.",
                nameof(cameraConditions));

        SanaWmCameraInputs inputs = new(batch);
        bool built = false;
        try
        {
            (Tensor cos, Tensor sin) = SanaWmCamRope.BuildCameraTables(backend, mainCos, mainSin, headDim);
            inputs.Cos = inputs.Own(cos);
            inputs.Sin = inputs.Own(sin);
            inputs.NegSin = inputs.Own(new Tensor(sin.Shape, DType.F32));
            backend.Scale(inputs.NegSin, sin, -1f);

            int tokens = checked(frames * height * width);
            TensorShape rayShape = new(tokens, 4, 4);
            for (int b = 0; b < batch; b++)
            {
                Tensor p = new(rayShape, DType.F32);
                Tensor pt = new(rayShape, DType.F32);
                Tensor pi = new(rayShape, DType.F32);
                try
                {
                    SanaWmCamRays.Build(cameraConditions.Slice(b * itemConditions, itemConditions), 1, frames, height, width,
                        p.AsSpan<float>(), pt.AsSpan<float>(), pi.AsSpan<float>());
                }
                catch
                {
                    p.Dispose();
                    pt.Dispose();
                    pi.Dispose();
                    throw;
                }

                inputs._rays[b] = inputs.Own(SanaWmTensors.Stage(backend, p));
                inputs._raysTransposed[b] = inputs.Own(SanaWmTensors.Stage(backend, pt));
                inputs._raysInverse[b] = inputs.Own(SanaWmTensors.Stage(backend, pi));
            }

            built = true;
            return inputs;
        }
        finally
        {
            if (!built)
                inputs.Dispose();
        }
    }

    /// <summary>Per-token <c>P</c> of batch item <paramref name="item"/>, <c>[N, 4, 4]</c>.</summary>
    public Tensor Rays(int item) => _rays[item];

    /// <summary>Per-token <c>Pᵀ</c> of batch item <paramref name="item"/>.</summary>
    public Tensor RaysTransposed(int item) => _raysTransposed[item];

    /// <summary>Per-token <c>P⁻¹</c> of batch item <paramref name="item"/>.</summary>
    public Tensor RaysInverse(int item) => _raysInverse[item];

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        SanaWmTensors.DisposeAll(_owned);
    }

    private Tensor Own(Tensor tensor)
    {
        _owned.Add(tensor);
        return tensor;
    }
}
