using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>Output-space conversion for x0-predicting flow-match models (Chroma Radiance, Zeta-Chroma). The
/// flow-match Euler scheduler steps on velocity, so an x0 (clean-sample) prediction must be converted via
/// <c>v = (x_t − x0) / t</c> before CFG combine and the Euler step. Mirrors ComfyUI's <c>_apply_x0_residual</c>.</summary>
public static unsafe class X0Prediction
{
    /// <summary>Converts an x0 prediction to velocity: <c>v = (x_t − x0) / max(t, eps)</c>. Inputs must be F32 with
    /// identical shapes; neither is disposed. The eps floor only matters at t→0 (the fully-denoised end of the
    /// schedule, which samplers never evaluate exactly).</summary>
    /// <param name="x0">Model's clean-sample prediction.</param>
    /// <param name="noisySample">The sample the model was conditioned on this step (x_t).</param>
    /// <param name="t">Current flow-match time/sigma in [0, 1].</param>
    public static Tensor ToVelocity(Tensor x0, Tensor noisySample, float t, float eps = 1e-6f)
    {
        if (x0.DType != DType.F32 || noisySample.DType != DType.F32)
            throw new ArgumentException($"ToVelocity requires F32 inputs; got x0={x0.DType}, noisySample={noisySample.DType}.");
        if (!x0.Shape.Equals(noisySample.Shape))
            throw new ArgumentException($"ToVelocity shape mismatch: x0={x0.Shape}, noisySample={noisySample.Shape}.");

        Tensor velocity = new Tensor(x0.Shape, DType.F32);
        float invT = 1.0f / MathF.Max(t, eps);
        float* x0Ptr = (float*)x0.DataPointer;
        float* xtPtr = (float*)noisySample.DataPointer;
        float* outPtr = (float*)velocity.DataPointer;
        long count = velocity.Shape.ElementCount;
        for (long i = 0; i < count; i++)
            outPtr[i] = (xtPtr[i] - x0Ptr[i]) * invT;
        return velocity;
    }

    /// <summary>Device twin of <see cref="ToVelocity"/>: the same <c>v = (x_t − x0)/max(t, eps)</c> as three
    /// backend elementwise ops, so a device-resident denoise loop never reads a DataPointer (the host loop was a
    /// full pipeline drain + D2H of the x0 prediction every step). Neither input is disposed.</summary>
    public static Tensor ToVelocityDevice(IBackend backend, Tensor x0, Tensor noisySample, float t, float eps = 1e-6f)
    {
        if (x0.DType != DType.F32 || noisySample.DType != DType.F32)
            throw new ArgumentException($"ToVelocityDevice requires F32 inputs; got x0={x0.DType}, noisySample={noisySample.DType}.");
        if (!x0.Shape.Equals(noisySample.Shape))
            throw new ArgumentException($"ToVelocityDevice shape mismatch: x0={x0.Shape}, noisySample={noisySample.Shape}.");

        float invT = 1.0f / MathF.Max(t, eps);
        Tensor negX0 = new Tensor(x0.Shape, DType.F32);
        backend.Scale(negX0, x0, -invT);
        Tensor xOverT = new Tensor(x0.Shape, DType.F32);
        backend.Scale(xOverT, noisySample, invT);
        Tensor velocity = new Tensor(x0.Shape, DType.F32);
        backend.Add(velocity, xOverT, negX0);
        negX0.Dispose();
        xOverT.Dispose();
        return velocity;
    }
}
