using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Utilities;

/// <summary>Centralizes the "cast a tensor to a target dtype if it isn't already" pattern duplicated across every pipeline that crosses an F16 ↔ F32 (or BF16) boundary — typically: UNet/transformer F16 output → F32 for scheduler arithmetic, or F32 latent → VAE-weights' dtype before decode. Routes through <see cref="IBackend"/> cast methods so GPU-resident tensors stay on-device.</summary>
public static class DtypeCastHelper
{
    /// <summary>Returns a tensor of <paramref name="target"/> dtype. If <paramref name="input"/> already matches, returns it unchanged (caller still owns it). Otherwise allocates a new tensor, dispatches to the backend-appropriate cast op, and — when <paramref name="disposeSourceOnCast"/> is true (default) — disposes the source so the caller doesn't have to track whether a cast happened.
    /// <para>Pattern: <c>x = DtypeCastHelper.EnsureDtype(backend, x, DType.F32);</c> after which <c>x</c> is always F32 and the old reference is gone if it differed.</para>
    /// </summary>
    public static Tensor EnsureDtype(IBackend backend, Tensor input, DType target, bool disposeSourceOnCast = true)
    {
        if (input.DType == target) return input;
        Tensor output = new Tensor(input.Shape, target);
        if (target == DType.F16) backend.CastToF16(output, input);
        else if (target == DType.F32) backend.CastToF32(output, input);
        else if (target == DType.BF16) backend.CastToBf16(output, input);
        else
        {
            output.Dispose();
            throw new InvalidOperationException($"DtypeCastHelper.EnsureDtype does not support target dtype {target}. Supported: F16, F32, BF16.");
        }
        if (disposeSourceOnCast) input.Dispose();
        return output;
    }

    /// <summary>Shorthand for <c>EnsureDtype(backend, input, DType.F32)</c>. By far the most common usage — UNet/transformer outputs come back in F16 on most pipelines, and scheduler arithmetic + CFG combine run in F32.</summary>
    public static Tensor EnsureF32(IBackend backend, Tensor input, bool disposeSourceOnCast = true)
        => EnsureDtype(backend, input, DType.F32, disposeSourceOnCast);
}
