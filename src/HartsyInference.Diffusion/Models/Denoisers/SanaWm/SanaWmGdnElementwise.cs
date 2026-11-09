using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>Elementwise <c>exp</c> and reciprocal that IBackend does not provide. Host-only until backends gain kernels;
/// device backends refuse rather than read device memory on the host.</summary>
internal static class SanaWmGdnElementwise
{
    internal static void Exp(IBackend backend, Tensor output, Tensor input)
    {
        RequireHost(backend, "Exp");
        RequireSameShape(output, input);
        ReadOnlySpan<float> src = input.AsReadOnlySpan<float>();
        Span<float> dst = output.AsSpan<float>();
        for (int i = 0; i < src.Length; i++)
        {
            dst[i] = MathF.Exp(src[i]);
        }
    }

    internal static void Reciprocal(IBackend backend, Tensor output, Tensor input)
    {
        RequireHost(backend, "Reciprocal");
        RequireSameShape(output, input);
        ReadOnlySpan<float> src = input.AsReadOnlySpan<float>();
        Span<float> dst = output.AsSpan<float>();
        for (int i = 0; i < src.Length; i++)
        {
            dst[i] = 1f / src[i];
        }
    }

    private static void RequireHost(IBackend backend, string op)
    {
        if (!backend.Device.IsCpu)
        {
            throw new NotSupportedException($"{op} has no IBackend kernel yet; SANA-WM GDN elementwise math runs on the CPU backend only.");
        }
    }

    private static void RequireSameShape(Tensor output, Tensor input)
    {
        if (output.ElementCount != input.ElementCount || output.DType != DType.F32 || input.DType != DType.F32)
        {
            throw new ArgumentException("Elementwise helper needs F32 tensors of equal element count.");
        }
    }
}
