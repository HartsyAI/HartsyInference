using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Rotary position embedding: one generic rotation over strided views, the fused norm-then-rotate form, and the
// device-resident table a graph decode indexes.
public partial interface IBackend
{
    /// <summary>Rotary position embedding in place on <c>call.X</c>: for every pair the spec names, with its entries
    /// <c>(c0, s0)</c> and <c>(c1, s1)</c>, <c>x0' = x0·c0 − x1·s0</c> and <c>x1' = x1·c1 + x0·s1</c>, each product and
    /// sum rounded once. The <see cref="RotarySpec"/> views decide the pairing (interleaved or rotate-half), the rotated
    /// width and offset, the table layout and sharing (per position, per head, per grid axis), and a device position
    /// for a captured decode step; build them with <see cref="RotaryViews"/>. Contract: <c>docs/BACKEND_OPS.md</c>;
    /// <see cref="RotaryReference.Apply"/> is its executable form.</summary>
    void Rotary(in RotaryCall call) => RotaryReference.Apply(call);

    /// <summary>Full-row RMS norm of token-major <paramref name="input"/> <c>[seq, heads·headDim]</c>, then the rotation
    /// <paramref name="rotation"/> describes over those rows (skipped when <paramref name="cos"/> is null), emitted as
    /// <paramref name="output"/>: token-major <c>[seq, heads·headDim]</c>, or head-major <c>[1, heads, seq, headDim]</c>
    /// when the output has four axes. A backend fuses the three where it has a kernel for the rotation's geometry;
    /// the default composes <see cref="RmsNorm"/>, <see cref="Rotary"/> and <see cref="Permute0213"/>.</summary>
    void RmsNormRotary(Tensor output, Tensor input, Tensor normWeight, float eps, Tensor? cos, Tensor? sin, in RotarySpec rotation)
        => ComposeRmsNormRotary(this, output, input, normWeight, eps, cos, sin, rotation);

    /// <summary>The unfused form of <see cref="RmsNormRotary"/>, callable by a backend whose fused kernel does not cover a call.</summary>
    static void ComposeRmsNormRotary(IBackend backend, Tensor output, Tensor input, Tensor normWeight, float eps, Tensor? cos, Tensor? sin, in RotarySpec rotation)
    {
        if (input.Shape.Rank != 2)
            throw new ArgumentException($"RmsNormRotary takes token-major rows [seq, heads·headDim]; got {input.Shape}.", nameof(input));
        bool headMajor = output.Shape.Rank == 4;
        if (!headMajor && output.Shape != input.Shape)
            throw new ArgumentException($"RmsNormRotary output must be the input's shape or head-major [1, heads, seq, headDim]; got {output.Shape} for {input.Shape}.", nameof(output));
        Tensor rows = headMajor ? new Tensor(input.Shape, input.DType) : output;
        try
        {
            backend.RmsNorm(rows, input, normWeight, eps);
            if (cos is not null && sin is not null) backend.Rotary(new RotaryCall(rows, cos, sin, rotation));
            if (headMajor) backend.Permute0213(output, rows, (int)output.Shape[2], (int)output.Shape[1], (int)output.Shape[3]);
        }
        finally
        {
            if (headMajor) rows.Dispose();
        }
    }

    /// <summary>Precomputes the full RoPE cos/sin table <c>[maxPos,headDim]</c> device-resident, so a decode step only indexes a row.</summary>
    /// <param name="splitHalfPartial">Split-half vs. interleaved rotary pairing when rotaryDim &lt; headDim; ignored for full rotary.</param>
    (Tensor cos, Tensor sin) BuildRopeTableDevice(int maxPos, int headDim, int rotaryDim, float theta, HartsyInference.Core.Rope.RopeScaling scaling, bool splitHalfPartial = false)
        => throw new NotSupportedException("BuildRopeTableDevice requires GraphDecodeSupported.");
}
