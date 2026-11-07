using HartsyInference.Core.Backends.Norm;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Host;

/// <summary>The executable form of the chunk-activate contract: input rows <c>[rows, count · width]</c> cut into <c>count</c> outputs
/// <c>[rows, width]</c>, chunk <c>c</c> passed through its <see cref="ChunkActivation"/>. The <see cref="IBackend"/> default and the
/// oracle every kernel is checked against.</summary>
public static unsafe class ChunkActivateReference
{
    /// <summary>Most chunks one call cuts.</summary>
    public const int MaxChunks = 8;

    /// <summary>Validates the operands and returns the row count and the chunk width.</summary>
    public static (long Rows, int Width) Validate(ReadOnlySpan<Tensor> outputs, Tensor input, ReadOnlySpan<ChunkActivation> activations)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (outputs.Length == 0 || outputs.Length > MaxChunks || outputs.Length != activations.Length)
            throw new ArgumentException($"ChunkActivate takes 1 to {MaxChunks} outputs with one activation each; got {outputs.Length} outputs and {activations.Length} activations.", nameof(outputs));
        if (input.DType != DType.F32)
            throw new NotSupportedException($"ChunkActivate takes an F32 input; got {input.DType}.");
        long packed = input.Shape[input.Shape.Rank - 1];
        if (packed % outputs.Length != 0)
            throw new ArgumentException($"The input's last extent {packed} is not {outputs.Length} chunks.", nameof(input));
        int width = (int)(packed / outputs.Length);
        long rows = input.ElementCount / packed;
        for (int c = 0; c < outputs.Length; c++)
        {
            Tensor output = outputs[c] ?? throw new ArgumentNullException(nameof(outputs), $"Output {c} is null.");
            if (output.DType != DType.F32 || output.ElementCount != rows * width)
                throw new ArgumentException($"Output {c} holds {rows} rows of {width} F32 elements; got {output.DType} {output.Shape}.", nameof(outputs));
            if (output.HasOverlappingHostStorageWithoutSync(input))
                throw new ArgumentException($"Output {c} overlaps the input.", nameof(outputs));
            if (!Enum.IsDefined(activations[c]))
                throw new ArgumentOutOfRangeException(nameof(activations), activations[c], $"Unknown activation for chunk {c}.");
        }
        return (rows, width);
    }

    public static void Apply(ReadOnlySpan<Tensor> outputs, Tensor input, ReadOnlySpan<ChunkActivation> activations)
    {
        (long rows, int width) = Validate(outputs, input, activations);
        float* src = (float*)input.DataPointer;
        for (int c = 0; c < outputs.Length; c++)
        {
            float* dst = (float*)outputs[c].DataPointer;
            ChunkActivation activation = activations[c];
            for (long r = 0; r < rows; r++)
            {
                float* from = src + r * width * outputs.Length + (long)c * width;
                float* to = dst + r * width;
                for (int d = 0; d < width; d++) to[d] = Activate(from[d], activation);
            }
        }
    }

    /// <summary>One chunk element through its activation.</summary>
    public static float Activate(float x, ChunkActivation activation) => activation switch
    {
        ChunkActivation.OnePlus => 1f + x,
        ChunkActivation.Tanh => MathF.Tanh(x),
        ChunkActivation.Sigmoid => 1f / (1f + MathF.Exp(-x)),
        _ => x,
    };
}
