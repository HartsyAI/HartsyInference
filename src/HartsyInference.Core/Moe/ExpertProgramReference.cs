namespace HartsyInference.Core.Moe;

/// <summary>
/// Scalar F32 execution of an <see cref="ExpertProgram"/>: the correctness oracle for every backend expert kernel. It is
/// deliberately simple; speed belongs to the backend kernels that are checked against it.
/// </summary>
public static class ExpertProgramReference
{
    /// <summary>
    /// Runs the program on <paramref name="rows"/> token rows: <c>y = Down · (act(clampGate(Gate·x)) * clampUp(Up·x))</c>.
    /// </summary>
    /// <param name="program">Activation and clamp bounds.</param>
    /// <param name="weights">Validated expert matrices.</param>
    /// <param name="x"><c>rows × H</c> inputs, row-major.</param>
    /// <param name="rows">Token rows.</param>
    /// <param name="y"><c>rows × H</c> outputs, overwritten.</param>
    public static void Apply(ExpertProgram program, F32ExpertWeights weights, ReadOnlySpan<float> x, int rows, Span<float> y)
    {
        ArgumentNullException.ThrowIfNull(program);
        ArgumentNullException.ThrowIfNull(weights);
        program.Validated();
        weights.Validated();
        int h = weights.Hidden, inter = weights.Intermediate;
        if (x.Length != (long)rows * h) throw new ArgumentException($"x must hold {rows} rows of {h}.", nameof(x));
        if (y.Length != (long)rows * h) throw new ArgumentException($"y must hold {rows} rows of {h}.", nameof(y));
        float[] hidden = new float[inter];
        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<float> row = x.Slice(r * h, h);
            for (int i = 0; i < inter; i++)
            {
                float gate = Dot(weights.Gate.AsSpan(i * h, h), row);
                float up = Dot(weights.Up.AsSpan(i * h, h), row);
                (float clampedGate, float clampedUp) = program.Clamp(gate, up);
                hidden[i] = Activate(program.Activation, clampedGate) * clampedUp;
            }
            Span<float> outRow = y.Slice(r * h, h);
            for (int d = 0; d < h; d++)
            {
                float acc = 0f;
                ReadOnlySpan<float> downRow = weights.Down.AsSpan(d * inter, inter);
                for (int i = 0; i < inter; i++) acc += downRow[i] * hidden[i];
                outRow[d] = acc;
            }
        }
    }

    /// <summary>Scalar activation of the gate branch.</summary>
    public static float Activate(ExpertActivation activation, float v) => activation switch
    {
        ExpertActivation.Silu => v / (1f + MathF.Exp(-v)),
        ExpertActivation.GeluTanh => 0.5f * v * (1f + MathF.Tanh(0.7978845608028654f * (v + 0.044715f * v * v * v))),
        ExpertActivation.Relu => MathF.Max(v, 0f),
        ExpertActivation.ReluSquared => MathF.Max(v, 0f) * MathF.Max(v, 0f),
        _ => throw new ArgumentOutOfRangeException(nameof(activation), activation, "Unknown activation."),
    };

    private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        float acc = 0f;
        for (int i = 0; i < a.Length; i++) acc += a[i] * b[i];
        return acc;
    }
}
