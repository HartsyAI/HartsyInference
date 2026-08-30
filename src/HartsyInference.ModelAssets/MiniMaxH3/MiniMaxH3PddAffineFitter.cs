using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.MiniMaxH3;

/// <summary>Fits dense H3 <c>SiLU(time_embedder(t))</c> onto a pruned checkpoint's curve coordinates in F64.</summary>
public static unsafe class MiniMaxH3PddAffineFitter
{
    private static readonly string[] _timeKeys =
    [
        "time_embedder.proj_in.weight", "time_embedder.proj_in.bias",
        "time_embedder.proj_out.weight", "time_embedder.proj_out.bias",
    ];

    /// <summary>Computes the affine curve map and rejects a mismatched trunk/table above <paramref name="maxResidual"/>.</summary>
    public static MiniMaxH3PddAffineBasis Fit(IReadOnlyDictionary<string, Tensor> fullBase,
        IReadOnlyDictionary<string, Tensor> prunedBase, double maxResidual = 1e-4,
        bool requirePublishedShape = true)
    {
        ArgumentNullException.ThrowIfNull(fullBase);
        ArgumentNullException.ThrowIfNull(prunedBase);
        if (!(maxResidual > 0.0) || !double.IsFinite(maxResidual))
            throw new ArgumentOutOfRangeException(nameof(maxResidual));
        Tensor table = Require(prunedBase, "adaln_t_table");
        if (table.Shape.Rank != 2 || table.Shape[0] < 2 || table.Shape[1] < 1)
            throw new HartsyInferenceException($"Pruned H3 adaln_t_table must be [rows,curve], got {table.Shape}.");

        Tensor inWeight = Require(fullBase, _timeKeys[0]);
        Tensor inBias = Require(fullBase, _timeKeys[1]);
        Tensor outWeight = Require(fullBase, _timeKeys[2]);
        Tensor outBias = Require(fullBase, _timeKeys[3]);
        ValidateTimeEmbedder(inWeight, inBias, outWeight, outBias);
        int rows = (int)table.Shape[0];
        int curve = (int)table.Shape[1];
        int frequency = (int)inWeight.Shape[1];
        int hidden = (int)inWeight.Shape[0];
        int dense = (int)outWeight.Shape[0];
        if (requirePublishedShape && (rows != 1025 || curve != 8 || dense != 2688))
        {
            throw new HartsyInferenceException(
                $"Published pruned H3 basis geometry is [1025,8] -> 2688; got [{rows},{curve}] -> {dense}.");
        }

        double[] design = BuildDesign(table, rows, curve);
        double[] denseCurve = BuildDenseCurve(inWeight, inBias, outWeight, outBias, rows, frequency, hidden, dense);
        int columns = curve + 1;
        (double[] q, double[] r) = QrFactor(design, rows, columns);
        double[] solution = SolveAll(q, r, denseCurve, rows, columns, dense);
        double residual = RelativeResidual(design, denseCurve, solution, rows, columns, dense);
        if (!double.IsFinite(residual) || residual > maxResidual)
        {
            throw new HartsyInferenceException(
                $"Dense/pruned H3 AdaLN affine fit residual {residual:E6} exceeds {maxResidual:E6}; "
                + "the full and pruned checkpoints do not share the same trunk curve.");
        }

        Tensor intercept = new Tensor(new TensorShape(dense), DType.F32);
        Tensor projection = new Tensor(new TensorShape(dense, curve), DType.F32);
        float* interceptPointer = (float*)intercept.DataPointer;
        float* projectionPointer = (float*)projection.DataPointer;
        for (int d = 0; d < dense; d++)
        {
            interceptPointer[d] = (float)solution[d];
            for (int k = 0; k < curve; k++)
            {
                projectionPointer[(long)d * curve + k] = (float)solution[(long)(k + 1) * dense + d];
            }
        }
        return new MiniMaxH3PddAffineBasis(intercept, projection, residual);
    }

    private static double[] BuildDesign(Tensor table, int rows, int curve)
    {
        int columns = curve + 1;
        double[] design = new double[rows * columns];
        for (int row = 0; row < rows; row++)
        {
            design[row * columns] = 1.0;
            for (int k = 0; k < curve; k++)
                design[row * columns + k + 1] = MiniMaxH3TensorReader.Read(table, (long)row * curve + k);
        }
        return design;
    }

    private static double[] BuildDenseCurve(Tensor inWeight, Tensor inBias, Tensor outWeight,
        Tensor outBias, int rows, int frequency, int hidden, int dense)
    {
        int half = frequency / 2;
        double[] frequencies = new double[half];
        for (int i = 0; i < half; i++) frequencies[i] = Math.Exp(-Math.Log(10000.0) * i / half);
        double[] embedding = new double[frequency];
        double[] hiddenValues = new double[hidden];
        double[] curve = new double[rows * dense];
        for (int row = 0; row < rows; row++)
        {
            double timestep = row / (double)(rows - 1);
            for (int i = 0; i < half; i++)
            {
                double argument = timestep * frequencies[i];
                embedding[i] = Math.Cos(argument);
                embedding[i + half] = Math.Sin(argument);
            }
            for (int h = 0; h < hidden; h++)
            {
                double value = MiniMaxH3TensorReader.Read(inBias, h);
                long weightBase = (long)h * frequency;
                for (int f = 0; f < frequency; f++)
                    value += MiniMaxH3TensorReader.Read(inWeight, weightBase + f) * embedding[f];
                hiddenValues[h] = Silu(value);
            }
            for (int d = 0; d < dense; d++)
            {
                double value = MiniMaxH3TensorReader.Read(outBias, d);
                long weightBase = (long)d * hidden;
                for (int h = 0; h < hidden; h++)
                    value += MiniMaxH3TensorReader.Read(outWeight, weightBase + h) * hiddenValues[h];
                curve[(long)row * dense + d] = Silu(value);
            }
        }
        return curve;
    }

    private static (double[] Q, double[] R) QrFactor(double[] design, int rows, int columns)
    {
        double[] q = new double[rows * columns];
        double[] r = new double[columns * columns];
        double[] work = new double[rows];
        for (int column = 0; column < columns; column++)
        {
            for (int row = 0; row < rows; row++) work[row] = design[row * columns + column];
            for (int pass = 0; pass < 2; pass++)
            {
                for (int previous = 0; previous < column; previous++)
                {
                    double dot = 0.0;
                    for (int row = 0; row < rows; row++) dot += q[row * columns + previous] * work[row];
                    r[previous * columns + column] += dot;
                    for (int row = 0; row < rows; row++) work[row] -= dot * q[row * columns + previous];
                }
            }
            double squared = 0.0;
            for (int row = 0; row < rows; row++) squared += work[row] * work[row];
            double norm = Math.Sqrt(squared);
            if (!(norm > 1e-14) || !double.IsFinite(norm))
                throw new HartsyInferenceException("Pruned H3 curve-table design matrix is rank deficient.");
            r[column * columns + column] = norm;
            for (int row = 0; row < rows; row++) q[row * columns + column] = work[row] / norm;
        }
        return (q, r);
    }

    private static double[] SolveAll(double[] q, double[] r, double[] denseCurve, int rows,
        int columns, int dense)
    {
        double[] solution = new double[columns * dense];
        double[] right = new double[columns];
        for (int d = 0; d < dense; d++)
        {
            for (int column = 0; column < columns; column++)
            {
                double dot = 0.0;
                for (int row = 0; row < rows; row++)
                    dot += q[row * columns + column] * denseCurve[(long)row * dense + d];
                right[column] = dot;
            }
            for (int row = columns - 1; row >= 0; row--)
            {
                double value = right[row];
                for (int column = row + 1; column < columns; column++)
                    value -= r[row * columns + column] * solution[(long)column * dense + d];
                solution[(long)row * dense + d] = value / r[row * columns + row];
            }
        }
        return solution;
    }

    private static double RelativeResidual(double[] design, double[] denseCurve, double[] solution,
        int rows, int columns, int dense)
    {
        double errorSquared = 0.0;
        double targetSquared = 0.0;
        for (int row = 0; row < rows; row++)
        {
            for (int d = 0; d < dense; d++)
            {
                double predicted = 0.0;
                for (int column = 0; column < columns; column++)
                    predicted += design[row * columns + column] * solution[(long)column * dense + d];
                double target = denseCurve[(long)row * dense + d];
                double error = predicted - target;
                errorSquared += error * error;
                targetSquared += target * target;
            }
        }
        return Math.Sqrt(errorSquared / targetSquared);
    }

    private static void ValidateTimeEmbedder(Tensor inWeight, Tensor inBias, Tensor outWeight, Tensor outBias)
    {
        if (inWeight.Shape.Rank != 2 || inBias.Shape.Rank != 1 || outWeight.Shape.Rank != 2
            || outBias.Shape.Rank != 1 || inWeight.Shape[0] != inBias.Shape[0]
            || outWeight.Shape[1] != inWeight.Shape[0] || outWeight.Shape[0] != outBias.Shape[0]
            || inWeight.Shape[1] % 2 != 0)
        {
            throw new HartsyInferenceException(
                $"Invalid H3 time embedder shapes: inW={inWeight.Shape}, inB={inBias.Shape}, "
                + $"outW={outWeight.Shape}, outB={outBias.Shape}.");
        }
    }

    private static Tensor Require(IReadOnlyDictionary<string, Tensor> tensors, string key) =>
        tensors.TryGetValue(key, out Tensor? tensor) ? tensor
            : throw new HartsyInferenceException($"H3 affine fitting requires '{key}'.");

    private static double Silu(double value) => value / (1.0 + Math.Exp(-value));
}
