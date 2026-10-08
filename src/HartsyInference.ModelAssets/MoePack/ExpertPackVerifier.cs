using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;

namespace HartsyInference.ModelAssets.MoePack;

/// <summary>Checks a pack against its source: checksums, then the dequantized values against the original F32 weights.</summary>
public static class ExpertPackVerifier
{
    /// <summary>Verifies the given experts. Checksum failures are reported, not thrown.</summary>
    /// <param name="reader">The open pack.</param>
    /// <param name="source">The original gate, up and down values of an expert, row-major.</param>
    /// <param name="keys">Experts to check; use <see cref="ExpertPackReader.Keys"/> for all of them.</param>
    public static ExpertPackVerification Verify(ExpertPackReader reader, Func<ExpertKey, (float[] Gate, float[] Up, float[] Down)> source,
            IEnumerable<ExpertKey> keys)
    {
        ArgumentNullException.ThrowIfNull(reader);
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(keys);
        List<ExpertKey> failures = [];
        int checkedCount = 0;
        double maxAbs = 0, errorSquares = 0, referenceSquares = 0;
        long values = 0;
        foreach (ExpertKey key in keys)
        {
            ExpertWeights weights;
            try
            {
                weights = reader.Resolve(key);
            }
            catch (InvalidDataException)
            {
                failures.Add(key);
                continue;
            }
            (float[] gate, float[] up, float[] down) = source(key);
            Compare(weights.W1.Weight, gate, ref maxAbs, ref errorSquares, ref referenceSquares, ref values);
            Compare(weights.W3.Weight, up, ref maxAbs, ref errorSquares, ref referenceSquares, ref values);
            Compare(weights.W2.Weight, down, ref maxAbs, ref errorSquares, ref referenceSquares, ref values);
            checkedCount++;
        }
        double relative = referenceSquares == 0 ? 0 : Math.Sqrt(errorSquares / referenceSquares);
        return new ExpertPackVerification(checkedCount, maxAbs, relative, failures);
    }

    private static void Compare(Tensor quantized, float[] reference, ref double maxAbs, ref double errorSquares,
            ref double referenceSquares, ref long values)
    {
        using Tensor decoded = GgufDequantizer.Dequantize(quantized, DType.F32);
        ReadOnlySpan<float> actual = decoded.AsSpan<float>();
        for (int i = 0; i < reference.Length; i++)
        {
            double error = actual[i] - reference[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(error));
            errorSquares += error * error;
            referenceSquares += (double)reference[i] * reference[i];
            values++;
        }
    }
}
