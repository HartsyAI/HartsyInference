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
        // Without checksums a corrupt record could pass the numeric tolerance unseen, and the verifier gates source deletion.
        if (!reader.VerifiesChecksums) throw new ArgumentException("Verification needs a reader opened with checksums enabled.", nameof(reader));
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
            try
            {
                (float[] gate, float[] up, float[] down) = source(key);
                int matrix = reader.Intermediate * reader.Hidden;
                ValidateReference(gate, matrix, "gate", key);
                ValidateReference(up, matrix, "up", key);
                ValidateReference(down, matrix, "down", key);
                Compare(weights.W1.Weight, gate, ref maxAbs, ref errorSquares, ref referenceSquares, ref values);
                Compare(weights.W3.Weight, up, ref maxAbs, ref errorSquares, ref referenceSquares, ref values);
                Compare(weights.W2.Weight, down, ref maxAbs, ref errorSquares, ref referenceSquares, ref values);
                checkedCount++;
            }
            finally
            {
                foreach (Tensor tensor in weights.Tensors) tensor.Dispose();
            }
        }
        // An all-zero source with any nonzero decoded value is a failure, not a perfect match.
        double relative = referenceSquares == 0
            ? (errorSquares == 0 ? 0 : double.PositiveInfinity)
            : Math.Sqrt(errorSquares / referenceSquares);
        return new ExpertPackVerification(checkedCount, maxAbs, relative, failures);
    }

    private static void ValidateReference(float[] reference, int expected, string role, ExpertKey key)
    {
        ArgumentNullException.ThrowIfNull(reference);
        if (reference.Length != expected)
            throw new ArgumentException($"The source {role} for {key} holds {reference.Length} values; the pack expects {expected}.");
    }

    /// <summary>Decodes a borrowed Q2_0 matrix; the GGUF dequantizer does not know this pack-local format.</summary>
    private static unsafe Tensor DecodeQ2_0(Tensor quantized)
    {
        int rows = checked((int)quantized.Shape[0]), cols = checked((int)quantized.Shape[1]);
        long byteCount = DType.Q2_0.ComputeByteCount(quantized.ElementCount);
        float[] values = Q2_0Codec.Decode(new ReadOnlySpan<byte>(quantized.DataPointer, checked((int)byteCount)), rows, cols);
        Tensor result = new(new TensorShape(rows, cols), DType.F32);
        values.AsSpan().CopyTo(result.AsSpan<float>());
        return result;
    }

    private static void Compare(Tensor quantized, float[] reference, ref double maxAbs, ref double errorSquares,
            ref double referenceSquares, ref long values)
    {
        using Tensor decoded = quantized.DType == DType.Q2_0 ? DecodeQ2_0(quantized) : GgufDequantizer.Dequantize(quantized, DType.F32);
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
