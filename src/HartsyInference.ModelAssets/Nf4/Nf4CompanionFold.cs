using System.Text.Json;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.Nf4;

/// <summary>Turns a bitsandbytes <c>Linear4bit</c> checkpoint back into ordinary weights: finds each NF4 weight's
/// companion tensors, reconstructs its scales, dequantizes it, and drops the companions.</summary>
/// <remarks><para>bitsandbytes stores a 4-bit weight as a flat U8 blob of nibbles plus four or five sibling tensors and
/// a JSON blob describing how to read them. None of that is architecture-specific, so folding it here — once, before
/// any converter runs — is what makes an <c>nf4</c> repack loadable by every model rather than by the ones somebody
/// wired it into.</para>
/// <para>This dequantizes rather than keeping the weight packed: there is no NF4 GEMM in any backend, so a packed NF4
/// weight has nowhere to go. The cost is real — 4 bits/param becomes 16 — and is the price of loading a format the
/// hardware path does not speak.</para>
/// <para><b>Everything is reconciled before anything is decoded.</b> The declared shape, block size and codebook must
/// agree with the actual byte counts of the packed weight and its scales, and the shipped codebook must be the NF4 one.
/// A layout this code read wrongly would otherwise decode to plausible noise; instead it refuses and names the key.</para></remarks>
public static class Nf4CompanionFold
{
    /// <summary>Suffix bitsandbytes gives the serialized non-tensor quant state, one per quantized weight.</summary>
    private const string QuantStatePrefix = ".quant_state.bitsandbytes__";

    /// <summary>Returns a dictionary with every NF4 weight dequantized and its companions removed; the input is returned unchanged when the checkpoint holds none.</summary>
    public static Dictionary<string, Tensor> Apply(Dictionary<string, Tensor> source)
    {
        ArgumentNullException.ThrowIfNull(source);
        List<string>? stateKeys = null;
        foreach (string key in source.Keys)
        {
            if (key.Contains(QuantStatePrefix, StringComparison.Ordinal))
                (stateKeys ??= new List<string>()).Add(key);
        }
        if (stateKeys is null)
            return source;

        Dictionary<string, Tensor> result = new(source.Count, StringComparer.Ordinal);
        HashSet<string> consumed = new(StringComparer.Ordinal);
        foreach (string stateKey in stateKeys)
        {
            int marker = stateKey.IndexOf(QuantStatePrefix, StringComparison.Ordinal);
            string weightKey = stateKey[..marker];
            string quantType = stateKey[(marker + QuantStatePrefix.Length)..];
            if (!string.Equals(quantType, "nf4", StringComparison.Ordinal))
            {
                throw new NotSupportedException(
                    $"'{weightKey}' is bitsandbytes '{quantType}', not nf4. Only NF4 4-bit checkpoints are read; use the "
                    + "model's BF16, fp8_scaled or GGUF build instead.");
            }
            if (!source.TryGetValue(weightKey, out Tensor? packed))
                throw new NotSupportedException($"'{stateKey}' describes a weight '{weightKey}' the checkpoint does not contain.");

            Nf4QuantState state = Nf4QuantState.Parse(source[stateKey], stateKey);
            result[weightKey] = Dequantize(packed, state, weightKey, source, consumed);
            consumed.Add(stateKey);
            consumed.Add(weightKey);
        }

        foreach (KeyValuePair<string, Tensor> entry in source)
        {
            if (!consumed.Contains(entry.Key))
                result.TryAdd(entry.Key, entry.Value);
        }
        Logs.Info($"NF4: dequantized {stateKeys.Count} bitsandbytes 4-bit weights at load.");
        return result;
    }

    private static Tensor Dequantize(Tensor packed, Nf4QuantState state, string weightKey,
        Dictionary<string, Tensor> source, HashSet<string> consumed)
    {
        if (packed.DType != DType.U8)
            throw new NotSupportedException($"NF4 weight '{weightKey}' must be U8 packed nibbles; got {packed.DType}.");
        long elements = state.Shape.ElementCount;
        long expectedBytes = (elements + 1) / 2;
        if (packed.ElementCount != expectedBytes)
        {
            throw new NotSupportedException(
                $"NF4 weight '{weightKey}' declares shape {state.Shape} ({elements} elements, {expectedBytes} packed "
                + $"bytes) but holds {packed.ElementCount} bytes. The companion layout does not describe this weight.");
        }

        Tensor codebook = Require(source, consumed, $"{weightKey}.quant_map", weightKey);
        RequireNf4Codebook(codebook, weightKey);

        Tensor storedAbsmax = Require(source, consumed, $"{weightKey}.absmax", weightKey);
        long blocks = (elements + state.BlockSize - 1) / state.BlockSize;
        Tensor? reconstructed = null;
        try
        {
            Tensor absmax;
            if (storedAbsmax.DType == DType.F32)
            {
                absmax = storedAbsmax;
            }
            else
            {
                // Double quant: absmax is itself 8-bit against a nested codebook, so it has to be rebuilt first.
                Tensor nestedAbsmax = Require(source, consumed, $"{weightKey}.nested_absmax", weightKey);
                Tensor nestedCodebook = Require(source, consumed, $"{weightKey}.nested_quant_map", weightKey);
                absmax = reconstructed = Nf4Codec.ReconstructDoubleQuantAbsmax(storedAbsmax, nestedAbsmax,
                    nestedCodebook, state.NestedOffset, state.NestedBlockSize);
            }
            if (absmax.ElementCount != blocks)
            {
                throw new NotSupportedException(
                    $"NF4 weight '{weightKey}' needs {blocks} block scales at blocksize {state.BlockSize}; its absmax "
                    + $"holds {absmax.ElementCount}. The companion layout does not describe this weight.");
            }

            using Tensor f32 = Nf4Codec.Dequantize(packed, absmax, state.Shape, state.BlockSize);
            return state.ComputeDType == DType.F32 ? f32.To(f32.Device) : f32.CastTo(state.ComputeDType);
        }
        finally
        {
            reconstructed?.Dispose();
        }
    }

    private static Tensor Require(Dictionary<string, Tensor> source, HashSet<string> consumed, string key, string weightKey)
    {
        if (!source.TryGetValue(key, out Tensor? tensor))
            throw new NotSupportedException($"NF4 weight '{weightKey}' has no '{key}' companion, so it cannot be decoded.");
        consumed.Add(key);
        return tensor;
    }

    /// <summary>Refuses a codebook that is not the NF4 one. bitsandbytes ships the table in the file, and decoding FP4 (or anything custom) through NF4's quantiles produces values that are wrong but entirely plausible.</summary>
    private static void RequireNf4Codebook(Tensor codebook, string weightKey)
    {
        if (codebook.DType != DType.F32 || codebook.ElementCount != Nf4Codec.Nf4Lut.Length)
        {
            throw new NotSupportedException(
                $"NF4 weight '{weightKey}' ships a {codebook.DType} codebook of {codebook.ElementCount} entries; NF4 is "
                + $"{Nf4Codec.Nf4Lut.Length} F32 quantiles.");
        }
        ReadOnlySpan<float> shipped = codebook.AsReadOnlySpan<float>();
        for (int i = 0; i < Nf4Codec.Nf4Lut.Length; i++)
        {
            if (MathF.Abs(shipped[i] - Nf4Codec.Nf4Lut[i]) > 1e-6f)
            {
                throw new NotSupportedException(
                    $"NF4 weight '{weightKey}' ships a codebook that is not bitsandbytes' NF4 table (entry {i} is "
                    + $"{shipped[i]}, expected {Nf4Codec.Nf4Lut[i]}).");
            }
        }
    }

    /// <summary>The non-tensor half of a bitsandbytes quant state, as serialized into a U8 JSON blob beside the weight.</summary>
    private sealed record Nf4QuantState(TensorShape Shape, int BlockSize, DType ComputeDType, int NestedBlockSize, float NestedOffset)
    {
        public static Nf4QuantState Parse(Tensor blob, string stateKey)
        {
            if (blob.DType != DType.U8)
                throw new NotSupportedException($"'{stateKey}' must be a U8 JSON blob; got {blob.DType}.");
            ReadOnlySpan<byte> bytes = blob.AsReadOnlySpan<byte>();
            int end = bytes.Length;
            while (end > 0 && bytes[end - 1] == 0) end--;

            using JsonDocument document = JsonDocument.Parse(bytes[..end].ToArray());
            JsonElement root = document.RootElement;
            if (!root.TryGetProperty("shape", out JsonElement shapeElement))
                throw new NotSupportedException($"'{stateKey}' declares no shape.");
            long[] dims = new long[shapeElement.GetArrayLength()];
            int index = 0;
            foreach (JsonElement dim in shapeElement.EnumerateArray()) dims[index++] = dim.GetInt64();

            int blockSize = root.TryGetProperty("blocksize", out JsonElement bs) ? bs.GetInt32() : Nf4Codec.DefaultBlockSize;
            int nestedBlockSize = root.TryGetProperty("nested_blocksize", out JsonElement nbs) ? nbs.GetInt32() : 256;
            float nestedOffset = root.TryGetProperty("nested_offset", out JsonElement offset) ? offset.GetSingle() : 0f;
            string dtype = root.TryGetProperty("dtype", out JsonElement dt) ? dt.GetString() ?? "" : "";
            return new Nf4QuantState(new TensorShape(dims.AsSpan()), blockSize, ParseComputeDType(dtype, stateKey),
                nestedBlockSize, nestedOffset);
        }

        /// <summary>The dtype the quantizer captured the weight from, which is what it dequantizes back to.</summary>
        private static DType ParseComputeDType(string dtype, string stateKey)
        {
            string name = dtype.StartsWith("torch.", StringComparison.Ordinal) ? dtype["torch.".Length..] : dtype;
            return name switch
            {
                "bfloat16" => DType.BF16,
                "float16" or "half" => DType.F16,
                "float32" or "float" or "" => DType.F32,
                _ => throw new NotSupportedException($"'{stateKey}' declares compute dtype '{dtype}', which is not a float type."),
            };
        }
    }
}
