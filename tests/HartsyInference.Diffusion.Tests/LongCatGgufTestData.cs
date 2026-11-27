using System.Runtime.InteropServices;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Gguf.Codecs;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Writes LongCat weights under the engine's names as a GGUF repack in the ComfyUI Wan layout.</summary>
internal static class LongCatGgufTestData
{
    /// <summary>Writes <c>model.gguf</c> into <paramref name="directory"/>: every block linear whose rows hold whole blocks quantized
    /// (<paramref name="valueQuant"/> for the two value projections), the rest F32. Returns each engine weight as the file decodes it, F32, which
    /// the caller disposes. <paramref name="skip"/> leaves one stored tensor out.</summary>
    public static Dictionary<string, Tensor> Write(string directory, IReadOnlyDictionary<string, Tensor> weights, string quant, string valueQuant,
        string? skip = null)
    {
        Dictionary<string, Tensor> decoded = new(StringComparer.Ordinal);
        List<Tensor> written = [];
        try
        {
            using GgufWriter writer = new(Path.Combine(directory, "model.gguf"));
            writer.SetMetadata("general.architecture", "wan");
            foreach ((string key, Tensor weight) in weights)
            {
                string[] names = ComfyNames(key);
                bool linear = weight.Shape.Rank == 2 && key.StartsWith("blocks.", StringComparison.Ordinal)
                    && key.EndsWith(".weight", StringComparison.Ordinal)
                    && !key.Contains("norm", StringComparison.Ordinal) && !key.Contains("adaLN", StringComparison.Ordinal)
                    && weight.Shape[1] % DTypeOf(quant).BlockElementCount == 0;
                long rows = weight.Shape[0] / names.Length;
                Tensor whole = new(weight.Shape, DType.F32);
                for (int part = 0; part < names.Length; part++)
                {
                    TensorShape shape = names.Length == 1 ? weight.Shape
                        : weight.Shape.Rank == 1 ? new TensorShape(rows) : new TensorShape(rows, weight.Shape[1]);
                    int count = checked((int)shape.ElementCount);
                    ReadOnlySpan<float> values = weight.AsReadOnlySpan<float>().Slice(part * count, count);
                    DType dtype = !linear ? DType.F32
                        : DTypeOf(names[part].Contains(".v.", StringComparison.Ordinal) ? valueQuant : quant);
                    Tensor stored = new(shape, dtype);
                    if (dtype == DType.F32)
                        values.CopyTo(stored.AsSpan<float>());
                    else
                        Quantize(values, stored);
                    Decode(stored).CopyTo(whole.AsSpan<float>().Slice(part * count, count));
                    written.Add(stored);
                    if (names[part] != skip)
                        writer.AddTensor(names[part], stored);
                }
                decoded[key] = whole;
            }
            writer.Flush();
        }
        finally
        {
            foreach (Tensor tensor in written) tensor.Dispose();
        }
        return decoded;
    }

    /// <summary>The ComfyUI Wan names of one engine tensor; two or three for a fused one, in its row order.</summary>
    private static string[] ComfyNames(string key)
    {
        int dot = key.LastIndexOf('.');
        string stem = key[..dot], kind = key[(dot + 1)..];
        if (stem.EndsWith(".attn.qkv", StringComparison.Ordinal))
            return [.. new[] { "q", "k", "v" }.Select(p => $"{stem[..^".attn.qkv".Length]}.self_attn.{p}.{kind}")];
        if (stem.EndsWith(".cross_attn.kv_linear", StringComparison.Ordinal))
            return [.. new[] { "k", "v" }.Select(p => $"{stem[..^".kv_linear".Length]}.{p}.{kind}")];
        // A "."-led fragment such as ".attn.proj." does not match inside "cross_attn.".
        string name = key.Replace("x_embedder.proj.", "patch_embedding.").Replace("y_embedder.y_proj.", "text_embedding.")
            .Replace("t_embedder.", "time_embedding.").Replace("final_layer.linear.", "head.head.")
            .Replace("final_layer.adaLN_modulation.", "head.modulation.").Replace(".adaLN_modulation.1.", ".modulation.1.")
            .Replace(".attn.proj.", ".self_attn.o.").Replace(".attn.q_norm.", ".self_attn.norm_q.").Replace(".attn.k_norm.", ".self_attn.norm_k.")
            .Replace(".cross_attn.q_linear.", ".cross_attn.q.").Replace(".cross_attn.proj.", ".cross_attn.o.")
            .Replace(".cross_attn.q_norm.", ".cross_attn.norm_q.").Replace(".cross_attn.k_norm.", ".cross_attn.norm_k.")
            .Replace(".pre_crs_attn_norm.", ".norm3.").Replace("audio_proj.", "multitalk_audio_proj.")
            .Replace(".audio_adaLN_modulation.1.", ".audio_modulation.1.").Replace(".pre_video_crs_attn_norm.", ".norm_x.");
        return [name];
    }

    public static DType DTypeOf(string name) => name switch
    {
        "Q8_0" => DType.Q8_0, "Q4_0" => DType.Q4_0, "Q4_1" => DType.Q4_1, "Q5_0" => DType.Q5_0, "Q5_1" => DType.Q5_1,
        _ => throw new ArgumentOutOfRangeException(nameof(name)),
    };

    private static unsafe void Quantize(ReadOnlySpan<float> values, Tensor into)
    {
        IGgufCodec codec = GgufCodecRegistry.Get(into.DType);
        if (codec.SupportsQuantize)
        {
            fixed (float* source = values)
                codec.QuantizeFromF32(source, (byte*)into.DataPointer, values.Length);
            return;
        }
        // A read-only type: seeded payload bytes behind an F16 scale (and offset) decode to some values, which is all a join check needs.
        Random rng = new(values.Length);
        Span<byte> bytes = into.AsSpan<byte>();
        rng.NextBytes(bytes);
        for (int start = 0; start < bytes.Length; start += into.DType.BlockByteSize)
        {
            MemoryMarshal.Write(bytes[start..], (Half)(0.001f + 0.01f * rng.NextSingle()));
            if (into.DType == DType.Q4_1 || into.DType == DType.Q5_1)
                MemoryMarshal.Write(bytes[(start + 2)..], (Half)(-0.05f * rng.NextSingle()));
        }
    }

    /// <summary>A tensor's values as F32.</summary>
    public static float[] Decode(Tensor tensor)
    {
        if (tensor.DType == DType.F32)
            return tensor.AsReadOnlySpan<float>().ToArray();
        if (tensor.DType == DType.BF16)
        {
            ReadOnlySpan<ushort> bits = tensor.AsReadOnlySpan<ushort>();
            float[] values = new float[bits.Length];
            for (int i = 0; i < values.Length; i++)
                values[i] = BFloat16Bits.ToFloat(bits[i]);
            return values;
        }
        using Tensor wide = GgufDequantizer.Dequantize(tensor, DType.F32);
        return wide.AsReadOnlySpan<float>().ToArray();
    }
}
