using Xunit;
using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common;

namespace HartsyInference.Cpu.Tests.Attention;

/// <summary>Loads the committed Python reference fixtures and builds tensors and latent sources from them.</summary>
public static unsafe class AttentionFixtures
{
    public static JsonElement Load(string name)
    {
        string path = Path.Combine(RepoRoot.Path, "tests", "python-reference", "deepseek_v41", "fixtures", name);
        return JsonDocument.Parse(File.ReadAllText(path)).RootElement;
    }

    public static float[] Floats(JsonElement e) => e.EnumerateArray().Select(v => v.ValueKind == JsonValueKind.String ? float.NaN : (float)v.GetDouble()).ToArray();

    public static int[] Ints(JsonElement e) => e.EnumerateArray().Select(v => v.GetInt32()).ToArray();

    public static byte[] Bytes(JsonElement e) => e.EnumerateArray().Select(v => (byte)v.GetInt32()).ToArray();

    public static Tensor F32(float[] data, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.F32);
        data.AsSpan().CopyTo(new Span<float>((float*)t.DataPointer, data.Length));
        return t;
    }

    public static Tensor I32(int[] data, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.I32);
        data.AsSpan().CopyTo(new Span<int>((int*)t.DataPointer, data.Length));
        return t;
    }

    public static Tensor U8(byte[] data, params long[] shape)
    {
        Tensor t = new(new TensorShape(shape), DType.U8);
        data.AsSpan().CopyTo(new Span<byte>((byte*)t.DataPointer, data.Length));
        return t;
    }

    public static Tensor Empty(DType type, params long[] shape) => new(new TensorShape(shape), type);

    public static float[] ReadF32(Tensor t) => new ReadOnlySpan<float>((float*)t.DataPointer, (int)t.ElementCount).ToArray();

    public static int[] ReadI32(Tensor t) => new ReadOnlySpan<int>((int*)t.DataPointer, (int)t.ElementCount).ToArray();

    public static byte[] ReadU8(Tensor t) => new ReadOnlySpan<byte>((byte*)t.DataPointer, (int)t.ElementCount).ToArray();

    /// <summary>A quantized source from fixture code and scale bytes; the caller disposes the tensors.</summary>
    public static LatentSource Source(LatentEncoding enc, byte[] codes, byte[] scales, int rows, int dim) =>
        new(enc, U8(codes, rows, LatentEncodings.CodeBytesPerRow(enc, dim)),
            U8(scales, rows, LatentEncodings.ScaleBytesPerRow(enc, dim)), rows, dim);

    public static void Dispose(in LatentSource s)
    {
        s.Codes?.Dispose();
        s.Scales?.Dispose();
    }

    public static float MaxAbsDiff(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float m = 0f;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }
}
