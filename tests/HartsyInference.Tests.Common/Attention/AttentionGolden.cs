using System.Security.Cryptography;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Tests.Common.Attention;

/// <summary>Shared kit of the attention golden suites (Waves A-E): seeded tensors, SHA-256 digests of output bits, case-owned tensor lists.</summary>
public static unsafe class AttentionGolden
{
    /// <summary>Fixture digest strings (<c>label=hex</c>) of every output, after releasing everything the case allocated.</summary>
    public static string Finish(List<Tensor> owned, params (string Label, Tensor Tensor)[] outputs)
    {
        string digests = string.Join(' ', outputs.Select(entry => Digest(entry.Label, entry.Tensor)));
        foreach (Tensor t in owned) t.Dispose();
        return digests;
    }

    /// <summary><c>label=sha256</c> of the tensor's bits; rejects non-finite values.</summary>
    public static string Digest(string label, Tensor t)
    {
        Tensor f32 = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        for (long i = 0; i < f32.ElementCount; i++)
            if (!float.IsFinite(((float*)f32.DataPointer)[i]))
                throw new InvalidOperationException($"Non-finite value in '{label}' at {i}.");
        if (!ReferenceEquals(f32, t)) f32.Dispose();
        return Hex(label, Bytes(t));
    }

    /// <summary><c>label=sha256</c> of host floats; rejects non-finite values.</summary>
    public static string Digest(string label, ReadOnlySpan<float> values)
    {
        for (int i = 0; i < values.Length; i++)
            if (!float.IsFinite(values[i]))
                throw new InvalidOperationException($"Non-finite value in '{label}' at {i}.");
        return Hex(label, System.Runtime.InteropServices.MemoryMarshal.AsBytes(values));
    }

    public static ReadOnlySpan<byte> Bytes(Tensor t) =>
        new(t.DataPointer, checked((int)(t.ElementCount * t.DType.SizeInBytes)));

    public static string Hex(string label, ReadOnlySpan<byte> bytes) => $"{label}={Convert.ToHexStringLower(SHA256.HashData(bytes))}";

    /// <summary>Deterministic F32 tensor in [-scale, scale) from <paramref name="seed"/> alone (no shared RNG state).</summary>
    public static Tensor Seeded(TensorShape shape, uint seed, float scale = 1f)
    {
        Tensor tensor = new(shape, DType.F32);
        float* data = (float*)tensor.DataPointer;
        uint state = seed * 2654435761u + 0x9E3779B9u;
        for (long i = 0; i < tensor.ElementCount; i++)
        {
            state = state * 1664525u + 1013904223u;
            data[i] = (((state >> 8) & 0xFFFF) / 32768f - 1f) * scale;
        }
        return tensor;
    }

    /// <summary>Loads <c>Fixtures/Attention/&lt;fixture&gt;.golden.txt</c> from the test bin: <c>case label=hex ...</c> lines, <c>#</c> comments skipped.</summary>
    public static Dictionary<string, string> LoadFixture(string fixture)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Attention", $"{fixture}.golden.txt");
        Dictionary<string, string> cases = new(StringComparer.Ordinal);
        foreach (string line in File.ReadLines(path))
        {
            if (line.Length == 0 || line[0] == '#')
                continue;
            int split = line.IndexOf(' ');
            cases.Add(line[..split], line[(split + 1)..]);
        }
        return cases;
    }

    /// <summary>FNV-1a of <paramref name="name"/> xor <paramref name="salt"/>; the seed of a name-seeded weight.</summary>
    public static uint NameSeed(string name, uint salt)
    {
        uint h = 2166136261u ^ salt;
        foreach (char c in name) h = (h ^ c) * 16777619u;
        return h;
    }
}

/// <summary>A list of tensors a golden case disposes after digesting.</summary>
public sealed class GoldenOwner : List<Tensor>
{
    public Tensor Own(Tensor t) { Add(t); return t; }
}

/// <summary>A checkpoint stand-in: each tensor is seeded from its own name (FNV-1a) plus a per-case salt.</summary>
public sealed unsafe class GoldenWeights(uint salt) : Dictionary<string, Tensor>
{
    public void Add(string name, long rows, long cols) => base.Add(name, AttentionGolden.Seeded(new TensorShape(rows, cols), Seed(name), 0.3f));

    public void Replace(string name, long rows, long cols) => this[name] = AttentionGolden.Seeded(new TensorShape(rows, cols), Seed(name), 0.3f);

    public void AddShape(string name, long[] shape, float scale) => base.Add(name, AttentionGolden.Seeded(new TensorShape(shape), Seed(name), scale));

    public void AddBias(string name, long n) => base.Add(name, AttentionGolden.Seeded(new TensorShape(n), Seed(name), 0.1f));

    /// <summary>A weight <c>[out, in]</c> under <c>prefix.weight</c>, plus <c>prefix.bias</c> when asked.</summary>
    public void Linear(string prefix, long outFeatures, long inFeatures, bool bias = false)
    {
        Add($"{prefix}.weight", outFeatures, inFeatures);
        if (bias) AddBias($"{prefix}.bias", outFeatures);
    }

    public void AddConst(string name, float value)
    {
        Tensor t = new(new TensorShape(1), DType.F32);
        ((float*)t.DataPointer)[0] = value;
        base.Add(name, t);
    }

    /// <summary>Norm scale near 1 (near 0 when stored zero-centered, as Krea2's are).</summary>
    public void AddNorm(string name, long n, bool zeroCentered = false)
    {
        Tensor t = AttentionGolden.Seeded(new TensorShape(n), Seed(name), 0.25f);
        if (!zeroCentered)
        {
            float* p = (float*)t.DataPointer;
            for (long i = 0; i < n; i++) p[i] += 1f;
        }
        base.Add(name, t);
    }

    /// <summary>Any-shape tensor named by seed, for 1-d/3-d/4-d parameters.</summary>
    public void AddTensor(string name, float scale, params long[] shape) => base.Add(name, AttentionGolden.Seeded(new TensorShape(shape), Seed(name), scale));

    public uint Seed(string name) => AttentionGolden.NameSeed(name, salt);
}
