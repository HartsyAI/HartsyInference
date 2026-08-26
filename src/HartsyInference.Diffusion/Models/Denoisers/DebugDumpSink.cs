using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Shared plumbing behind the per-model debug-dump hooks: dump-dir knob resolution, <c>layers/</c> creation, and raw-F32 tensor writes. Cached mode resolves the knob once per process; per-call mode re-reads it on every access, for hooks whose parity tests point the same knob at per-test directories.</summary>
internal sealed unsafe class DebugDumpSink
{
    private readonly Knob<string?> _knob;
    private readonly bool _perCall;
    private readonly string? _cachedDir;
    private readonly object _lock = new object();
    private bool _initialized;

    public DebugDumpSink(Knob<string?> knob, bool perCallResolve = false)
    {
        _knob = knob;
        _perCall = perCallResolve;
        if (!perCallResolve)
            _cachedDir = Resolve(knob);
    }

    /// <summary>Dump root directory, or null when dumping is disabled.</summary>
    public string? Dir => _perCall ? Resolve(_knob) : _cachedDir;

    /// <summary>True when the knob points at a dump directory.</summary>
    public bool Enabled => Dir is not null;

    private static string? Resolve(Knob<string?> knob)
    {
        string? dir = knob.Value;
        return string.IsNullOrEmpty(dir) ? null : dir;
    }

    /// <summary>Creates <c>{dir}/layers</c> — double-checked once in cached mode, on every call in per-call mode (the dir may change between calls).</summary>
    public void EnsureLayersDir(string dir)
    {
        if (_perCall)
        {
            Directory.CreateDirectory(Path.Combine(dir, "layers"));
            return;
        }
        if (_initialized) return;
        lock (_lock)
        {
            if (_initialized) return;
            Directory.CreateDirectory(Path.Combine(dir, "layers"));
            _initialized = true;
        }
    }

    /// <summary>Writes the tensor's data as raw little-endian F32 to <paramref name="path"/>, host-casting non-F32 inputs.</summary>
    public void WriteRawF32(string path, Tensor t)
    {
        long count = t.Shape.ElementCount;
        byte[] buffer = new byte[count * sizeof(float)];
        if (t.DType == DType.F32)
        {
            fixed (byte* dst = buffer)
                Buffer.MemoryCopy((float*)t.DataPointer, dst, buffer.Length, buffer.Length);
        }
        else
        {
            using Tensor cast = t.CastTo(DType.F32);
            fixed (byte* dst = buffer)
                Buffer.MemoryCopy((float*)cast.DataPointer, dst, buffer.Length, buffer.Length);
        }
        File.WriteAllBytes(path, buffer);
    }
}
