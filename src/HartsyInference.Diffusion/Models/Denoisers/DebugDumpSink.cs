using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Shared plumbing behind the per-model debug-dump hooks: dump-dir knob resolution, <c>layers/</c>
/// creation, and raw-F32 tensor writes.</summary>
/// <remarks>The directory is resolved on every access. Caching it in the constructor froze the knob at
/// type-initialization — every sink is held in a <c>static readonly</c> field — so a per-request or per-test
/// value could not reach it, and the opt-out that existed for that was set by exactly one of the nineteen
/// sinks: the one whose parity tests needed the knob to work.</remarks>
internal sealed unsafe class DebugDumpSink(Knob<string?> knob)
{
    private readonly Knob<string?> _knob = knob;

    /// <summary>Dump root directory, or null when dumping is disabled.</summary>
    public string? Dir
    {
        get
        {
            string? dir = _knob.Value;
            return string.IsNullOrEmpty(dir) ? null : dir;
        }
    }

    /// <summary>True when the knob points at a dump directory.</summary>
    public bool Enabled => Dir is not null;

    /// <summary>Creates <c>{dir}/layers</c>. Unconditional: the directory can change between calls, so a
    /// created-once flag would send later dumps to a directory that was never made.</summary>
    public void EnsureLayersDir(string dir) => Directory.CreateDirectory(Path.Combine(dir, "layers"));

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
