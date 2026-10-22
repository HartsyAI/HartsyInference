using System.Runtime.InteropServices;

namespace HartsyInference.Cuda;

/// <summary>A trace entry's target (b, h, query tile) and its device buffers (Kernels/attention/int8_attn_v2.cu, V2Trace).</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct Int8V2TraceArgs
{
    /// <summary>INT32 QK products <c>[BR, kvPad]</c>.</summary>
    public ulong Dots;

    /// <summary>P bytes <c>[BR, kvPad]</c> in logical key order.</summary>
    public ulong Probabilities;

    /// <summary>Score maximum per row and tile <c>[BR, tiles]</c>.</summary>
    public ulong TileMax;

    /// <summary>Byte sum per row and tile <c>[BR, tiles]</c>.</summary>
    public ulong Sums;

    /// <summary>INT32 PV partials <c>[BR, tiles, 128]</c>.</summary>
    public ulong Partials;

    /// <summary>Traced batch.</summary>
    public uint Batch;

    /// <summary>Traced head.</summary>
    public uint Head;

    /// <summary>Traced query tile.</summary>
    public uint QueryTile;
}
