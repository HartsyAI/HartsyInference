namespace HartsyInference.Cuda;

/// <summary>The PV formats INT8 attention v2's kernel entries run.</summary>
internal enum Int8AttentionV2Pv
{
    /// <summary>u8 P times s8 V with one V scale per channel over the sequence.</summary>
    U8SequenceScales,

    /// <summary>u8 P times s8 V with one V scale per channel per key tile.</summary>
    U8TileScales,

    /// <summary>F16 P times F16 V against the running maximum (the sm_86 fallback).</summary>
    F16,
}
