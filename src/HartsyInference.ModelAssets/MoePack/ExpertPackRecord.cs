namespace HartsyInference.ModelAssets.MoePack;

/// <summary>
/// Where one expert lives in <c>experts.bin</c>: its identity, byte offset and length, and the SHA-256 of its bytes.
/// The record is three contiguous quantized projections: gate <c>[I, H]</c>, up <c>[I, H]</c>, down <c>[H, I]</c>.
/// </summary>
public sealed record ExpertPackRecord(int Layer, int Expert, ushort Bank, long Offset, long Length, string Sha256);
