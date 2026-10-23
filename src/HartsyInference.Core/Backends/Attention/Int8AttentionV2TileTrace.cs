namespace HartsyInference.Core.Backends;

/// <summary>One u8 key tile as a query row drains it: <paramref name="batchHead"/> is <c>b·Hq + h</c>; the P bytes cover the
/// tile's key slots (0 where a key is excluded), the INT32 PV partial is per padded channel and the sum is the bytes' total.</summary>
public delegate void Int8AttentionV2TileTrace(int batchHead, int row, int tile, float tileMax, ReadOnlySpan<byte> probabilities,
    ReadOnlySpan<float> partial, int sum);
