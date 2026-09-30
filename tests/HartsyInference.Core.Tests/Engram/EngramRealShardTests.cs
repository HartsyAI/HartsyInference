using HartsyInference.Core.Engram;
using HartsyInference.Core.IO;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Engram;

/// <summary>Row gather from the real 475 GiB checkpoint's Engram shards. Skips (or fails under HARTSY_REQUIRE_REAL_WEIGHTS=1) when the shards are not staged.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class EngramRealShardTests
{
    private const string DirEnvVar = "HARTSY_DSV41_FLASH_DIR";
    private const long Layer1Rows = 384006168;
    private readonly ITestOutputHelper _output;

    public EngramRealShardTests(ITestOutputHelper output) => _output = output;

    private static string Shard(string name) =>
        Path.Combine(Environment.GetEnvironmentVariable(DirEnvVar) ?? Path.Combine(TestPaths.ModelsDir, "DeepSeek-V4.1-Flash"), name);

    [Fact]
    public void Layer1_RandomRowsMatchAnIndependentDecode()
    {
        string shard = Shard("model-00047-of-00048.safetensors");
        if (!RealWeightGate.Require(_output.WriteLine, shard))
            return;
        // Offsets pinned from the shard header at revision dba1be0a: F8_E4M3 [rows,256] at data start 664, F8_E8M0 [rows,8] right after.
        OfficialFp8E8M0RowLayout layout = new(Layer1Rows, 664, 664 + 98305579008);
        using PreadByteSource source = new(shard);
        using EngramTableStore store = new(layout, source, EngramBacking.Storage, EngramRowRange.All(Layer1Rows), 256L * 1024 * 1024);

        Random rng = new(1301);
        const int Batch = 1000;
        long[] rows = new long[Batch];
        ushort[] dest = new ushort[Batch * 256];
        byte[] fp8 = new byte[256];
        byte[] scales = new byte[8];
        for (int batch = 0; batch < 10; batch++)
        {
            for (int i = 0; i < Batch; i++)
                rows[i] = rng.NextInt64(Layer1Rows);
            store.Gather(rows, dest);
            for (int i = 0; i < Batch; i++)
            {
                source.ReadAt(664 + rows[i] * 256, fp8);
                source.ReadAt(664 + 98305579008 + rows[i] * 8, scales);
                for (int c = 0; c < 256; c++)
                {
                    ushort expected = Reference(fp8[c], scales[c / 32]);
                    if (expected != dest[i * 256 + c] && !(IsNaN(expected) && IsNaN(dest[i * 256 + c])))
                        Assert.Fail($"row {rows[i]} col {c}: expected 0x{expected:X4} got 0x{dest[i * 256 + c]:X4}");
                }
            }
        }
        _output.WriteLine($"10000 random rows matched; {store.Stats}");
    }

    private static bool IsNaN(ushort bits) => (bits & 0x7FFF) > 0x7F80;

    /// <summary>Slow, independent decode: exact double arithmetic, then round to bf16 (nearest even).</summary>
    private static ushort Reference(byte code, byte e8m0)
    {
        if (code == 0x7F || code == 0xFF || e8m0 == 255)
            return 0x7FC0;
        int sign = code >> 7;
        int exp = (code >> 3) & 0xF;
        int man = code & 7;
        double value = exp == 0 ? man / 8.0 * Math.Pow(2, -6) : (1 + man / 8.0) * Math.Pow(2, exp - 7);
        double scaled = value * Math.Pow(2, e8m0 - 127);
        float f = (float)scaled;
        if (sign == 1)
            f = -f;
        return Bf16Rounding.FromSingle(f);
    }
}
