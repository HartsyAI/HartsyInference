using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>The row every dense head-major call plans under each value of <c>numerics.vkTiledAttention</c>. The expected
/// rows are written out here from the rule, not read back from the backend, so a change of admission fails this test.
/// A device that cannot run the tiled kernel fails it too: nothing here passes by checking the fallback against itself.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("AttentionSemantics")]
public sealed class VulkanTiledAttentionRouteTests(ITestOutputHelper output)
{
    private const int ScalarMaxHeadDim = 128, TiledMaxHeadDim = 256;
    private static readonly int[] HeadDims = [32, 64, 96, 128, 192, 256, 320];

    /// <summary>Operand storage of one cell: all four tensors in one type, or F32 queries and output over F16 keys and values.</summary>
    private enum Storage { F32, F16, Bf16, F32OverF16Kv }

    private static DType QueryType(Storage s) => s switch { Storage.F16 => DType.F16, Storage.Bf16 => DType.BF16, _ => DType.F32 };

    private static DType KvType(Storage s) => s switch { Storage.F32 => DType.F32, Storage.Bf16 => DType.BF16, _ => DType.F16 };

    /// <summary>The cooperative-matrix row: head dim 64 or 128, no BF16, and F16 operands or the F16 permission.</summary>
    private static bool Cm2Takes(int headDim, Storage storage, bool f16Permission, bool cm2) =>
        cm2 && headDim is 64 or 128 && storage != Storage.Bf16 && (storage == Storage.F16 || f16Permission);

    private static string Expected(VulkanTiledAttention mode, int headDim, Storage storage, bool f16Permission, bool cm2)
    {
        if (Cm2Takes(headDim, storage, f16Permission, cm2)) return "cm2.flash";
        int tiledUpTo = mode switch { VulkanTiledAttention.Off => 0, VulkanTiledAttention.Auto => ScalarMaxHeadDim, _ => TiledMaxHeadDim };
        if (storage == Storage.F32 && headDim <= tiledUpTo) return "tiled.f32";
        return headDim <= ScalarMaxHeadDim ? "flash.scalar" : "naive";
    }

    private static string Planned(VulkanBackend backend, VulkanTiledAttention mode, in AttentionCall call)
    {
        backend.TiledAttention = mode;
        return backend.PlanAttention(in call).RowId ?? "none";
    }

    /// <summary>The knob's own default, read from its declaration so an override in the environment cannot change the answer.</summary>
    [Fact]
    public void TheKnobDefaultsToAuto() => Assert.Equal("auto", EngineKnobs.VkTiledAttention.Default);

    [VulkanFact]
    public void EveryDenseCallPlansTheRowTheKnobValueGivesIt()
    {
        const int queryTokens = 24, keyTokens = 40;
        using VulkanBackend backend = new();
        bool cm2 = backend.SupportsTokenMajorAttention;
        int cells = 0, moved = 0;
        List<string> wrong = [];
        foreach (int headDim in HeadDims)
            foreach (Storage storage in Enum.GetValues<Storage>())
                foreach (bool f16Permission in (bool[])[false, true])
                {
                    HashSet<string> off = [], auto = [], all = [];
                    foreach (bool withBias in (bool[])[false, true])
                        foreach (int kvHeads in (int[])[4, 2])
                            foreach (int batch in (int[])[1, 2])
                            {
                                using Tensor q = new(new TensorShape(batch, 4, queryTokens, headDim), QueryType(storage));
                                using Tensor k = new(new TensorShape(batch, kvHeads, keyTokens, headDim), KvType(storage));
                                using Tensor v = new(k.Shape, KvType(storage));
                                using Tensor o = new(q.Shape, QueryType(storage));
                                using Tensor? bias = withBias ? new(new TensorShape(queryTokens, keyTokens), DType.F32) : null;
                                AttentionCall call = AttentionCall.FromScaledDotProductAttention(o, q, k, v, bias, 0.1f, f16Permission);
                                string cell = $"D {headDim}, {storage}, F16 permission {f16Permission}, bias {withBias}, kv heads {kvHeads}, batch {batch}";
                                string offRow = Planned(backend, VulkanTiledAttention.Off, in call);
                                string autoRow = Planned(backend, VulkanTiledAttention.Auto, in call);
                                string allRow = Planned(backend, VulkanTiledAttention.All, in call);
                                foreach ((VulkanTiledAttention mode, string row) in ((VulkanTiledAttention, string)[])[
                                    (VulkanTiledAttention.Off, offRow), (VulkanTiledAttention.Auto, autoRow), (VulkanTiledAttention.All, allRow)])
                                {
                                    string expected = Expected(mode, headDim, storage, f16Permission, cm2);
                                    if (row != expected) wrong.Add($"{cell}: {mode} planned {row}, expected {expected}");
                                }
                                cells++;
                                if (autoRow != offRow)
                                {
                                    moved++;
                                    if (storage != Storage.F32 || headDim > ScalarMaxHeadDim || offRow != "flash.scalar" || autoRow != "tiled.f32")
                                        wrong.Add($"{cell}: moved from {offRow} to {autoRow}, which is not an F32 call leaving flash.scalar at head dim up to 128");
                                }
                                off.Add(offRow);
                                auto.Add(autoRow);
                                all.Add(allRow);
                            }
                    output.WriteLine($"ROUTE | {headDim} | {storage} | {(f16Permission ? "yes" : "no")} | {string.Join("/", off)} | "
                        + $"{string.Join("/", auto)} | {string.Join("/", all)}");
                }
        output.WriteLine($"ROUTE cells {cells}, moved by the default {moved}, cooperative-matrix row on this device {cm2}");
        Assert.True(wrong.Count == 0, string.Join(Environment.NewLine, wrong));
        Assert.True(moved > 0, "No cell moved: the device does not run the tiled kernel, so the default was not exercised.");
    }

    /// <summary>Past head dim 128 the default takes the tiled kernel only for the call the materialized row cannot tile:
    /// one with a bias whose score matrix is past 1 GiB. Planned only, with a key-only bias, so nothing large is computed.</summary>
    [VulkanTheory]
    [InlineData(2049, true, "tiled.f32")]
    [InlineData(2048, true, "naive")]
    [InlineData(2049, false, "naive")]
    public void PastHeadDim128TheDefaultTakesTheTiledKernelOnlyForABiasedCallWhoseScoresPassOneGibibyte(int tokens, bool withBias, string expected)
    {
        const int heads = 64, headDim = 129;
        using VulkanBackend backend = new();
        using Tensor q = new(new TensorShape(1, heads, tokens, headDim), DType.F32);
        using Tensor k = new(q.Shape, DType.F32);
        using Tensor v = new(q.Shape, DType.F32);
        using Tensor o = new(q.Shape, DType.F32);
        using Tensor? bias = withBias ? new(new TensorShape(1, 1, 1, tokens), DType.F32) : null;
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(o, q, k, v, bias, 0.1f, allowF16: false);
        long scoreBytes = (long)heads * tokens * tokens * sizeof(float);
        output.WriteLine($"ROUTE256 | {tokens} tokens, bias {withBias}, scores {scoreBytes} bytes (1 GiB is {1L << 30})");

        Assert.Equal(expected, Planned(backend, VulkanTiledAttention.Auto, in call));
        Assert.Equal("naive", Planned(backend, VulkanTiledAttention.Off, in call));
        Assert.Equal("tiled.f32", Planned(backend, VulkanTiledAttention.All, in call));
    }
}
