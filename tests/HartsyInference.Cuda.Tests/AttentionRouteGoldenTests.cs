using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>The dispatch route every attention shape class takes on CUDA, pinned before the ladder became planner rows.
/// The rows must reproduce these routes exactly; a change here is a dispatch change and is reviewed as one.</summary>
/// <remarks>Shapes are small enough that no route depends on free VRAM or the card's SM count, so the table holds on the
/// 3060 and the 4090 alike. Each case gets a fresh backend, built after its knobs are set (four are read by the ctor).</remarks>
[Collection("CudaSerial")]
public sealed unsafe class AttentionRouteGoldenTests
{
    private readonly ITestOutputHelper _output;
    public AttentionRouteGoldenTests(ITestOutputHelper output) => _output = output;

    /// <param name="Entry">sdpa (head-major), tm (token-major), tmg (the grouped token-major overload), krea2 (that overload with
    /// Krea2's own shapes: [1,S,H,D] operands, a [1,S,H·D] output), flash (host positions) or dev (device positions).</param>
    /// <param name="Mask">none, bias (<c>[1,1,Sq,Skv]</c>), perhead (<c>[1,H,Sq,Skv]</c>) or key (<c>[1,1,1,Skv]</c>).</param>
    private sealed record Case(string Name, string Entry, DType Dtype, int Hq, int Hkv, int Sq, int Skv, int D, string Mask,
        bool AllowF16, string Expected, DType? KvDtype = null, bool Sink = false, Action? Knobs = null);

    private static readonly Case[] Table =
    [
        new("f16-d128", "sdpa", DType.F16, 4, 4, 256, 256, 128, "none", true, "S.3"),
        new("f16-d128-8k", "sdpa", DType.F16, 2, 2, 8192, 8192, 128, "none", true, "S.1"),
        new("f16-d256", "sdpa", DType.F16, 2, 2, 256, 256, 256, "none", true, "S.2"),
        new("f16-d40", "sdpa", DType.F16, 4, 4, 256, 256, 40, "none", true, "S.3"),
        new("f16-d160", "sdpa", DType.F16, 4, 4, 256, 256, 160, "none", true, "S.3"),
        new("f16-d36", "sdpa", DType.F16, 4, 4, 256, 256, 36, "none", true, "S.15"),
        new("f16-d80", "sdpa", DType.F16, 4, 4, 256, 256, 80, "none", true, "S.3"),
        new("f16-d128-bias", "sdpa", DType.F16, 4, 4, 256, 256, 128, "bias", true, "S.3"),
        new("f16-d64-perhead", "sdpa", DType.F16, 4, 4, 256, 256, 64, "perhead", true, "S.3"),
        new("f16-cudnn-off", "sdpa", DType.F16, 4, 4, 256, 256, 128, "none", true, "S.15",
            Knobs: () => KnobStore.Set(EngineKnobs.SdpaCudnn, false)),
        new("f32-allowf16", "sdpa", DType.F32, 4, 4, 256, 256, 128, "none", true, "S.5"),
        new("f32", "sdpa", DType.F32, 4, 4, 256, 256, 128, "none", false, "S.14"),
        new("f32-forcetiled", "sdpa", DType.F32, 4, 4, 256, 256, 128, "none", false, "S.9",
            Knobs: () => KnobStore.Set(EngineKnobs.SdpaForceTiled, true)),
        new("f32-key-allowf16", "sdpa", DType.F32, 4, 4, 256, 256, 128, "key", true, "S.5"),
        new("f32-key-forcetiled", "sdpa", DType.F32, 4, 4, 256, 256, 128, "key", false, "S.10",
            Knobs: () => KnobStore.Set(EngineKnobs.SdpaForceTiled, true)),
        new("f32-perhead", "sdpa", DType.F32, 4, 4, 256, 256, 64, "perhead", false, "S.14"),
        new("f32-perhead-allowf16", "sdpa", DType.F32, 4, 4, 256, 256, 64, "perhead", true, "S.5"),
        new("f32-d40-allowf16", "sdpa", DType.F32, 4, 4, 256, 256, 40, "none", true, "S.5"),
        new("f32-allowf16-nof16", "sdpa", DType.F32, 4, 4, 256, 256, 128, "none", true, "S.14",
            Knobs: () => KnobStore.Set(EngineKnobs.SdpaNoF16, true)),
        new("bf16", "sdpa", DType.BF16, 4, 4, 256, 256, 128, "none", true, "S.12"),
        new("bf16-forcetiled", "sdpa", DType.BF16, 4, 4, 256, 256, 128, "none", true, "S.11",
            Knobs: () => KnobStore.Set(EngineKnobs.SdpaForceTiled, true)),
        new("tm-f16-d128", "tm", DType.F16, 4, 4, 256, 256, 128, "none", true, "M.2"),
        new("tm-f16-d256", "tm", DType.F16, 2, 2, 256, 256, 256, "none", true, "M.1"),
        new("tm-f16-d40", "tm", DType.F16, 4, 4, 256, 256, 40, "none", true, "M.2"),
        new("tm-f16-d128-8k", "tm", DType.F16, 2, 2, 8192, 8192, 128, "none", true, "M.2"),
        new("tm-f32-allowf16", "tm", DType.F32, 4, 4, 256, 256, 128, "none", true, "M.2"),
        new("tm-f32-d40-allowf16", "tm", DType.F32, 4, 4, 256, 256, 40, "none", true, "M.2"),
        new("tm-f32-d160-allowf16", "tm", DType.F32, 4, 4, 256, 256, 160, "none", true, "M.2"),
        new("tm-f32", "tm", DType.F32, 4, 4, 256, 256, 128, "none", false, "M.3/S.14"),
        new("tm-f16-d128-perhead", "tm", DType.F16, 4, 4, 256, 256, 128, "perhead", true, "M.2"),
        new("single-f16-d128", "single", DType.F16, 24, 24, 256, 256, 128, "none", true, "M.2"),
        new("single-f16-d128-bias", "single", DType.F16, 24, 24, 256, 256, 128, "bias", true, "M.2"),
        new("single-f32-allowf16", "single", DType.F32, 24, 24, 256, 256, 128, "none", true, "M.2"),
        new("tm-gqa-f16-d128", "tmg", DType.F16, 48, 12, 256, 256, 128, "none", true, "G.1"),
        new("tm-gqa-f16-16-8", "tmg", DType.F16, 16, 8, 256, 256, 128, "none", true, "G.1"),
        new("tm-gqa-f16-sage-8k", "tmg", DType.F16, 4, 2, 8192, 8192, 128, "none", true, "MG/S.1"),
        new("tm-gqa-f16-d256", "tmg", DType.F16, 4, 2, 256, 256, 256, "none", true, "MG/S.2"),
        new("tm-gqa-f32-allowf16", "tmg", DType.F32, 48, 12, 256, 256, 128, "none", true, "MG/S.5"),
        new("tm-gqa-f16-mha", "tmg", DType.F16, 4, 4, 256, 256, 128, "none", true, "MG/S.3"),
        new("tm-gqa-f16-key", "tmg", DType.F16, 48, 12, 256, 256, 128, "key", true, "G.1"),
        new("tm-gqa-f16-perhead", "tmg", DType.F16, 48, 12, 256, 256, 128, "perhead", true, "G.1"),
        new("krea2-main", "krea2", DType.F16, 48, 12, 256, 256, 128, "none", true, "G.1"),
        new("krea2-fusion-key", "krea2", DType.F16, 24, 24, 256, 256, 128, "key", true, "MG/S.3"),
        new("flash-decode", "flash", DType.F32, 8, 2, 1, 256, 128, "none", false, "F.4"),
        new("flash-decode-nogqa", "flash", DType.F32, 8, 2, 1, 256, 128, "none", false, "F.5",
            Knobs: () => KnobStore.Set(EngineKnobs.FlashDecodeGqa, false)),
        new("flash-decode-mono", "flash", DType.F32, 8, 2, 1, 256, 128, "none", false, "F.7",
            Knobs: () => { KnobStore.Set(EngineKnobs.FlashDecodeGqa, false); KnobStore.Set(EngineKnobs.FlashSplitOff, true); }),
        new("flash-decode-f16kv-mono", "flash", DType.F32, 8, 2, 1, 256, 128, "none", false, "F.6", KvDtype: DType.F16,
            Knobs: () => { KnobStore.Set(EngineKnobs.FlashDecodeGqa, false); KnobStore.Set(EngineKnobs.FlashSplitOff, true); }),
        new("flash-decode-sink", "flash", DType.F32, 8, 2, 1, 256, 128, "none", false, "F.7", Sink: true),
        new("flash-prefill-tight", "flash", DType.F32, 8, 2, 64, 64, 128, "none", false, "F.2"),
        new("flash-prefill-f16kv", "flash", DType.F32, 8, 2, 64, 64, 128, "none", false, "F.3", KvDtype: DType.F16),
        new("flash-prefill-cudnn-off", "flash", DType.F32, 8, 2, 64, 64, 128, "none", false, "F.3",
            Knobs: () => KnobStore.Set(EngineKnobs.SdpaCudnn, false)),
        new("flash-prefill-nofa2", "flash", DType.F32, 8, 2, 64, 64, 128, "none", false, "F.6", KvDtype: DType.F16,
            Knobs: () => KnobStore.Set(EngineKnobs.Fa2Prefill, false)),
        new("flash-d1032", "flash", DType.F32, 1, 1, 1, 4, 1032, "none", false, "F.1"),
        new("dev-decode", "dev", DType.F32, 8, 2, 1, 256, 128, "none", false, "V.2"),
        new("dev-decode-nogqa", "dev", DType.F32, 8, 2, 1, 256, 128, "none", false, "V.3",
            Knobs: () => KnobStore.Set(EngineKnobs.FlashDecodeGqa, false)),
        new("dev-short-f16kv", "dev", DType.F32, 8, 2, 1, 16, 128, "none", false, "V.4", KvDtype: DType.F16,
            Knobs: () => KnobStore.Set(EngineKnobs.FlashDecodeGqa, false)),
        new("dev-short", "dev", DType.F32, 8, 2, 1, 16, 128, "none", false, "V.5",
            Knobs: () => KnobStore.Set(EngineKnobs.FlashDecodeGqa, false)),
        new("dev-pos0", "dev0", DType.F32, 8, 2, 1, 256, 128, "none", false, "V.0/F.4"),
    ];

    public static IEnumerable<object[]> Names => Table.Select(c => new object[] { c.Name });

    private static readonly Action[] ClearKnobs =
    [
        () => KnobStore.Clear(EngineKnobs.SdpaCudnn), () => KnobStore.Clear(EngineKnobs.SdpaForceTiled),
        () => KnobStore.Clear(EngineKnobs.SdpaNoF16), () => KnobStore.Clear(EngineKnobs.FlashDecodeGqa),
        () => KnobStore.Clear(EngineKnobs.FlashSplitOff), () => KnobStore.Clear(EngineKnobs.Fa2Prefill),
    ];

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        return Directory.Exists(dir) ? dir : Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
    }

    private static Tensor Random(TensorShape shape, DType dt, int seed, float lo = -1f, float hi = 1f)
    {
        Tensor f32 = new(shape, DType.F32);
        float* p = (float*)f32.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < shape.ElementCount; i++) p[i] = lo + (float)rng.NextDouble() * (hi - lo);
        if (dt == DType.F32) return f32;
        try { return f32.CastTo(dt); } finally { f32.Dispose(); }
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [MemberData(nameof(Names))]
    public void TakesTheGoldenRoute(string name)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        CudnnRuntime.EnsureProbed();
        if (!CudnnRuntime.SupportsSdpa) { _output.WriteLine($"SKIPPED: {CudnnRuntime.Reason}"); return; }
        Case c = Table.Single(x => x.Name == name);
        foreach (Action clear in ClearKnobs) clear();
        c.Knobs?.Invoke();
        try
        {
            using CudaBackend backend = new(0, PtxDir());
            Run(backend, c);
            backend.Sync();
            _output.WriteLine($"{c.Name}: {CudaBackend.LastAttentionRoute}");
            Assert.Equal(c.Expected, CudaBackend.LastAttentionRoute);
        }
        finally
        {
            foreach (Action clear in ClearKnobs) clear();
        }
    }

    private static void Run(CudaBackend backend, Case c)
    {
        IBackend b = backend;
        float scale = 1f / MathF.Sqrt(c.D);
        if (c.Entry is "flash" or "dev" or "dev0")
        {
            DType kv = c.KvDtype ?? DType.F32;
            using Tensor q = Random(new TensorShape(1, c.Hq, c.Sq, c.D), DType.F32, 1);
            using Tensor k = Random(new TensorShape(1, c.Hkv, c.Skv, c.D), kv, 2);
            using Tensor v = Random(new TensorShape(1, c.Hkv, c.Skv, c.D), kv, 3);
            using Tensor o = new(new TensorShape(1, c.Hq, c.Sq, c.D), DType.F32);
            int qOffset = c.Skv - c.Sq;
            if (c.Entry == "flash")
            {
                using Tensor? sink = c.Sink ? Random(new TensorShape(c.Hq), DType.F32, 4) : null;
                b.FlashAttention(o, q, k, v, c.Skv, c.Hq / c.Hkv, causal: true, qOffset, scale, sink: sink);
                return;
            }
            ulong pos = c.Entry == "dev0" ? 0 : b.AllocDevicePos();
            try
            {
                if (pos != 0) b.WriteDevicePos(pos, c.Skv, qOffset);
                b.FlashAttentionDev(o, q, k, v, c.Skv, c.Hq / c.Hkv, causal: true, qOffset, scale, pos);
            }
            finally
            {
                if (pos != 0) b.FreeDevicePos(pos);
            }
            return;
        }
        bool tm = c.Entry is "tm" or "tmg" or "krea2" or "single";
        TensorShape qs = tm ? new(1, c.Sq, c.Hq, c.D) : new(1, c.Hq, c.Sq, c.D);
        TensorShape ks = tm ? new(1, c.Skv, c.Hkv, c.D) : new(1, c.Hkv, c.Skv, c.D);
        using Tensor qt = Random(qs, c.Dtype, 1);
        using Tensor kt = Random(ks, c.Dtype, 2);
        using Tensor vt = Random(ks, c.Dtype, 3);
        // Krea 2 and the single-stream blocks hand the entry a [1, S, H·D] output.
        using Tensor ot = new(c.Entry is "krea2" or "single" ? new TensorShape(1, c.Sq, c.Hq * c.D) : qs, c.Dtype);
        using Tensor? mask = c.Mask switch
        {
            "bias" => Random(new TensorShape(1, 1, c.Sq, c.Skv), DType.F32, 5, -2f, 2f),
            "perhead" => Random(new TensorShape(1, c.Hq, c.Sq, c.Skv), DType.F32, 5, -2f, 2f),
            "key" => Random(new TensorShape(1, 1, 1, c.Skv), DType.F32, 5, -2f, 2f),
            _ => null,
        };
        if (c.Entry is "tmg" or "krea2") b.ScaledDotProductAttentionTokenMajor(ot, qt, kt, vt, mask, c.Hq, c.Hkv, c.D, scale, c.AllowF16);
        else if (tm) b.ScaledDotProductAttentionTokenMajor(ot, qt, kt, vt, mask, c.Hq, c.D, scale, c.AllowF16);
        else b.ScaledDotProductAttention(ot, qt, kt, vt, mask, scale, c.AllowF16);
    }
}
