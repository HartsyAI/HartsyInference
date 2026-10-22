using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Cuda.Profiling;

namespace HartsyInference.GpuBenchmarks.Attention;

/// <summary>One way of calling attention: input dtype, the knobs that steer dispatch, and the kernel it must reach.</summary>
/// <param name="Expect">Kernel the arm exists to measure; a call that lands elsewhere is reported as not applicable
/// rather than timed under the wrong name. Null means "whatever production dispatch picks".</param>
/// <param name="Grant">Permissions the call carries through <see cref="IBackend.Attention"/>; null calls the legacy entry.</param>
public sealed record AttentionArm(string Name, string Entry, DType Dtype, DType KvDtype, bool AllowF16, string? Expect,
    Action Apply, string Description, AttentionNumerics? Grant = null);

/// <summary>The arm catalog the table runner times.</summary>
/// <remarks>Arms set knobs before the backend is constructed: <c>sdpaCudnn</c>, <c>flashF16</c>, <c>sdpaF16</c> and
/// <c>sdpaNoF16</c> are captured by the CUDA constructor, so setting them afterwards changes nothing.</remarks>
public static class AttentionArms
{
    private static readonly Action[] Clears =
    [
        () => KnobStore.Clear(EngineKnobs.SageAttn), () => KnobStore.Clear(EngineKnobs.SdpaCudnn),
        () => KnobStore.Clear(EngineKnobs.FlashF16), () => KnobStore.Clear(EngineKnobs.SageF16MinSkv),
        () => KnobStore.Clear(EngineKnobs.SdpaForceTiled), () => KnobStore.Clear(EngineKnobs.SdpaF16),
        () => KnobStore.Clear(EngineKnobs.SdpaNoF16), () => KnobStore.Clear(EngineKnobs.NoTf32),
        () => KnobStore.Clear(EngineKnobs.Fa2Prefill), () => KnobStore.Clear(EngineKnobs.FlashDecodeGqa),
        () => KnobStore.Clear(EngineKnobs.FlashSplitOff), () => KnobStore.Clear(EngineKnobs.FlashSplitForce),
        () => KnobStore.Clear(EngineKnobs.Profile), () => KnobStore.Clear(EngineKnobs.ProfileShapes),
    ];

    /// <summary>Knob ids whose provenance is printed with every row, so a settings-file override cannot pass as a default.</summary>
    public static readonly string[] ReportedKnobIds =
    [
        EngineKnobs.SageAttn.Id, EngineKnobs.SdpaCudnn.Id, EngineKnobs.FlashF16.Id, EngineKnobs.SageF16MinSkv.Id,
        EngineKnobs.SdpaForceTiled.Id, EngineKnobs.SdpaF16.Id, EngineKnobs.SdpaNoF16.Id, EngineKnobs.NoTf32.Id,
        EngineKnobs.Fa2Prefill.Id, EngineKnobs.FlashDecodeGqa.Id, EngineKnobs.FlashSplitOff.Id, EngineKnobs.FlashSplitForce.Id,
    ];

    /// <summary>Every arm the runner knows.</summary>
    public static readonly AttentionArm[] All =
    [
        new("def-f16", "sdpa", DType.F16, DType.F16, true, null, () => { },
            "native F16 I/O, production knobs"),
        new("cudnn-f16", "sdpa", DType.F16, DType.F16, true, "cudnn",
            () => { KnobStore.Set(EngineKnobs.SageAttn, false); KnobStore.Set(EngineKnobs.FlashF16, false); },
            "native F16 through cuDNN fused attention"),
        new("sage-f16", "sdpa", DType.F16, DType.F16, true, "sage-int8",
            () => KnobStore.Set(EngineKnobs.SageF16MinSkv, 1),
            "native F16 through SageAttention v1 INT8 at any length"),
        new("flash-f16", "sdpa", DType.F16, DType.F16, true, "flash-f16",
            () => KnobStore.Set(EngineKnobs.SageAttn, false),
            "native F16 through the engine's F16 flash kernel (D=256)"),
        new("mat-f16", "sdpa", DType.F16, DType.F16, true, "materialized",
            () => { KnobStore.Set(EngineKnobs.SageAttn, false); KnobStore.Set(EngineKnobs.SdpaCudnn, false); KnobStore.Set(EngineKnobs.FlashF16, false); },
            "native F16 through the materialized cuBLAS score matrix"),
        new("def-f32", "sdpa", DType.F32, DType.F32, false, null, () => { },
            "F32 I/O, allowF16=false: the unbounded-score caller"),
        new("tiled-f32", "sdpa", DType.F32, DType.F32, false, "tiled",
            () => KnobStore.Set(EngineKnobs.SdpaForceTiled, true),
            "F32 I/O through the query-tiled core"),
        new("cast-f16", "sdpa", DType.F32, DType.F32, true, null, () => { },
            "F32 I/O, allowF16=true: the bounded-score caller (casts to F16 inside)"),
        new("bf16-cast", "sdpa", DType.F32, DType.F32, false, "cudnn", () => { },
            "F32 I/O with the BF16 grant: cuDNN on BF16 copies (S.16)", AttentionNumerics.Bf16Storage),
        new("bf16", "sdpa", DType.BF16, DType.BF16, true, null, () => { },
            "BF16 I/O, production knobs"),
        new("fa-def", "flash", DType.F32, DType.F32, false, null, () => { },
            "LLM entry, F32 Q/O and F32 K/V, production knobs"),
        new("fa-f16kv", "flash", DType.F32, DType.F16, false, null, () => { },
            "LLM entry, F16 K/V cache storage"),
        new("fa-nocudnn", "flash", DType.F32, DType.F32, false, null,
            () => KnobStore.Set(EngineKnobs.SdpaCudnn, false),
            "LLM entry without the cuDNN causal-prefill route"),
        new("fa-nofa2", "flash", DType.F32, DType.F16, false, null,
            () => KnobStore.Set(EngineKnobs.Fa2Prefill, false),
            "LLM entry, F16 K/V, FA2 prefill off (the per-row kernel)"),
        new("fa-nogqa", "flash", DType.F32, DType.F16, false, null,
            () => KnobStore.Set(EngineKnobs.FlashDecodeGqa, false),
            "LLM entry, F16 K/V, grouped decode kernel off"),
        new("fa-splitoff", "flash", DType.F32, DType.F16, false, null,
            () => { KnobStore.Set(EngineKnobs.FlashDecodeGqa, false); KnobStore.Set(EngineKnobs.FlashSplitOff, true); },
            "LLM entry, F16 K/V, monolithic decode"),
        new("fa-splitforce", "flash", DType.F32, DType.F16, false, null,
            () => { KnobStore.Set(EngineKnobs.FlashDecodeGqa, false); KnobStore.Set(EngineKnobs.FlashSplitForce, true); },
            "LLM entry, F16 K/V, split-K decode forced"),
    ];

    /// <summary>Looks an arm up by name.</summary>
    public static AttentionArm Get(string name) =>
        All.FirstOrDefault(a => a.Name == name) ?? throw new ArgumentException($"unknown arm '{name}'; known: {string.Join(", ", All.Select(a => a.Name))}");

    /// <summary>Drops every override the catalog can set, so the next arm starts from the declared defaults.</summary>
    public static void ClearAll()
    {
        foreach (Action clear in Clears) clear();
    }

    /// <summary><c>id=source</c> for every reported knob.</summary>
    public static string Provenance() =>
        string.Join(",", ReportedKnobIds.Select(id => $"{id}={KnobStore.SourceOf(id)}"));

    /// <summary>Execution counters of the kernels that keep one.</summary>
    public readonly record struct Counters(long Sage, long Cudnn, long FlashF16, long Tiled)
    {
        /// <summary>Reads the counters; all zero on a backend that keeps none.</summary>
        public static Counters Read(IBackend backend) => backend is CudaBackend c
            ? new(c.SageAttentionExecutionCount, c.CudnnSdpaExecutionCount, c.FlashF16ExecutionCount, c.SdpaTiledExecutionCount)
            : default;
    }

    /// <summary>Runs <paramref name="call"/> once with the profiler on and names the kernel that served it.</summary>
    /// <remarks>Counters name the kernels that keep one; profiler labels name the FA2 prefill kernel; anything else on the
    /// dense entry is the materialized path and anything else on the LLM entry is the decode family.</remarks>
    public static string Identify(IBackend backend, string entry, Action call)
    {
        if (backend is not CudaBackend)
        {
            call();
            backend.Sync();
            return backend.Capabilities.Name.StartsWith("Vulkan", StringComparison.Ordinal) ? "vulkan" : "cpu";
        }
        Counters before = Counters.Read(backend);
        KnobStore.Set(EngineKnobs.Profile, true);
        NvtxRange.ResetProfile();
        HashSet<string> labels;
        try
        {
            call();
            backend.Sync();
            string tmp = Path.Combine(Path.GetTempPath(), $"attn-ident-{Environment.ProcessId}.txt");
            NvtxRange.DumpProfile(tmp);
            labels = File.Exists(tmp)
                ? File.ReadLines(tmp).Skip(1).Select(l => l.Split(' ', StringSplitOptions.RemoveEmptyEntries).FirstOrDefault() ?? "").ToHashSet()
                : [];
            File.Delete(tmp);
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.Profile);
            NvtxRange.ResetProfile();
        }
        Counters after = Counters.Read(backend);
        if (after.Sage > before.Sage) return "sage-int8";
        if (after.FlashF16 > before.FlashF16) return "flash-f16";
        if (after.Cudnn > before.Cudnn) return entry == "flash" ? "cudnn-causal" : "cudnn";
        if (after.Tiled > before.Tiled) return "tiled";
        if (entry == "flash") return labels.Contains("FlashAttention.Fa2") ? "fa2-causal" : "flash-decode";
        return "materialized";
    }
}
