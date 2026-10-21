using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>A fact that runs only under tests/cudnn-probes.sh, which pins one card and names it in
/// HARTSY_CUDNN_PROBE_GPU. Anywhere else it reports Skipped, never a silent pass.</summary>
public sealed class CudnnProbeFactAttribute : FactAttribute
{
    public CudnnProbeFactAttribute()
    {
        if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable(CudnnSdpaProbe.GpuEnvVar)))
            Skip = $"cuDNN capability probe: run it through tests/cudnn-probes.sh ({CudnnSdpaProbe.GpuEnvVar} is unset).";
    }
}

/// <summary>cuDNN 9.24 SDPA capability probes P0–P15 (attention plan Phase 0 item 6; probe plan in the cuDNN graph
/// research report §3). One fact per probe, one process per fact (tests/cudnn-probes.sh). Every spec is planned with
/// each heuristic mode, executed and compared against a float64 reference. A capability no engine accepts is recorded
/// as unsupported. The fact fails on a wrong environment, a plan that finalizes and runs wrong, a refused production
/// baseline, or an accepted negative control. Each fact writes one JSON line to HARTSY_CUDNN_PROBE_OUT (and its
/// per-mode rows to the same path + ".tsv"), or to the test output when that is unset.</summary>
[Collection("CudaSerial")]
public sealed class CudnnSdpaCapabilityProbeTests
{
    private const string OutEnvVar = "HARTSY_CUDNN_PROBE_OUT";
    private const long FluxHeads = 24, FluxSeq = 4608, FluxDim = 128;

    private static readonly JsonSerializerOptions _json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
        Converters = { new JsonStringEnumConverter() },
    };

    private readonly ITestOutputHelper _output;
    public CudnnSdpaCapabilityProbeTests(ITestOutputHelper output) => _output = output;

    [CudnnProbeFact]
    [Trait("Category", "GpuIntegration")]
    [Trait("Category", "CudnnProbe")]
    public void P00_BaselineBuildCost() => Run("P0",
    [
        Flux("flux-4608") with { Expect = ProbeExpect.Required, Note = "today's production graph" },
        .. new long[] { 1024, 2048, 4096, 8192 }.Select(n => Flux($"flux-s{n}") with { Sq = n, Skv = n }),
    ]);

    [CudnnProbeFact]
    [Trait("Category", "GpuIntegration")]
    [Trait("Category", "CudnnProbe")]
    public void P01_Bf16() => Run("P1",
    [
        Flux("flux-bf16-head-major") with { Dtype = ProbeDtype.BF16 },
        Flux("flux-bf16-token-major") with { Dtype = ProbeDtype.BF16, Layout = ProbeLayout.TokenMajor },
    ]);

    [CudnnProbeFact]
    [Trait("Category", "GpuIntegration")]
    [Trait("Category", "CudnnProbe")]
    public void P02_Gqa()
    {
        ProbeSpec krea2 = Spec("krea2-gqa-48-12", 48, 4608, 4608, 128, hkv: 12) with { Layout = ProbeLayout.TokenMajor };
        Run("P2",
        [
            krea2,
            krea2 with { Name = "krea2-gqa-48-12-f32-bias", Bias = new ProbeBias(1, 1, 4608) },
            Llm("llm-gqa-16-8-s8192", 8, 8192, 8192),
            krea2 with
            {
                Name = "p2b-kv-head-stride0", KvHeadStrideZero = true, Layout = ProbeLayout.HeadMajor, Expect = ProbeExpect.Unknown,
                Note = "diagnostic: one stored K/V head declared as 48 heads with head stride 0",
            },
        ]);
    }

    [CudnnProbeFact]
    [Trait("Category", "GpuIntegration")]
    [Trait("Category", "CudnnProbe")]
    public void P03_HeadDims()
    {
        List<ProbeSpec> specs =
        [
            Spec("sd15-self-d40", 8, 4096, 4096, 40, b: 2),
            Spec("sd15-cross-d40", 8, 4096, 77, 40, b: 2),
        ];
        foreach (long d in new long[] { 32, 48, 80, 96, 160, 192, 256 })
            specs.Add(Spec($"d{d}", 8, 1024, 1024, d));
        foreach (long d in new long[] { 36, 44 })
            specs.Add(Spec($"neg-d{d}", 8, 1024, 1024, d) with { Expect = ProbeExpect.Refused, Note = "negative control: not a multiple of 8" });
        specs.Add(Spec("decode-d256", 8, 1, 4096, 256) with { Expect = ProbeExpect.Unknown, Note = "docs: decode head dim <= 128" });
        Run("P3", [.. specs]);
    }

    [CudnnProbeFact]
    [Trait("Category", "GpuIntegration")]
    [Trait("Category", "CudnnProbe")]
    public void P04_BiasShapes()
    {
        ProbeSpec t5 = Spec("t5-24h-f32-bias", 24, 512, 512, 64) with { Bias = new ProbeBias(1, 24, 512) };
        ProbeSpec keyOnly = Spec("key-only-bias", 4, 128, 512, 128) with
        {
            Bias = new ProbeBias(1, 1, 1), Expect = ProbeExpect.Unknown, Note = "undocumented [1,1,1,Skv] form",
        };
        Run("P4",
        [
            t5,
            t5 with { Name = "t5-24h-f16-bias", Bias = new ProbeBias(1, 24, 512, ProbeBiasType.F16) },
            t5 with { Name = "t5xxl-64h-f32-bias", Hq = 64, Bias = new ProbeBias(1, 64, 512) },
            Spec("batch-bias-2x1", 24, 1024, 1024, 64, b: 2) with { Bias = new ProbeBias(2, 1, 1024) },
            Spec("batch-head-bias-2x24", 24, 1024, 1024, 64, b: 2) with { Bias = new ProbeBias(2, 24, 1024) },
            keyOnly,
            keyOnly with { Name = "sq-stride0-bias", Bias = new ProbeBias(1, 1, 128, SqStrideZero: true), Note = "[1,1,Sq,Skv] with Sq stride 0" },
        ]);
    }

    [CudnnProbeFact]
    [Trait("Category", "GpuIntegration")]
    [Trait("Category", "CudnnProbe")]
    public void P05_Causal()
    {
        ProbeSpec c1 = Llm("c1-band-gqa-16-8", 8, 8192, 8192) with { Causal = ProbeCausal.TopLeft };
        ProbeSpec resume = Llm("p5b-bottom-right-resume", 8, 4096, 8192) with { Causal = ProbeCausal.BottomRight };
        Run("P5",
        [
            c1,
            c1 with { Name = "c1-band-mha-16-16", Hkv = 16 },
            c1 with { Name = "c1-band-fill-1e30", Fill = -1e30f, Expect = ProbeExpect.Unknown },
            c1 with { Name = "c2-unified-band-subgraph", Unified = true, Expect = ProbeExpect.Unknown },
            c1 with
            {
                Name = "c3-legacy-chain", MaskForm = ProbeMaskForm.LegacyChain, Expect = ProbeExpect.Unknown,
                Note = "diagnostic: the pre-9.21 GEN_INDEX/CMP_GE/BINARY_SELECT chain",
            },
            resume,
            resume with { Name = "p5c-top-left-shift-4096", Causal = ProbeCausal.TopLeft, ShiftRightBound = 4096 },
            Llm("neg-decode-causal-gqa", 8, 1, 4096) with
            {
                Causal = ProbeCausal.BottomRight, Expect = ProbeExpect.KnownIssue,
                Note = "9.24 known issue: s_q = 1 causal with Hq != Hkv may mismatch or NaN",
            },
        ]);
    }

    [CudnnProbeFact]
    [Trait("Category", "GpuIntegration")]
    [Trait("Category", "CudnnProbe")]
    public void P06_SlidingWindow()
    {
        ProbeSpec window = Llm("window-128-int32", 8, 8192, 8192) with { Causal = ProbeCausal.TopLeft, Window = 128 };
        Run("P6",
        [
            window,
            window with { Name = "window-128-float", BoundType = ProbeBoundType.Float },
            window with { Name = "window-4096-int32", Window = 4096 },
            window with { Name = "window-4096-float", Window = 4096, BoundType = ProbeBoundType.Float },
            Llm("p6b-noncausal-window-512", 8, 2048, 2048) with
            {
                Window = 512, Expect = ProbeExpect.Unknown, Note = "9.14 known-issue zone: non-causal window",
            },
        ]);
    }

    [CudnnProbeFact]
    [Trait("Category", "GpuIntegration")]
    [Trait("Category", "CudnnProbe")]
    public void P07_Padding()
    {
        int[] lengths = [1024, 700, 513, 1];
        ProbeSpec padded = Spec("composite-padding", 8, 1024, 1024, 128, b: 4) with
        {
            Padding = ProbePadding.Composite, SeqQ = lengths, SeqKv = lengths,
        };
        Run("P7",
        [
            padded,
            padded with { Name = "composite-padding-kv-capacity-2048", KvCapacity = 2048 },
            padded with { Name = "p7b-unified-seq-len", Padding = ProbePadding.Unified, Unified = true, Expect = ProbeExpect.Unknown },
            padded with
            {
                Name = "p7b-unified-seq-len-kv-capacity-2048", Padding = ProbePadding.Unified, Unified = true, KvCapacity = 2048,
                Expect = ProbeExpect.Unknown,
            },
        ]);
    }

    [CudnnProbeFact(Skip = "P8 (paged KV through PAGED_CACHE_LOAD) is deferred by the attention plan: no Phase 1 consumer.")]
    [Trait("Category", "GpuIntegration")]
    [Trait("Category", "CudnnProbe")]
    public void P08_PagedKv()
    {
    }

    [CudnnProbeFact]
    [Trait("Category", "GpuIntegration")]
    [Trait("Category", "CudnnProbe")]
    public void P09_Alibi() => Run("P9",
    [
        Spec("alibi-causal-16h", 16, 2048, 2048, 128) with { Alibi = true, Causal = ProbeCausal.TopLeft },
    ]);

    [CudnnProbeFact]
    [Trait("Category", "GpuIntegration")]
    [Trait("Category", "CudnnProbe")]
    public void P10_Softcap() => Run("P10",
    [
        Spec("softcap-50-window-4096", 8, 4096, 4096, 256, hkv: 4) with
        {
            Softcap = 50, Causal = ProbeCausal.TopLeft, Window = 4096, Expect = ProbeExpect.Unknown,
        },
    ]);

    [CudnnProbeFact]
    [Trait("Category", "GpuIntegration")]
    [Trait("Category", "CudnnProbe")]
    public void P11_Sinks() => Run("P11",
    [
        Spec("sink-64-8-window-128", 64, 4096, 4096, 64, hkv: 8) with { Sink = true, Causal = ProbeCausal.TopLeft, Window = 128 },
    ]);

    [CudnnProbeFact]
    [Trait("Category", "GpuIntegration")]
    [Trait("Category", "CudnnProbe")]
    public void P12_LseStats() => Run("P12", [Flux("flux-lse-stats") with { Stats = true }]);

    [CudnnProbeFact]
    [Trait("Category", "GpuIntegration")]
    [Trait("Category", "CudnnProbe")]
    public void P13_UnifiedOp()
    {
        ProbeSpec t5 = Spec("unified-t5-f16-bias", 24, 512, 512, 64) with
        {
            Unified = true, Bias = new ProbeBias(1, 24, 512, ProbeBiasType.F16), Expect = ProbeExpect.Unknown,
        };
        Run("P13",
        [
            Flux("unified-flux") with { Unified = true, Expect = ProbeExpect.Unknown },
            t5,
            t5 with { Name = "unified-t5-f32-bias", Bias = new ProbeBias(1, 24, 512) },
            Spec("unified-krea2-gqa-48-12", 48, 4608, 4608, 128, hkv: 12) with
            {
                Unified = true, Layout = ProbeLayout.TokenMajor, Expect = ProbeExpect.Unknown,
            },
        ]);
    }

    [CudnnProbeFact]
    [Trait("Category", "GpuIntegration")]
    [Trait("Category", "CudnnProbe")]
    public void P14_BlockMask() => Run("P14",
    [
        Flux("unified-block-mask") with
        {
            Unified = true, BlockMask = true, Expect = ProbeExpect.Unknown,
            Note = "one INT8 per 128x128 tile, nonzero attends; the layout is the research report's reading, unconfirmed",
        },
    ]);

    [CudnnProbeFact]
    [Trait("Category", "GpuIntegration")]
    [Trait("Category", "CudnnProbe")]
    public void P15_ByValueScale()
    {
        float scale = 1f / MathF.Sqrt(FluxDim);
        Run("P15",
        [
            Flux("flux-device-scale") with { Seed = 15, KeepOutput = true, Expect = ProbeExpect.Required },
            Flux("flux-by-value-scale") with
            {
                Seed = 15, ScaleByValue = true, Scales = [scale, 0.05f, 0.3f], KeepOutput = true,
                Note = "one plan executed with three scales",
            },
        ], results =>
        {
            ushort[]? device = results[0].Output, byValue = results[1].Output;
            object? identical = device is null || byValue is null ? null : device.AsSpan().SequenceEqual(byValue);
            return new Dictionary<string, object?> { ["byteIdenticalToDeviceScale"] = identical };
        });
    }

    private static ProbeSpec Spec(string name, long hq, long sq, long skv, long d, long hkv = 0, long b = 1) =>
        new() { Name = name, B = b, Hq = hq, Hkv = hkv, Sq = sq, Skv = skv, D = d };

    private static ProbeSpec Flux(string name) => Spec(name, FluxHeads, FluxSeq, FluxSeq, FluxDim);

    private static ProbeSpec Llm(string name, long hkv, long sq, long skv) => Spec(name, 16, sq, skv, 128, hkv);

    private void Run(string probeId, ProbeSpec[] specs,
        Func<IReadOnlyList<ProbeSpecResult>, Dictionary<string, object?>>? extras = null)
    {
        List<ProbeSpecResult> results = new();
        CudnnProbeEnvironment? env = null;
        IReadOnlyList<string> libraryProblems = [];
        Dictionary<string, object?>? extra = null;
        string? aborted = null;
        try
        {
            using CudaContext ctx = new(0);
            env = CudnnSdpaProbe.Inspect(ctx);
            Assert.True(env.Problems.Count == 0, $"{probeId}: environment rejected, nothing ran: {string.Join("; ", env.Problems)}");
            using (CudnnSdpaProbe probe = new(ctx, env))
            {
                foreach (ProbeSpec spec in specs)
                {
                    ProbeSpecResult result = probe.Run(spec);
                    results.Add(result);
                    _output.WriteLine($"{probeId} {spec.Name}: {result.Outcome} [{string.Join(", ", result.Modes.Select(m => $"{m.Mode}={m.Verdict}"))}]");
                }
                libraryProblems = probe.LibraryProblems();
            }
            extra = extras?.Invoke(results);
        }
        catch (Exception ex)
        {
            aborted = $"{ex.GetType().Name}: {ex.Message}";
            throw;
        }
        finally
        {
            Emit(probeId, env, results, libraryProblems, extra, aborted);
        }
        List<string> failures = [.. libraryProblems.Select(p => $"library: {p}"), .. Judge(results)];
        Assert.True(failures.Count == 0, $"{probeId} failed:\n  " + string.Join("\n  ", failures));
    }

    private static IEnumerable<string> Judge(IReadOnlyList<ProbeSpecResult> results)
    {
        foreach (ProbeSpecResult r in results)
        {
            ProbeSpec s = r.Spec;
            ProbeModeResult[] bad = [.. r.Modes.Where(m => m.Verdict is CudnnSdpaProbe.VerdictWrong or CudnnSdpaProbe.VerdictExecError)];
            if (bad.Length > 0 && s.Expect != ProbeExpect.KnownIssue)
                yield return $"{s.Name}: a plan finalized and then ran wrong: " + string.Join("; ", bad.Select(Describe));
            if (s.Expect == ProbeExpect.Required && !r.Modes.Any(m => m.Verdict == CudnnSdpaProbe.VerdictOk))
                yield return $"{s.Name}: the production graph was refused or wrong in every heuristic mode: {r.Modes.FirstOrDefault()?.Error}";
            if (s.Expect == ProbeExpect.Refused && r.Modes.Any(m => m.Finalized))
                yield return $"{s.Name}: negative control was accepted by mode(s) "
                    + string.Join(", ", r.Modes.Where(m => m.Finalized).Select(m => m.Mode).Distinct());
        }
    }

    private static string Describe(ProbeModeResult m) => string.Create(CultureInfo.InvariantCulture,
        $"{m.Mode} scale {m.Scale:G4} relL2 {m.RelL2:E2} maxAbs {m.MaxAbs:E2} NaN {m.NanCount} {m.Error}");

    private void Emit(string probeId, CudnnProbeEnvironment? env, List<ProbeSpecResult> results,
        IReadOnlyList<string> libraryProblems, Dictionary<string, object?>? extras, string? aborted)
    {
        var record = new
        {
            probe = probeId,
            utc = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture),
            environment = env,
            libraryProblemsAfterRun = libraryProblems,
            aborted,
            extras,
            specs = results,
        };
        string line = JsonSerializer.Serialize(record, _json);
        _output.WriteLine(line);
        string? path = Environment.GetEnvironmentVariable(OutEnvVar);
        if (string.IsNullOrEmpty(path))
            return;
        File.AppendAllText(path, line + "\n");
        StringBuilder tsv = new();
        foreach (ProbeSpecResult r in results)
            foreach (ProbeModeResult m in r.Modes)
                tsv.Append(string.Create(CultureInfo.InvariantCulture,
                    $"{probeId}\t{r.Spec.Name}\t{r.Spec.Expect}\t{r.Outcome}\t{m.Mode}\t{m.Scale:G4}\t{m.Verdict}\t{m.RelL2:E2}\t{m.MaxAbs:E2}\t{m.PlanBuildMs:F1}\t{m.ExecMedianMs:F3}\t{m.EngineGlobalIndex}\t{Clip(m.Error)}\n"));
        if (aborted is not null)
            tsv.Append($"{probeId}\t-\t-\tABORTED\t-\t-\t-\t-\t-\t-\t-\t-\t{Clip(aborted)}\n");
        File.AppendAllText(path + ".tsv", tsv.ToString());
    }

    private static string Clip(string? text)
    {
        if (string.IsNullOrEmpty(text))
            return "-";
        string flat = text.Replace('\t', ' ').Replace('\n', ' ').Replace('\r', ' ');
        return flat.Length <= 200 ? flat : flat[..200];
    }
}
