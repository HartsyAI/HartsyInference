using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.Kyutai;
using HartsyInference.Cpu;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>The definitive validation of the Kyutai generation LOOP (<see cref="MoshiTtsGenerator.Generate"/>):
/// the ring-buffer acoustic-delay feedback + the text scheduler integration. The Python reference drives moshi's
/// own authoritative <c>LMGen</c> + <c>StateMachine</c> greedily on fixed entries + a fixed voice and dumps the
/// emitted codes; this replays the SAME greedy decode in C# and requires the codes to match bit-for-bit (greedy,
/// so any loop-logic bug diverges). Gated on <c>KYUTAI_TTS_WEIGHTS</c> + <c>KYUTAI_REF_GENERATE</c>; optional
/// <c>GSV_CUDA</c>/<c>GSV_PTX</c>.</summary>
public sealed unsafe class KyutaiGenerateParityTests
{
    private readonly ITestOutputHelper _out;
    public KyutaiGenerateParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void Generate_MatchesMoshiGreedyReference()
    {
        string? wp = Environment.GetEnvironmentVariable("KYUTAI_TTS_WEIGHTS");
        string? rp = Environment.GetEnvironmentVariable("KYUTAI_REF_GENERATE");
        if (string.IsNullOrEmpty(wp) || !File.Exists(wp) || string.IsNullOrEmpty(rp) || !File.Exists(rp)) return;

        using SafeTensorsLoader weights = new(); weights.Load(wp);
        using SafeTensorsLoader io = new(); io.Load(rp);
        IReadOnlyDictionary<string, Tensor> w = weights.GetAllTensors();
        IReadOnlyDictionary<string, Tensor> d = io.GetAllTensors();

        Tensor refCodes = d["codes"];   // [32, R] int32
        Tensor voiceT = d["voice"];     // [1,8,512] f32
        int refN = (int)refCodes.Shape[1];

        // Rebuild the exact Entry list from the dumped tokens (entry_0, entry_1, ...).
        List<KyutaiTextScheduler.Entry> entries = new();
        for (int i = 0; d.ContainsKey($"entry_{i}"); i++)
        {
            Tensor e = d[$"entry_{i}"];
            int len = (int)e.Shape[0]; int* ep = (int*)e.DataPointer;
            int[] toks = new int[len]; for (int j = 0; j < len; j++) toks[j] = ep[j];
            entries.Add(new KyutaiTextScheduler.Entry(toks, $"w{i}"));
        }

        using MoshiTtsGenerator gen = new();
        gen.LoadWeights(w);
        gen.SetZeroToken(-1);
        IBackend backend = Environment.GetEnvironmentVariable("GSV_CUDA") == "1"
            ? new HartsyInference.Cuda.CudaBackend(0, Environment.GetEnvironmentVariable("GSV_PTX")!)
            : new CpuBackend();

        using Tensor cross = gen.Conditioner.ComputeCross(backend, voiceT);
        using Tensor sumCond = gen.Conditioner.ComputeSum(backend, MoshiConditioner.CfgBin(1.0f));
        KyutaiTextScheduler scheduler = new();
        int[,] mine = gen.Generate(backend, scheduler, entries, cross, sumCond, maxFrames: refN + 6);
        (backend as IDisposable)?.Dispose();
        int n = mine.GetLength(1);

        // The reference includes the leading forced-silence (-1) warmup frames; my Generate trims them.
        // Align by the reference's first fully-valid (all codebooks in [0,2048)) frame.
        int* rc = (int*)refCodes.DataPointer;
        int firstValid = 0;
        while (firstValid < refN && !RefFrameValid(rc, refN, firstValid)) firstValid++;
        int refValidN = refN - firstValid;
        _out.WriteLine($"entries={entries.Count}; mine={n} frames; ref={refN} (firstValid={firstValid} → {refValidN} real).");

        Assert.True(n > 0, "C# generation emitted no valid frames");
        int compare = Math.Min(n, refValidN);
        int mismatches = 0; int firstMismatchCb = -1, firstMismatchF = -1;
        for (int f = 0; f < compare; f++)
            for (int k = 0; k < MoshiTtsGenerator.NumCodebooks; k++)
                if (mine[k, f] != rc[(long)k * refN + (firstValid + f)])
                {
                    if (mismatches == 0) { firstMismatchCb = k; firstMismatchF = f; }
                    mismatches++;
                }
        _out.WriteLine($"compared {compare} frames × 32 cb: {mismatches} mismatches" +
            (mismatches > 0 ? $" (first at frame {firstMismatchF} cb {firstMismatchCb}: mine={mine[firstMismatchCb, firstMismatchF]} ref={rc[(long)firstMismatchCb * refN + firstValid + firstMismatchF]})" : ""));
        _out.WriteLine($"  mine f0[:8]=[{Row(mine, 0, 8)}]");
        _out.WriteLine($"  ref  f0[:8]=[{RefRow(rc, refN, firstValid, 8)}]");
        Assert.Equal(refValidN, n);
        Assert.Equal(0, mismatches);
    }

    private static bool RefFrameValid(int* rc, int refN, int f)
    {
        for (int k = 0; k < MoshiTtsGenerator.NumCodebooks; k++)
        {
            int c = rc[(long)k * refN + f];
            if (c < 0 || c >= MoshiTtsGenerator.AudioCard) return false;
        }
        return true;
    }

    private static string Row(int[,] m, int f, int cnt)
    {
        string[] s = new string[cnt];
        for (int k = 0; k < cnt; k++) s[k] = m[k, f].ToString();
        return string.Join(",", s);
    }

    private static string RefRow(int* rc, int refN, int f, int cnt)
    {
        string[] s = new string[cnt];
        for (int k = 0; k < cnt; k++) s[k] = rc[(long)k * refN + f].ToString();
        return string.Join(",", s);
    }
}
