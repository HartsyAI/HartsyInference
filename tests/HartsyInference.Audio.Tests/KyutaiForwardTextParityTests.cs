using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.Kyutai;
using HartsyInference.Cpu;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>End-to-end parity for the Kyutai TTS "forward_text" front (<see cref="MoshiTtsGenerator"/>): the
/// demuxing text embedding + 32 audio-code embeddings + sum-condition, run through the backbone and the text
/// head. Diffs both the per-frame context (transformer_out) and the text logits against the moshi reference
/// (whose fixed code sequence includes a multiplexed token to exercise the demux). Gated on
/// <c>KYUTAI_TTS_WEIGHTS</c> + <c>KYUTAI_REF_FWDTEXT</c>.</summary>
public sealed unsafe class KyutaiForwardTextParityTests
{
    private readonly ITestOutputHelper _out;
    public KyutaiForwardTextParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void ForwardText_MatchesMoshiReference()
    {
        string? wp = Environment.GetEnvironmentVariable("KYUTAI_TTS_WEIGHTS");
        string? rp = Environment.GetEnvironmentVariable("KYUTAI_REF_FWDTEXT");
        if (string.IsNullOrEmpty(wp) || !File.Exists(wp) || string.IsNullOrEmpty(rp) || !File.Exists(rp)) return;

        using SafeTensorsLoader weights = new(); weights.Load(wp);
        using SafeTensorsLoader io = new(); io.Load(rp);
        IReadOnlyDictionary<string, Tensor> w = weights.GetAllTensors();
        IReadOnlyDictionary<string, Tensor> d = io.GetAllTensors();

        Tensor seq = d["seq"];                 // [1,33,T] int32
        Tensor sumCond = d["sum_cond"];        // [1,1,2048]
        Tensor cross = d["cross"];             // [1,5,2048]
        Tensor refTout = d["transformer_out"]; // [1,T,2048]
        Tensor refLogits = d["text_logits"];   // [1,T,8000]
        int t = (int)seq.Shape[2];
        int* sp = (int*)seq.DataPointer;

        using MoshiTtsGenerator gen = new();
        gen.LoadWeights(w);
        IBackend backend = Environment.GetEnvironmentVariable("GSV_CUDA")=="1" ? new HartsyInference.Cuda.CudaBackend(0, Environment.GetEnvironmentVariable("GSV_PTX")!) : new CpuBackend();

        List<Tensor> frames = new();
        for (int f = 0; f < t; f++)
        {
            int textTok = sp[0 * t + f];
            int[] codes = new int[MoshiTtsGenerator.NumCodebooks];
            for (int cb = 0; cb < MoshiTtsGenerator.NumCodebooks; cb++) codes[cb] = sp[(1 + cb) * t + f];
            frames.Add(gen.EmbedFrame(backend, textTok, codes));
        }
        using Tensor tout = gen.ForwardText(backend, frames, sumCond, cross, out Tensor textLogits);
        foreach (Tensor f in frames) f.Dispose();

        double toutMax = MaxAbs((float*)tout.DataPointer, (float*)refTout.DataPointer, (long)t * MoshiTtsGenerator.Dim);
        double logitMax = MaxAbs((float*)textLogits.DataPointer, (float*)refLogits.DataPointer, (long)t * MoshiTtsGenerator.TextCard);
        textLogits.Dispose();
        _out.WriteLine($"forward_text T={t}: transformer_out maxAbs={toutMax:E4}; text_logits maxAbs={logitMax:E4}.");
        Assert.True(toutMax < 5e-3, $"transformer_out diverges ({toutMax:E4}).");
        Assert.True(logitMax < 2e-2, $"text_logits diverges ({logitMax:E4}).");
    }

    private static double MaxAbs(float* a, float* b, long n)
    {
        double m = 0;
        for (long i = 0; i < n; i++) m = Math.Max(m, Math.Abs(a[i] - b[i]));
        return m;
    }
}
