using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.Codecs.Mimi;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Parity for Mimi's split EMA-RVQ decode (codes -> quantized embeddings) vs `transformers` MimiModel.
/// Gated on <c>MIMI_WEIGHTS</c> (kyutai/mimi safetensors) + <c>MIMI_RVQ_REF</c> (dump_mimi_rvq.py).</summary>
public sealed unsafe class MimiRvqParityTests
{
    private readonly ITestOutputHelper _out;
    public MimiRvqParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void SplitRvqDecode_MatchesTransformers()
    {
        string? wPath = Environment.GetEnvironmentVariable("MIMI_WEIGHTS");
        string? refP = Environment.GetEnvironmentVariable("MIMI_RVQ_REF");
        if (string.IsNullOrEmpty(wPath) || !File.Exists(wPath) || string.IsNullOrEmpty(refP) || !File.Exists(refP))
            return;

        SafeTensorsLoader wl = new(); wl.Load(wPath);
        IReadOnlyDictionary<string, Tensor> w = wl.GetAllTensors();
        SafeTensorsLoader rl = new(); rl.Load(refP);
        IReadOnlyDictionary<string, Tensor> r = rl.GetAllTensors();

        Tensor codesRef = r["codes"];   // [1,32,T] int32
        Tensor quantRef = r["quant"];   // [1,512,T]
        int nq = (int)codesRef.Shape[1];
        int t = (int)codesRef.Shape[2];

        MimiSplitRvq rvq = new("quantizer", numSemantic: 1, numTotal: nq, codebookDim: 256, latentDim: 512);
        rvq.LoadWeights(w);

        using CpuBackend backend = new();
        Tensor quant = rvq.Decode(backend, codesRef, batch: 1, t: t);
        Assert.Equal((int)quantRef.ElementCount, (int)quant.ElementCount);

        float* a = (float*)quant.DataPointer; float* b = (float*)quantRef.DataPointer;
        long n = quant.ElementCount;
        double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0, mx = 0;
        for (long i = 0; i < n; i++)
        {
            double x = a[i], y = b[i];
            sa += x; sb += y; saa += x * x; sbb += y * y; sab += x * y;
            mx = Math.Max(mx, Math.Abs(x - y));
        }
        double cov = sab - sa * sb / n, va = saa - sa * sa / n, vb = sbb - sb * sb / n;
        double corr = cov / (Math.Sqrt(va * vb) + 1e-12);
        _out.WriteLine($"Mimi split-RVQ decode corr={corr:F6}  maxAbs={mx:E4}");
        Assert.True(corr > 0.9999, $"RVQ corr too low ({corr:F6}).");
        Assert.True(mx < 1e-3, $"RVQ maxAbs too high ({mx:E4}).");
        quant.Dispose();
    }
}
