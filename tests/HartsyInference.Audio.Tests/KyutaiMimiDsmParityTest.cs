using System.Globalization;
using HartsyInference.Audio.Models.Codecs.Mimi;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>DSM Mimi decode parity: our <see cref="Mimi"/> decode of fixed codes vs the moshi reference decode.
/// Gated on <c>KYUTAI_MIMI</c> (the tokenizer-*.safetensors) + <c>MIMI_REF_CODES</c>/<c>MIMI_REF_AUDIO</c> text dumps.</summary>
public sealed unsafe class KyutaiMimiDsmParityTest
{
    private readonly ITestOutputHelper _out;
    public KyutaiMimiDsmParityTest(ITestOutputHelper o) => _out = o;

    [Fact]
    public void Decode_Matches_Moshi()
    {
        string? mp = Environment.GetEnvironmentVariable("KYUTAI_MIMI");
        string? cp = Environment.GetEnvironmentVariable("MIMI_REF_CODES");
        string? ap = Environment.GetEnvironmentVariable("MIMI_REF_AUDIO");
        if (mp is null || cp is null || ap is null || !File.Exists(mp) || !File.Exists(cp) || !File.Exists(ap))
        { _out.WriteLine("gated"); return; }

        string[] cl = File.ReadAllText(cp).Split('\n');
        string[] hdr = cl[0].Split(' ', StringSplitOptions.RemoveEmptyEntries);
        int K = int.Parse(hdr[0]), T = int.Parse(hdr[1]);
        string[] cv = cl[1].Split(' ', StringSplitOptions.RemoveEmptyEntries);
        Tensor codes = new(new TensorShape(1, K, T), DType.I32);
        int* cptr = (int*)codes.DataPointer;
        for (int i = 0; i < K * T; i++) cptr[i] = int.Parse(cv[i]);

        using SafeTensorsLoader w = new(); w.Load(mp);
        Mimi mimi = new(MimiConfig.Mimi24kHzDsm);
        mimi.LoadWeights(w.GetAllTensors());
        using Tensor audio = mimi.Decode(new CpuBackend(), codes, batch: 1, tFrames: T);
        codes.Dispose();

        int n = (int)audio.Shape[audio.Shape.Rank - 1];
        float* aptr = (float*)audio.DataPointer;

        string[] rv = File.ReadAllText(ap).Split('\n')[1].Split(' ', StringSplitOptions.RemoveEmptyEntries);
        int m = Math.Min(n, rv.Length);
        double dot = 0, na = 0, nb = 0, maxAbs = 0;
        for (int i = 0; i < m; i++)
        {
            double x = aptr[i], y = double.Parse(rv[i], CultureInfo.InvariantCulture);
            dot += x * y; na += x * x; nb += y * y; maxAbs = Math.Max(maxAbs, Math.Abs(x - y));
        }
        double corr = dot / (Math.Sqrt(na * nb) + 1e-12);
        _out.WriteLine($"ours n={n} ref n={rv.Length}; corr={corr:F6} maxAbs={maxAbs:F5}");
        Assert.True(corr > 0.99, $"decode corr {corr:F6} < 0.99");
    }
}
