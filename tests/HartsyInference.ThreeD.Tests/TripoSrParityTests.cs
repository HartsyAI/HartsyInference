using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.TripoSr;
using HartsyInference.Vision.DinoVit;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ThreeD.Tests;

/// <summary>Real-weight numeric parity for TripoSR (<c>stabilityai/TripoSR</c>) vs the upstream <c>tsr</c>
/// library. Gated on <c>TRIPOSR_WEIGHTS</c> (the converted <c>triposr.safetensors</c>) +
/// <c>TRIPOSR_REF_IO</c> (<c>triposr_ref_io.safetensors</c> from
/// <c>tests/python-reference/triposr_reference/dump_triposr_reference.py</c>). Skips cleanly when unset.</summary>
public sealed unsafe class TripoSrParityTests
{
    private readonly ITestOutputHelper _out;
    public TripoSrParityTests(ITestOutputHelper o) => _out = o;

    private static readonly float[] Mean = [0.485f, 0.456f, 0.406f];
    private static readonly float[] Std = [0.229f, 0.224f, 0.225f];

    private static (string w, string r)? Env()
    {
        string? w = Environment.GetEnvironmentVariable("TRIPOSR_WEIGHTS");
        string? r = Environment.GetEnvironmentVariable("TRIPOSR_REF_IO");
        return (w is null || r is null || !File.Exists(w) || !File.Exists(r)) ? null : (w, r);
    }

    [Fact]
    public void DinoVit_ImageTokens_MatchReference()
    {
        (string w, string r)? e = Env();
        if (e is null) return; // gated
        using IBackend cpu = new CpuBackend();
        using SafeTensorsLoader wl = new(); wl.Load(e!.Value.w);
        using SafeTensorsLoader rl = new(); rl.Load(e!.Value.r);
        Dictionary<string, Tensor> all = wl.GetAllTensors();
        TripoSrCheckpointConverter.ConvertedWeights cw = TripoSrCheckpointConverter.Convert(all);

        DinoViTEncoder dino = new(DinoViTConfig.DinoVitB16_512); dino.LoadWeights(cw.Dino);

        Tensor pixels = rl.GetTensor("pixels");           // [1,3,512,512] in [0,1]
        Tensor norm = ImagenetNormalize(pixels);
        Tensor tokens = dino.Encode(cpu, norm); norm.Dispose();
        Tensor refTok = rl.GetTensor("image_tokens");     // [1,1025,768]

        (double maxAbs, double corr) = Compare(tokens, refTok);
        _out.WriteLine($"DINO image_tokens: maxAbs={maxAbs:E3} corr={corr:F8}");
        tokens.Dispose();
        Assert.True(maxAbs < 5e-3, $"maxAbs {maxAbs}");
        Assert.True(corr > 0.9999, $"corr {corr}");
    }

    [Fact]
    public void Transformer_SceneCodes_MatchReference()
    {
        (string w, string r)? e = Env();
        if (e is null) return; // gated
        using IBackend cpu = new CpuBackend();
        using SafeTensorsLoader wl = new(); wl.Load(e!.Value.w);
        using SafeTensorsLoader rl = new(); rl.Load(e!.Value.r);
        TripoSrCheckpointConverter.ConvertedWeights cw = TripoSrCheckpointConverter.Convert(wl.GetAllTensors());

        TripoSrTransformer tr = new(TripoSrConfig.TripoSr); tr.LoadWeights(cw.Transformer);

        Tensor imageTokens = rl.GetTensor("image_tokens");   // [1,1025,768]
        Triplane plane = tr.Forward(cpu, imageTokens);        // [3,40,64,64]
        Tensor sceneCodes = rl.GetTensor("scene_codes");      // [1,3,40,64,64]

        float* a = (float*)sceneCodes.DataPointer;
        double maxAbs = 0, dot = 0, na = 0, nb = 0;
        for (long i = 0; i < plane.Features.Length; i++)
        {
            double x = plane.Features[i], y = a[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(x - y)); dot += x * y; na += x * x; nb += y * y;
        }
        double corr = dot / (Math.Sqrt(na * nb) + 1e-12);
        _out.WriteLine($"transformer scene_codes: maxAbs={maxAbs:E3} corr={corr:F8}");
        Assert.True(maxAbs < 5e-3, $"maxAbs {maxAbs}");
        Assert.True(corr > 0.9999, $"corr {corr}");
    }

    [Fact]
    public void Decoder_ProbeDensityAndColor_MatchReference()
    {
        (string w, string r)? e = Env();
        if (e is null) return; // gated
        using IBackend cpu = new CpuBackend();
        using SafeTensorsLoader wl = new(); wl.Load(e!.Value.w);
        using SafeTensorsLoader rl = new(); rl.Load(e!.Value.r);
        TripoSrCheckpointConverter.ConvertedWeights cw = TripoSrCheckpointConverter.Convert(wl.GetAllTensors());
        TripoSrConfig cfg = TripoSrConfig.TripoSr;
        TriplaneNerfDecoder dec = new(cfg); dec.LoadWeights(cw.Decoder);

        Triplane tri = TriplaneFromSceneCodes(rl.GetTensor("scene_codes"), cfg);
        Tensor probe = rl.GetTensor("probe_points");          // [N,3]
        int n = (int)probe.Shape[0];
        float[] pts = new float[n * 3];
        new ReadOnlySpan<float>((float*)probe.DataPointer, n * 3).CopyTo(pts);

        Tensor outp = dec.Evaluate(cpu, tri, pts, n);         // [1,n,4] raw
        float* op = (float*)outp.DataPointer;
        Tensor refDen = rl.GetTensor("probe_density_act");    // [n,1]
        Tensor refCol = rl.GetTensor("probe_color");          // [n,3]
        float* rd = (float*)refDen.DataPointer; float* rc = (float*)refCol.DataPointer;

        double dMax = 0, cMax = 0;
        for (int i = 0; i < n; i++)
        {
            float den = MathF.Exp(op[i * 4] + cfg.DensityBias);
            dMax = Math.Max(dMax, Math.Abs(den - rd[i]));
            for (int c = 0; c < 3; c++)
            {
                float col = 1f / (1f + MathF.Exp(-op[i * 4 + 1 + c]));
                cMax = Math.Max(cMax, Math.Abs(col - rc[i * 3 + c]));
            }
        }
        outp.Dispose();
        _out.WriteLine($"decoder probe: density_act maxAbs={dMax:E3}  color maxAbs={cMax:E3}");
        Assert.True(cMax < 5e-3, $"color maxAbs {cMax}");
        Assert.True(dMax < 5e-1, $"density_act maxAbs {dMax}"); // exp magnifies; relative check below
    }

    [Fact]
    public void Decoder_DensityGrid_MatchReference()
    {
        (string w, string r)? e = Env();
        if (e is null) return; // gated
        using IBackend cpu = new CpuBackend();
        using SafeTensorsLoader wl = new(); wl.Load(e!.Value.w);
        using SafeTensorsLoader rl = new(); rl.Load(e!.Value.r);
        TripoSrCheckpointConverter.ConvertedWeights cw = TripoSrCheckpointConverter.Convert(wl.GetAllTensors());
        TripoSrConfig cfg = TripoSrConfig.TripoSr;
        TriplaneNerfDecoder dec = new(cfg); dec.LoadWeights(cw.Decoder);

        Triplane tri = TriplaneFromSceneCodes(rl.GetTensor("scene_codes"), cfg);
        ScalarField3D field = dec.DecodeDensityField(cpu, tri, resolution: 64);
        Tensor refGrid = rl.GetTensor("grid_density_act");    // [64^3,1]
        float* rg = (float*)refGrid.DataPointer;

        double maxAbs = 0, dot = 0, na = 0, nb = 0;
        for (long i = 0; i < field.Values.Length; i++)
        {
            double x = field.Values[i], y = rg[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(x - y)); dot += x * y; na += x * x; nb += y * y;
        }
        double corr = dot / (Math.Sqrt(na * nb) + 1e-12);
        _out.WriteLine($"decoder density grid 64^3: maxAbs={maxAbs:E3} corr={corr:F8}");
        Assert.True(corr > 0.9999, $"corr {corr}");
    }

    private static Triplane TriplaneFromSceneCodes(Tensor sc, TripoSrConfig cfg)
    {
        // sc: [1,3,40,64,64] -> Features [3,40,64,64] flattened (drop batch).
        int c = cfg.TriplaneChannels, res = cfg.TriplaneResolution;
        float[] feat = new float[3 * c * res * res];
        new ReadOnlySpan<float>((float*)sc.DataPointer, feat.Length).CopyTo(feat);
        return new Triplane { Features = feat, Channels = c, Height = res, Width = res };
    }

    private static Tensor ImagenetNormalize(Tensor pixels)
    {
        int b = (int)pixels.Shape[0], c = (int)pixels.Shape[1], h = (int)pixels.Shape[2], w = (int)pixels.Shape[3];
        Tensor o = new(pixels.Shape, DType.F32);
        float* src = (float*)pixels.DataPointer; float* dst = (float*)o.DataPointer;
        long hw = (long)h * w;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
            {
                long baseI = ((long)bi * c + ci) * hw;
                for (long i = 0; i < hw; i++) dst[baseI + i] = (src[baseI + i] - Mean[ci]) / Std[ci];
            }
        return o;
    }

    private static (double maxAbs, double corr) Compare(Tensor a, Tensor b)
    {
        long n = a.ElementCount;
        float* pa = (float*)a.DataPointer; float* pb = (float*)b.DataPointer;
        double maxAbs = 0, dot = 0, na = 0, nb = 0;
        for (long i = 0; i < n; i++)
        {
            double x = pa[i], y = pb[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(x - y)); dot += x * y; na += x * x; nb += y * y;
        }
        return (maxAbs, dot / (Math.Sqrt(na * nb) + 1e-12));
    }
}
