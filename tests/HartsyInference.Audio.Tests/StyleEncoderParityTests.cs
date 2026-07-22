using System.Globalization;
using HartsyInference.Audio.Models.StyleTts2;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.PyTorch;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Real-weight A/B for the StyleTTS2 <see cref="StyleEncoder"/> (acoustic + prosodic) vs the Python
/// <c>yl4579/StyleTTS2</c> reference. The reference (<c>scratchpad/style_enc_ref.py</c>) dumps the exact input
/// mel and the target 128-d style vectors; this feeds the identical mel through the engine and asserts a high
/// correlation, isolating the encoder from the mel front-end. Gated on <c>STYLE_CKPT</c> (LibriTTS <c>.pth</c>)
/// + <c>STYLE_REF_DIR</c> (dir with <c>mel.txt</c> / <c>style.txt</c>).</summary>
public sealed unsafe class StyleEncoderParityTests
{
    private readonly ITestOutputHelper _out;
    public StyleEncoderParityTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public void StyleEncoder_Matches_Reference()
    {
        string? ckpt = Environment.GetEnvironmentVariable("STYLE_CKPT");
        string? refDir = Environment.GetEnvironmentVariable("STYLE_REF_DIR");
        if (ckpt is null || !File.Exists(ckpt) || refDir is null || !File.Exists(Path.Combine(refDir, "mel.txt")))
        {
            _out.WriteLine("STYLE_CKPT / STYLE_REF_DIR not set — skipping.");
            return;
        }

        PytorchPickleLoader loader = new();
        loader.Load(ckpt, recursiveFlatten: true);
        Dictionary<string, Tensor> w = StyleTts2Weights.Adapt(loader.GetAllTensors());

        StyleEncoder styleEnc = new(styleDim: 128);
        styleEnc.LoadWeights(w, "style_encoder");
        StyleEncoder predEnc = new(styleDim: 128);
        predEnc.LoadWeights(w, "predictor_encoder");

        Tensor mel = ReadMel(Path.Combine(refDir, "mel.txt"));
        (float[] refS, float[] refP) = ReadStyles(Path.Combine(refDir, "style.txt"));

        using CpuBackend backend = new();
        using Tensor sAcoustic = styleEnc.Forward(backend, mel);
        using Tensor sProsodic = predEnc.Forward(backend, mel);
        mel.Dispose();

        float corrS = Corr(Span(sAcoustic), refS);
        float corrP = Corr(Span(sProsodic), refP);
        _out.WriteLine($"style_encoder corr={corrS:F6} (ourNorm={Norm(Span(sAcoustic)):F4} refNorm={Norm(refS):F4})");
        _out.WriteLine($"predictor_encoder corr={corrP:F6} (ourNorm={Norm(Span(sProsodic)):F4} refNorm={Norm(refP):F4})");
        Assert.True(corrS > 0.99f, $"style_encoder corr {corrS} < 0.99");
        Assert.True(corrP > 0.99f, $"predictor_encoder corr {corrP} < 0.99");
    }

    [Fact]
    public void HifiGanGenerator_Matches_Reference()
    {
        string? ckpt = Environment.GetEnvironmentVariable("STYLE_CKPT");
        string? refDir = Environment.GetEnvironmentVariable("HIFIGAN_REF_DIR");
        if (ckpt is null || !File.Exists(ckpt) || refDir is null || !File.Exists(Path.Combine(refDir, "out.txt")))
        {
            _out.WriteLine("STYLE_CKPT / HIFIGAN_REF_DIR not set — skipping."); return;
        }
        PytorchPickleLoader loader = new();
        loader.Load(ckpt, recursiveFlatten: true);
        Dictionary<string, Tensor> w = StyleTts2Weights.Adapt(loader.GetAllTensors());
        HartsyInference.Audio.Models.StyleTts2.StyleHifiGanGenerator gen = new(24_000);
        gen.LoadWeights(w, "decoder.generator");

        (int[] xs, float[] xv) = ReadArr(Path.Combine(refDir, "x.txt"));
        (int[] ss, float[] sv) = ReadArr(Path.Combine(refDir, "s.txt"));
        (int[] fs, float[] fv) = ReadArr(Path.Combine(refDir, "f0.txt"));
        (_, float[] refOut) = ReadArr(Path.Combine(refDir, "out.txt"));

        using Tensor x = ToTensor([1, xs[0], xs[1]], xv);
        using Tensor s = ToTensor([1, ss[0]], sv);
        using Tensor f0 = ToTensor([1, fs[0]], fv);
        using CpuBackend backend = new();

        // Source isolation: engine GenerateHarmonicSource vs the reference m_source (both noise-free).
        if (File.Exists(Path.Combine(refDir, "har.txt")))
        {
            (_, float[] refHar) = ReadArr(Path.Combine(refDir, "har.txt"));
            using Tensor f0c = ToTensor([1, 1, fs[0]], fv);
            float[] ourHar = HartsyInference.Audio.Models.StyleTts2.StyleSineGen.Source(
                f0c, 300, 24_000, HartsyInference.Audio.Models.Whisper.WhisperOps.EnsureF32(w["decoder.generator.m_source.l_linear.weight"]),
                HartsyInference.Audio.Models.Whisper.WhisperOps.EnsureF32(w["decoder.generator.m_source.l_linear.bias"]));
            int hn = Math.Min(ourHar.Length, refHar.Length);
            _out.WriteLine($"harmonic source corr={Corr(ourHar[..hn], refHar[..hn]):F6} (ourLen={ourHar.Length} refLen={refHar.Length})");
        }

        float[] audio = gen.Forward(backend, x, s, f0);

        int n = Math.Min(audio.Length, refOut.Length);
        float corr = Corr(audio[..n], refOut[..n]);
        double maxAbs = 0; for (int i = 0; i < n; i++) maxAbs = Math.Max(maxAbs, Math.Abs(audio[i] - refOut[i]));
        _out.WriteLine($"HiFiGAN out: ourLen={audio.Length} refLen={refOut.Length} corr={corr:F6} maxAbs={maxAbs:F5} ourRMS={Norm(audio) / MathF.Sqrt(audio.Length):F4} refRMS={Norm(refOut) / MathF.Sqrt(refOut.Length):F4}");
        Assert.True(corr > 0.99f, $"HiFiGAN generator corr {corr} < 0.99");
    }

    private static (int[] shape, float[] data) ReadArr(string path)
    {
        string[] lines = File.ReadAllLines(path);
        int[] shape = [.. lines[0].Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse)];
        float[] data = [.. lines[1].Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(x => float.Parse(x, CultureInfo.InvariantCulture))];
        return (shape, data);
    }

    private static unsafe Tensor ToTensor(int[] shape, float[] data)
    {
        Tensor t = new(new TensorShape([.. shape.Select(d => (long)d)]), DType.F32);
        fixed (float* dp = data) Buffer.MemoryCopy(dp, (void*)t.DataPointer, (long)data.Length * 4, (long)data.Length * 4);
        return t;
    }

    [Fact]
    public void StyleTts2_LibriTTS_Submodules_Load()
    {
        string? ckpt = Environment.GetEnvironmentVariable("STYLE_CKPT");
        if (ckpt is null || !File.Exists(ckpt)) { _out.WriteLine("STYLE_CKPT not set — skipping."); return; }

        PytorchPickleLoader loader = new();
        loader.Load(ckpt, recursiveFlatten: true);
        Dictionary<string, Tensor> w = StyleTts2Weights.Adapt(loader.GetAllTensors());
        HartsyInference.Audio.Models.Kokoro.KokoroConfig cfg = HartsyInference.Audio.Models.Kokoro.KokoroConfig.V1;

        void Try(string name, Action load)
        {
            try { load(); _out.WriteLine($"  {name}: LOADED ✓"); }
            catch (Exception ex) { _out.WriteLine($"  {name}: FAILED — {ex.GetType().Name}: {ex.Message}"); throw; }
        }
        Try("plBert", () => new HartsyInference.Audio.Models.Kokoro.KokoroPlBert(cfg).LoadWeights(w));
        Try("textEncoder", () => new HartsyInference.Audio.Models.Kokoro.KokoroTextEncoder(cfg).LoadWeights(w));
        Try("prosodyPredictor", () => new HartsyInference.Audio.Models.Kokoro.KokoroProsodyPredictor(cfg).LoadWeights(w));
        Try("iStftNetDecoder", () => new HartsyInference.Audio.Models.Kokoro.KokoroIStftNetDecoder(cfg).LoadWeights(w));
        Try("styleEncoder", () => new StyleEncoder(128).LoadWeights(w, "style_encoder"));
        Try("predictorEncoder", () => new StyleEncoder(128).LoadWeights(w, "predictor_encoder"));
    }

    private static float[] Span(Tensor t)
    {
        float[] a = new float[t.ElementCount];
        for (int i = 0; i < a.Length; i++) a[i] = ((float*)t.DataPointer)[i];
        return a;
    }

    private static Tensor ReadMel(string path)
    {
        string[] lines = File.ReadAllLines(path);
        string[] hdr = lines[0].Split(' ', StringSplitOptions.RemoveEmptyEntries);
        int m = int.Parse(hdr[0]), t = int.Parse(hdr[1]);
        string[] vals = lines[1].Split(' ', StringSplitOptions.RemoveEmptyEntries);
        Tensor mel = new(new TensorShape(1, m, t), DType.F32);
        float* mp = (float*)mel.DataPointer;
        for (int i = 0; i < m * t; i++) mp[i] = float.Parse(vals[i], CultureInfo.InvariantCulture);
        return mel;
    }

    private static (float[] s, float[] p) ReadStyles(string path)
    {
        float[] s = [], p = [];
        foreach (string line in File.ReadAllLines(path))
        {
            if (line.StartsWith("style_s:", StringComparison.Ordinal)) s = Parse(line[8..]);
            else if (line.StartsWith("style_p:", StringComparison.Ordinal)) p = Parse(line[8..]);
        }
        return (s, p);
        static float[] Parse(string body) =>
            [.. body.Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(x => float.Parse(x, CultureInfo.InvariantCulture))];
    }

    private static float Corr(float[] a, float[] b)
    {
        int n = Math.Min(a.Length, b.Length);
        double ma = 0, mb = 0;
        for (int i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
        ma /= n; mb /= n;
        double num = 0, da = 0, db = 0;
        for (int i = 0; i < n; i++) { double x = a[i] - ma, y = b[i] - mb; num += x * y; da += x * x; db += y * y; }
        return (float)(num / (Math.Sqrt(da * db) + 1e-12));
    }

    private static float Norm(float[] a)
    {
        double s = 0; foreach (float x in a) s += (double)x * x; return (float)Math.Sqrt(s);
    }
}
