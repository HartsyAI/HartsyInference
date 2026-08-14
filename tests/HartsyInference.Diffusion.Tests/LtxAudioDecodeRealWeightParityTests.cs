using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Stage-by-stage numeric parity for the LTX-2.5 audio decode (latent → log-mel → waveform) against
/// dumps captured from ComfyUI's own <c>AudioVAE</c> on the SAME checkpoint and the SAME latent. This is the
/// first absolute check on the audio chain: everything before it compared our output to our own prior output,
/// which cannot see a level defect that was present from the start.
/// <para>Reference dumps are produced by <c>ref_audio_vae.py</c>; the directory is passed in
/// <c>HARTSY_LTX2_AUDIO_REFDIR</c> and the test skips when it is absent.</para></summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed unsafe class LtxAudioDecodeRealWeightParityTests
{
    private readonly ITestOutputHelper _output;
    public LtxAudioDecodeRealWeightParityTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void AudioVaeAndVocoder_OnReferenceLatent_MatchComfyUi()
    {
        string? refDir = Environment.GetEnvironmentVariable("HARTSY_LTX2_AUDIO_REFDIR");
        string checkpoint = Environment.GetEnvironmentVariable("HARTSY_LTX2_AUDIO_VAE")
            ?? "/home/hartsy/Desktop/HartsyInference/Models/VAE/LTX-2/ltx-2.5-audio-vae-bf16.safetensors";
        if (refDir is null || !Directory.Exists(refDir))
        {
            _output.WriteLine("SKIPPED: HARTSY_LTX2_AUDIO_REFDIR not set / missing.");
            return;
        }
        if (!File.Exists(checkpoint))
        {
            _output.WriteLine($"SKIPPED: audio VAE checkpoint not found at {checkpoint}.");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: CUDA not available.");
            return;
        }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found at {ptxDir}.");
            return;
        }

        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(checkpoint);
        Dictionary<string, Tensor> all = new(loader.GetAllTensors());
        Dictionary<string, Tensor> vaeW = new();
        Dictionary<string, Tensor> vocW = new();
        foreach (KeyValuePair<string, Tensor> kv in all)
        {
            if (kv.Key.StartsWith("audio_vae.", StringComparison.Ordinal))
                vaeW[kv.Key["audio_vae.".Length..]] = kv.Value;
            else vocW[kv.Key] = kv.Value;
        }
        _output.WriteLine($"audio_vae keys: {vaeW.Count}, vocoder keys: {vocW.Count}");

        float[] mean = ReadStat(vaeW, "per_channel_statistics.mean-of-means");
        float[] std = ReadStat(vaeW, "per_channel_statistics.std-of-means");
        Assert.Equal(128, mean.Length);
        Assert.Equal(128, std.Length);

        using CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);

        Dictionary<string, Tensor> vaeF32 = new();
        foreach (KeyValuePair<string, Tensor> kv in vaeW)
            vaeF32[kv.Key] = kv.Value.DType == DType.F32 ? kv.Value : kv.Value.CastTo(DType.F32);

        LtxAudioVaeDecoder vae = new LtxAudioVaeDecoder();
        vae.LoadWeights(vaeF32);

        // --- Stage 1: denormalize + unpack, exactly as LtxVideo2Pipeline.UnpackAudioLatents does.
        (float[] latFlat, long[] latShape) = ReadBin(refDir, "latent_norm");     // [1, 8, T, 16]
        int channels = (int)latShape[1], frames = (int)latShape[2], melLat = (int)latShape[3];
        Tensor unpacked = new Tensor(new TensorShape(1L, channels, frames, melLat), DType.F32);
        float* up = (float*)unpacked.DataPointer;
        for (int c = 0; c < channels; c++)
            for (int fI = 0; fI < frames; fI++)
                for (int m = 0; m < melLat; m++)
                {
                    long i = ((long)c * frames + fI) * melLat + m;
                    int si = c * melLat + m;
                    up[i] = latFlat[i] * std[si] + mean[si];
                }
        Report("latent (denormalized)", unpacked);
        CompareTo(refDir, "latent_denorm", unpacked, "denormalized latent");

        // --- Stage 2: audio VAE → log-mel.
        Tensor mel = vae.Decode(backend, unpacked);
        backend.Sync();
        Report("mel (log-mel out of VAE)", mel);
        CompareTo(refDir, "mel", mel, "audio VAE log-mel");

        // --- Stage 3: vocoder → waveform.
        LtxAudioVocoder vocoder = new LtxAudioVocoder();
        vocoder.LoadWeights(vocW);
        Tensor wave = vocoder.Forward(backend, mel);
        backend.Sync();
        Report("vocoder waveform", wave);
        CompareTo(refDir, "wave", wave, "vocoder waveform");

        mel.Dispose();
        unpacked.Dispose();
        wave.Dispose();
        vocoder.Dispose();
    }

    /// <summary>Absolute level guard. Every audio check before this one compared our output to our own previous
    /// output, so a level defect present from the start passed forever. This decodes a deterministic unit-Gaussian
    /// latent — the distribution the checkpoint's own <c>per_channel_statistics</c> normalize to — and asserts the
    /// waveform lands in a plausible loudness band. Measured against ComfyUI on the same checkpoint, a unit-Gaussian
    /// latent decodes at RMS ≈ 0.033 (−29.7 dBFS); the band below is deliberately wide (−46 … −16 dBFS) so it fails
    /// only on a real gain defect, not on a numerics drift.</summary>
    [Fact]
    public void AudioDecode_UnitGaussianLatent_LandsInAPlausibleLoudnessBand()
    {
        string checkpoint = Environment.GetEnvironmentVariable("HARTSY_LTX2_AUDIO_VAE")
            ?? "/home/hartsy/Desktop/HartsyInference/Models/VAE/LTX-2/ltx-2.5-audio-vae-bf16.safetensors";
        if (!File.Exists(checkpoint)) { _output.WriteLine($"SKIPPED: {checkpoint} not found."); return; }
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA not available."); return; }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: no PTX at {ptxDir}."); return; }

        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(checkpoint);
        Dictionary<string, Tensor> vaeW = new(), vocW = new();
        foreach (KeyValuePair<string, Tensor> kv in loader.GetAllTensors())
        {
            if (kv.Key.StartsWith("audio_vae.", StringComparison.Ordinal))
                vaeW[kv.Key["audio_vae.".Length..]] = kv.Value.DType == DType.F32 ? kv.Value : kv.Value.CastTo(DType.F32);
            else vocW[kv.Key] = kv.Value;
        }
        float[] mean = ReadStat(vaeW, "per_channel_statistics.mean-of-means");
        float[] std = ReadStat(vaeW, "per_channel_statistics.std-of-means");

        using CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
        LtxAudioVaeDecoder vae = new LtxAudioVaeDecoder();
        vae.LoadWeights(vaeW);

        const int channels = 8, melLat = 16, frames = 101;
        Tensor latent = new Tensor(new TensorShape(1L, channels, frames, melLat), DType.F32);
        float* lp = (float*)latent.DataPointer;
        uint state = 0x9E3779B9;
        for (int c = 0; c < channels; c++)
            for (int f = 0; f < frames; f++)
                for (int m = 0; m < melLat; m++)
                {
                    lp[((long)c * frames + f) * melLat + m] = NextGaussian(ref state) * std[c * melLat + m] + mean[c * melLat + m];
                }

        Tensor mel = vae.Decode(backend, latent);
        LtxAudioVocoder vocoder = new LtxAudioVocoder();
        vocoder.LoadWeights(vocW);
        Tensor wave = vocoder.Forward(backend, mel);
        backend.Sync();

        float* wp = (float*)wave.DataPointer;
        long n = wave.Shape.ElementCount;
        double sumSq = 0; float peak = 0;
        for (long i = 0; i < n; i++) { sumSq += (double)wp[i] * wp[i]; peak = Math.Max(peak, Math.Abs(wp[i])); }
        double rms = Math.Sqrt(sumSq / n);
        _output.WriteLine($"decoded waveform: n={n} peak={peak:F5} rms={rms:F6} ({20 * Math.Log10(rms):F1} dBFS)");

        latent.Dispose(); mel.Dispose(); wave.Dispose(); vocoder.Dispose();

        Assert.True(rms > 0.005, $"decoded audio is far too quiet: RMS {rms:F6} ({20 * Math.Log10(rms):F1} dBFS); "
            + "a unit-Gaussian latent should decode near −30 dBFS. Something in the audio decode chain lost gain.");
        Assert.True(rms < 0.16, $"decoded audio is far too loud: RMS {rms:F6} ({20 * Math.Log10(rms):F1} dBFS).");
        Assert.True(peak <= 1.0f, $"waveform exceeds full scale: peak {peak:F5}.");
    }

    /// <summary>Box–Muller over a deterministic xorshift — the test must not depend on a framework RNG.</summary>
    private static float NextGaussian(ref uint state)
    {
        static double NextUnit(ref uint s)
        {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            return (s & 0xFFFFFF) / (double)0x1000000;
        }
        double u1 = Math.Max(NextUnit(ref state), 1e-12), u2 = NextUnit(ref state);
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }

    private static float[] ReadStat(Dictionary<string, Tensor> w, string key)
    {
        Tensor t = w[key];
        Tensor f32 = t.DType == DType.F32 ? t : t.CastTo(DType.F32);
        float[] result = new float[f32.Shape.ElementCount];
        float* p = (float*)f32.DataPointer;
        for (int i = 0; i < result.Length; i++) result[i] = p[i];
        return result;
    }

    private static (float[] Data, long[] Shape) ReadBin(string dir, string name)
    {
        long[] shape = System.Text.Json.JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, name + ".json")))
            .RootElement.GetProperty("shape").EnumerateArray().Select(e => e.GetInt64()).ToArray();
        byte[] raw = File.ReadAllBytes(Path.Combine(dir, name + ".bin"));
        float[] data = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, data, 0, raw.Length);
        return (data, shape);
    }

    private void Report(string label, Tensor t)
    {
        float* p = (float*)t.DataPointer;
        long n = t.Shape.ElementCount;
        float mn = float.MaxValue, mx = float.MinValue;
        double sum = 0, sumSq = 0;
        for (long i = 0; i < n; i++)
        {
            float v = p[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v; sumSq += (double)v * v;
        }
        _output.WriteLine($"{label,-34} shape={t.Shape} min={mn,9:F5} max={mx,9:F5} "
            + $"mean={sum / n,9:F5} rms={Math.Sqrt(sumSq / n),9:F5}");
    }

    private void CompareTo(string dir, string name, Tensor ours, string what)
    {
        (float[] refData, long[] refShape) = ReadBin(dir, name);
        _output.WriteLine($"  [{what}] ref shape=[{string.Join(",", refShape)}] n={refData.Length}; "
            + $"ours shape={ours.Shape} n={ours.Shape.ElementCount}");
        long n = Math.Min(refData.Length, ours.Shape.ElementCount);
        float* p = (float*)ours.DataPointer;
        double num = 0, den = 0, maxAbs = 0;
        double refSumSq = 0, ourSumSq = 0;
        for (long i = 0; i < n; i++)
        {
            double d = p[i] - refData[i];
            num += d * d; den += (double)refData[i] * refData[i];
            maxAbs = Math.Max(maxAbs, Math.Abs(d));
            refSumSq += (double)refData[i] * refData[i];
            ourSumSq += (double)p[i] * p[i];
        }
        double relL2 = den > 0 ? Math.Sqrt(num / den) : double.NaN;
        _output.WriteLine($"  [{what}] relL2={relL2:E3} maxAbsDiff={maxAbs:F5} "
            + $"refRms={Math.Sqrt(refSumSq / n):F5} ourRms={Math.Sqrt(ourSumSq / n):F5}");
    }
}
