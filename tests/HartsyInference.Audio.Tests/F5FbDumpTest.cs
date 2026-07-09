using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Preprocessing;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

public sealed class F5FbDumpTest
{
    private readonly ITestOutputHelper _out;
    public F5FbDumpTest(ITestOutputHelper o) => _out = o;

    [Fact]
    public void DumpOurResampledJfk()
    {
        string jfk = Path.Combine(AudioModelCache.CacheRoot, "test-clips", "jfk.wav");
        if (!File.Exists(jfk)) { _out.WriteLine("no jfk"); return; }
        WavFile.DecodedAudio a = WavFile.Read(jfk);
        float[] r = Resampler.Create(a.SampleRate, 24_000).Resample(a.Channels[0]);
        WavFile.WriteMono16("/tmp/jfk_ours24k.wav", r, 24_000);
        _out.WriteLine($"our resample {a.SampleRate}->24k: {r.Length} samples");
    }

    [Fact]
    public void DumpF5FilterbankSums()
    {
        // Same params as F5VocosConfig: sr 24000, nFft 1024, 100 mels, [0,12000], HTK, no Slaney norm.
        float[,] fb = MelFilterbank.Get(24_000, 1024, 100, 0.0, 12_000.0, MelScale.Htk, slaneyNorm: false);
        int nMels = fb.GetLength(0), numBins = fb.GetLength(1);
        double[] sums = new double[nMels];
        for (int m = 0; m < nMels; m++) { double s = 0; for (int k = 0; k < numBins; k++) s += fb[m, k]; sums[m] = s; }
        System.IO.File.WriteAllText("/tmp/ours_fbsum.txt", string.Join(" ", System.Array.ConvertAll(sums, x => x.ToString("F4"))));
        _out.WriteLine($"OURS fb [{nMels},{numBins}] filter sums — bins 0-4: {string.Join(",", System.Array.ConvertAll(sums[..5], x => x.ToString("F3")))}");
        _out.WriteLine($"  bins 48-52: {string.Join(",", System.Array.ConvertAll(sums[48..53], x => x.ToString("F3")))}");
        _out.WriteLine($"  bins 95-99: {string.Join(",", System.Array.ConvertAll(sums[95..], x => x.ToString("F3")))}");
    }
}
