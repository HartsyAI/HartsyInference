using HartsyInference.Audio.Models.VoiceIsolator;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using Xunit;

namespace HartsyInference.Audio.Tests.VoiceIsolator;

public sealed class VoiceIsolatorUnitTests : IDisposable
{
    // The released files' own metadata (HF inworld/voice-isolator @ 9b0285b3, safetensors __metadata__["config"]).
    private const string OfflineConfig = "{\"band_plan\": [[100, 1000], [250, 4000], [500, 8000], [1000, 16000], [2000, 24000]], "
        + "\"block_types\": [\"transformer\", \"mamba\", \"transformer\", \"mamba\", \"transformer\", \"mamba\", \"transformer\"], "
        + "\"causal\": false, \"chunk_seconds\": 8.0, \"correction_bound\": 0.5, \"dual_mode\": false, \"exclusive_axes\": \"time\", "
        + "\"exclusive_self_attention\": true, \"hop_length\": 480, \"lookahead_frames\": 0, \"lookahead_frames_choices\": [], "
        + "\"mamba_conv\": 4, \"mamba_expand\": 2, \"mamba_state\": 16, \"mode_film\": false, \"n_fft\": 2048, \"num_heads\": 8, "
        + "\"overlap_seconds\": 2.0, \"sample_rate\": 48000, \"width\": 512}";

    private const string RealtimeConfig = "{\"attention_window_frames\": 1200, \"band_plan\": [[100, 1000], [250, 4000], [500, 8000], "
        + "[1000, 16000], [2000, 24000]], \"block_types\": [\"transformer\", \"mamba\", \"transformer\", \"mamba\", \"transformer\", "
        + "\"mamba\", \"transformer\"], \"causal\": true, \"chunk_seconds\": 30.0, \"correction_bound\": 0.5, \"dual_mode\": false, "
        + "\"exclusive_axes\": \"time\", \"exclusive_self_attention\": true, \"hop_length\": 480, \"lookahead_frames\": 2, "
        + "\"lookahead_frames_choices\": [0, 2, 4], \"mamba_conv\": 4, \"mamba_expand\": 2, \"mamba_state\": 16, \"mode_film\": false, "
        + "\"n_fft\": 2048, \"num_heads\": 8, \"overlap_seconds\": 2.0, \"sample_rate\": 48000, \"width\": 512}";

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "voice-isolator-tests-" + Guid.NewGuid().ToString("N"));

    public VoiceIsolatorUnitTests() => Directory.CreateDirectory(_dir);

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    [Theory]
    [InlineData(OfflineConfig)]
    [InlineData(RealtimeConfig)]
    public void ReleasedConfigsCutFortySixBandsCoveringEveryBinOnce(string json)
    {
        VoiceIsolatorConfig config = VoiceIsolatorConfig.FromJson(json);
        (int Start, int End)[] edges = VoiceIsolatorBands.Edges(config);
        Assert.Equal(46, edges.Length);
        Assert.Equal(0, edges[0].Start);
        Assert.Equal(1025, edges[^1].End);
        for (int i = 1; i < edges.Length; i++)
        {
            Assert.Equal(edges[i - 1].End, edges[i].Start);
        }
    }

    [Fact]
    public void ParsesTheReleasedConfigs()
    {
        VoiceIsolatorConfig offline = VoiceIsolatorConfig.FromJson(OfflineConfig);
        Assert.False(offline.Causal);
        Assert.Equal(512, offline.Width);
        Assert.Equal(7, offline.BlockTypes.Count);
        Assert.Equal(8.0, offline.ChunkSeconds);
        VoiceIsolatorConfig realtime = VoiceIsolatorConfig.FromJson(RealtimeConfig);
        Assert.True(realtime.Causal);
        Assert.Equal(1200, realtime.AttentionWindowFrames);
        Assert.Equal(2, realtime.LookaheadFrames);
        Assert.Equal([0, 2, 4], realtime.LookaheadChoices);
    }

    [Theory]
    [InlineData("{\"width\": 512, \"block_types\": [\"transformer\"], \"band_plan\": [[100, 1000]], \"dual_mode\": true}")]
    [InlineData("{\"width\": 512, \"block_types\": [\"transformer\"], \"band_plan\": [[100, 1000]], \"exclusive_axes\": \"both\"}")]
    [InlineData("{\"width\": 512, \"block_types\": [\"conformer\"], \"band_plan\": [[100, 1000]]}")]
    [InlineData("{\"block_types\": [\"mamba\"], \"band_plan\": [[100, 1000]]}")]
    public void RefusesWhatItDoesNotImplement(string json) =>
        Assert.Throws<InvalidDataException>(() => VoiceIsolatorConfig.FromJson(json));

    [Fact]
    public void AnalysisAndSynthesisRoundTrip()
    {
        VoiceIsolatorSpectral spectral = new(2048, 480);
        float[] audio = Signal(9_000, 1);
        foreach (bool reflect in new[] { true, false })
        {
            int frames = spectral.FrameCount(audio.Length);
            float[] re = new float[frames * spectral.Bins];
            float[] im = new float[frames * spectral.Bins];
            spectral.Analyze(audio, re, im, reflect);
            float[] back = spectral.Synthesize(re, im, frames, audio.Length);
            float worst = 0f;
            for (int i = 0; i < audio.Length; i++)
            {
                worst = Math.Max(worst, Math.Abs(back[i] - audio[i]));
            }
            Assert.True(worst < 1e-5f, $"reflect={reflect} round trip max abs error {worst}");
        }
    }

    [Fact]
    public void WindowsFormAPartitionOfUnity()
    {
        foreach ((int n, int chunk, int overlap) in new[] { (100_000, 30_000, 8_000), (65_001, 30_000, 15_000), (30_001, 30_000, 8_000) })
        {
            List<int> starts = VoiceIsolatorRenderer.WindowStarts(n, chunk, overlap);
            float[] total = new float[n];
            float[] window = new float[chunk];
            for (int i = 0; i < starts.Count; i++)
            {
                VoiceIsolatorRenderer.FillWindow(window, starts, i, chunk, overlap, n);
                for (int j = 0; j < chunk; j++)
                {
                    total[starts[i] + j] += window[j];
                }
            }
            Assert.Equal(n - chunk, starts[^1]);
            for (int j = 0; j < n; j++)
            {
                Assert.True(Math.Abs(total[j] - 1f) < 1e-5f, $"n={n}: weight at {j} is {total[j]}");
            }
        }
    }

    [Fact]
    public void ACausalNetworkIgnoresAudioBeyondItsLookahead()
    {
        string path = TinyVoiceIsolatorCheckpoint.Write(_dir, causal: true, seed: 5);
        using VoiceIsolatorNetwork network = VoiceIsolatorNetwork.Load(path);
        using CpuBackend backend = new();
        float[] a = Signal(1_600, 3);
        float[] b = (float[])a.Clone();
        const int perturbFrom = 900;
        for (int i = perturbFrom; i < b.Length; i++)
        {
            b[i] += 0.3f * MathF.Sin(i * 0.37f);
        }
        const int lookahead = 2;
        float[] ya = network.Isolate(backend, a, lookahead);
        float[] yb = network.Isolate(backend, b, lookahead);
        int hop = TinyVoiceIsolatorCheckpoint.Hop;
        int half = TinyVoiceIsolatorCheckpoint.NFft / 2;
        int checkedSamples = 0;
        for (int s = 0; s < a.Length; s++)
        {
            int lastFrame = ((s + half) / hop) + lookahead;
            if (lastFrame * hop + half >= perturbFrom)
            {
                break;
            }
            Assert.True(Math.Abs(ya[s] - yb[s]) < 1e-6f, $"sample {s} changed by {Math.Abs(ya[s] - yb[s])}");
            checkedSamples++;
        }
        Assert.True(checkedSamples > 600, "the causal check covered too few samples");
        Assert.True(Math.Abs(ya[^1] - yb[^1]) > 1e-4f, "the perturbation must reach the tail");
    }

    [Fact]
    public void AnOfflineNetworkSeesTheWholeClipAndIsDeterministic()
    {
        string path = TinyVoiceIsolatorCheckpoint.Write(_dir, causal: false, seed: 9);
        using VoiceIsolatorNetwork network = VoiceIsolatorNetwork.Load(path);
        using CpuBackend backend = new();
        float[] a = Signal(1_200, 4);
        float[] b = (float[])a.Clone();
        for (int i = 1_000; i < b.Length; i++)
        {
            b[i] += 0.3f;
        }
        float[] first = network.Isolate(backend, a, 0);
        float[] again = network.Isolate(backend, a, 0);
        float[] other = network.Isolate(backend, b, 0);
        Assert.Equal(first, again);
        Assert.All(first, v => Assert.True(float.IsFinite(v)));
        float earlyChange = 0f;
        for (int i = 0; i < 300; i++)
        {
            earlyChange = Math.Max(earlyChange, Math.Abs(first[i] - other[i]));
        }
        Assert.True(earlyChange > 1e-6f, "a two-sided network must react to later audio");
    }

    [Fact]
    public void ALongClipRendersInWindowsAndAShortOneInOnePass()
    {
        string path = TinyVoiceIsolatorCheckpoint.Write(_dir, causal: false, seed: 13);
        using VoiceIsolatorNetwork network = VoiceIsolatorNetwork.Load(path);
        using CpuBackend backend = new();
        float[] shortClip = Signal(1_500, 2);
        Assert.Equal(network.Isolate(backend, shortClip, 0), VoiceIsolatorRenderer.Render(network, backend, shortClip));
        float[] longClip = Signal(4_100, 6);
        int windows = 0;
        float[] rendered = VoiceIsolatorRenderer.Render(network, backend, longClip, progress: (_, total) => windows = total);
        Assert.Equal(longClip.Length, rendered.Length);
        Assert.All(rendered, v => Assert.True(float.IsFinite(v)));
        Assert.Equal(VoiceIsolatorRenderer.WindowStarts(4_100, 1_600, 400).Count, windows);
    }

    [Fact]
    public void TheCorrectionTouchesEveryBinOfALongClip()
    {
        const int frames = 300;
        const int bins = 33;
        Random rng = new(2);
        float[] raw = new float[frames * bins * 4];
        float[] re = new float[frames * bins];
        float[] im = new float[frames * bins];
        for (int i = 0; i < raw.Length; i++)
        {
            raw[i] = (float)(rng.NextDouble() * 4 - 2);
        }
        for (int i = 0; i < re.Length; i++)
        {
            re[i] = (float)(rng.NextDouble() * 2 - 1);
            im[i] = (float)(rng.NextDouble() * 2 - 1);
        }
        float[] expectedRe = new float[re.Length];
        float[] expectedIm = new float[im.Length];
        for (int i = 0; i < re.Length; i++)
        {
            double mr = 2 * Math.Tanh(raw[i * 4]);
            double mi = 2 * Math.Tanh(raw[i * 4 + 1]);
            double rr = Math.Tanh(raw[i * 4 + 2]);
            double ri = Math.Tanh(raw[i * 4 + 3]);
            double scale = 0.5 * Math.Sqrt((double)re[i] * re[i] + (double)im[i] * im[i]) / Math.Max(Math.Sqrt(rr * rr + ri * ri), 1.0);
            expectedRe[i] = (float)(re[i] * mr - im[i] * mi + rr * scale);
            expectedIm[i] = (float)(re[i] * mi + im[i] * mr + ri * scale);
        }
        VoiceIsolatorNetwork.Correct(raw, re, im, frames, bins, 0.5f);
        for (int i = 0; i < re.Length; i++)
        {
            Assert.True(Math.Abs(re[i] - expectedRe[i]) < 1e-5f && Math.Abs(im[i] - expectedIm[i]) < 1e-5f, $"bin {i} was not corrected");
        }
    }

    [Fact]
    public void ARefusesWeightsThatDoNotMatchTheConfig()
    {
        // the metadata names three taps, the tensors four
        string path = TinyVoiceIsolatorCheckpoint.Write(_dir, causal: false, seed: 17, metadataConv: 3);
        Assert.Throws<InvalidDataException>(() => VoiceIsolatorNetwork.Load(path));
    }

    private static float[] Signal(int n, int seed)
    {
        Random rng = new(seed);
        float[] x = new float[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = 0.3f * MathF.Sin(i * 0.05f) + 0.1f * ((float)rng.NextDouble() - 0.5f);
        }
        return x;
    }
}
