using System.Text.Json;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>LongCat-Video-Avatar's multi-speaker path against the pinned reference (tests/python-reference/longcat_avatar/
/// dump_longcat_avatar_multi.py @ 58eb336), CPU F32 on a tiny avatar DiT: the token masks of every source size and box layout, bit for bit; each
/// block's reference attention map, query and key positions and audio cross-attention output, and the velocity, from an image, in a continuation,
/// through the KV cache, with a silent third track and with a region that has no token; and the guided denoising loops. The empty region's expected
/// values are upstream pull request 159's, since the pinned code divides by zero there. Skipped, and so not evidence, until
/// HARTSY_LONGCAT_AVATAR_MULTI_DIR names the dump directory.</summary>
[Trait("Category", "Integration")]
public sealed class LongCatAvatarMultiSpeakerParityTests(ITestOutputHelper output)
{
    private const string EnvName = "HARTSY_LONGCAT_AVATAR_MULTI_DIR";
    private const double Tolerance = 1e-5;

    private static readonly string[] BlockStages = ["ref_attn_map", "query_positions", "key_positions", "audio_cross_attn"];

    [LongCatAvatarMultiFact]
    public void EveryTokenMaskMatchesTheReference()
    {
        string dir = Environment.GetEnvironmentVariable(EnvName)!;
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "meta.json")));
        List<string> failures = [];
        int index = 0;
        foreach (JsonElement item in document.RootElement.GetProperty("mask_cases").EnumerateArray())
        {
            int[] source = Ints(item.GetProperty("src")), target = Ints(item.GetProperty("target")), grid = Ints(item.GetProperty("grid"));
            (int regions, float[] masks) = LongCatSpeakerMasks.TokenMasks(source[0], source[1], Box(item.GetProperty("first")),
                Box(item.GetProperty("second")), Boxes(item.GetProperty("others")), target[0], target[1], grid[0], grid[1]);
            float[] expected = LongCatReferenceDump.Read(dir, document, $"mask.{index}");
            if (expected.Length != regions * grid[0] * grid[1] || !expected.AsSpan().SequenceEqual(masks))
                failures.Add($"case {index} ({source[0]}x{source[1]} to {target[0]}x{target[1]}): {Differences(expected, masks)} tokens differ");
            index++;
        }
        output.WriteLine($"{index} mask cases, {failures.Count} differ");
        Assert.True(index > 60, "The dump should hold the fixed and random mask cases.");
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    [LongCatAvatarMultiFact]
    public void EveryScenarioMatchesTheReference()
    {
        string dir = Environment.GetEnvironmentVariable(EnvName)!;
        using Reference reference = new(dir);
        List<string> failures = [];
        foreach (JsonProperty scenario in reference.Root.GetProperty("scenarios").EnumerateObject())
        {
            string name = scenario.Name;
            JsonElement s = scenario.Value;
            int tracks = s.GetProperty("tracks").GetInt32(), cond = s.GetProperty("num_cond").GetInt32();
            LongCatConditioning conditioning = s.GetProperty("num_ref").GetInt32() > 0
                ? LongCatConditioning.Avatar(cond, s.GetProperty("ref_index").GetInt32())
                : new LongCatConditioning(NumCondLatents: cond);
            LongCatSpeakers speakers = reference.Speakers(s.GetProperty("layout").GetString()!, tracks);
            float[] tokens = reference.AudioTokens(tracks, silent: false);

            Dictionary<string, float[]> got = [];
            using CpuBackend backend = new();
            using Tensor encoder = reference.Encoder();
            Tensor velocity;
            if (s.GetProperty("cached").GetBoolean())
            {
                using Tensor condLatent = reference.Latent(0, cond);
                using LongCatKvCache cache = reference.Dit.BuildKvCache(backend, condLatent, conditioning);
                using Tensor noisy = reference.Latent(cond, reference.Frames - cond);
                using Tensor audio = Shaped(tokens[(tokens.Length / reference.Frames * cond)..], tracks * (reference.Frames - cond)
                    * reference.Config.AudioContextTokens, reference.Config.AudioOutputDim);
                reference.Dit.Probe = (stage, t) => got[stage] = t.AsReadOnlySpan<float>().ToArray();
                velocity = reference.Dit.ForwardCached(backend, noisy, encoder, [500f], in conditioning, cache, audioTokens: audio,
                    speakers: speakers);
            }
            else
            {
                using Tensor latent = reference.Latent(0, reference.Frames);
                using Tensor audio = Shaped(tokens, tracks * reference.Frames * reference.Config.AudioContextTokens, reference.Config.AudioOutputDim);
                reference.Dit.Probe = (stage, t) => got[stage] = t.AsReadOnlySpan<float>().ToArray();
                velocity = reference.Dit.Forward(backend, latent, encoder, [500f], in conditioning, audioTokens: audio, speakers: speakers);
            }
            using (velocity)
            {
                reference.Dit.Probe = null;
                for (int block = 0; block < reference.Config.Depth; block++)
                    foreach (string stage in BlockStages)
                        Compare(failures, $"{name}/block{block}.{stage}", got.GetValueOrDefault($"block{block}.{stage}"),
                            reference.Read($"{name}.block{block}.{stage}"));
                Compare(failures, $"{name}/velocity", velocity.AsReadOnlySpan<float>().ToArray(), reference.Read($"{name}.velocity"));
            }
        }
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    [LongCatAvatarMultiFact]
    public void TheGuidedLoopsMatchTheReference()
    {
        string dir = Environment.GetEnvironmentVariable(EnvName)!;
        using Reference reference = new(dir);
        JsonElement pipeline = reference.Root.GetProperty("pipeline");
        int steps = pipeline.GetProperty("steps").GetInt32(), cond = pipeline.GetProperty("cond_frames").GetInt32();
        float shift = (float)pipeline.GetProperty("shift").GetDouble();
        float text = (float)pipeline.GetProperty("text_scale").GetDouble(), audio = (float)pipeline.GetProperty("audio_scale").GetDouble();
        using CpuBackend backend = new();
        LongCatAvatarPipeline runner = new(backend, reference.Dit);
        int textLength = reference.Root.GetProperty("text_len").GetInt32();
        LongCatPrompt positive = new(reference.Read("inputs.encoder"), textLength, textLength);
        LongCatPrompt negative = new(reference.Read("pipeline.negative"), textLength, textLength);
        LongCatSpeakers speakers = reference.Speakers("two", 2);
        float[] latent = reference.Read("inputs.latent");
        int channels = reference.Config.InChannels, hw = reference.Height * reference.Width;
        List<string> failures = [];

        LongCatAvatarRun ai2v = new(channels, reference.Frames, reference.Height, reference.Width, latent, positive, negative,
            reference.AudioTokens(2, silent: false), reference.AudioTokens(2, silent: true), steps, LongCatAvatarSchedule.Kind.Linear, shift,
            text, audio, LongCatAvatarMode.Ai2v, NumCondLatents: 1, CondLatents: FirstFrames(latent, channels, reference.Frames, hw, 1),
            Speakers: speakers);
        Compare(failures, "pipeline/ai2v", runner.Denoise(ai2v).Latents, reference.Read("pipeline.ai2v.latents"));

        LongCatAvatarRun avc = ai2v with
        {
            Mode = LongCatAvatarMode.Avc, NumCondLatents = cond, CondLatents = FirstFrames(latent, channels, reference.Frames, hw, cond),
            RefImgIndex = 10, UseKvCache = true,
        };
        Compare(failures, "pipeline/avc", runner.Denoise(avc).Latents, reference.Read("pipeline.avc.latents"));
        Assert.True(failures.Count == 0, string.Join(Environment.NewLine, failures));
    }

    private void Compare(List<string> failures, string name, float[]? actual, float[] expected)
    {
        if (actual is null)
        {
            failures.Add($"{name}: the C# forward did not report it");
            return;
        }
        double error = actual.Length == expected.Length ? LongCatTestData.RelRms(actual, expected) : double.PositiveInfinity;
        string line = $"{name}: relRMS {error:E2} (limit {Tolerance:E0}, {actual.Length} vs {expected.Length} values)";
        output.WriteLine(line);
        if (!(error <= Tolerance))
            failures.Add(line);
    }

    private static float[] FirstFrames(float[] latent, int channels, int frames, int hw, int count)
    {
        float[] result = new float[channels * count * hw];
        for (int c = 0; c < channels; c++)
            Array.Copy(latent, c * frames * hw, result, c * count * hw, count * hw);
        return result;
    }

    private static Tensor Shaped(float[] values, params long[] shape)
    {
        Tensor tensor = new(new TensorShape(shape), DType.F32);
        values.CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static int Differences(float[] expected, float[] actual) =>
        expected.Length != actual.Length ? Math.Max(expected.Length, actual.Length) : expected.Where((value, i) => value != actual[i]).Count();

    private static int[] Ints(JsonElement array) => [.. array.EnumerateArray().Select(static e => e.GetInt32())];

    private static LongCatSpeakerBox? Box(JsonElement box)
    {
        if (box.ValueKind == JsonValueKind.Null)
            return null;
        int[] v = Ints(box);
        return new LongCatSpeakerBox(v[0], v[1], v[2], v[3]);
    }

    private static LongCatSpeakerBox[] Boxes(JsonElement boxes) => [.. boxes.EnumerateArray().Select(box => Box(box)!.Value)];

    /// <summary>The dump's model and inputs.</summary>
    private sealed class Reference : IDisposable
    {
        private readonly string _dir;
        private readonly JsonDocument _document;
        private readonly Dictionary<string, Tensor> _weights = [];
        private readonly CpuBackend _backend = new();

        public Reference(string dir)
        {
            _dir = dir;
            _document = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "meta.json")));
            Config = LongCatAvatarDitParityTests.TinyAvatarConfig();
            foreach (JsonProperty tensor in Root.GetProperty("tensors").EnumerateObject())
                if (tensor.Name.StartsWith("dit.", StringComparison.Ordinal))
                    _weights[tensor.Name["dit.".Length..]] = LongCatReferenceDump.Load(dir, _document, tensor.Name);
            Dit = new LongCatTransformer(Config);
            Dit.LoadWeights(_weights);
        }

        public JsonElement Root => _document.RootElement;
        public LongCatConfig Config { get; }
        public LongCatTransformer Dit { get; }
        public int Frames => Root.GetProperty("frames").GetInt32();
        public int Height => Root.GetProperty("h").GetInt32();
        public int Width => Root.GetProperty("w").GetInt32();

        public float[] Read(string name) => LongCatReferenceDump.Read(_dir, _document, name);

        public Tensor Encoder() => Shaped(Read("inputs.encoder"), Root.GetProperty("text_len").GetInt32(), Config.CaptionChannels);

        /// <summary><paramref name="count"/> latent frames from <paramref name="from"/> as <c>[1, C, count, H, W]</c>.</summary>
        public Tensor Latent(int from, int count)
        {
            float[] all = Read("inputs.latent");
            int channels = Config.InChannels, hw = Height * Width;
            float[] part = new float[channels * count * hw];
            for (int c = 0; c < channels; c++)
                Array.Copy(all, (c * Frames + from) * hw, part, c * count * hw, count * hw);
            return Shaped(part, 1, channels, count, Height, Width);
        }

        /// <summary>The speaker conditioning of a layout, its masks built by the engine from the layout's boxes and compared with the
        /// dump's.</summary>
        public LongCatSpeakers Speakers(string layout, int tracks)
        {
            JsonElement item = Root.GetProperty("layouts").GetProperty(layout);
            int[] source = Ints(item.GetProperty("src"));
            int gridH = Height / 2, gridW = Width / 2;
            (int regions, float[] masks) = LongCatSpeakerMasks.TokenMasks(source[0], source[1], Box(item.GetProperty("first")),
                Box(item.GetProperty("second")), Boxes(item.GetProperty("others")), Height * 8, Width * 8, gridH, gridW);
            Assert.Equal(Read($"layout.{layout}.tokens"), masks);
            return new LongCatSpeakers(tracks, regions, gridH, gridW, masks);
        }

        /// <summary>Every track's raw windows through the audio projection, joined per frame; <paramref name="silent"/> zeroes the raw windows,
        /// the unconditional audio.</summary>
        public float[] AudioTokens(int tracks, bool silent)
        {
            float[] raw = Read("inputs.audio");
            int rows = 1 + Config.AudioVaeScale * (Frames - 1), width = Config.AudioWindow * Config.AudioBlock * Config.AudioChannel;
            float[][] perTrack = new float[tracks][];
            for (int t = 0; t < tracks; t++)
            {
                using Tensor windows = Shaped(silent ? new float[rows * width] : raw[(t * rows * width)..((t + 1) * rows * width)], rows, width);
                using Tensor projected = Dit.ProjectAudio(_backend, windows, Frames);
                perTrack[t] = projected.AsReadOnlySpan<float>().ToArray();
            }
            return LongCatAvatarAudio.JoinTracks(perTrack, Frames);
        }

        public void Dispose()
        {
            Dit.Dispose();
            foreach (Tensor tensor in _weights.Values)
                tensor.Dispose();
            _backend.Dispose();
            _document.Dispose();
        }
    }

    private sealed class LongCatAvatarMultiFactAttribute : FactAttribute
    {
        public LongCatAvatarMultiFactAttribute()
        {
            string? dir = Environment.GetEnvironmentVariable(EnvName);
            if (string.IsNullOrWhiteSpace(dir) || !File.Exists(Path.Combine(dir, "meta.json")))
                Skip = $"SKIPPED: {EnvName} is unset or has no meta.json; the LongCat multi-speaker reference comparison did not run.";
        }
    }
}
