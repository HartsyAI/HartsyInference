using System.Text.Json;
using HartsyInference.Audio.Models.ControlFoley;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Tiny random instances of the official <c>AudioGenerationNetwork</c> (<c>tools/controlfoley/network_reference.py</c>,
/// float32 CPU) must be reproduced by the port: condition preprocessing, empty conditions, velocities with and without
/// each condition, CFG, the rope tables and a full euler trajectory.</summary>
public sealed unsafe class ControlFoleyNetworkParityTests
{
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "ControlFoley");

    private sealed class Fixture : IDisposable
    {
        private readonly SafeTensorsLoader _loader = new();

        internal Fixture(string file)
        {
            _loader.Load(Path.Combine(Dir, file));
            foreach (string name in _loader.Descriptors.Keys)
            {
                Tensors[name] = _loader.GetTensor(name);
            }
        }

        internal Dictionary<string, Tensor> Tensors { get; } = new();

        internal Tensor this[string name] => Tensors[name];

        public void Dispose() => _loader.Dispose();
    }

    private static ControlFoleyNetworkConfig Config(JsonElement e) => new()
    {
        V2 = e.GetProperty("v2").GetBoolean(),
        LatentDim = e.GetProperty("latent_dim").GetInt32(),
        ClipDim = e.GetProperty("clip_dim").GetInt32(),
        VisualDim = e.GetProperty("visual_dim").GetInt32(),
        SyncDim = e.GetProperty("sync_dim").GetInt32(),
        TextDim = e.GetProperty("text_dim").GetInt32(),
        AudioDim = e.GetProperty("audio_dim").GetInt32(),
        TimbreDim = e.GetProperty("timbre_dim").GetInt32(),
        HiddenDim = e.GetProperty("hidden_dim").GetInt32(),
        Depth = e.GetProperty("depth").GetInt32(),
        FusedDepth = e.GetProperty("fused_depth").GetInt32(),
        NumHeads = e.GetProperty("num_heads").GetInt32(),
        MlpRatio = e.GetProperty("mlp_ratio").GetSingle(),
        LatentSeqLen = e.GetProperty("latent_seq_len").GetInt32(),
        ClipSeqLen = e.GetProperty("clip_seq_len").GetInt32(),
        VisualSeqLen = e.GetProperty("visual_seq_len").GetInt32(),
        SyncSeqLen = e.GetProperty("sync_seq_len").GetInt32(),
        TextSeqLen = e.GetProperty("text_seq_len").GetInt32(),
    };

    private static JsonElement Meta()
    {
        using JsonDocument doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(Dir, "controlfoley_net.json")));
        return doc.RootElement.Clone();
    }

    private static void AssertClose(Tensor expected, Tensor actual, string what, float tol = 1e-4f)
    {
        Assert.Equal(expected.ElementCount, actual.ElementCount);
        float* e = (float*)expected.DataPointer, a = (float*)actual.DataPointer;
        float worst = 0f, scale = 1e-6f;
        for (long i = 0; i < expected.ElementCount; i++)
        {
            worst = MathF.Max(worst, MathF.Abs(e[i] - a[i]));
            scale = MathF.Max(scale, MathF.Abs(e[i]));
        }

        Assert.True(worst <= tol * MathF.Max(1f, scale), $"{what}: max |Δ| = {worst} (max |ref| = {scale})");
    }

    private static void AssertConditions(Fixture f, string prefix, ControlFoleyConditions c, string what)
    {
        AssertClose(f[$"{prefix}.clip_f"], c.ClipF, $"{what} clip_f");
        AssertClose(f[$"{prefix}.sync_f"], c.SyncF, $"{what} sync_f");
        AssertClose(f[$"{prefix}.text_f"], c.TextF, $"{what} text_f");
        AssertClose(f[$"{prefix}.audio_f"], c.AudioF, $"{what} audio_f");
        AssertClose(f[$"{prefix}.timbre_f"], c.TimbreF, $"{what} timbre_f");
        AssertClose(f[$"{prefix}.clip_f_c"], c.ClipFC, $"{what} clip_f_c");
        AssertClose(f[$"{prefix}.text_f_c"], c.TextFC, $"{what} text_f_c");
    }

    private static ControlFoleyConditions Preprocess(IBackend backend, ControlFoleyNetwork net, Fixture f)
        => net.PreprocessConditions(backend, f["cond.clip_f"], f["cond.visual_f"], f["cond.sync_f"], f["cond.text_f"],
            f["cond.audio_f"], f["cond.timbre_f"]);

    [Theory]
    [InlineData("v1")]
    [InlineData("v2")]
    public void PreprocessEmptyAndFlow_MatchOfficialImplementation(string variant)
    {
        JsonElement meta = Meta();
        using Fixture f = new($"controlfoley_net_{variant}.safetensors");
        using IBackend backend = new CpuBackend();
        using ControlFoleyNetwork net = new(Config(meta.GetProperty(variant)));
        net.LoadWeights(f.Tensors);
        float[] ts = meta.GetProperty("timesteps").EnumerateArray().Select(e => e.GetSingle()).ToArray();
        float cfg = meta.GetProperty("cfg").GetSingle();
        int bs = meta.GetProperty("batch").GetInt32();

        using ControlFoleyConditions cond = Preprocess(backend, net, f);
        AssertConditions(f, "ref.pre", cond, "preprocess");
        using ControlFoleyConditions empty = net.GetEmptyConditions(backend, bs);
        AssertConditions(f, "ref.empty", empty, "empty");
        using ControlFoleyConditions emptyNeg = net.GetEmptyConditions(backend, bs, f["cond.negative_text_f"]);
        AssertConditions(f, "ref.emptyneg", emptyNeg, "empty+negative text");

        AssertClose(f["ref.rot.latent"], RotationTensor(net.LatentRope), "latent rope");
        AssertClose(f["ref.rot.clip"], RotationTensor(net.ClipRope), "clip rope");

        for (int i = 0; i < ts.Length; i++)
        {
            float[] t = Enumerable.Repeat(ts[i], bs).ToArray();
            using Tensor flow = net.PredictFlow(backend, f["cond.latent"], t, cond, out Tensor multimodal);
            using (multimodal)
            {
                if (i == 0)
                {
                    AssertClose(f["ref.multimodal"], multimodal, "multimodal");
                }
            }

            AssertClose(f[$"ref.flow.t{i}"], flow, $"flow t{i}");
            using Tensor guided = net.OdeWrapper(backend, ts[i], f["cond.latent"], cond, empty, cfg);
            AssertClose(f[$"ref.ode_cfg.t{i}"], guided, $"cfg flow t{i}");
            if (variant == "v1")
            {
                using Tensor guidedNeg = net.OdeWrapper(backend, ts[i], f["cond.latent"], cond, emptyNeg, cfg);
                AssertClose(f[$"ref.ode_cfgneg.t{i}"], guidedNeg, $"cfg+negative flow t{i}");
                using Tensor plain = net.OdeWrapper(backend, ts[i], f["cond.latent"], cond, empty, 0.5f);
                AssertClose(f[$"ref.ode_nocfg.t{i}"], plain, $"no-cfg flow t{i}");
            }
        }
    }

    [Fact]
    public void MissingModalities_MatchOfficialEmptyFeatures()
    {
        JsonElement meta = Meta();
        using Fixture f = new("controlfoley_net_v1.safetensors");
        using IBackend backend = new CpuBackend();
        using ControlFoleyNetwork net = new(Config(meta.GetProperty("v1")));
        net.LoadWeights(f.Tensors);
        float t1 = meta.GetProperty("timesteps")[1].GetSingle();
        int bs = meta.GetProperty("batch").GetInt32();

        AssertClose(f["ref.getter.clip_f"], net.GetEmptyClipSequence(bs), "empty clip sequence");
        AssertClose(f["ref.getter.visual_f"], net.GetEmptyVisualSequence(bs), "empty visual sequence");
        AssertClose(f["ref.getter.sync_f"], net.GetEmptySyncSequence(bs), "empty sync sequence");
        AssertClose(f["ref.getter.text_f"], net.GetEmptyStringSequence(bs), "empty string sequence");
        AssertClose(f["ref.getter.audio_f"], net.GetEmptyAudioSequence(bs), "empty audio sequence");
        AssertClose(f["ref.getter.timbre_f"], net.GetEmptyTimbreSequence(bs), "empty timbre sequence");

        foreach (string name in new[] { "clip_f", "visual_f", "sync_f", "text_f", "audio_f", "timbre_f" })
        {
            Tensor[] inputs = new[] { "clip_f", "visual_f", "sync_f", "text_f", "audio_f", "timbre_f" }
                .Select(n => n == name ? EmptyFor(net, n, bs) : f[$"cond.{n}"]).ToArray();
            using ControlFoleyConditions cond = net.PreprocessConditions(backend, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]);
            using Tensor flow = net.PredictFlow(backend, f["cond.latent"], Enumerable.Repeat(t1, bs).ToArray(), cond, out Tensor mm);
            mm.Dispose();
            AssertClose(f[$"ref.drop.{name}"], flow, $"flow without {name}");
            foreach (Tensor t in inputs.Where(t => !f.Tensors.ContainsValue(t)))
            {
                t.Dispose();
            }
        }
    }

    private static Tensor EmptyFor(ControlFoleyNetwork net, string name, int bs) => name switch
    {
        "clip_f" => net.GetEmptyClipSequence(bs),
        "visual_f" => net.GetEmptyVisualSequence(bs),
        "sync_f" => net.GetEmptySyncSequence(bs),
        "text_f" => net.GetEmptyStringSequence(bs),
        "audio_f" => net.GetEmptyAudioSequence(bs),
        _ => net.GetEmptyTimbreSequence(bs),
    };

    [Fact]
    public void Bf16Weights_TrackTheFloat32Reference()
    {
        JsonElement meta = Meta();
        using Fixture f = new("controlfoley_net_v1.safetensors");
        using IBackend backend = new CpuBackend();
        List<Tensor> converted = [];
        Dictionary<string, Tensor> weights = new();
        foreach (KeyValuePair<string, Tensor> kv in f.Tensors)
        {
            bool isWeight = !kv.Key.StartsWith("cond.", StringComparison.Ordinal) && !kv.Key.StartsWith("ref.", StringComparison.Ordinal);
            if (isWeight && kv.Value.Shape.Rank is 2 or 3)
            {
                Tensor bf16 = kv.Value.CastTo(DType.BF16);
                converted.Add(bf16);
                weights[kv.Key] = bf16;
            }
            else
            {
                weights[kv.Key] = kv.Value;
            }
        }

        using ControlFoleyNetwork net = new(Config(meta.GetProperty("v1")));
        net.LoadWeights(weights);
        int bs = meta.GetProperty("batch").GetInt32();
        float t0 = meta.GetProperty("timesteps")[0].GetSingle();
        using ControlFoleyConditions cond = Preprocess(backend, net, f);
        using Tensor flow = net.PredictFlow(backend, f["cond.latent"], Enumerable.Repeat(t0, bs).ToArray(), cond, out Tensor multimodal);
        multimodal.Dispose();
        AssertClose(f["ref.flow.t0"], flow, "bf16 flow", tol: 5e-2f);
        foreach (Tensor t in converted)
        {
            t.Dispose();
        }
    }

    [Fact]
    public void EulerTrajectory_MatchesOfficialFlowMatching()
    {
        JsonElement meta = Meta();
        using Fixture f = new("controlfoley_net_v1.safetensors");
        using IBackend backend = new CpuBackend();
        using ControlFoleyNetwork net = new(Config(meta.GetProperty("v1")));
        net.LoadWeights(f.Tensors);
        int bs = meta.GetProperty("batch").GetInt32(), steps = meta.GetProperty("steps").GetInt32();
        float cfg = meta.GetProperty("cfg").GetSingle();
        using ControlFoleyConditions cond = Preprocess(backend, net, f);
        using ControlFoleyConditions empty = net.GetEmptyConditions(backend, bs);

        using Tensor normalized = ControlFoleySampler.Sample(backend, net, cond, empty, f["cond.noise"], steps, cfg, unnormalize: false,
            onState: (i, state) => AssertClose(f[$"ref.traj.state{i}"], state, $"state {i}"));
        AssertClose(f["ref.traj.final_normalized"], normalized, "final (normalized)");
        using Tensor latent = ControlFoleySampler.Sample(backend, net, cond, empty, f["cond.noise"], steps, cfg);
        AssertClose(f["ref.traj.final"], latent, "final (un-normalized)");
    }

    [Fact]
    public void UpdatedSequenceLengths_MatchOfficialImplementation()
    {
        JsonElement meta = Meta();
        using Fixture weights = new("controlfoley_net_v1.safetensors");
        using Fixture f = new("controlfoley_net_resized.safetensors");
        using IBackend backend = new CpuBackend();
        using ControlFoleyNetwork net = new(Config(meta.GetProperty("v1")));
        net.LoadWeights(weights.Tensors);
        JsonElement r = meta.GetProperty("resized");
        net.UpdateSequenceLengths(net.Config with
        {
            LatentSeqLen = r.GetProperty("latent_seq_len").GetInt32(),
            ClipSeqLen = r.GetProperty("clip_seq_len").GetInt32(),
            VisualSeqLen = r.GetProperty("visual_seq_len").GetInt32(),
            SyncSeqLen = r.GetProperty("sync_seq_len").GetInt32(),
        });
        AssertClose(f["ref.rot.latent"], RotationTensor(net.LatentRope), "resized latent rope");
        AssertClose(f["ref.rot.clip"], RotationTensor(net.ClipRope), "resized clip rope");
        using ControlFoleyConditions cond = net.PreprocessConditions(backend, f["cond.clip_f"], f["cond.visual_f"], f["cond.sync_f"],
            f["cond.text_f"], f["cond.audio_f"], f["cond.timbre_f"]);
        using Tensor flow = net.PredictFlow(backend, f["cond.latent"], [r.GetProperty("t").GetSingle()], cond, out Tensor mm);
        mm.Dispose();
        AssertClose(f["ref.flow"], flow, "resized flow");
    }

    [Fact]
    public void TemporalConfig_MatchesOfficialLengths()
    {
        foreach (JsonElement row in Meta().GetProperty("temporal").EnumerateArray())
        {
            ControlFoleyTemporalConfig c = ControlFoleyTemporalConfig.Default44k with { TotalTimeSeconds = row.GetProperty("seconds").GetDouble() };
            if (row.TryGetProperty("error", out _))
            {
                Assert.Throws<InvalidOperationException>(() => c.SyncSequenceLength);
                continue;
            }

            Assert.Equal(row.GetProperty("latent").GetInt32(), c.LatentSequenceLength);
            Assert.Equal(row.GetProperty("clip").GetInt32(), c.ClipSequenceLength);
            Assert.Equal(row.GetProperty("visual").GetInt32(), c.VisualSequenceLength);
            Assert.Equal(row.GetProperty("sync").GetInt32(), c.SyncSequenceLength);
            Assert.Equal(row.GetProperty("samples").GetInt32(), c.TotalAudioSampleCount);
        }

        ControlFoleyNetworkConfig large = ControlFoleyNetworkConfig.Large44k.WithSequenceLengths(ControlFoleyTemporalConfig.Default44k);
        Assert.Equal((345, 64, 32, 192), (large.LatentSeqLen, large.ClipSeqLen, large.VisualSeqLen, large.SyncSeqLen));
    }

    [Fact]
    public void Sampler_TimeGridAndNoiseAreDeterministic()
    {
        float[] grid = ControlFoleySampler.TimeGrid(4);
        Assert.Equal([0f, 0.25f, 0.5f, 0.75f, 1f], grid);
        using Tensor a = ControlFoleySampler.CreateNoise(1, 6, 4, 7);
        using Tensor b = ControlFoleySampler.CreateNoise(1, 6, 4, 7);
        using Tensor c = ControlFoleySampler.CreateNoise(1, 6, 4, 8);
        Assert.True(new ReadOnlySpan<float>((void*)a.DataPointer, 24).SequenceEqual(new ReadOnlySpan<float>((void*)b.DataPointer, 24)));
        Assert.False(new ReadOnlySpan<float>((void*)a.DataPointer, 24).SequenceEqual(new ReadOnlySpan<float>((void*)c.DataPointer, 24)));
    }

    // The python table is [1, N, D/2, 2, 2] = [[cos, -sin], [sin, cos]] per pair; compare cos and sin.
    private static Tensor RotationTensor(ControlFoleyRope rope)
    {
        Tensor t = new(new TensorShape(1, rope.Length, rope.Pairs, 4), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < rope.Length * rope.Pairs; i++)
        {
            p[4 * i] = rope.Cos[i];
            p[4 * i + 1] = -rope.Sin[i];
            p[4 * i + 2] = rope.Sin[i];
            p[4 * i + 3] = rope.Cos[i];
        }

        return t;
    }
}
