using System.Text.Json;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Streaming;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The streaming chunk planner and the self-forcing schedule against fixtures generated from upstream NVlabs/Sana
/// <c>b3bad956</c> itself (<c>Fixtures/generate_sana_wm_streaming_fixtures.py</c>): <c>chunk_index_from_chunk_size</c> for every
/// split strategy and <c>SelfForcingFlowEuler.create_autoregressive_segments</c> for totals up to 64, and diffusers'
/// <c>FlowMatchEulerDiscreteScheduler</c> driven as <c>SelfForcingFlowEulerCamCtrl</c> drives it.</summary>
public sealed class SanaWmStreamingUpstreamTests
{
    private static readonly Lazy<JsonDocument> Fixture = new(() =>
        JsonDocument.Parse(File.ReadAllBytes(Path.Combine(AppContext.BaseDirectory, "Fixtures", "sana_wm_streaming_upstream.json"))));

    [Theory]
    [InlineData("uniform", ChunkSplitStrategy.Uniform)]
    [InlineData("first_frame", ChunkSplitStrategy.FirstFrameAlone)]
    [InlineData("first_plus_one", ChunkSplitStrategy.FirstChunkPlusOne)]
    [InlineData("first_chunk_plus_one", ChunkSplitStrategy.FirstChunkPlusOne)]
    public void Boundaries_MatchUpstreamChunkIndex(string upstreamStrategy, ChunkSplitStrategy strategy)
    {
        int checkedCases = 0;
        foreach (JsonElement entry in Fixture.Value.RootElement.GetProperty("chunk_index").EnumerateArray())
        {
            if (entry.GetProperty("strategy").GetString() != upstreamStrategy)
                continue;
            int baseFrames = entry.GetProperty("base").GetInt32();
            int total = entry.GetProperty("total").GetInt32();
            Assert.False(entry.TryGetProperty("error", out _), $"upstream refused {upstreamStrategy} base {baseFrames} total {total}");
            int[] expected = [.. entry.GetProperty("starts").EnumerateArray().Select(e => e.GetInt32()), total];
            int[] actual = ChunkPlanner.Boundaries(total, baseFrames, strategy);
            Assert.True(expected.SequenceEqual(actual),
                $"{upstreamStrategy} base {baseFrames} total {total}: upstream [{string.Join(",", expected)}], ours [{string.Join(",", actual)}]");
            checkedCases++;
        }

        Assert.Equal(128, checkedCases);
    }

    [Fact]
    public void FirstAbsorbsRemainder_MatchesUpstreamAutoregressiveSegments()
    {
        int checkedCases = 0;
        foreach (JsonElement entry in Fixture.Value.RootElement.GetProperty("ar_segments").EnumerateArray())
        {
            int baseFrames = entry.GetProperty("base").GetInt32();
            int total = entry.GetProperty("total").GetInt32();
            int[] expected = [.. entry.GetProperty("bounds").EnumerateArray().Select(e => e.GetInt32())];
            int[] actual = ChunkPlanner.Boundaries(total, baseFrames, ChunkSplitStrategy.FirstAbsorbsRemainder);
            Assert.True(expected.SequenceEqual(actual),
                $"base {baseFrames} total {total}: upstream [{string.Join(",", expected)}], ours [{string.Join(",", actual)}]");
            checkedCases++;
        }

        Assert.True(checkedCases > 100);
    }

    [Fact]
    public void Schedule_MatchesDiffusersTimestepsAndSigmas()
    {
        JsonElement fixture = Fixture.Value.RootElement.GetProperty("schedule");
        int[] steps = [.. fixture.GetProperty("steps").EnumerateArray().Select(e => e.GetInt32())];
        float[] timesteps = [.. fixture.GetProperty("timesteps").EnumerateArray().Select(e => e.GetSingle())];
        float[] sigmas = [.. fixture.GetProperty("sigmas").EnumerateArray().Select(e => e.GetSingle())];
        SelfForcingFlowSchedule schedule = SelfForcingFlowSchedule.Parse(string.Join(",", steps));

        Assert.Equal(timesteps.Length, schedule.Steps);
        for (int i = 0; i < schedule.Steps; i++)
        {
            Assert.Equal(timesteps[i], schedule.Timestep(i));
            Assert.Equal(sigmas[i], schedule.Sigma(i), 6);
        }
    }

    /// <summary>The softmax KV window over chunks 0 to 6 against upstream's own <c>_accumulate_softmax_kv_cache</c>: before each
    /// chunk, the frames the window lists (sink first, then history oldest first) are the frames upstream's accumulated K
    /// holds, for 1, 2, 3 and unbounded cached chunks, with and without the sink.</summary>
    [Theory]
    [InlineData(-1, false)]
    [InlineData(-1, true)]
    [InlineData(1, false)]
    [InlineData(1, true)]
    [InlineData(2, false)]
    [InlineData(2, true)]
    [InlineData(3, false)]
    [InlineData(3, true)]
    public void RecentChunksWindow_KeepsWhatUpstreamAccumulates(int cachedChunks, bool sink)
    {
        JsonElement[] entries = [.. Fixture.Value.RootElement.GetProperty("window").EnumerateArray()
            .Where(e => e.GetProperty("cached").GetInt32() == cachedChunks && e.GetProperty("sink").GetBoolean() == sink)
            .OrderBy(e => e.GetProperty("chunk").GetInt32())];
        int[] bounds = [0, 4, 7, 10, 13, 16, 19, 22];
        using Core.Backends.Cpu.CpuBackend backend = new();
        using RollingTokenWindow window = new(backend, TokenWindowPolicy.RecentChunks(cachedChunks, sink), streams: 1, tokenAxis: 0);
        Core.Tensors.Tensor[] parts = new Core.Tensors.Tensor[bounds.Length];
        Assert.Equal(bounds.Length - 1, entries.Length);

        for (int chunk = 0; chunk < bounds.Length - 1; chunk++)
        {
            int[] expected = [.. entries[chunk].GetProperty("kept_frames").EnumerateArray().Select(e => e.GetInt32())];
            int count = window.CopyParts(0, parts);
            int[] actual = [.. parts.Take(count).SelectMany(p => p.AsReadOnlySpan<float>().ToArray()).Select(v => (int)v)];
            Assert.True(expected.SequenceEqual(actual),
                $"cached {cachedChunks}, sink {sink}, before chunk {chunk}: upstream [{string.Join(",", expected)}], ours [{string.Join(",", actual)}]");

            int frames = bounds[chunk + 1] - bounds[chunk];
            Core.Tensors.Tensor keys = new(new Core.Tensors.TensorShape(frames), Core.Tensors.DType.F32);
            keys.AsSpan<float>().Fill(chunk);
            window.Stage([keys], frames);
            window.Commit();
        }
    }

    /// <summary>A whole chunk walk (four steps) with frame 0 clean: our per-frame Euler step against diffusers'
    /// <c>step(-v, t, x, per_token_timesteps)</c> on the same velocities.</summary>
    [Fact]
    public void Step_MatchesDiffusersPerTokenEulerWithACleanFrame()
    {
        JsonElement fixture = Fixture.Value.RootElement.GetProperty("step");
        int[] shape = [.. fixture.GetProperty("shape").EnumerateArray().Select(e => e.GetInt32())];
        int channels = shape[1], frames = shape[2];
        float[] latents = [.. fixture.GetProperty("x0").EnumerateArray().Select(e => e.GetSingle())];
        bool[] clean = new bool[frames];
        foreach (JsonElement f in fixture.GetProperty("clean_frames").EnumerateArray())
            clean[f.GetInt32()] = true;
        JsonElement[] velocities = [.. fixture.GetProperty("velocities").EnumerateArray()];
        JsonElement[] after = [.. fixture.GetProperty("after").EnumerateArray()];
        SelfForcingFlowSchedule schedule = SelfForcingFlowSchedule.Parse("1000,960,889,727,0");
        float[] frameTimesteps = new float[frames];

        for (int step = 0; step < schedule.Steps; step++)
        {
            schedule.FrameTimesteps(step, clean, frameTimesteps);
            Assert.Equal(0f, frameTimesteps[0]);
            Assert.All(frameTimesteps.Skip(1), t => Assert.Equal(schedule.Timestep(step), t));
            float[] velocity = [.. velocities[step].EnumerateArray().Select(e => e.GetSingle())];
            schedule.Step(latents, velocity, step, channels, clean, roundToBf16: false);
            float[] expected = [.. after[step].EnumerateArray().Select(e => e.GetSingle())];
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - latents[i]) <= 1e-6f,
                    $"step {step} element {i}: diffusers {expected[i]}, ours {latents[i]}");
        }
    }
}
