using HartsyInference.Audio.Models.VoiceIsolator;
using HartsyInference.Core.Backends.Cpu;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests.VoiceIsolator.Parity;

/// <summary>Parity of the C# network, offline renderer and stream against the upstream PyTorch code on tiny random-weight
/// checkpoints (fp32, CPU), dumped by tests/python-reference/voice_isolator/dump_tiny_reference.py on the reference image from
/// inworld-ai/voice-isolator @ c73b3fab. <c>VOICE_ISOLATOR_TINY_REF</c> overrides the folder; without it the tests return having
/// is reported as skipped, and <c>HARTSY_REQUIRE_REAL_WEIGHTS=1</c> makes a missing folder a failure.</summary>
[Trait("Category", "Integration")]
public sealed class VoiceIsolatorTinyReferenceTests(ITestOutputHelper output)
{
    private const float Tolerance = 1e-4f;

    [RealWeightsFact("VOICE_ISOLATOR_TINY_REF")]
    public void OfflineShortClipMatchesUpstream()
    {
        if (Folder() is not { } dir)
        {
            return;
        }
        using VoiceIsolatorNetwork network = VoiceIsolatorNetwork.Load(Path.Combine(dir, "tiny_offline.safetensors"));
        using CpuBackend backend = new();
        float[] actual = VoiceIsolatorRenderer.Render(network, backend, NpyFloat32.Read(Path.Combine(dir, "offline_short_input.npy")));
        Compare("offline short", actual, NpyFloat32.Read(Path.Combine(dir, "offline_short_expected.npy")));
    }

    [RealWeightsFact("VOICE_ISOLATOR_TINY_REF")]
    public void OfflineLongClipRenderMatchesUpstream()
    {
        if (Folder() is not { } dir)
        {
            return;
        }
        using VoiceIsolatorNetwork network = VoiceIsolatorNetwork.Load(Path.Combine(dir, "tiny_offline.safetensors"));
        using CpuBackend backend = new();
        float[] actual = VoiceIsolatorRenderer.Render(network, backend, NpyFloat32.Read(Path.Combine(dir, "offline_long_input.npy")));
        Compare("offline long (3 windows)", actual, NpyFloat32.Read(Path.Combine(dir, "offline_long_expected.npy")));
    }

    [RealWeightsFact("VOICE_ISOLATOR_TINY_REF")]
    public void CausalPlainStreamMatchesUpstream() => CausalStreamMatchesUpstream("plain", null, null, null);

    [RealWeightsFact("VOICE_ISOLATOR_TINY_REF")]
    public void CausalWindowedStreamMatchesUpstream() => CausalStreamMatchesUpstream("windowed", 40, 12, 40);

    private void CausalStreamMatchesUpstream(string name, int? window, int? crossfade, int? cache)
    {
        if (Folder() is not { } dir)
        {
            return;
        }
        using VoiceIsolatorNetwork network = VoiceIsolatorNetwork.Load(Path.Combine(dir, "tiny_causal.safetensors"));
        using CpuBackend backend = new();
        float[] audio = NpyFloat32.Read(Path.Combine(dir, "causal_input.npy"));
        using VoiceIsolatorStream stream = new(network, backend, blockFrames: 4, cache, window, crossfade);
        List<float> produced = [];
        for (int i = 0; i < audio.Length; i += 500)
        {
            produced.AddRange(stream.Push(audio.AsSpan(i, Math.Min(500, audio.Length - i))));
        }
        produced.AddRange(stream.Flush());
        Compare($"causal stream {name}", [.. produced], NpyFloat32.Read(Path.Combine(dir, $"causal_stream_{name}_expected.npy")));
    }

    [RealWeightsFact("VOICE_ISOLATOR_TINY_REF")]
    public void CausalWholeClipEqualsTheUpstreamStreamBeforeAnyWindow()
    {
        if (Folder() is not { } dir)
        {
            return;
        }
        using VoiceIsolatorNetwork network = VoiceIsolatorNetwork.Load(Path.Combine(dir, "tiny_causal.safetensors"));
        using CpuBackend backend = new();
        float[] audio = NpyFloat32.Read(Path.Combine(dir, "causal_input.npy"));
        Compare("causal whole clip vs upstream plain stream", network.Isolate(backend, audio, 2),
            NpyFloat32.Read(Path.Combine(dir, "causal_stream_plain_expected.npy")));
    }

    private void Compare(string name, float[] actual, float[] expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double maxAbs = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            maxAbs = Math.Max(maxAbs, Math.Abs(actual[i] - expected[i]));
        }
        output.WriteLine($"{name}: {expected.Length} samples, max abs error {maxAbs:E3}");
        Assert.True(maxAbs <= Tolerance, $"{name}: max abs error {maxAbs:E3} exceeds {Tolerance:E1}");
    }

    private static string? Folder()
    {
        string? env = Environment.GetEnvironmentVariable("VOICE_ISOLATOR_TINY_REF");
        if (!string.IsNullOrEmpty(env) && Directory.Exists(env))
        {
            return env;
        }
        Assert.False(Environment.GetEnvironmentVariable("HARTSY_REQUIRE_REAL_WEIGHTS") == "1", "The tiny reference tensors are missing; nothing was validated.");
        return null;
    }
}
