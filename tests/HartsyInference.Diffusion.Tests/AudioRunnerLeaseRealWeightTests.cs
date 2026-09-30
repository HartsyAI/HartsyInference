using System.Security.Cryptography;
using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Io;
using HartsyInference.Engine;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight CPU checks that a lease means what the service call means on the same spec and input: Kokoro
/// through a lease encodes to the service's exact WAV bytes, and Whisper-tiny gives the service's exact JFK
/// transcript, both at the model's rate and through the resampler. Skips through <see cref="RealWeightGate"/> when the
/// weights are not on the box.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class AudioRunnerLeaseRealWeightTests
{
    private const string Sentence = "The weather tomorrow is clear and mild with a gentle breeze.";

    private readonly ITestOutputHelper _output;

    public AudioRunnerLeaseRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task KokoroLease_MatchesTheServiceBytes_OnCpu()
    {
        string repoDir = AudioModelCache.GetRepoDirectory("hexgrad/Kokoro-82M", "tts");
        string repack = Path.Combine(AudioModelCache.GetRepoDirectory("Hartsy/kokoro-82m-safetensors", "tts"), "kokoro-82m.safetensors");
        string weights = File.Exists(repack) ? repack : Path.Combine(repoDir, "kokoro-82m.safetensors");
        if (!RealWeightGate.Require(_output.WriteLine, Path.Combine(repoDir, "config.json"), weights,
            Path.Combine(repoDir, "voices", "af_heart.bin"), AudioModelRoot.SharedFile("cmudict.dict")))
        {
            return;
        }

        using InferenceEngine engine = new("cpu");
        ModelSpec spec = ModelResolver.Resolve("kokoro", modelPathArg: null, Modality.Speech);
        SpeechRequest request = new() { Text = Sentence, Voice = "af_heart" };
        using ISynthesizerLease lease = await engine.Speech.OpenSynthesizerAsync(spec);

        float[] leased = lease.Synthesize(Sentence, request);
        float[] again = lease.Synthesize(Sentence, request);
        AudioResult served = await engine.Speech.SynthesizeAsync(spec, request);
        byte[] leasedWav = AudioClipCodec.EncodeWav(leased, null, lease.SampleRate);
        _output.WriteLine($"Kokoro lease: {leased.Length} samples @ {lease.SampleRate} Hz, PCM digest {PcmDigest.Of(leased)}; "
            + $"service WAV {served.Data.Length} bytes");

        Assert.Equal(PcmDigest.Of(leased), PcmDigest.Of(again));
        Assert.Equal(lease.SampleRate, served.SampleRate);
        Assert.Equal(Convert.ToHexString(SHA256.HashData(served.Data)), Convert.ToHexString(SHA256.HashData(leasedWav)));
    }

    [Fact]
    public async Task WhisperTinyLease_MatchesTheServiceTranscript_OnCpu()
    {
        string repoDir = AudioModelCache.GetRepoDirectory("openai/whisper-tiny", "stt");
        string jfk = Path.Combine(RepoPaths.RepoRoot(), "tests", "python-reference", "silerovad_reference", "jfk.wav");
        if (!RealWeightGate.Require(_output.WriteLine, Path.Combine(repoDir, "model.safetensors"),
            Path.Combine(repoDir, "added_tokens.json"), jfk))
        {
            return;
        }

        using InferenceEngine engine = new("cpu");
        ModelSpec spec = ModelResolver.Resolve("whisper:openai/whisper-tiny", modelPathArg: null, Modality.Transcribe);
        AudioClip clip = new() { Data = await File.ReadAllBytesAsync(jfk), Format = "wav" };
        AudioRequest request = new() { Audio = clip };
        using ITranscriberLease lease = await engine.Transcribe.OpenTranscriberAsync(spec);

        // At the model's rate: the samples the service decodes, handed to the lease as PCM.
        string served = (await engine.Transcribe.RunAsync(spec, request)).Text;
        string leased = lease.Transcribe(AudioClipCodec.DecodeMono(clip, 16_000), 16_000, request);
        _output.WriteLine($"16 kHz: service \"{served}\" / lease \"{leased}\"");
        Assert.Contains("country", served, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(served, leased);

        // Through the resampler: a 24 kHz clip goes to the lease at its own rate, and the lease must resample it
        // exactly as the service's decode does.
        float[] upsampled = Resampler.Create(16_000, 24_000).Resample(AudioClipCodec.DecodeMono(clip, 16_000));
        AudioClip clip24k = new() { Data = AudioClipCodec.EncodeWav(upsampled, null, 24_000), Format = "wav" };
        string served24k = (await engine.Transcribe.RunAsync(spec, new AudioRequest { Audio = clip24k })).Text;
        string leased24k = lease.Transcribe(AudioClipCodec.DecodeMono(clip24k, 24_000), 24_000, request);
        _output.WriteLine($"24 kHz: service \"{served24k}\" / lease \"{leased24k}\"");
        Assert.Contains("country", served24k, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(served24k, leased24k);
    }
}
