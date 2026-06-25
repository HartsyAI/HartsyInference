using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Chatterbox;
using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>End-to-end Chatterbox text -> 24 kHz WAV on the real ResembleAI/chatterbox checkpoints. Merges the
/// three shipped files (<c>t3_cfg.safetensors</c> under a <c>t3.</c> prefix, <c>s3gen.safetensors</c> under
/// <c>s3gen.</c>) and loads them through <see cref="ChatterboxPipeline.LoadWeights"/> — which validates every
/// expected key. Drives the default voice from the precomputed conditionals exported out of <c>conds.pt</c>
/// (<c>chatterbox_conds.safetensors</c>: <c>t3_speaker_emb</c> [1,256] + <c>flow_embedding</c> [1,192]).
///
/// <para>Gated on <c>CHATTERBOX_DIR</c> (the folder holding the four checkpoint files + the conds export +
/// <c>tokenizer.json</c>). Optional: <c>CHATTERBOX_TEXT</c>, <c>CHATTERBOX_OUT_WAV</c>. Verifies the loader
/// finds all keys, the T3 emits speech tokens, and the output is finite + non-silent. CosyVoice2-0.5B is the
/// exact S3Gen config (verified: SpeakerEmbedDim 192, HiFT [8,5,3], 10 Euler steps).</para></summary>
public sealed class ChatterboxEndToEndTests
{
    private readonly ITestOutputHelper _out;
    public ChatterboxEndToEndTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void Text_To_Wav_OnRealWeights()
    {
        string? dir = Environment.GetEnvironmentVariable("CHATTERBOX_DIR");
        if (string.IsNullOrEmpty(dir) || !File.Exists(Path.Combine(dir, "t3_cfg.safetensors")))
            return; // gated

        string text = Environment.GetEnvironmentVariable("CHATTERBOX_TEXT") ?? "Hello there, this is a test.";
        string outWav = Environment.GetEnvironmentVariable("CHATTERBOX_OUT_WAV")
            ?? Path.Combine(Path.GetTempPath(), "chatterbox_out.wav");

        // Merge the separate checkpoint files under the prefixes ChatterboxPipeline.LoadWeights expects.
        Dictionary<string, Tensor> merged = new();
        MergeInto(merged, Path.Combine(dir, "t3_cfg.safetensors"), "t3.");
        MergeInto(merged, Path.Combine(dir, "s3gen.safetensors"), "s3gen.");
        _out.WriteLine($"Merged {merged.Count} tensors (t3. + s3gen.).");

        ChatterboxConfig cfg = ChatterboxConfig.Default;
        if (int.TryParse(Environment.GetEnvironmentVariable("CHATTERBOX_MAXNEW"), out int mn)) cfg = cfg with { MaxNewTokens = mn };
        CosyVoiceConfig cosy = CosyVoiceConfig.V2_0_5B;   // Chatterbox S3Gen == CosyVoice2-0.5B
        using ChatterboxT3 t3 = new(cfg);
        using CosyVoiceFlow flow = new(cosy);
        using HiFTNetVocoder vocoder = new(cosy.Hift);
        using CamPlusSpeakerEncoder spkEnc = new(cosy.Flow.SpeakerEmbedDim);
        using ChatterboxPipeline pipe = new(cfg, t3, flow, vocoder, spkEnc);

        // This is the key-validation gate: a missing/renamed key throws here.
        pipe.LoadWeights(merged);
        _out.WriteLine("LoadWeights OK — all expected keys present.");

        // Precomputed default-voice conditionals (exported from conds.pt).
        SafeTensorsLoader condLoader = new();
        condLoader.Load(Path.Combine(dir, "chatterbox_conds.safetensors"));
        IReadOnlyDictionary<string, Tensor> conds = condLoader.GetAllTensors();
        using Tensor refSpk = Flatten(conds["t3_speaker_emb"], cfg.SpeakerEmbedDim);   // [256]
        Tensor flowSpk = conds["flow_embedding"];                                       // [1,192]

        using Stream tokStream = File.OpenRead(Path.Combine(dir, "tokenizer.json"));
        ChatterboxEnTokenizer tok = new(tokStream);
        int[] textTokens = tok.EncodeWithStartStop(text);   // wraps with START(255)/STOP(0) text tokens
        _out.WriteLine($"Text: \"{text}\" -> {textTokens.Length} tokens.");

        using CpuBackend backend = new();
        long t0 = Environment.TickCount64;
        float[] pcm = pipe.Synthesize(backend, textTokens, refSpk, cfg.Exaggeration, seed: 0,
            flowSpeakerEmbed: flowSpk);
        double secs = (Environment.TickCount64 - t0) / 1000.0;

        Assert.NotEmpty(pcm);
        double sumSq = 0; float peak = 0;
        foreach (float v in pcm) { Assert.True(float.IsFinite(v)); sumSq += (double)v * v; peak = Math.Max(peak, Math.Abs(v)); }
        double rms = Math.Sqrt(sumSq / pcm.Length);
        WavFile.WriteMono16(outWav, pcm, cfg.SampleRate);
        _out.WriteLine($"Generated {pcm.Length} samples ({pcm.Length / (double)cfg.SampleRate:0.00}s) in {secs:0.0}s. RMS={rms:0.0000} peak={peak:0.0000}. WAV: {outWav}");
        Assert.True(rms > 1e-4, "output should not be silent");
    }

    private static void MergeInto(Dictionary<string, Tensor> dst, string path, string prefix)
    {
        SafeTensorsLoader loader = new();
        loader.Load(path);
        foreach (KeyValuePair<string, Tensor> kv in loader.GetAllTensors())
            dst[prefix + kv.Key] = kv.Value;
    }

    private static unsafe Tensor Flatten(Tensor src, int n)
    {
        Tensor t = new(new TensorShape(n), DType.F32);
        Buffer.MemoryCopy((void*)src.DataPointer, (void*)t.DataPointer, (long)n * 4, (long)n * 4);
        return t;
    }
}
