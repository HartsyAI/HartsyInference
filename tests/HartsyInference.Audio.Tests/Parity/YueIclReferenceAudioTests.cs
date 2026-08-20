using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Models.Codecs.XCodec;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>End-to-end reference-audio (ICL) prompt assembly on CPU: real audio → x-codec codebook-0 codes →
/// infer.py's <c>audio_prompt_codec</c> offsets → the <c>sentence_ids</c> block YuE Stage-1 consumes.
///
/// <para>Gated on <c>YUE_XCODEC_PATH</c> (an encode-capable safetensors export of the <c>codec_model</c> state dict —
/// one WITHOUT the encode roots makes <c>CanEncode</c> false and the test asserts that rather than skipping) and
/// <c>YUE_TOKENIZER_PATH</c> (the mm_tokenizer_v0.2 <c>tokenizer.model</c>). The waveform is
/// <c>tests/python-reference/silerovad_reference/jfk.wav</c>, trimmed to 2.0 s so the HuBERT branch stays cheap.</para></summary>
public sealed class YueIclReferenceAudioTests
{
    private readonly ITestOutputHelper _out;
    public YueIclReferenceAudioTests(ITestOutputHelper o) => _out = o;

    private const int SampleRate = 16_000;
    private const int HopSize = 320;         // 8*5*4*2 -> 50 Hz frames
    private const int ClipSamples = 32_000;  // 2.0 s -> exactly 100 frames (no pad-fallback)
    private const int Offset = 45_334;

    private static string? FindJfkWav()
    {
        string? dir = AppContext.BaseDirectory;
        for (int up = 0; up < 8 && dir is not null; up++, dir = Path.GetDirectoryName(dir))
        {
            string candidate = Path.Combine(dir, "tests", "python-reference", "silerovad_reference", "jfk.wav");
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }

    /// <summary>The 2.0 s 16 kHz mono excerpt, downmixed by channel mean then resampled — upstream
    /// <c>load_audio_mono</c>'s order.</summary>
    private static float[]? LoadClip()
    {
        string? wav = FindJfkWav();
        if (wav is null) return null;
        WavFile.DecodedAudio decoded = WavFile.Read(wav);
        float[] mono = decoded.ToMono();
        if (decoded.SampleRate != SampleRate)
        {
            mono = Resampler.Create(decoded.SampleRate, SampleRate).Resample(mono);
        }
        if (mono.Length < ClipSamples) return null;
        float[] clip = new float[ClipSamples];
        Array.Copy(mono, clip, ClipSamples);
        return clip;
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ReferenceAudio_EncodesToCb0_AndAssemblesTheUpstreamIclPrompt()
    {
        string? ckpt = Environment.GetEnvironmentVariable("YUE_XCODEC_PATH");
        string? tokenizerPath = Environment.GetEnvironmentVariable("YUE_TOKENIZER_PATH");
        if (string.IsNullOrEmpty(ckpt) || !File.Exists(ckpt)) return;             // gated
        if (string.IsNullOrEmpty(tokenizerPath) || !File.Exists(tokenizerPath)) return;  // gated
        float[]? clip = LoadClip();
        if (clip is null) return;                                                 // gated

        (Dictionary<string, Tensor> w, SafeTensorsLoader loader) =
            YueCheckpointConverter.LoadXCodec(ckpt, castToF32: true, forEncode: true);
        using SafeTensorsLoader _ = loader;
        XCodec codec = new(XCodecConfig.XCodec16kHz);
        codec.LoadWeights(w);
        Assert.True(codec.CanEncode, "the export carries no encode branch — repack with forEncode.");

        using CpuBackend backend = new();
        int[] cb0 = Encode(codec, backend, clip);

        // 2.0 s at a 320-sample hop.
        Assert.Equal(ClipSamples / HopSize, cb0.Length);
        Assert.Equal(100, cb0.Length);
        foreach (int c in cb0) Assert.InRange(c, 0, XCodecConfig.XCodec16kHz.CodebookSize - 1);
        // A constant code sequence would mean the encoder collapsed; real audio must move through the codebook.
        Assert.True(new HashSet<int>(cb0).Count > 5, "codebook-0 codes are near-constant — the encode path is wrong.");
        _out.WriteLine($"cb0[0..8] = {string.Join(",", cb0[..8])} ({new HashSet<int>(cb0).Count} distinct)");

        // ── infer.py: code_ids = codectool.npy2ids(raw_codes[0]) ⇒ 45334 + code for codebook 0 ──
        int[] promptCodec = YueTokenizer.BuildAudioPromptCodec(cb0, [], 0.0, 30.0);
        Assert.Equal(cb0.Length, promptCodec.Length);
        for (int i = 0; i < cb0.Length; i++) Assert.Equal(Offset + cb0[i], promptCodec[i]);

        // ── infer.py: sentence_ids = tokenize("[start_of_reference]") + [soa] + sep_ids + codes + [eoa]
        //                            + tokenize("[end_of_reference]") ──
        using YueTokenizer tokenizer = new(tokenizerPath);
        Assert.Equal(32_001, tokenizer.Soa);
        Assert.Equal(32_002, tokenizer.Eoa);
        Assert.Equal(32_016, tokenizer.Xcodec);   // codectool.sep_ids == [<xcodec>]

        int[] startMarker = [.. tokenizer.EncodeRaw("[start_of_reference]")];
        int[] endMarker = [.. tokenizer.EncodeRaw("[end_of_reference]")];
        _out.WriteLine($"[start_of_reference] = {string.Join(",", startMarker)}");
        _out.WriteLine($"[end_of_reference]   = {string.Join(",", endMarker)}");
        // Pinned mm_tokenizer_v0.2 ids so this gates the literal sequence rather than only self-consistency. The two
        // markers share the "_of_reference]" tail (29918,974,29918,5679,29962), differing only in start(2962)/end(355).
        Assert.Equal([518, 2962, 29_918, 974, 29_918, 5679, 29_962], startMarker);
        Assert.Equal([518, 355, 29_918, 974, 29_918, 5679, 29_962], endMarker);

        List<int> expected = [.. startMarker, tokenizer.Soa, tokenizer.Xcodec, .. promptCodec, tokenizer.Eoa, .. endMarker];
        int[] block = tokenizer.EncodeReferenceBlock(promptCodec);
        Assert.Equal(expected, block);

        // Position-level structure, so a reordering that still round-trips through the same builder is caught.
        Assert.Equal(startMarker.Length + 2 + promptCodec.Length + 1 + endMarker.Length, block.Length);
        Assert.Equal(tokenizer.Soa, block[startMarker.Length]);
        Assert.Equal(tokenizer.Xcodec, block[startMarker.Length + 1]);
        Assert.Equal(promptCodec[0], block[startMarker.Length + 2]);
        Assert.Equal(promptCodec[^1], block[^(endMarker.Length + 2)]);
        Assert.Equal(tokenizer.Eoa, block[^(endMarker.Length + 1)]);

        // ── dual-track: the same clip as both stems interleaves v,i,v,i and doubles the token count ──
        int[] dual = YueTokenizer.BuildAudioPromptCodec(cb0, cb0, 0.0, 30.0);
        Assert.Equal(cb0.Length * 2, dual.Length);
        for (int i = 0; i < dual.Length; i++) Assert.Equal(Offset + cb0[i >> 1], dual[i]);
        // A 0.5 s window at 100 tokens/s is 50 tokens, opening on a vocal frame.
        int[] dualWindow = YueTokenizer.BuildAudioPromptCodec(cb0, cb0, 0.0, 0.5);
        Assert.Equal(50, dualWindow.Length);
        Assert.Equal(Offset + cb0[0], dualWindow[0]);
    }

    /// <summary>Runs the codec's encode path at <c>target_bw = 0.5</c> (codebook 0 only) over a 16 kHz mono clip.</summary>
    private static int[] Encode(XCodec codec, CpuBackend backend, float[] clip)
    {
        using Tensor pcm = new(new TensorShape(1, 1, clip.Length), DType.F32);
        clip.CopyTo(pcm.AsSpan<float>());
        using Tensor codes = codec.Encode(backend, pcm, clip.Length, nQ: 1);
        return codes.AsReadOnlySpan<int>().ToArray();
    }
}
