using System;
using System.Collections.Generic;
using System.IO;
using HartsyInference.Audio.Models.Codecs.XCodec;
using HartsyInference.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Numerical parity for the YuE x-codec (SoundStream) ENCODE path against the standalone torch
/// reference in <c>tests/python-reference/dump_yue_xcodec_encode_reference.py</c>, which re-derives
/// <c>SoundStream.encode</c> straight from the real <c>ckpt_00360000.pth</c>. Gates every stage boundary:
/// <list type="bullet">
///   <item><c>hubert_feats</c> — the HuBERT-base branch, mean over all 13 hidden states (NOT layer 9), on the
///   RAW (unnormalized) 160-padded waveform.</item>
///   <item><c>semantic_enc_out</c> — the RepCodec <c>encoder_semantic</c>.</item>
///   <item><c>acoustic_enc_out</c> — the dac2 encoder with the 256-wide latent override.</item>
///   <item><c>fc_prior_out</c> — concat order (acoustic first) + the 1024→1024 channel projection.</item>
///   <item><c>codes_i32</c> — all 12 RVQ stages, EXACT index match.</item>
///   <item>the <c>pad_*</c> set at S=8080, where the two branches disagree (25 vs 26 frames) and the acoustic
///   encoder is re-run on the padded waveform — the common case, not an edge case.</item>
/// </list>
/// Gated on <c>YUE_XCODEC_PATH</c> (a safetensors export of the <c>codec_model</c> state dict) +
/// <c>YUE_XCODEC_ENCODE_REF_IO</c>. See PARITY_VERIFICATION.md.</summary>
public sealed unsafe class YueXCodecEncodeParityTests
{
    private readonly ITestOutputHelper _out;
    public YueXCodecEncodeParityTests(ITestOutputHelper o) => _out = o;

    [Fact]
    [Trait("Category", "Integration")]
    public void XCodecEncode_MatchesTorchReference()
    {
        string? ckpt = Environment.GetEnvironmentVariable("YUE_XCODEC_PATH");
        string? refIo = Environment.GetEnvironmentVariable("YUE_XCODEC_ENCODE_REF_IO");
        if (string.IsNullOrEmpty(ckpt) || !File.Exists(ckpt) || string.IsNullOrEmpty(refIo) || !File.Exists(refIo))
            return; // gated

        (Dictionary<string, Tensor> w, SafeTensorsLoader loader) = YueCheckpointConverter.LoadXCodec(ckpt, castToF32: true, forEncode: true);
        using SafeTensorsLoader _ = loader;
        _out.WriteLine($"x-codec tensors after encode mapping: {w.Count}");

        XCodec codec = new(XCodecConfig.XCodec16kHz);
        codec.LoadWeights(w);
        Assert.True(codec.CanEncode, "encode weights were dropped by the converter.");

        SafeTensorsLoader io = new();
        io.Load(refIo);
        IReadOnlyDictionary<string, Tensor> d = io.GetAllTensors();

        using CpuBackend backend = new();
        RunOne(codec, w, backend, d, "", expectFallback: false);
        RunOne(codec, w, backend, d, "pad_", expectFallback: true);
    }

    /// <summary>Runs one reference input (the plain set or the <c>pad_</c> fallback set) and gates every stage.</summary>
    private void RunOne(XCodec codec, IReadOnlyDictionary<string, Tensor> w, CpuBackend backend,
        IReadOnlyDictionary<string, Tensor> d, string tag, bool expectFallback)
    {
        Tensor wavRef = d[$"{tag}wav_in"];
        int s = (int)wavRef.Shape[2];
        Tensor pcm = new(new TensorShape(1, 1, s), DType.F32);
        long bytes = (long)s * sizeof(float);
        Buffer.MemoryCopy((void*)wavRef.DataPointer, (void*)pcm.DataPointer, bytes, bytes);
        _out.WriteLine($"-- [{(tag.Length == 0 ? "plain" : "pad-fallback")}] S={s}");

        XCodec.XCodecEncodeStages stages = codec.EncodeStages(backend, pcm, s);
        _out.WriteLine($"   frames={stages.Frames}  padFallback={stages.UsedPadFallback}");
        Assert.Equal(expectFallback, stages.UsedPadFallback);

        // Every stage is measured before any of them asserts, so one regression still prints the whole table.
        List<string> failures = [];

        // hubert_feats is only dumped for the plain set (channels-last [1, T, 768]).
        if (d.TryGetValue($"{tag}hubert_feats", out Tensor? featsRef))
        {
            Tensor feats = HubertFeats(codec, w, backend, pcm, s);
            Report(failures, $"{tag}hubert_feats", feats, featsRef, refChannelsLast: true);
            feats.Dispose();
        }
        Report(failures, $"{tag}semantic_enc_out", stages.Semantic, d[$"{tag}semantic_enc_out"], refChannelsLast: false);
        Report(failures, $"{tag}acoustic_enc_out", stages.Acoustic, d[$"{tag}acoustic_enc_out"], refChannelsLast: false);
        Report(failures, $"{tag}fc_prior_out", stages.Prior, d[$"{tag}fc_prior_out"], refChannelsLast: false);

        Tensor codesRef = d[$"{tag}codes_i32"];
        int nq = (int)codesRef.Shape[0];
        Tensor codes = codec.Encode(backend, pcm, s, nQ: nq);
        Assert.Equal(codesRef.ElementCount, codes.ElementCount);
        int* mine = (int*)codes.DataPointer;
        int* theirs = (int*)codesRef.DataPointer;
        long total = codes.ElementCount, exact = 0, cb0Exact = 0;
        int firstBadStage = -1, firstBadPos = -1;
        for (long i = 0; i < total; i++)
        {
            if (mine[i] == theirs[i]) { exact++; if (i < stages.Frames) cb0Exact++; }
            else if (firstBadStage < 0) { firstBadStage = (int)(i / stages.Frames); firstBadPos = (int)(i % stages.Frames); }
        }
        _out.WriteLine($"   {tag}codes_i32 [{nq},1,{stages.Frames}] exact {exact}/{total} = {100.0 * exact / total:F3}%"
            + $"  (cb0 {cb0Exact}/{stages.Frames})");
        if (firstBadStage >= 0)
        {
            long idx = (long)firstBadStage * stages.Frames + firstBadPos;
            _out.WriteLine($"   first mismatch: stage {firstBadStage} pos {firstBadPos} mine={mine[idx]} ref={theirs[idx]}");
            failures.Add($"{tag}codes_i32 mismatch: {total - exact} of {total} indices differ.");
        }

        // Round-trip the encoder's own codes through the (already-verified) decoder — this is what proves the
        // EncoderRates/EncoderLatentDim additions to ToDacConfig left the decode path alone.
        if (d.TryGetValue($"{tag}rvq_quantized_out", out Tensor? quantRef))
        {
            Tensor quant = codec.DecodeToLatent(backend, codes, 1, stages.Frames);
            Report(failures, $"{tag}rvq_quantized_out", quant, quantRef, refChannelsLast: false);
            quant.Dispose();
        }
        if (d.TryGetValue($"{tag}decoded_out", out Tensor? decodedRef))
        {
            Tensor pcmOut = codec.Decode(backend, codes, 1, stages.Frames);
            Report(failures, $"{tag}decoded_out", pcmOut, decodedRef, refChannelsLast: false);
            pcmOut.Dispose();
        }

        // The residual RMS must fall monotonically across the RVQ stages — the cheapest check that the residual
        // loop actually subtracts the chosen codeword rather than re-quantizing the same vector 12 times.
        double[] rms = ResidualRms(codec, w, backend, stages.Prior, codes, nq, stages.Frames);
        _out.WriteLine("   residual rms: " + string.Join(" ", Array.ConvertAll(rms, r => r.ToString("F4"))));
        for (int i = 1; i < rms.Length; i++)
            if (rms[i] >= rms[i - 1]) failures.Add($"residual rms did not decrease at stage {i}: {rms[i - 1]:F4} -> {rms[i]:F4}");

        codes.Dispose();
        stages.Dispose();
        pcm.Dispose();
        Assert.True(failures.Count == 0, string.Join("\n", failures));
    }

    /// <summary>Reproduces the semantic branch's HuBERT half (160-sample zero pad + mean over 13 hidden states).</summary>
    private static Tensor HubertFeats(XCodec codec, IReadOnlyDictionary<string, Tensor> w, CpuBackend backend, Tensor pcm, int s)
    {
        int pad = codec.Config.SemanticPad;
        Tensor padded = new(new TensorShape(1, 1, s + 2 * pad), DType.F32);
        float* dst = (float*)padded.DataPointer;
        for (long i = 0; i < padded.ElementCount; i++) dst[i] = 0f;
        long bytes = (long)s * sizeof(float);
        Buffer.MemoryCopy((void*)pcm.DataPointer, (void*)(dst + pad), bytes, bytes);
        using Models.Hubert.Hubert hubert = new(codec.Config.ToHubertConfig());
        hubert.LoadWeights(w, "semantic_model.");
        Tensor feats = hubert.ForwardMeanHiddenStates(backend, padded, s + 2 * pad);
        padded.Dispose();
        return feats;
    }

    /// <summary>Per-stage residual RMS of the RVQ loop, recomputed from the reference-matched codes.</summary>
    private static double[] ResidualRms(XCodec codec, IReadOnlyDictionary<string, Tensor> w, CpuBackend backend,
        Tensor prior, Tensor codes, int nq, int t)
    {
        int dim = codec.LatentDim;
        XCodecEmaResidualVectorQuantizer rvq = new(codec.NCodebooks, codec.Config.CodebookSize, dim);
        rvq.LoadWeights(w, "quantizer");

        double[] rms = new double[nq + 1];
        float[] residual = new float[dim * t];
        float* pp = (float*)prior.DataPointer;
        for (int i = 0; i < residual.Length; i++) residual[i] = pp[i];
        rms[0] = Rms(residual);
        // Decode reads codebook i from row i of its input, so each stage must pass the FULL prefix [0..q] and
        // subtract the cumulative sum — feeding a single row would re-apply codebook 0 every time.
        int* cp = (int*)codes.DataPointer;
        for (int q = 0; q < nq; q++)
        {
            Tensor prefix = new(new TensorShape(q + 1, 1, t), DType.I32);
            int* sp = (int*)prefix.DataPointer;
            for (long i = 0; i < prefix.ElementCount; i++) sp[i] = cp[i];
            Tensor cumulative = rvq.Decode(backend, prefix, 1, t);
            float* lp = (float*)cumulative.DataPointer;
            for (int i = 0; i < residual.Length; i++) residual[i] = pp[i] - lp[i];
            rms[q + 1] = Rms(residual);
            cumulative.Dispose();
            prefix.Dispose();
        }
        return rms;
    }

    private static double Rms(float[] v)
    {
        double s = 0;
        for (int i = 0; i < v.Length; i++) s += (double)v[i] * v[i];
        return Math.Sqrt(s / v.Length);
    }

    /// <summary>Scale-relative parity gate: the reference peaks span 5.3 (<c>hubert_feats</c>) to 32.5
    /// (<c>semantic_enc_out</c>), so a flat absolute tolerance means something different at each stage. On
    /// large-peak stages this ALLOWS a larger absolute maxAbs than a flat 1e-4 would; the whole-tensor relL2 is
    /// gated at the same fraction to compensate, and the hard gate downstream is exact code equality.</summary>
    private const double RelTol = 1e-5;

    /// <summary>Diffs a channels-first <c>[1, C, T]</c> tensor against a reference that is channels-first, or
    /// channels-last <c>[1, T, C]</c> when <paramref name="refChannelsLast"/>; prints maxAbs + relative L2.</summary>
    private void Report(List<string> failures, string name, Tensor mine, Tensor reference, bool refChannelsLast)
    {
        int c = (int)mine.Shape[1], t = (int)mine.Shape[2];
        int refC = refChannelsLast ? (int)reference.Shape[2] : (int)reference.Shape[1];
        int refT = refChannelsLast ? (int)reference.Shape[1] : (int)reference.Shape[2];
        Assert.Equal(refC, c);
        Assert.Equal(refT, t);

        float* a = (float*)mine.DataPointer;
        float* b = (float*)reference.DataPointer;
        double maxAbs = 0, sumSqDiff = 0, sumSqRef = 0, peak = 0;
        for (int ci = 0; ci < c; ci++)
            for (int ti = 0; ti < t; ti++)
            {
                double x = a[(long)ci * t + ti];
                double y = refChannelsLast ? b[(long)ti * c + ci] : b[(long)ci * t + ti];
                double diff = Math.Abs(x - y);
                if (diff > maxAbs) maxAbs = diff;
                sumSqDiff += diff * diff;
                sumSqRef += y * y;
                peak = Math.Max(peak, Math.Abs(y));
            }
        double relL2 = Math.Sqrt(sumSqDiff) / (Math.Sqrt(sumSqRef) + 1e-12);
        double relMax = maxAbs / (peak + 1e-12);
        _out.WriteLine($"   {name,-24} [{c},{t}]  maxAbs={maxAbs:E4}  maxAbs/peak={relMax:E4}  relL2={relL2:E4}  (ref peak {peak:F4})");
        if (relMax >= RelTol || relL2 >= RelTol)
            failures.Add($"{name} diverges (maxAbs/peak={relMax:E4}, relL2={relL2:E4}, tol={RelTol:E1}).");
    }
}
