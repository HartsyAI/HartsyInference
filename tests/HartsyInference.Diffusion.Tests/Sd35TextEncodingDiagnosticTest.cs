using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Diagnostic: encodes the same prompt through the C# CLIP-L + CLIP-G + zero-T5 path that <see cref="HartsyInference.Diffusion.Pipelines.Sd3Pipeline.EncodePrompt"/> uses, then dumps the intermediate tensors so they can be diffed against a Python reference (dump_sd35_pipeline_inputs.py).</summary>
public unsafe class Sd35TextEncodingDiagnosticTest
{
    private readonly ITestOutputHelper _output;
    public Sd35TextEncodingDiagnosticTest(ITestOutputHelper output) => _output = output;

    [Fact]
    public void EncodePrompt_DumpsIntermediates()
    {
        string ckpt = TestPaths.Sd35.Medium;
        if (!File.Exists(ckpt))
        {
            _output.WriteLine($"SKIPPED: Checkpoint not found: {ckpt}");
            return;
        }
        if (!File.Exists(TestPaths.Tokenizers.ClipVocab) || !File.Exists(TestPaths.Tokenizers.ClipMerges))
        {
            _output.WriteLine("SKIPPED: CLIP tokenizer files not found");
            return;
        }

        string repoRoot = RepoRoot.Path;
        string outDir = Path.Combine(repoRoot, "Output", "sd35_text_encoding_dump");
        Directory.CreateDirectory(outDir);

        // ── Load + convert checkpoint ──
        _output.WriteLine($"Loading checkpoint: {Path.GetFileName(ckpt)}");
        Stopwatch sw = Stopwatch.StartNew();
        (Sd3CheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            Sd3CheckpointConverter.LoadAndConvert(ckpt);
        _output.WriteLine($"  Converted in {sw.ElapsedMilliseconds}ms");

        using (loader)
        {
            // ── Load CLIP-L + CLIP-G ──
            ClipTextEncoder clipL = new(ClipTextEncoderConfig.Sd3ClipL);
            clipL.LoadWeights(converted.ClipL, "text_model");
            ClipTextEncoder clipG = new(ClipTextEncoderConfig.SdxlClipG);
            clipG.LoadWeights(converted.ClipG, "text_model");
            _output.WriteLine("  CLIPs loaded.");

            // ── Tokenize the same prompt as the Python reference ──
            string prompt = "A photograph of an astronaut riding a horse";
            using ClipTokenizer tok = new(TestPaths.Tokenizers.ClipVocab, TestPaths.Tokenizers.ClipMerges);
            int[] tokenIds = tok.Encode(prompt);
            int eosPos = ClipTokenizer.FindEosPosition(tokenIds);
            _output.WriteLine($"  Tokens: count={tokenIds.Length}, eos_pos={eosPos}");
            _output.WriteLine($"  First 12: [{string.Join(", ", tokenIds[..Math.Min(12, tokenIds.Length)])}]");

            using IBackend backend = new CpuBackend();

            int[][] batch = [tokenIds];
            int[] eosBatch = [eosPos];

            // ── CLIP-L encode (penultimate hidden + pooled via text_projection) ──
            _output.WriteLine("\n[CLIP-L] EncodePenultimate ...");
            (Tensor clipLHidden, Tensor? clipLPooled) = clipL.EncodePenultimate(backend, batch, eosBatch);
            DumpStats("clip_l_hidden", clipLHidden);
            DumpStats("clip_l_pooled", clipLPooled!);
            DumpRawF32(Path.Combine(outDir, "clip_l_hidden_penultimate.bin"), clipLHidden);
            DumpRawF32(Path.Combine(outDir, "clip_l_pooled.bin"), clipLPooled!);

            // ── CLIP-G encode ──
            _output.WriteLine("\n[CLIP-G] EncodePenultimate ...");
            (Tensor clipGHidden, Tensor? clipGPooled) = clipG.EncodePenultimate(backend, batch, eosBatch);
            DumpStats("clip_g_hidden", clipGHidden);
            DumpStats("clip_g_pooled", clipGPooled!);
            DumpRawF32(Path.Combine(outDir, "clip_g_hidden_penultimate.bin"), clipGHidden);
            DumpRawF32(Path.Combine(outDir, "clip_g_pooled.bin"), clipGPooled!);

            // ── Replicate Sd3Pipeline.EncodePrompt: concat hidden, pad to 4096, concat with zero T5, concat pooled ──
            int seqLen = tokenIds.Length;
            int targetDim = 4096;
            int lgDim = 768 + 1280;

            // pooled = concat(clip_l_pooled, clip_g_pooled, dim=-1) → [1, 2048]
            Tensor pooled = new(new TensorShape(1, 2048), DType.F32);
            float* lpPtr = (float*)clipLPooled!.DataPointer;
            float* gpPtr = (float*)clipGPooled!.DataPointer;
            float* poPtr = (float*)pooled.DataPointer;
            for (int i = 0; i < 768; i++) poPtr[i] = lpPtr[i];
            for (int i = 0; i < 1280; i++) poPtr[768 + i] = gpPtr[i];

            // lgHidden = concat(clip_l_hidden, clip_g_hidden, dim=-1) → [1, 77, 2048]
            Tensor lgHidden = new(new TensorShape(1, seqLen, lgDim), DType.F32);
            float* lhPtr = (float*)clipLHidden.DataPointer;
            float* ghPtr = (float*)clipGHidden.DataPointer;
            float* lgPtr = (float*)lgHidden.DataPointer;
            for (int s = 0; s < seqLen; s++)
            {
                for (int d = 0; d < 768; d++) lgPtr[s * lgDim + d] = lhPtr[s * 768 + d];
                for (int d = 0; d < 1280; d++) lgPtr[s * lgDim + 768 + d] = ghPtr[s * 1280 + d];
            }

            // pad to 4096
            Tensor lgPadded = new(new TensorShape(1, seqLen, targetDim), DType.F32);
            float* lpdPtr = (float*)lgPadded.DataPointer;
            for (long i = 0; i < lgPadded.ElementCount; i++) lpdPtr[i] = 0f;
            for (int s = 0; s < seqLen; s++)
                for (int d = 0; d < lgDim; d++)
                    lpdPtr[s * targetDim + d] = lgPtr[s * lgDim + d];

            // concat with zero T5 → [1, 154, 4096]
            int totalSeq = seqLen + seqLen;
            Tensor context = new(new TensorShape(1, totalSeq, targetDim), DType.F32);
            float* ctxPtr = (float*)context.DataPointer;
            for (int s = 0; s < seqLen; s++)
                for (int d = 0; d < targetDim; d++)
                    ctxPtr[s * targetDim + d] = lpdPtr[s * targetDim + d];
            for (int s = seqLen; s < totalSeq; s++)
                for (int d = 0; d < targetDim; d++)
                    ctxPtr[s * targetDim + d] = 0f;

            DumpStats("final_context_pre", context);
            DumpStats("final_pooled", pooled);
            DumpRawF32(Path.Combine(outDir, "final_context_pre.bin"), context);
            DumpRawF32(Path.Combine(outDir, "final_pooled.bin"), pooled);

            clipLHidden.Dispose();
            clipLPooled?.Dispose();
            clipGHidden.Dispose();
            clipGPooled?.Dispose();
            pooled.Dispose();
            lgHidden.Dispose();
            lgPadded.Dispose();
            context.Dispose();
        }

        _output.WriteLine($"\nDumps written to: {outDir}");
        _output.WriteLine("Run: tests/python-reference/.venv/bin/python tests/python-reference/diff_sd35_text_encoding.py");
    }

    private void DumpStats(string name, Tensor t)
    {
        float* p = (float*)t.DataPointer;
        long n = t.ElementCount;
        double sum = 0, sumAbs = 0, sumSq = 0;
        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        for (long i = 0; i < n; i++)
        {
            float v = p[i];
            sum += v; sumAbs += Math.Abs(v); sumSq += v * v;
            if (v < min) min = v;
            if (v > max) max = v;
        }
        double mean = sum / n, absMean = sumAbs / n, std = Math.Sqrt(sumSq / n - mean * mean);
        _output.WriteLine($"  {name,-22} shape={t.Shape}  mean={mean:E4}  std={std:E4}  abs_mean={absMean:E4}  min={min:F4}  max={max:F4}");
        _output.WriteLine($"    first_8: [{string.Join(", ", Enumerable.Range(0, Math.Min(8, (int)n)).Select(i => p[i].ToString("F6")))}]");
    }

    private static void DumpRawF32(string path, Tensor t)
    {
        long count = t.ElementCount;
        if (t.DType != DType.F32)
            throw new InvalidOperationException($"Expected F32, got {t.DType}");
        byte[] buf = new byte[count * sizeof(float)];
        fixed (byte* dst = buf)
        {
            Buffer.MemoryCopy((float*)t.DataPointer, dst, buf.Length, buf.Length);
        }
        File.WriteAllBytes(path, buf);
    }
}
