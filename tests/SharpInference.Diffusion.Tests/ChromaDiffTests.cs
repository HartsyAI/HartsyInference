using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Cuda;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.ModelHandler.CheckpointConverters;
using SharpInference.ModelHandler.SafeTensors;
using SharpInference.Tests.Common;

namespace SharpInference.Diffusion.Tests;

/// <summary>Layer-by-layer diff harness for Chroma. Loads the Python-saved synthetic inputs (packed latent +
/// raw T5 context + attention mask + timestep), runs the C# `ChromaTransformer` on CPU/GPU with
/// `CHROMA_DEBUG_DIR` set, then leaves the dump for `diff_chroma_layers.py`.
///
/// Skips cleanly when the Python reference dump or the Chroma checkpoint is missing.</summary>
public unsafe class ChromaDiffTests
{
    private readonly ITestOutputHelper _output;
    public ChromaDiffTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Transformer_Matches_PythonReference_LayerByLayer_Cpu() =>
        RunDiff(useGpu: false, dumpDirName: "chroma_csharp_dump");

    [Fact]
    public void Transformer_Matches_PythonReference_LayerByLayer_Gpu() =>
        RunDiff(useGpu: true, dumpDirName: "chroma_csharp_gpu_dump");

    private void RunDiff(bool useGpu, string dumpDirName)
    {
        string repoRoot = RepoRoot.Path;
        string refDir = Path.Combine(repoRoot, "tests/python-reference/chroma_reference_tensors/full_forward");
        string inputsDir = Path.Combine(refDir, "inputs");

        if (!Directory.Exists(inputsDir))
        {
            _output.WriteLine($"SKIPPED: Python reference dump not found at {inputsDir}");
            _output.WriteLine("  Run: tests/python-reference/.venv/bin/python tests/python-reference/dump_chroma_full_forward.py");
            return;
        }

        string ckpt = TestPaths.Chroma.V1;
        if (!File.Exists(ckpt))
        {
            _output.WriteLine($"SKIPPED: Chroma checkpoint not found: {ckpt}");
            return;
        }

        string assemblyDir = Path.GetDirectoryName(typeof(ChromaDiffTests).Assembly.Location)!;
        string ptxDir = Path.Combine(assemblyDir, "Ptx");
        if (useGpu && !Directory.Exists(ptxDir))
        {
            _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}");
            return;
        }

        string csDumpDir = Path.Combine(repoRoot, "Output", dumpDirName);
        if (Directory.Exists(csDumpDir)) Directory.Delete(csDumpDir, recursive: true);
        Directory.CreateDirectory(Path.Combine(csDumpDir, "layers"));
        Environment.SetEnvironmentVariable("CHROMA_DEBUG_DIR", csDumpDir);

        // Synthetic inputs (must match dump_chroma_full_forward.py): latent [1, 16, 32, 32] (memory-saving
        // 256-token equivalent), context [1, 64, 4096], attention_mask [1, 64] (long), timestep [1] (sigma=0.5).
        // The Python script packs the latent via 2×2 patchify; we reconstruct it here by loading and re-packing.
        Tensor latentBchw = LoadF32Tensor(Path.Combine(inputsDir, "latent.bin"), 1, 16, 32, 32);
        Tensor contextPre = LoadF32Tensor(Path.Combine(inputsDir, "context_pre.bin"), 1, 64, 4096);
        Tensor attentionMask = LoadInt64AsF32Mask(Path.Combine(inputsDir, "attention_mask.bin"), 1, 64);
        float timestep = LoadF32Scalar(Path.Combine(inputsDir, "timestep.bin"));
        Tensor packedLatent = PackLatent2x2(latentBchw);
        latentBchw.Dispose();
        _output.WriteLine($"Loaded inputs: packedLatent={packedLatent.Shape}, context_pre={contextPre.Shape}, mask={attentionMask.Shape}, t={timestep}");

        _output.WriteLine($"Loading checkpoint: {Path.GetFileName(ckpt)}");
        Stopwatch sw = Stopwatch.StartNew();
        (ChromaCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            ChromaCheckpointConverter.LoadAndConvert(ckpt);
        _output.WriteLine($"  Converted in {sw.ElapsedMilliseconds}ms ({converted.Transformer.Count} keys)");

        using (loader)
        {
            Dictionary<string, Tensor> trans = useGpu ? converted.Transformer : CastWeightsToF32(converted.Transformer);
            ChromaConfig config = ChromaConfig.V1;

            using ChromaTransformer transformer = new(config);
            transformer.LoadWeights(trans);
            _output.WriteLine("  Weights loaded.");

            using IBackend backend = useGpu
                ? new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir)
                : (IBackend)new CpuBackend();
            if (useGpu)
            {
                ((CudaBackend)backend).PreloadWeights(transformer.EnumerateWeights());
                _output.WriteLine($"  GPU backend ready: {backend.Capabilities.Name}");
            }

            sw.Restart();
            _output.WriteLine($"\nRunning ChromaTransformer.Forward ({(useGpu ? "GPU" : "CPU")}) ...");
            // Chroma forward: (backend, packedLatent, encoderHidden, timestep, txtSeqLen, hPacked, wPacked, attentionMask)
            // txtSeqLen=64 (Python SEQ_T), hPacked=wPacked=16 (= 32/2 after 2×2 patchify of [1,16,32,32]).
            Tensor velocity = transformer.Forward(backend, packedLatent, contextPre, timestep, 64, 16, 16, attentionMask);
            sw.Stop();
            _output.WriteLine($"Forward done in {sw.ElapsedMilliseconds}ms.");
            _output.WriteLine($"Output velocity: shape={velocity.Shape}");

            float* vptr = (float*)velocity.DataPointer;
            int n = (int)velocity.ElementCount;
            double sum = 0, sumAbs = 0, sumSq = 0;
            for (int i = 0; i < n; i++) { sum += vptr[i]; sumAbs += Math.Abs(vptr[i]); sumSq += vptr[i] * vptr[i]; }
            double mean = sum / n, absMean = sumAbs / n, std = Math.Sqrt(sumSq / n - mean * mean);
            _output.WriteLine($"  Stats: mean={mean:F6}, abs_mean={absMean:F6}, std={std:F6}");

            velocity.Dispose();
        }

        packedLatent.Dispose();
        contextPre.Dispose();
        attentionMask.Dispose();
        Environment.SetEnvironmentVariable("CHROMA_DEBUG_DIR", null);

        _output.WriteLine($"\nC# dump written to: {csDumpDir}");
        _output.WriteLine("Now run: tests/python-reference/.venv/bin/python tests/python-reference/diff_chroma_layers.py");
    }

    private static unsafe Tensor LoadF32Tensor(string path, params long[] dims)
    {
        TensorShape shape = dims.Length switch
        {
            2 => new TensorShape(dims[0], dims[1]),
            3 => new TensorShape(dims[0], dims[1], dims[2]),
            4 => new TensorShape(dims[0], dims[1], dims[2], dims[3]),
            _ => throw new ArgumentException($"Unsupported rank: {dims.Length}"),
        };
        Tensor t = new(shape, DType.F32);
        byte[] raw = File.ReadAllBytes(path);
        long expected = shape.ElementCount * sizeof(float);
        if (raw.LongLength != expected)
            throw new InvalidDataException($"{path}: expected {expected} bytes, got {raw.LongLength}");
        fixed (byte* src = raw) Buffer.MemoryCopy(src, (void*)t.DataPointer, raw.LongLength, raw.LongLength);
        return t;
    }

    /// <summary>The Python ref saves the attention mask as int64. We reinterpret as int64 and convert to F32 (1.0/0.0).</summary>
    private static unsafe Tensor LoadInt64AsF32Mask(string path, long b, long s)
    {
        byte[] raw = File.ReadAllBytes(path);
        long n = b * s;
        if (raw.LongLength != n * sizeof(long))
            throw new InvalidDataException($"{path}: expected {n * sizeof(long)} bytes for int64 mask, got {raw.LongLength}");
        Tensor t = new(new TensorShape(b, s), DType.F32);
        float* dst = (float*)t.DataPointer;
        fixed (byte* src = raw)
        {
            long* asLong = (long*)src;
            for (long i = 0; i < n; i++) dst[i] = asLong[i] != 0 ? 1.0f : 0.0f;
        }
        return t;
    }

    private static unsafe float LoadF32Scalar(string path)
    {
        byte[] raw = File.ReadAllBytes(path);
        fixed (byte* src = raw) return *(float*)src;
    }

    /// <summary>Packs a [B, C, H, W] latent into [B, H/2 * W/2, C*4] via 2×2 patchify (matches Flux/Chroma `_pack_latents`).</summary>
    private static unsafe Tensor PackLatent2x2(Tensor latent)
    {
        int b = (int)latent.Shape[0];
        int c = (int)latent.Shape[1];
        int h = (int)latent.Shape[2];
        int w = (int)latent.Shape[3];
        int hP = h / 2, wP = w / 2;
        int seq = hP * wP;
        int patchDim = c * 4;
        Tensor packed = new(new TensorShape(b, seq, patchDim), DType.F32);
        float* inPtr = (float*)latent.DataPointer;
        float* outPtr = (float*)packed.DataPointer;
        for (int bb = 0; bb < b; bb++)
        for (int ph = 0; ph < hP; ph++)
        for (int pw = 0; pw < wP; pw++)
        {
            int seqIdx = ph * wP + pw;
            int outBase = (bb * seq + seqIdx) * patchDim;
            for (int cc = 0; cc < c; cc++)
            {
                int inChanBase = (bb * c + cc) * h * w;
                int patchBase = outBase + cc * 4;
                outPtr[patchBase + 0] = inPtr[inChanBase + (ph * 2 + 0) * w + (pw * 2 + 0)];
                outPtr[patchBase + 1] = inPtr[inChanBase + (ph * 2 + 0) * w + (pw * 2 + 1)];
                outPtr[patchBase + 2] = inPtr[inChanBase + (ph * 2 + 1) * w + (pw * 2 + 0)];
                outPtr[patchBase + 3] = inPtr[inChanBase + (ph * 2 + 1) * w + (pw * 2 + 1)];
            }
        }
        return packed;
    }

    private static Dictionary<string, Tensor> CastWeightsToF32(IReadOnlyDictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> f32 = new(weights.Count);
        foreach (KeyValuePair<string, Tensor> kvp in weights)
        {
            DType dt = kvp.Value.DType;
            // Chroma1-HD is fp8-mixed: most layers are F8E4M3, a few are BF16 (norms, embed). All must
            // be cast to F32 for the CPU forward — otherwise the F32 backend reads 1-byte FP8 tensors
            // as F32 and walks off the end of the buffer (AccessViolationException).
            bool needsCast = dt == DType.F16 || dt == DType.BF16 || dt == DType.F8E4M3 || dt == DType.F8E5M2;
            f32[kvp.Key] = needsCast ? kvp.Value.CastTo(DType.F32) : kvp.Value;
        }
        return f32;
    }
}
