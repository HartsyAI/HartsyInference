using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The Wan S2V audio injector attended one latent frame at a time: three slices, one attention call and a place
/// in a concat for every frame. It now makes one call with a batch item a frame. The former forward is kept here, op for
/// op, as the oracle. The class had no test before this.
/// <para>The tolerance was fixed before the first run: 1e-5 relative to the larger of 1 and the value, on the CPU
/// backend. The two forms reach the CPU attention by different entries (one batch item takes the token-major entry, a
/// batch the head-major one), so the last bits may differ and bit-equality is not asked.</para>
/// <para>On a GPU every case requires the batched forward to equal the former forward on the same backend bit for bit.
/// Against the former forward on the CPU the bound is 1e-4 on Vulkan and 2e-3 on CUDA, whose F32 dense row multiplies
/// in TF32; it is asserted below dim 512 and reported at it, because at dim 512 with these random weights CUDA's
/// forward is 8.7e-3 from the CPU whether batched or not (its five linears and the attention all multiply in TF32).</para></summary>
[Collection("AttentionEvidence")]
public sealed class WanS2VAudioInjectorTests(ITestOutputHelper output)
{
    private const float Tolerance = 1e-5f, GpuTolerance = 1e-4f, Tf32Tolerance = 2e-3f, AdaLnEps = 1e-5f, QkEps = 1e-6f;

    private static Tensor Random(int seed, float scale, params long[] shape)
    {
        Tensor tensor = new(new TensorShape(shape), DType.F32);
        Random random = new(seed);
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = (random.NextSingle() * 2f - 1f) * scale;
        return tensor;
    }

    private static Dictionary<string, Tensor> Weights(int count, int dim)
    {
        Dictionary<string, Tensor> w = [];
        int seed = 900;
        for (int i = 0; i < count; i++)
        {
            string p = $"audio_injector.injector.{i}";
            foreach (string name in (string[])["q", "k", "v", "o"])
            {
                w[$"{p}.{name}.weight"] = Random(seed++, 0.3f, dim, dim);
                w[$"{p}.{name}.bias"] = Random(seed++, 0.1f, dim);
            }
            w[$"{p}.norm_q.weight"] = Random(seed++, 1f, dim);
            w[$"{p}.norm_k.weight"] = Random(seed++, 1f, dim);
            w[$"audio_injector.injector_adain_layers.{i}.linear.weight"] = Random(seed++, 0.3f, 2 * dim, dim);
            w[$"audio_injector.injector_adain_layers.{i}.linear.bias"] = Random(seed++, 0.1f, 2 * dim);
        }
        return w;
    }

    /// <summary>Frees the weights and what the injector made of them at load (the AdaLN halves and any F32 norm copies),
    /// which belong to nobody else in a test.</summary>
    private static void Release(WanS2VAudioInjector? injector, Dictionary<string, Tensor> weights)
    {
        HashSet<Tensor> given = new(weights.Values, ReferenceEqualityComparer.Instance);
        if (injector is not null)
            foreach (Tensor t in injector.EnumerateWeights().Distinct<Tensor>(ReferenceEqualityComparer.Instance))
                if (!given.Contains(t)) t.Dispose();
        foreach (Tensor t in weights.Values) t.Dispose();
    }

    /// <summary>The first or second half of the rows of a <c>[2 · rows, …]</c> weight or <c>[2 · rows]</c> bias.</summary>
    private static Tensor Half(Tensor x, int rows, bool second)
    {
        int perRow = (int)(x.ElementCount / (2L * rows));
        Tensor half = new(x.Shape.Rank == 1 ? new TensorShape(rows) : new TensorShape(rows, x.Shape[1]), DType.F32);
        x.AsReadOnlySpan<float>().Slice(second ? rows * perRow : 0, rows * perRow).CopyTo(half.AsSpan<float>());
        return half;
    }

    /// <summary>The former forward: per-frame slices, one attention call a frame, a concat.</summary>
    private static Tensor FormerForward(IBackend backend, AttentionLayer attention, Dictionary<string, Tensor> w, int index, int dim,
        Tensor hidden, Tensor audioLocal, Tensor audioGlobal, int seqLen)
    {
        string p = $"audio_injector.injector.{index}", ada = $"audio_injector.injector_adain_layers.{index}.linear";
        int t = (int)audioLocal.Shape[0], nTok = (int)audioLocal.Shape[1], n = seqLen / t, totalS = (int)(hidden.ElementCount / dim);
        using Tensor shiftW = Half(w[$"{ada}.weight"], dim, false), scaleW = Half(w[$"{ada}.weight"], dim, true);
        using Tensor shiftB = Half(w[$"{ada}.bias"], dim, false), scaleB = Half(w[$"{ada}.bias"], dim, true);

        using Tensor gAct = new(new TensorShape(t, dim), DType.F32);
        backend.Silu(gAct, audioGlobal);
        using Tensor shift = new(new TensorShape(t, dim), DType.F32);
        backend.Linear(shift, gAct, shiftW, shiftB);
        using Tensor scale1 = new(new TensorShape(t, dim), DType.F32);
        backend.Linear(scale1, gAct, scaleW, scaleB);
        using Tensor scale1p = new(scale1.Shape, DType.F32);
        backend.AddScalar(scale1p, scale1, 1f);

        using Tensor top = new(new TensorShape(t, n, dim), DType.F32);
        backend.SliceRows(top, hidden, 0);
        using Tensor normed = new(top.Shape, DType.F32);
        backend.LayerNormNoAffine(normed, top, AdaLnEps);
        using Tensor adain = new(top.Shape, DType.F32);
        backend.AffineBroadcastLastDim(adain, normed, scale1p, shift);

        using Tensor q = new(new TensorShape(seqLen, dim), DType.F32);
        backend.Linear(q, adain, w[$"{p}.q.weight"], w[$"{p}.q.bias"]);
        using Tensor qn = new(q.Shape, DType.F32);
        backend.RmsNorm(qn, q, w[$"{p}.norm_q.weight"], QkEps);
        int kvRows = t * nTok;
        using Tensor k = new(new TensorShape(kvRows, dim), DType.F32);
        backend.Linear(k, audioLocal, w[$"{p}.k.weight"], w[$"{p}.k.bias"]);
        using Tensor kn = new(k.Shape, DType.F32);
        backend.RmsNorm(kn, k, w[$"{p}.norm_k.weight"], QkEps);
        using Tensor v = new(new TensorShape(kvRows, dim), DType.F32);
        backend.Linear(v, audioLocal, w[$"{p}.v.weight"], w[$"{p}.v.bias"]);

        Tensor[] frameOuts = new Tensor[t];
        for (int ti = 0; ti < t; ti++)
        {
            Tensor qF = new(new TensorShape(n, dim), DType.F32); backend.SliceRows(qF, qn, ti * n);
            Tensor kF = new(new TensorShape(nTok, dim), DType.F32); backend.SliceRows(kF, kn, ti * nTok);
            Tensor vF = new(new TensorShape(nTok, dim), DType.F32); backend.SliceRows(vF, v, ti * nTok);
            frameOuts[ti] = attention.AttendFlat(backend, qF, kF, vF, new TensorShape(n, dim), default);
        }
        using Tensor attnFlat = new(new TensorShape(seqLen, dim), DType.F32);
        backend.Concat(attnFlat, frameOuts, 0);
        foreach (Tensor f in frameOuts) f.Dispose();

        using Tensor residual = new(new TensorShape(seqLen, dim), DType.F32);
        backend.Linear(residual, attnFlat, w[$"{p}.o.weight"], w[$"{p}.o.bias"]);
        Tensor topAdded = new(new TensorShape(seqLen, dim), DType.F32);
        backend.Add(topAdded, top, residual);
        if (totalS == seqLen) return topAdded;
        using Tensor rest = new(new TensorShape(totalS - seqLen, dim), DType.F32);
        backend.SliceRows(rest, hidden, seqLen);
        Tensor outFull = new(new TensorShape(totalS, dim), DType.F32);
        backend.Concat(outFull, [topAdded, rest], 0);
        topAdded.Dispose();
        return outFull;
    }

    private static void ForwardMatches(int frames, int tokensAFrame, int audioTokens, int trailing, int dim, int heads, int index,
        IBackend? device = null, float tolerance = Tolerance)
    {
        const int count = 2;
        int seqLen = frames * tokensAFrame;
        Dictionary<string, Tensor> weights = Weights(count, dim);
        WanS2VAudioInjector? injector = null;
        try
        {
            using CpuBackend backend = new();
            injector = new(count, dim, heads, dim / heads);
            injector.LoadWeights(weights);
            using Tensor hidden = Random(11, 1f, seqLen + trailing, dim);
            using Tensor audioLocal = Random(12, 1f, frames, audioTokens, dim), audioGlobal = Random(13, 1f, frames, 1, dim);
            using Tensor expected = FormerForward(backend, injector.AttentionStage, weights, index, dim, hidden, audioLocal, audioGlobal, seqLen);
            using Tensor actual = injector.Forward(device ?? backend, hidden, index, audioLocal, audioGlobal, seqLen);
            device?.Sync();

            Assert.Equal(expected.Shape, actual.Shape);
            ReadOnlySpan<float> e = expected.AsReadOnlySpan<float>(), a = actual.AsReadOnlySpan<float>(), h = hidden.AsReadOnlySpan<float>();
            double worst = 0, moved = 0;
            int at = -1;
            for (int i = 0; i < e.Length; i++)
            {
                double error = Math.Abs(e[i] - a[i]) / Math.Max(1f, Math.Abs(e[i]));
                if (!(error <= worst)) (worst, at) = (error, i);
                if (i < seqLen * dim) moved = Math.Max(moved, Math.Abs(e[i] - h[i]));
            }
            string what = $"{frames} frames of {tokensAFrame} tokens, {audioTokens} audio tokens, {trailing} trailing rows, dim {dim}, {heads} heads";
            Assert.True(worst <= tolerance, $"{what}: element {at} is {(at < 0 ? 0f : a[at])}, expected {(at < 0 ? 0f : e[at])} (relative error {worst:G3})");
            // The injection is not a no-op in this test, and the rows past seqLen pass through untouched.
            Assert.True(moved > 0.01, $"{what}: the injection moves the hidden state by only {moved}");
            for (int i = seqLen * dim; i < e.Length; i++) Assert.Equal(h[i], a[i]);
        }
        finally
        {
            Release(injector, weights);
        }
    }

    private static readonly (int Frames, int Tokens, int Audio, int Trailing, int Dim, int Heads, int Index)[] Cases =
        [(1, 12, 5, 0, 32, 4, 0), (4, 12, 5, 0, 32, 4, 0), (7, 9, 5, 18, 32, 4, 1), (5, 30, 3, 7, 64, 8, 1), (3, 16, 1, 0, 32, 2, 0),
            (20, 400, 5, 64, 512, 8, 0)];

    /// <summary>Every case on a GPU backend against the former forward on the CPU, the kernel rows that ran, and the wall
    /// time of the former per-frame forward and of the batched one on that backend for the largest case.</summary>
    private void OnAGpu(IBackend device, float tolerance)
    {
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(4096);
        try
        {
            foreach ((int frames, int tokens, int audio, int trailing, int dim, int heads, int index) in Cases)
            {
                if (dim < 512) ForwardMatches(frames, tokens, audio, trailing, dim, heads, index, device, tolerance);
                SameBackendMatches(frames, tokens, audio, trailing, dim, heads, index, device);
            }
        }
        finally
        {
            foreach (IGrouping<string, AttentionEvidenceEvent> row in evidence.Snapshot().Where(e => e.Kind == AttentionEvidenceKind.Execution)
                         .GroupBy(e => $"{e.Implementation} batch {e.Batch}, heads {e.Heads}, seq {e.QueryTokens}"))
                output.WriteLine($"{row.Key}: {row.Count()} call(s)");
        }

        const int f = 20, n = 400, a = 5, d = 512, h = 8, seqLen = f * n;
        Dictionary<string, Tensor> weights = Weights(1, d);
        WanS2VAudioInjector? injector = null;
        try
        {
            injector = new(1, d, h, d / h);
            injector.LoadWeights(weights);
            using Tensor hidden = Random(31, 1f, seqLen, d), audioLocal = Random(32, 1f, f, a, d), audioGlobal = Random(33, 1f, f, 1, d);
            // Where a difference from the CPU comes from: the same former forward on this backend against the CPU's, and
            // the batched forward against the former one on this backend.
            using (CpuBackend cpu = new())
            using (Tensor onCpu = FormerForward(cpu, injector.AttentionStage, weights, 0, d, hidden, audioLocal, audioGlobal, seqLen))
            using (Tensor formerHere = FormerForward(device, injector.AttentionStage, weights, 0, d, hidden, audioLocal, audioGlobal, seqLen))
            using (Tensor batchedHere = injector.Forward(device, hidden, 0, audioLocal, audioGlobal, seqLen))
            {
                device.Sync();
                output.WriteLine($"{f} frames of {n} tokens, dim {d}: former forward here against the CPU {Worst(onCpu, formerHere):G3}, "
                    + $"batched against former, both here {Worst(formerHere, batchedHere):G3} (largest relative difference)");
            }
            double former = double.MaxValue, batched = double.MaxValue;
            for (int run = 0; run < 6; run++)
            {
                long start = Stopwatch.GetTimestamp();
                using (FormerForward(device, injector.AttentionStage, weights, 0, d, hidden, audioLocal, audioGlobal, seqLen)) device.Sync();
                if (run > 0) former = Math.Min(former, Stopwatch.GetElapsedTime(start).TotalMilliseconds);
                start = Stopwatch.GetTimestamp();
                using (injector.Forward(device, hidden, 0, audioLocal, audioGlobal, seqLen)) device.Sync();
                if (run > 0) batched = Math.Min(batched, Stopwatch.GetElapsedTime(start).TotalMilliseconds);
            }
            output.WriteLine($"{f} frames of {n} tokens, dim {d}: former per-frame forward {former:F1} ms, batched {batched:F1} ms (best of 5 after a warm-up)");
        }
        finally
        {
            Release(injector, weights);
        }
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void OnCudaTheBatchedForwardMatchesTheFormerForwardOnTheCpu()
    {
        if (!CudaContext.IsAvailable()) { output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = new(0, PtxDirectory());
        OnAGpu(backend, Tf32Tolerance);
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void OnVulkanTheBatchedForwardMatchesTheFormerForwardOnTheCpu()
    {
        VulkanBackend backend;
        try { backend = new VulkanBackend(); }
        catch (Exception ex) { output.WriteLine($"SKIPPED: Vulkan unavailable ({ex.Message})"); return; }
        using (backend) OnAGpu(backend, GpuTolerance);
    }

    /// <summary>On one backend the batched forward is the former per-frame forward bit for bit: batching changes which
    /// call a frame rides in, not its arithmetic.</summary>
    private static void SameBackendMatches(int frames, int tokensAFrame, int audioTokens, int trailing, int dim, int heads, int index,
        IBackend device)
    {
        int seqLen = frames * tokensAFrame;
        Dictionary<string, Tensor> weights = Weights(2, dim);
        WanS2VAudioInjector? injector = null;
        try
        {
            injector = new(2, dim, heads, dim / heads);
            injector.LoadWeights(weights);
            using Tensor hidden = Random(11, 1f, seqLen + trailing, dim);
            using Tensor audioLocal = Random(12, 1f, frames, audioTokens, dim), audioGlobal = Random(13, 1f, frames, 1, dim);
            using Tensor former = FormerForward(device, injector.AttentionStage, weights, index, dim, hidden, audioLocal, audioGlobal, seqLen);
            using Tensor batched = injector.Forward(device, hidden, index, audioLocal, audioGlobal, seqLen);
            device.Sync();
            ReadOnlySpan<float> e = former.AsReadOnlySpan<float>(), a = batched.AsReadOnlySpan<float>();
            Assert.Equal(e.Length, a.Length);
            int differ = 0;
            for (int i = 0; i < e.Length; i++)
                if (BitConverter.SingleToInt32Bits(e[i]) != BitConverter.SingleToInt32Bits(a[i])) differ++;
            Assert.True(differ == 0, $"{frames} frames of {tokensAFrame} tokens, dim {dim}: {differ} of {e.Length} values differ "
                + $"between the batched and the former forward on the same backend (largest relative difference {Worst(former, batched):G3})");
        }
        finally
        {
            Release(injector, weights);
        }
    }

    private static double Worst(Tensor expected, Tensor actual)
    {
        ReadOnlySpan<float> e = expected.AsReadOnlySpan<float>(), a = actual.AsReadOnlySpan<float>();
        double worst = 0;
        for (int i = 0; i < e.Length; i++) worst = Math.Max(worst, Math.Abs(e[i] - a[i]) / Math.Max(1f, Math.Abs(e[i])));
        return worst;
    }

    private static string PtxDirectory()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return candidate;
        }
        throw new DirectoryNotFoundException("src/HartsyInference.Cuda/Ptx was not found above the test directory.");
    }

    [Theory]
    [InlineData(1, 12, 5, 0, 32, 4, 0)]      // one frame: a batch of one
    [InlineData(4, 12, 5, 0, 32, 4, 0)]
    [InlineData(7, 9, 5, 18, 32, 4, 1)]      // trailing reference rows, the second injector
    [InlineData(5, 30, 3, 7, 64, 8, 1)]
    [InlineData(3, 16, 1, 0, 32, 2, 0)]      // one audio token a frame
    [InlineData(20, 400, 5, 64, 512, 8, 0)]  // the GPU tests' large case, with trailing rows
    public void TheBatchedForwardMatchesTheFormerPerFrameForward(int frames, int tokensAFrame, int audioTokens, int trailing, int dim,
        int heads, int index) => ForwardMatches(frames, tokensAFrame, audioTokens, trailing, dim, heads, index);

    /// <summary>A frame attends only to its own audio tokens: changing another frame's audio leaves its rows as they were.
    /// This is what "one batch item a frame" has to keep.</summary>
    [Fact]
    public void AFramesOutputDoesNotDependOnAnotherFramesAudio()
    {
        const int frames = 4, tokensAFrame = 10, audioTokens = 5, dim = 32, heads = 4, seqLen = frames * tokensAFrame;
        Dictionary<string, Tensor> weights = Weights(1, dim);
        WanS2VAudioInjector? injector = null;
        try
        {
            using CpuBackend backend = new();
            injector = new(1, dim, heads, dim / heads);
            injector.LoadWeights(weights);
            using Tensor hidden = Random(21, 1f, seqLen, dim), audioGlobal = Random(23, 1f, frames, 1, dim);
            using Tensor audio = Random(22, 1f, frames, audioTokens, dim), changed = new(audio.Shape, DType.F32);
            audio.AsReadOnlySpan<float>().CopyTo(changed.AsSpan<float>());
            // Frame 2's audio tokens are replaced; frames 0, 1 and 3 keep theirs.
            Span<float> c = changed.AsSpan<float>();
            for (int i = 2 * audioTokens * dim; i < 3 * audioTokens * dim; i++) c[i] = -c[i] + 0.5f;
            using Tensor before = injector.Forward(backend, hidden, 0, audio, audioGlobal, seqLen);
            using Tensor after = injector.Forward(backend, hidden, 0, changed, audioGlobal, seqLen);
            ReadOnlySpan<float> x = before.AsReadOnlySpan<float>(), y = after.AsReadOnlySpan<float>();
            double changedFrame = 0;
            for (int i = 0; i < x.Length; i++)
            {
                int frame = i / (tokensAFrame * dim);
                if (frame == 2) changedFrame = Math.Max(changedFrame, Math.Abs(x[i] - y[i]));
                else Assert.Equal(x[i], y[i]);
            }
            Assert.True(changedFrame > 0.01, $"frame 2 moved by only {changedFrame}");
        }
        finally
        {
            Release(injector, weights);
        }
    }
}
