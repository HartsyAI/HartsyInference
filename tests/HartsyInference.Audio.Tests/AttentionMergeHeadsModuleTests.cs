using System.Reflection;
using HartsyInference.Audio.Models.ControlFoley;
using HartsyInference.Audio.Models.PocketTts;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Runs the three modules that merge attention heads back to token-major layout forward with synthetic weights: the
/// Pocket-TTS Mimi decoder, ControlFoley's <c>SelfAttention</c> (the CAV-MAE blocks) and the Synchformer's spatial aggregation
/// layer. Each result is recorded on the CPU backend (a checksum, the sum of magnitudes and sampled elements, held to 1e-4
/// relative so another instruction set cannot fail it) and CUDA and Vulkan are held to the CPU's full output (2e-3 on CUDA,
/// whose F32 dense row multiplies in TF32, and 1e-4 on Vulkan, as the other audio sites on these rows are).
/// <para>A head merge that disposed its input twice would not show here, because <c>Tensor.Dispose</c> is idempotent. What the
/// tests catch is a merge that reads a tensor after it has been disposed (an exception on the host, garbage on a device) and
/// one that leaks a tensor (on CUDA the count of cached activations must return to its starting value).</para></summary>
public sealed class AttentionMergeHeadsModuleTests(ITestOutputHelper output)
{
    private const float Tolerance = 1e-4f, Tf32Tolerance = 2e-3f;

    private static float[] Values(int count, int seed, float scale, float offset = 0f)
    {
        Random random = new(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = offset + (random.NextSingle() * 2f - 1f) * scale;
        }

        return values;
    }

    private static Tensor Make(int seed, float scale, float offset, params long[] shape)
    {
        Tensor tensor = new(new TensorShape(shape), DType.F32);
        float[] values = Values((int)tensor.ElementCount, seed, scale, offset);
        values.CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static int Seed(string key) => key.Aggregate(17, (h, c) => unchecked(h * 31 + c));

    private static void Add(Dictionary<string, Tensor> weights, string key, float scale, float offset, params long[] shape)
        => weights[key] = Make(Seed(key), scale, offset, shape);

    private static void AssertClose(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, float tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0;
        int at = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            double error = Math.Abs(expected[i] - actual[i]) / Math.Max(1f, Math.Abs(expected[i]));
            if (!(error <= worst))
            {
                (worst, at) = (error, i);
            }
        }

        Assert.True(worst <= tolerance, $"{what}: element {at} is {(at < 0 ? 0f : actual[at])}, expected "
            + $"{(at < 0 ? 0f : expected[at])} (relative error {worst:G3}, bound {tolerance})");
    }

    /// <summary>Checksum of a result: the sum, the sum of magnitudes, and eight elements at fixed fractions of its length.</summary>
    private static float[] Stats(ReadOnlySpan<float> values)
    {
        double sum = 0, magnitude = 0;
        foreach (float v in values)
        {
            Assert.True(float.IsFinite(v), "the result holds a value that is not finite");
            sum += v;
            magnitude += Math.Abs(v);
        }

        float[] stats = new float[10];
        stats[0] = (float)sum;
        stats[1] = (float)magnitude;
        for (int i = 0; i < 8; i++)
        {
            stats[2 + i] = values[(int)((long)(values.Length - 1) * (2 * i + 1) / 16)];
        }

        return stats;
    }

    private void AssertStats(float[] expected, ReadOnlySpan<float> values, string what)
    {
        float[] actual = Stats(values);
        output.WriteLine($"{what}: " + string.Join(", ", actual.Select(v => v.ToString("R", System.Globalization.CultureInfo.InvariantCulture) + "f")));
        for (int i = 0; i < expected.Length; i++)
        {
            double error = Math.Abs(expected[i] - actual[i]) / Math.Max(1.0, Math.Abs(expected[i]));
            Assert.True(error <= Tolerance, $"{what}: statistic {i} is {actual[i]}, expected {expected[i]} (relative error {error:G3})");
        }
    }

    /// <summary>Cached activations the CUDA backend's residency cache holds now; read by reflection because the type is internal
    /// to the CUDA assembly. Null when it cannot be read.</summary>
    private static int? CudaActivationCount()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        PropertyInfo? property = typeof(CudaBackend).Assembly.GetType("HartsyInference.Cuda.GpuTransferHelper")
            ?.GetProperty("CachedActivationCount", BindingFlags.Static | BindingFlags.NonPublic | BindingFlags.Public);
        return property?.GetValue(null) as int?;
    }

    private static string PtxDirectory()
    {
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate))
            {
                return candidate;
            }
        }

        throw new DirectoryNotFoundException("src/HartsyInference.Cuda/Ptx was not found above the test directory.");
    }

    // ── Pocket-TTS Mimi decoder ─────────────────────────────────────────────────────────────────────────────────────

    private static Dictionary<string, Tensor> PocketWeights()
    {
        Dictionary<string, Tensor> w = [];
        const float s = 0.05f;
        Add(w, "mimi.quantizer.output_proj.weight", s, 0f, 512, 32, 1);
        Add(w, "mimi.upsample.convtr.convtr.weight", s, 0f, 512, 1, 32);
        for (int i = 0; i < 2; i++)
        {
            string p = $"mimi.decoder_transformer.transformer.layers.{i}";
            Add(w, $"{p}.norm1.weight", 0.1f, 1f, 512);
            Add(w, $"{p}.norm1.bias", 0.1f, 0f, 512);
            Add(w, $"{p}.norm2.weight", 0.1f, 1f, 512);
            Add(w, $"{p}.norm2.bias", 0.1f, 0f, 512);
            Add(w, $"{p}.self_attn.in_proj.weight", 0.08f, 0f, 1536, 512);
            Add(w, $"{p}.self_attn.out_proj.weight", 0.08f, 0f, 512, 512);
            Add(w, $"{p}.linear1.weight", 0.05f, 0f, 2048, 512);
            Add(w, $"{p}.linear2.weight", 0.05f, 0f, 512, 2048);
            Add(w, $"{p}.layer_scale_1.scale", 0.5f, 1f, 512);
            Add(w, $"{p}.layer_scale_2.scale", 0.5f, 1f, 512);
        }

        Add(w, "mimi.decoder.model.0.conv.weight", 0.02f, 0f, 512, 512, 7);
        Add(w, "mimi.decoder.model.0.conv.bias", 0.05f, 0f, 512);
        int[] upIdx = [2, 5, 8], rbIdx = [3, 6, 9], ratios = [6, 5, 4];
        int channels = 512;
        for (int stage = 0; stage < 3; stage++)
        {
            int outCh = channels / 2;
            Add(w, $"mimi.decoder.model.{upIdx[stage]}.convtr.weight", 0.03f, 0f, channels, outCh, 2 * ratios[stage]);
            Add(w, $"mimi.decoder.model.{upIdx[stage]}.convtr.bias", 0.05f, 0f, outCh);
            Add(w, $"mimi.decoder.model.{rbIdx[stage]}.block.1.conv.weight", 0.05f, 0f, outCh / 2, outCh, 3);
            Add(w, $"mimi.decoder.model.{rbIdx[stage]}.block.1.conv.bias", 0.05f, 0f, outCh / 2);
            Add(w, $"mimi.decoder.model.{rbIdx[stage]}.block.3.conv.weight", 0.05f, 0f, outCh, outCh / 2, 1);
            Add(w, $"mimi.decoder.model.{rbIdx[stage]}.block.3.conv.bias", 0.05f, 0f, outCh);
            channels = outCh;
        }

        Add(w, "mimi.decoder.model.11.conv.weight", 0.1f, 0f, 1, 64, 3);
        Add(w, "mimi.decoder.model.11.conv.bias", 0.05f, 0f, 1);
        return w;
    }

    /// <summary>The decoder's output for a fixed latent of <paramref name="frames"/> frames, as a host array.</summary>
    private static float[] PocketForward(IBackend backend, int batch, int frames)
    {
        Dictionary<string, Tensor> weights = PocketWeights();
        try
        {
            PocketTtsMimiDecoder decoder = new("mimi");
            decoder.LoadWeights(weights);
            using Tensor latent = Make(4242, 1f, 0f, batch, 32, frames);
            using Tensor pcm = decoder.Forward(backend, latent, batch, frames);
            backend.Sync();
            return pcm.AsReadOnlySpan<float>().ToArray();
        }
        finally
        {
            foreach (Tensor t in weights.Values)
            {
                t.Dispose();
            }
        }
    }

    // Recorded on the CPU backend before the merge moved onto the shared layer (batch 2, 3 frames).
    private static readonly float[] PocketCpuStats = [-573.892f, 578.72943f, -0.035805613f, -0.017486608f, -0.029979678f, -0.003725144f, -0.002890177f, 0.017869834f, 0.0026051404f, -0.011795155f];

    [Fact]
    public void PocketTtsDecoderOnTheCpuMatchesItsRecordedOutput()
    {
        using CpuBackend backend = new();
        float[] pcm = PocketForward(backend, 2, 3);
        Assert.Equal(2 * 3 * 16 * 120, pcm.Length);
        AssertStats(PocketCpuStats, pcm, "Pocket-TTS decoder on the CPU");
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void PocketTtsDecoderOnCudaMatchesTheCpu()
    {
        if (!CudaContext.IsAvailable())
        {
            output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using CpuBackend cpu = new();
        float[] expected = PocketForward(cpu, 2, 3);
        using CudaBackend backend = new(0, PtxDirectory());
        PocketForward(backend, 2, 3);
        int? before = CudaActivationCount();
        Assert.NotNull(before);
        float[] actual = PocketForward(backend, 2, 3);
        int? after = CudaActivationCount();
        AssertClose(expected, actual, Tf32Tolerance, "Pocket-TTS decoder on CUDA");
        output.WriteLine($"CUDA cached activations before {before}, after {after}");
        Assert.Equal(before, after);
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void PocketTtsDecoderOnVulkanMatchesTheCpu()
    {
        VulkanBackend backend;
        try
        {
            backend = new VulkanBackend();
        }
        catch (Exception ex)
        {
            output.WriteLine($"SKIPPED: Vulkan unavailable ({ex.Message})");
            return;
        }

        using (backend)
        {
            using CpuBackend cpu = new();
            float[] expected = PocketForward(cpu, 2, 3);
            float[] actual = PocketForward(backend, 2, 3);
            AssertClose(expected, actual, Tolerance, "Pocket-TTS decoder on Vulkan");
        }
    }

    // ── ControlFoley self-attention (the CAV-MAE blocks) ──────────────────────────────────────────────────────────

    private static float[] SelfAttentionForward(IBackend backend)
    {
        const int batch = 2, tokens = 5, dim = 32, heads = 4;
        using Tensor x = Make(11, 1f, 0f, batch, tokens, dim);
        using Tensor qkvW = Make(12, 0.3f, 0f, 3 * dim, dim), qkvB = Make(13, 0.1f, 0f, 3 * dim);
        using Tensor projW = Make(14, 0.3f, 0f, dim, dim), projB = Make(15, 0.1f, 0f, dim);
        using Tensor result = ControlFoleyVitOps.SelfAttention(backend, x, qkvW, qkvB, projW, projB, heads);
        backend.Sync();
        return result.AsReadOnlySpan<float>().ToArray();
    }

    private static readonly float[] SelfAttentionCpuStats = [18.653866f, 69.5295f, 0.21077907f, 0.06958215f, -0.013473634f, 0.16145971f, 0.5160211f, -0.031462386f, -0.100392f, -0.52674335f];

    [Fact]
    public void ControlFoleySelfAttentionOnTheCpuMatchesItsRecordedOutput()
    {
        using CpuBackend backend = new();
        AssertStats(SelfAttentionCpuStats, SelfAttentionForward(backend), "ControlFoley self-attention on the CPU");
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void ControlFoleySelfAttentionOnCudaMatchesTheCpu()
    {
        if (!CudaContext.IsAvailable())
        {
            output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using CpuBackend cpu = new();
        float[] expected = SelfAttentionForward(cpu);
        using CudaBackend backend = new(0, PtxDirectory());
        SelfAttentionForward(backend);
        int? before = CudaActivationCount();
        Assert.NotNull(before);
        float[] actual = SelfAttentionForward(backend);
        int? after = CudaActivationCount();
        AssertClose(expected, actual, Tf32Tolerance, "ControlFoley self-attention on CUDA");
        output.WriteLine($"CUDA cached activations before {before}, after {after}");
        Assert.Equal(before, after);
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void ControlFoleySelfAttentionOnVulkanMatchesTheCpu()
    {
        VulkanBackend backend;
        try
        {
            backend = new VulkanBackend();
        }
        catch (Exception ex)
        {
            output.WriteLine($"SKIPPED: Vulkan unavailable ({ex.Message})");
            return;
        }

        using (backend)
        {
            using CpuBackend cpu = new();
            AssertClose(SelfAttentionForward(cpu), SelfAttentionForward(backend), Tolerance, "ControlFoley self-attention on Vulkan");
        }
    }

    // ── Synchformer spatial aggregation ─────────────────────────────────────────────────────────────────────────

    private static ControlFoleySynchformerConfig SynchformerConfig() => new()
    {
        EmbedDim = 16, Depth = 1, Heads = 2, PatchSize = 8, InputSize = 16, TemporalPatch = 2, SegmentFrames = 4, StrideFrames = 2,
        MlpRatio = 2,
    };

    private static Dictionary<string, Tensor> SynchformerWeights(ControlFoleySynchformerConfig c)
    {
        Dictionary<string, Tensor> w = [];
        int d = c.EmbedDim, hidden = d * c.MlpRatio, p = c.PatchSize, tp = c.TemporalPatch;
        const string prefix = "vfeat_extractor.";
        Add(w, prefix + "patch_embed_3d.proj.weight", 0.2f, 0f, d, 3, tp, p, p);
        Add(w, prefix + "patch_embed_3d.proj.bias", 0.1f, 0f, d);
        Add(w, prefix + "cls_token", 0.5f, 0f, 1, 1, d);
        Add(w, prefix + "pos_embed", 0.5f, 0f, 1, c.SpatialTokens + 1, d);
        Add(w, prefix + "temp_embed", 0.5f, 0f, 1, c.TemporalTokens, d);
        Add(w, prefix + "norm.weight", 0.1f, 1f, d);
        Add(w, prefix + "norm.bias", 0.1f, 0f, d);
        for (int i = 0; i < c.Depth; i++)
        {
            string b = $"{prefix}blocks.{i}";
            foreach (string n in new[] { "norm1", "norm2", "norm3" })
            {
                Add(w, $"{b}.{n}.weight", 0.1f, 1f, d);
                Add(w, $"{b}.{n}.bias", 0.1f, 0f, d);
            }

            foreach (string a in new[] { "attn", "timeattn" })
            {
                Add(w, $"{b}.{a}.qkv.weight", 0.3f, 0f, 3 * d, d);
                Add(w, $"{b}.{a}.qkv.bias", 0.1f, 0f, 3 * d);
                Add(w, $"{b}.{a}.proj.weight", 0.3f, 0f, d, d);
                Add(w, $"{b}.{a}.proj.bias", 0.1f, 0f, d);
            }

            Add(w, $"{b}.mlp.fc1.weight", 0.3f, 0f, hidden, d);
            Add(w, $"{b}.mlp.fc1.bias", 0.1f, 0f, hidden);
            Add(w, $"{b}.mlp.fc2.weight", 0.3f, 0f, d, hidden);
            Add(w, $"{b}.mlp.fc2.bias", 0.1f, 0f, d);
        }

        string agg = prefix + "spatial_attn_agg.";
        Add(w, agg + "cls_token", 0.5f, 0f, 1, 1, d);
        Add(w, agg + "self_attn.in_proj_weight", 0.3f, 0f, 3 * d, d);
        Add(w, agg + "self_attn.in_proj_bias", 0.1f, 0f, 3 * d);
        Add(w, agg + "self_attn.out_proj.weight", 0.3f, 0f, d, d);
        Add(w, agg + "self_attn.out_proj.bias", 0.1f, 0f, d);
        Add(w, agg + "norm1.weight", 0.1f, 1f, d);
        Add(w, agg + "norm1.bias", 0.1f, 0f, d);
        Add(w, agg + "norm2.weight", 0.1f, 1f, d);
        Add(w, agg + "norm2.bias", 0.1f, 0f, d);
        Add(w, agg + "linear1.weight", 0.3f, 0f, hidden, d);
        Add(w, agg + "linear1.bias", 0.1f, 0f, hidden);
        Add(w, agg + "linear2.weight", 0.3f, 0f, d, hidden);
        Add(w, agg + "linear2.bias", 0.1f, 0f, d);
        return w;
    }

    private static float[] SynchformerForward(IBackend backend)
    {
        ControlFoleySynchformerConfig config = SynchformerConfig();
        Dictionary<string, Tensor> weights = SynchformerWeights(config);
        try
        {
            using ControlFoleySynchformer model = new(config);
            model.LoadWeights(weights);
            float[] frames = Values(4 * 3 * 16 * 16, 99, 1f);
            float[] tokens = model.Encode(backend, frames);
            backend.Sync();
            return tokens;
        }
        finally
        {
            foreach (Tensor t in weights.Values)
            {
                t.Dispose();
            }
        }
    }

    private static readonly float[] SynchformerCpuStats = [5.1733427f, 18.938334f, 1.0481282f, 0.6990382f, -0.014437579f, -0.46071115f, 1.0501195f, 1.0193229f, -0.22406289f, -0.32509643f];

    [Fact]
    public void SynchformerOnTheCpuMatchesItsRecordedOutput()
    {
        using CpuBackend backend = new();
        float[] tokens = SynchformerForward(backend);
        Assert.Equal(2 * 16, tokens.Length);
        AssertStats(SynchformerCpuStats, tokens, "Synchformer on the CPU");
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void SynchformerOnCudaMatchesTheCpu()
    {
        if (!CudaContext.IsAvailable())
        {
            output.WriteLine("SKIPPED: CUDA unavailable");
            return;
        }

        using CpuBackend cpu = new();
        float[] expected = SynchformerForward(cpu);
        using CudaBackend backend = new(0, PtxDirectory());
        SynchformerForward(backend);
        int? before = CudaActivationCount();
        Assert.NotNull(before);
        float[] actual = SynchformerForward(backend);
        int? after = CudaActivationCount();
        AssertClose(expected, actual, Tf32Tolerance, "Synchformer on CUDA");
        output.WriteLine($"CUDA cached activations before {before}, after {after}");
        Assert.Equal(before, after);
    }

    [Fact]
    [Trait("Category", "GpuIntegration")]
    public void SynchformerOnVulkanMatchesTheCpu()
    {
        VulkanBackend backend;
        try
        {
            backend = new VulkanBackend();
        }
        catch (Exception ex)
        {
            output.WriteLine($"SKIPPED: Vulkan unavailable ({ex.Message})");
            return;
        }

        using (backend)
        {
            using CpuBackend cpu = new();
            AssertClose(SynchformerForward(cpu), SynchformerForward(backend), Tolerance, "Synchformer on Vulkan");
        }
    }
}
