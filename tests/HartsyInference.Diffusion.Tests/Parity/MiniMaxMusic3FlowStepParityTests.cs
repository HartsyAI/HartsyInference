using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Bisects the flow stage's first denoising step against the reference's own captured internals
/// (<c>--stage flowprobe</c>): the condition tensor, each guidance branch's velocity, and the latents the step
/// produced. Each assertion isolates one link, so a failure names the stage rather than the pipeline.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "Slow")]
public sealed unsafe class MiniMaxMusic3FlowStepParityTests(ITestOutputHelper output)
{
    [Fact]
    public void ConditionEncoder_MatchesTheWindowTheReferenceFed()
    {
        if (!TryLoad(out MiniMaxMusic3Reference? reference, out List<SafeTensorsLoader> loaders) || !reference!.Has("probe_condition"))
        {
            return;
        }
        try
        {
            using MiniMaxMusic3ConditionEncoder encoder = new MiniMaxMusic3ConditionEncoder();
            encoder.LoadWeights(Open("condition_encoder", loaders));
            float[] frameHiddens = reference.Read("flow_frame_hiddens");
            int windowFrames = MiniMaxMusic3FlowPipelineWindowFrames(reference);
            using Tensor condition = encoder.Encode(Backend(), frameHiddens, frameOffset: 0, frameCount: windowFrames);
            Report(reference, "condition", condition.AsReadOnlySpan<float>(), "probe_condition");
        }
        finally
        {
            Close(loaders);
        }
    }

    [Fact]
    public void Dit_MatchesBothBranchesOnTheReferencesOwnStepZeroInputs()
    {
        if (!TryLoad(out MiniMaxMusic3Reference? reference, out List<SafeTensorsLoader> loaders) || !reference!.Has("probe_velocity_cond"))
        {
            return;
        }
        try
        {
            using MiniMaxMusic3Dit dit = new MiniMaxMusic3Dit();
            dit.LoadWeights(Open("transformer", loaders));
            IBackend backend = Backend();
            using Tensor latents = Load(reference, "probe_latents_step0");
            using Tensor conditional = Load(reference, "probe_encoder_cond");
            using Tensor unconditional = Load(reference, "probe_encoder_uncond");
            float timestep = (float)reference.Meta("probe_timesteps")[0].GetDouble();

            using Tensor conditionalVelocity = dit.Forward(backend, latents, timestep, conditional);
            Report(reference, "velocity(cond)", conditionalVelocity.AsReadOnlySpan<float>(), "probe_velocity_cond");
            using Tensor unconditionalVelocity = dit.Forward(backend, latents, timestep, unconditional);
            Report(reference, "velocity(uncond)", unconditionalVelocity.AsReadOnlySpan<float>(), "probe_velocity_uncond");

            // The Euler update itself, driven by the REFERENCE's velocities so only the integrator is under test.
            using Tensor referenceCond = Load(reference, "probe_velocity_cond");
            using Tensor referenceUncond = Load(reference, "probe_velocity_uncond");
            using Tensor guided = CfgHelper.ApplyCfg(referenceUncond, referenceCond, MiniMaxMusic3FlowPipeline.DefaultCfgScale);
            int steps = reference.Meta("flow_steps").GetInt32();
            float[] stepped = [.. Load(reference, "probe_latents_step0").AsReadOnlySpan<float>().ToArray()];
            ReadOnlySpan<float> velocity = guided.AsReadOnlySpan<float>();
            for (int i = 0; i < stepped.Length; i++)
            {
                stepped[i] += velocity[i] / steps;
            }
            Report(reference, "euler(step0->step1)", stepped, "probe_latents_step1");
        }
        finally
        {
            Close(loaders);
        }
    }


    /// <summary>The same 137-latent case the CPU run matches at &lt;1e-3, on whatever backend <see cref="Backend"/>
    /// picks. A CUDA-only failure here means the divergence is in a backend op, not in the model.</summary>
    [Fact]
    public void Dit_MatchesReferenceOnTheSmallCase()
    {
        if (!TryLoad(out MiniMaxMusic3Reference? reference, out List<SafeTensorsLoader> loaders) || !reference!.Has("dit_out_t0_cond"))
        {
            return;
        }
        try
        {
            using MiniMaxMusic3Dit dit = new MiniMaxMusic3Dit();
            dit.LoadWeights(Open("transformer", loaders));
            IBackend backend = Backend();
            output.WriteLine($"[MiniMaxMusic3Step] backend={backend.GetType().Name} tokenMajorAttention={backend.SupportsTokenMajorAttention}");
            using Tensor latents = Load(reference, "dit_latents");
            using Tensor condition = Load(reference, "cond_out");
            using Tensor velocity = dit.Forward(backend, latents, 0f, condition);
            Report(reference, "smallcase velocity(t0,cond)", velocity.AsReadOnlySpan<float>(), "dit_out_t0_cond");
        }
        finally
        {
            Close(loaders);
        }
    }

    private static int MiniMaxMusic3FlowPipelineWindowFrames(MiniMaxMusic3Reference reference)
    {
        int frames = reference.Meta("flow_frames").GetInt32();
        return Math.Min(MiniMaxMusic3FlowPipeline.ChunkFrames, frames);
    }

    private void Report(MiniMaxMusic3Reference reference, string label, ReadOnlySpan<float> actual, string expectedName)
    {
        float[] expected = reference.Read(expectedName);
        (double meanAbs, double maxAbs, double correlation) = MiniMaxMusic3Reference.Compare(actual, expected);
        output.WriteLine($"[MiniMaxMusic3Step] {label}: n={actual.Length}/{expected.Length} "
            + $"meanAbs={meanAbs:E3} maxAbs={maxAbs:E3} corr={correlation:F8}");
        Assert.Equal(expected.Length, actual.Length);
        // The reference is F32 on CPU; a CUDA run differs by cuBLAS accumulation order, so the GPU gate is the
        // looser one — the same 10x split the T5 encoder parity uses.
        double tolerance = Environment.GetEnvironmentVariable("HARTSY_MM3_FORCE_CPU") == "1" ? 1e-3 : 1e-2;
        Assert.True(meanAbs < tolerance, $"{label} diverges: meanAbs={meanAbs:E3}, maxAbs={maxAbs:E3}, corr={correlation:F8}");
        Assert.True(correlation > 0.9999, $"{label} decorrelates: corr={correlation:F8}, meanAbs={meanAbs:E3}");
    }

    private static Tensor Load(MiniMaxMusic3Reference reference, string name)
    {
        int[] shape = reference.Shape(name);
        float[] values = reference.Read(name);
        Tensor tensor = new Tensor(new TensorShape(shape[0], shape[1], shape[2]), DType.F32);
        values.CopyTo(new Span<float>((float*)tensor.DataPointer, values.Length));
        return tensor;
    }

    private static IBackend Backend()
    {
        // The CPU path is the reference-precision one; the override exists so a divergence can be attributed to a
        // backend rather than to the model.
        if (Environment.GetEnvironmentVariable("HARTSY_MM3_FORCE_CPU") == "1")
        {
            return new CpuBackend();   // tier-lint: guarded
        }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            return new CpuBackend();   // tier-lint: guarded
        }
        try
        {
            return new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
        }
        catch (Exception)
        {
            return new CpuBackend();   // tier-lint: guarded
        }
    }

    private static bool TryLoad(out MiniMaxMusic3Reference? reference, out List<SafeTensorsLoader> loaders)
    {
        reference = MiniMaxMusic3Reference.TryLoad();
        loaders = [];
        return Environment.GetEnvironmentVariable("HARTSY_MINIMAX_MUSIC3_PATH") is not null && reference is not null;
    }

    private static Dictionary<string, Tensor> Open(string subfolder, List<SafeTensorsLoader> loaders)
    {
        string checkpoint = Environment.GetEnvironmentVariable("HARTSY_MINIMAX_MUSIC3_PATH")!;
        string[] shards = Directory.GetFiles(Path.Combine(checkpoint, subfolder), "*.safetensors");
        Array.Sort(shards, StringComparer.Ordinal);
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (string shard in shards)
        {
            SafeTensorsLoader loader = new SafeTensorsLoader();
            loader.Load(shard);
            loaders.Add(loader);
            foreach (KeyValuePair<string, Tensor> entry in loader.GetAllTensors())
            {
                weights[entry.Key] = entry.Value;
            }
        }
        return weights;
    }

    private static void Close(List<SafeTensorsLoader> loaders)
    {
        foreach (SafeTensorsLoader loader in loaders)
        {
            loader.Dispose();
        }
    }
}
