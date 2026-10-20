using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary><see cref="MossImageParityTests"/>' tiny reference on every GPU backend, through the device kernels the real
/// model runs (fused RMSNorm-modulate, SiTU-GLU, token-major attention, F16 branches where the backend has them), plus
/// the step-graph path: a branch set run through the fixed latent repeatedly, so the warm-up, capture and replay calls
/// all have to agree with the eager answer.</summary>
[Trait("Category", "GpuIntegration")]
public sealed unsafe class MossImageGpuParityTests(ITestOutputHelper output)
{
    private const int Height = MossImageFixture.Height, Width = MossImageFixture.Width;

    [Theory]
    [MemberData(nameof(BackendGate.GpuKinds), MemberType = typeof(BackendGate))]
    public void Forward_MatchesReference(string kind)
    {
        if (!BackendGate.TryOpen(kind, output.WriteLine, out IBackend? gpu)) return;
        using IBackend backend = gpu!;
        Run(backend, (transformer, f) =>
        {
            using MossImageConditioning conditioning = transformer.PrepareConditioning(backend, f["inputs.context"], Height, Width);
            using Tensor full = transformer.Forward(backend, f["inputs.latent"], 500f, conditioning);
            TensorAssert.Close(full, f["full.velocity"], rtol: 3e-3f, atol: 3e-3f, because: $"full forward on {kind}");
            using Tensor skipped = transformer.Forward(backend, f["inputs.latent"], 937f, conditioning, [1]);
            TensorAssert.Close(skipped, f["skip.velocity"], rtol: 3e-3f, atol: 3e-3f, because: $"skipped layer on {kind}");
        });
    }

    /// <summary>Six steps through the fixed latent with the graph forced on: steps 1–2 warm up, 3 captures, 4–6 replay.</summary>
    [Theory]
    [MemberData(nameof(BackendGate.GpuKinds), MemberType = typeof(BackendGate))]
    public void StepGraph_ReplaysTheEagerBranches(string kind)
    {
        if (!BackendGate.TryOpen(kind, output.WriteLine, out IBackend? gpu)) return;
        using IBackend backend = gpu!;
        KnobStore.Set(EngineKnobs.DitGraph, true);
        try
        {
            Run(backend, (transformer, f) =>
            {
                using MossImageConditioning cond = transformer.PrepareConditioning(backend, f["inputs.context"], Height, Width);
                MossImageBranch[] branches = [new(cond, []), new(cond, [1])];
                using Tensor expectedFull = transformer.Forward(backend, f["inputs.latent"], 937f, cond);
                Tensor latent = transformer.PrepareGraphLatent(backend, f["inputs.latent"]);
                Assert.True(transformer.GraphEligible(backend), $"step graph not eligible on {kind}");
                for (int step = 1; step <= 6; step++)
                {
                    (Tensor[] velocities, bool callerOwns) = transformer.ForwardBranches(backend, latent, 937f, branches);
                    Assert.False(callerOwns);
                    // The fixed buffers back the captured graph: snapshot them rather than reading them directly.
                    using Tensor cond0 = Snapshot(backend, velocities[0]);
                    using Tensor skip1 = Snapshot(backend, velocities[1]);
                    TensorAssert.Close(cond0, expectedFull, rtol: 1e-4f, atol: 1e-4f, because: $"cond branch, step {step} on {kind}");
                    TensorAssert.Close(skip1, f["skip.velocity"], rtol: 3e-3f, atol: 3e-3f, because: $"skip branch, step {step} on {kind}");
                }
                transformer.InvalidateStepGraph(backend);
            });
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.DitGraph);
        }
    }

    [Theory]
    [MemberData(nameof(BackendGate.GpuKinds), MemberType = typeof(BackendGate))]
    public void Edit_MatchesReference(string kind)
    {
        if (!BackendGate.TryOpen(kind, output.WriteLine, out IBackend? gpu)) return;
        using IBackend backend = gpu!;
        Run(backend, (transformer, f, references) =>
        {
            using MossImageConditioning joint = transformer.PrepareConditioning(backend, f["inputs.context"], Height, Width,
                references, ReferenceAttentionMode.Joint);
            using Tensor jointVelocity = transformer.Forward(backend, f["inputs.latent"], 500f, joint);
            TensorAssert.Close(jointVelocity, f["edit.velocity"], rtol: 3e-3f, atol: 3e-3f, because: $"joint edit on {kind}");
            using MossImageConditioning cached = transformer.PrepareConditioning(backend, f["inputs.context"], Height, Width,
                references, ReferenceAttentionMode.CachedKv);
            using Tensor cachedVelocity = transformer.Forward(backend, f["inputs.latent"], 500f, cached);
            TensorAssert.Close(cachedVelocity, f["edit_cached.velocity"], rtol: 3e-3f, atol: 3e-3f, because: $"cached edit on {kind}");
            using Tensor skipped = transformer.Forward(backend, f["inputs.latent"], 937f, cached, [1]);
            TensorAssert.Close(skipped, f["edit_cached_skip.velocity"], rtol: 3e-3f, atol: 3e-3f, because: $"cached edit, skip on {kind}");
        });
    }

    /// <summary>The step graph over edit branches: the cached keys/values and the joint segment tables must sit at fixed
    /// addresses, so replays agree with the eager answer.</summary>
    [Theory]
    [MemberData(nameof(BackendGate.GpuKinds), MemberType = typeof(BackendGate))]
    public void StepGraph_ReplaysEditBranches(string kind)
    {
        if (!BackendGate.TryOpen(kind, output.WriteLine, out IBackend? gpu)) return;
        using IBackend backend = gpu!;
        KnobStore.Set(EngineKnobs.DitGraph, true);
        try
        {
            Run(backend, (transformer, f, references) =>
            {
                foreach (ReferenceAttentionMode mode in new[] { ReferenceAttentionMode.CachedKv, ReferenceAttentionMode.Joint })
                {
                    using MossImageConditioning cond = transformer.PrepareConditioning(backend, f["inputs.context"], Height, Width,
                        references, mode);
                    MossImageBranch[] branches = [new(cond, []), new(cond, [1])];
                    using Tensor expected0 = transformer.Forward(backend, f["inputs.latent"], 937f, cond);
                    using Tensor expected1 = transformer.Forward(backend, f["inputs.latent"], 937f, cond, [1]);
                    Tensor latent = transformer.PrepareGraphLatent(backend, f["inputs.latent"]);
                    for (int step = 1; step <= 6; step++)
                    {
                        (Tensor[] velocities, bool callerOwns) = transformer.ForwardBranches(backend, latent, 937f, branches);
                        Assert.False(callerOwns);
                        using Tensor v0 = Snapshot(backend, velocities[0]);
                        using Tensor v1 = Snapshot(backend, velocities[1]);
                        TensorAssert.Close(v0, expected0, rtol: 1e-4f, atol: 1e-4f, because: $"{mode} cond branch, step {step} on {kind}");
                        TensorAssert.Close(v1, expected1, rtol: 1e-4f, atol: 1e-4f, because: $"{mode} skip branch, step {step} on {kind}");
                    }
                    transformer.InvalidateStepGraph(backend);
                }
            });
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.DitGraph);
        }
    }

    private static Tensor Snapshot(IBackend backend, Tensor fixedBuffer)
    {
        Tensor copy = new(fixedBuffer.Shape, fixedBuffer.DType);
        backend.CopyInto(copy, fixedBuffer);
        return copy;
    }

    private static void Run(IBackend backend, Action<MossImageTransformer, IReadOnlyDictionary<string, Tensor>> body) =>
        Run(backend, (transformer, tensors, _) => body(transformer, tensors));

    private static void Run(IBackend backend, Action<MossImageTransformer, IReadOnlyDictionary<string, Tensor>, ReferenceLatentTokens[]> body)
    {
        using MossImageFixture fixture = new();
        backend.PreloadWeights(fixture.Transformer.EnumerateWeights());
        try
        {
            body(fixture.Transformer, fixture.Tensors, fixture.References());
        }
        finally
        {
            backend.FreeWeights(fixture.Transformer.EnumerateWeights());
        }
    }
}
