using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

/// <summary>The Anima reference path against ComfyUI-Anima-Ref-Adapter, on the real base checkpoint and the released adapter. The engine
/// applies the adapter through the gated LoRA path (live during the reference prepass and the reference-conditioned forward), builds the
/// reference cache, and joins it. Its per-block reference keys and values and its velocity must match the node's dump.
/// <para>Needs three inputs, and skips when any is unset: <c>HARTSY_ANIMA_REF_DUMP</c> (the output of <c>dump_anima_ref_adapter.py</c>),
/// <c>HARTSY_ANIMA_CHECKPOINT</c> (anima-base-v1.0.safetensors) and <c>HARTSY_ANIMA_ADAPTER</c> (anima_ref_adapter_v2_r64.safetensors).
/// The reference RoPE tables are compared first, so a position mismatch is not mistaken for a block mismatch. Each value's error is
/// divided by <c>max(1, |expected|)</c>, so values below 1 are compared on an absolute scale. The tolerance is a first bound, not a
/// tuned one: the run reports the observed maximum, which is what to tighten against.</para></summary>
[Collection("CudaSerial")]
public sealed unsafe class AnimaReferenceParityGpuTests
{
    private const float Tolerance = 1e-3f;

    private readonly ITestOutputHelper _output;

    public AnimaReferenceParityGpuTests(ITestOutputHelper output) => _output = output;

    [Trait("Category", "GpuIntegration")]
    [Fact]
    public void ReferenceKvAndVelocity_MatchComfyUi()
    {
        string? dumpPath = Environment.GetEnvironmentVariable("HARTSY_ANIMA_REF_DUMP");
        string? checkpointPath = Environment.GetEnvironmentVariable("HARTSY_ANIMA_CHECKPOINT");
        string? adapterPath = Environment.GetEnvironmentVariable("HARTSY_ANIMA_ADAPTER");
        if (dumpPath is null || checkpointPath is null || adapterPath is null)
        {
            _output.WriteLine("SKIPPED: set HARTSY_ANIMA_REF_DUMP, HARTSY_ANIMA_CHECKPOINT and HARTSY_ANIMA_ADAPTER");
            return;
        }
        if (!CudaContext.IsAvailable())
        {
            _output.WriteLine("SKIPPED: no CUDA device");
            return;
        }
        // The dump runs in F32 with TF32 off. Under default numerics the GEMMs cast F32 activations to BF16 against the checkpoint's
        // BF16 weights, which the dump does not do, so the comparison runs under the reference profile. The backend reads its GEMM
        // knobs at construction, so the profile must be pushed before the backend is created.
        using IDisposable numerics = KnobProfiles.Reference.Push();

        using SafeTensorsLoader dump = new();
        dump.Load(dumpPath);
        using CheckpointSource source = CheckpointSource.Open(checkpointPath);
        AnimaCheckpointConverter.ConvertedWeights converted = AnimaCheckpointConverter.Convert(source.Weights);
        Dictionary<string, Tensor> weights = new(converted.Transformer);
        using CudaBackend cuda = new(0, PtxDir());

        using LoraFile adapterFile = LoraFile.Load(adapterPath);
        AdjunctGate gate = new();
        using LoraStack stack = new();
        stack.AddGated(adapterFile, gate, strength: 1.0f);
        stack.ApplyTo(weights, LoraTarget.Transformer, cuda);

        using AnimaTransformer transformer = new(AnimaConfig.AnimaPreview3);
        transformer.LoadWeights(weights);
        cuda.PreloadWeights(transformer.EnumerateWeights());
        cuda.Sync();

        using Tensor latent = OnDevice(dump.GetTensor("in.latent"), cuda);
        using Tensor reference = OnDevice(dump.GetTensor("in.reference"), cuda);
        using Tensor context = OnDevice(dump.GetTensor("in.context"), cuda);
        float sigma = ReadFirst(dump.GetTensor("in.sigma"));

        // The node's placement: frame 1, one target height down, column 0. The target grid is the latent's patched height.
        int targetGridH = (int)latent.Shape[2] / 2;
        ReferencePlacement placement = new(1, 1.0f, 0);
        using IDisposable scope = gate.Enable(1.0f);
        using ReferenceKvCache cache = transformer.BuildReferenceCache(cuda, reference, targetGridH, placement, identity: 1);
        cuda.Sync();

        // The RoPE rows the reference and the target read, as the engine's host tables hold them, against the node's. A position error
        // shows here before it shows in any block.
        AnimaRope rope = new(AnimaConfig.AnimaPreview3.HeadDim, AnimaConfig.AnimaPreview3.RopeTheta, AnimaConfig.AnimaPreview3.RopeScale);
        int referenceGridH = (int)reference.Shape[2] / 2;
        int referenceGridW = (int)reference.Shape[3] / 2;
        using (RopeTableLease referenceTables = rope.AcquireFrameTables(cuda, 1, referenceGridH, referenceGridW,
            placement.RowOffset(targetGridH), placement.ColumnOffset, out _, out _))
        {
            using Tensor cos = HostTensor(referenceTables.HostCos);
            using Tensor sin = HostTensor(referenceTables.HostSin);
            CompareFlat(cuda, "rope.ref.cos", dump.GetTensor("rope.ref.cos"), cos);
            CompareFlat(cuda, "rope.ref.sin", dump.GetTensor("rope.ref.sin"), sin);
        }
        using (RopeTableLease targetTables = rope.AcquireFrameTables(cuda, 0, targetGridH, (int)latent.Shape[3] / 2, 0, 0, out _, out _))
        {
            using Tensor cos = HostTensor(targetTables.HostCos);
            using Tensor sin = HostTensor(targetTables.HostSin);
            CompareFlat(cuda, "rope.target.cos", dump.GetTensor("rope.target.cos"), cos);
            CompareFlat(cuda, "rope.target.sin", dump.GetTensor("rope.target.sin"), sin);
        }
        rope.ReleaseGpuTables(cuda);

        for (int i = 0; i < cache.Layers; i++)
        {
            // Borrowed from the cache: read them, never dispose them.
            Tensor keys = cache.Keys(i)!;
            Tensor values = cache.Values(i)!;
            CompareFlat(cuda, $"ref.k.{i}", dump.GetTensor($"ref.k.{i}"), keys);
            CompareFlat(cuda, $"ref.v.{i}", dump.GetTensor($"ref.v.{i}"), values);
        }

        using Tensor velocity = transformer.Forward(cuda, latent, sigma, context, null, cache);
        CompareFlat(cuda, "out.velocity", dump.GetTensor("out.velocity"), velocity);
        transformer.ReleaseDeviceCache(cuda);
    }

    /// <summary>A host tensor holding a lease's table values, so the comparison reads them the way it reads the dump's.</summary>
    private static Tensor HostTensor(ReadOnlySpan<float> values)
    {
        Tensor tensor = new(new TensorShape(values.Length), DType.F32);
        values.CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    /// <summary>Copies a host tensor to the device as a device-authored input, so a host round trip in the path would show.</summary>
    private static Tensor OnDevice(Tensor host, CudaBackend backend)
    {
        Tensor device = new(host.Shape, DType.F32);
        backend.Scale(device, host, 1.0f);
        return device;
    }

    private static float ReadFirst(Tensor tensor) => ((float*)tensor.DataPointer)[0];

    /// <summary>Compares two tensors of equal element count, reporting the observed maximum error before asserting.</summary>
    private void CompareFlat(CudaBackend backend, string name, Tensor expected, Tensor actual)
    {
        backend.Sync();
        int count = (int)expected.ElementCount;
        Assert.Equal(count, (int)actual.ElementCount);
        float* e = (float*)expected.DataPointer;
        float* a = (float*)actual.DataPointer;
        float worst = 0f, worstRelative = 0f;
        int worstIndex = 0;
        for (int i = 0; i < count; i++)
        {
            float error = MathF.Abs(e[i] - a[i]);
            float relative = error / MathF.Max(1f, MathF.Abs(e[i]));
            if (relative > worstRelative)
            {
                worstRelative = relative;
                worst = error;
                worstIndex = i;
            }
        }
        _output.WriteLine($"{name}: max |diff| {worst:G6} (relative {worstRelative:G6}) at element {worstIndex} of {count}");
        Assert.True(worstRelative <= Tolerance, $"{name}: relative error {worstRelative} at element {worstIndex} exceeds {Tolerance}");
    }

    private static string PtxDir()
    {
        string dir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(dir))
            dir = Path.Combine(RepoRoot.Path, "src", "HartsyInference.Cuda", "Ptx");
        return dir;
    }
}
