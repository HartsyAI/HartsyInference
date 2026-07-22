using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.Trellis;
using HartsyInference.ThreeD.Pipelines.Requests;
using HartsyInference.Vision.Dinov2;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>TRELLIS single-image → 3D Gaussian-splat pipeline: a DINOv2-with-registers conditioner encodes the
/// image, a rectified-flow stage denoises a sparse 16³ occupancy grid to pick the active voxels, a second
/// rectified-flow stage denoises a structured latent (SLAT) over those voxels, and the GS decoder turns the SLAT
/// into a <see cref="GaussianSplatCloud"/>. The flexicubes mesh decoder and radiance-field decoder are not yet
/// ported (see <c>docs/Checklists/TRELLIS_BUILD_PLAN.md</c>), so this pipeline only ever populates
/// <see cref="ThreeDResult.Splats"/>, never <see cref="ThreeDResult.Mesh"/>.
/// <para><b>Numerics validation-pending</b> — every network stage is parity-verified against the reference
/// TRELLIS in isolation (see <c>TrellisStage1/2ParityTests</c>, <c>TrellisGsDecoderParityTests</c>), but the
/// assembled pipeline's rendered splat output has not yet been visually cross-checked against the reference
/// pipeline's render, the way <see cref="TripoSrPipeline"/>'s mesh output has.</para></summary>
public sealed unsafe class TrellisImageTo3DPipeline : ThreeDPipelineBase
{
    // TRELLIS-image-large's fixed SLAT normalization constants (upstream `pipeline.json` → slat_normalization).
    // Per-model constants, not sample-dependent, so they're baked in here rather than requiring a runtime dump.
    private static readonly float[] SlatMean =
    [
        -2.1687545776367188f, -0.004347046371549368f, -0.13352349400520325f, -0.08418072760105133f,
        -0.5271206498146057f, 0.7238689064979553f, -1.1414450407028198f, 1.2039363384246826f,
    ];
    private static readonly float[] SlatStd =
    [
        2.377650737762451f, 2.386378288269043f, 2.124418020248413f, 2.1748552322387695f,
        2.663944721221924f, 2.371192216873169f, 2.6217446327209473f, 2.684523105621338f,
    ];

    private readonly TrellisImageConditioner _conditioner;
    private readonly SparseStructureFlow _ssFlow;
    private readonly SparseStructureDecoder _ssDecoder;
    private readonly SlatFlowModel _slatFlow;
    private readonly SlatGaussianDecoder _gsDecoder;
    private readonly Dinov2ImagePreprocessor _preprocessor;
    private readonly List<IDisposable> _ownedLoaders = [];

    public TrellisImageTo3DPipeline(IBackend backend, TrellisImageConditioner conditioner, SparseStructureFlow ssFlow,
        SparseStructureDecoder ssDecoder, SlatFlowModel slatFlow, SlatGaussianDecoder gsDecoder) : base(backend)
    {
        _conditioner = conditioner;
        _ssFlow = ssFlow;
        _ssDecoder = ssDecoder;
        _slatFlow = slatFlow;
        _gsDecoder = gsDecoder;
        _preprocessor = new Dinov2ImagePreprocessor(Dinov2Preset.LargeReg.ImageSize);
    }

    /// <summary>Loads a TRELLIS image-to-3D pipeline from a checkpoint directory. Expects the four upstream
    /// <c>microsoft/TRELLIS-image-large</c> component files by their canonical <c>ckpts/</c> names
    /// (<c>ss_flow_img_dit_L_16l8_fp16</c>, <c>ss_dec_conv3d_16l8_fp16</c>, <c>slat_flow_img_dit_L_64l8p2_fp16</c>,
    /// <c>slat_dec_gs_swin8_B_64l8gs32_fp16</c>) plus a DINOv2-with-registers-large conditioner file named
    /// <c>dinov2_vitl14_reg.safetensors</c> (HF <c>facebook/dinov2-with-registers-large</c> key format), all
    /// under <paramref name="modelPath"/>.</summary>
    public static TrellisImageTo3DPipeline LoadFromPath(IBackend backend, string modelPath)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (!Directory.Exists(modelPath))
            throw new DirectoryNotFoundException($"TRELLIS checkpoint directory not found: '{modelPath}'.");

        List<IDisposable> loaders = [];
        IReadOnlyDictionary<string, Tensor> LoadNamed(string fileName)
        {
            string path = Directory.GetFiles(modelPath, fileName, SearchOption.AllDirectories).FirstOrDefault()
                ?? throw new FileNotFoundException($"'{fileName}' not found under '{modelPath}'.");
            SafeTensorsLoader loader = new();
            loader.Load(path);
            loaders.Add(loader);
            return loader.GetAllTensors();
        }

        SparseStructureFlow ssFlow = new(); ssFlow.LoadWeights(LoadNamed("ss_flow_img_dit_L_16l8_fp16.safetensors"));
        SparseStructureDecoder ssDecoder = new(); ssDecoder.LoadWeights(LoadNamed("ss_dec_conv3d_16l8_fp16.safetensors"));
        SlatFlowModel slatFlow = new(); slatFlow.LoadWeights(LoadNamed("slat_flow_img_dit_L_64l8p2_fp16.safetensors"));
        SlatGaussianDecoder gsDecoder = new(); gsDecoder.LoadWeights(LoadNamed("slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors"));

        TrellisImageConditioner conditioner = new();
        conditioner.LoadWeights(LoadNamed("dinov2_vitl14_reg.safetensors"));

        TrellisImageTo3DPipeline pipeline = new(backend, conditioner, ssFlow, ssDecoder, slatFlow, gsDecoder);
        pipeline._ownedLoaders.AddRange(loaders);
        return pipeline;
    }

    /// <summary>Generates a Gaussian-splat cloud from a single conditioning image.</summary>
    /// <remarks><b>Input contract:</b> <see cref="ImageTo3DRequest.ImageRgb"/> must already be a foreground-isolated
    /// image on a neutral background (premultiply-alpha crop + resize), matching TRELLIS's own preprocessing.
    /// Passing a raw photo with a real background produces a degenerate result, same caveat as
    /// <see cref="TripoSrPipeline"/>/<see cref="Hunyuan3DShapePipeline"/>.</remarks>
    public ThreeDResult Generate(ImageTo3DRequest request, Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(request);
        int seed = request.Seed ?? SeedGenerator.RandomSeed();
        int steps = request.Steps > 0 ? request.Steps : 25;
        float cfg = request.CfgScale > 0 ? request.CfgScale : 5.0f;

        // 1. Image → DINOv2-reg conditioning tokens.
        Tensor pixels = _preprocessor.Preprocess(request.ImageRgb, request.Width, request.Height);
        Backend.PreloadWeights(_conditioner.EnumerateWeights());
        Tensor cond = _conditioner.Encode(Backend, pixels);
        pixels.Dispose();
        Backend.FreeWeights(_conditioner.EnumerateWeights());
        Tensor negCond = new(cond.Shape, DType.F32); Backend.Fill(negCond, 0f);
        onProgress?.Invoke(new GenerationProgress(1, 3, 0));

        // 2. Stage 1: sparse structure flow → active-voxel coordinates.
        Backend.PreloadWeights(_ssFlow.EnumerateWeights());
        Tensor noise1 = SeedGenerator.CreateNoise(new TensorShape(new long[] { 1, 8, 16, 16, 16 }), seed);
        Tensor zS = new TrellisSparseStructureSampler().Sample(Backend, _ssFlow, noise1, cond, negCond, steps, cfg);
        noise1.Dispose();
        Backend.FreeWeights(_ssFlow.EnumerateWeights());
        Backend.PreloadWeights(_ssDecoder.EnumerateWeights());
        Tensor occ = _ssDecoder.Decode(Backend, zS);
        zS.Dispose();
        Backend.FreeWeights(_ssDecoder.EnumerateWeights());
        int[] coords = ActiveCoords(occ);
        occ.Dispose();
        int nv = coords.Length / 4;
        onProgress?.Invoke(new GenerationProgress(2, 3, 0));
        if (nv == 0)
        {
            cond.Dispose(); negCond.Dispose();
            return new ThreeDResult { Splats = null, Seed = seed };
        }

        // 3. Stage 2: structured latent over the active voxels → GS decode.
        Backend.PreloadWeights(_slatFlow.EnumerateWeights());
        Tensor slatNoise = SeedGenerator.CreateNoise(new TensorShape(1, nv, 8), seed + 1);
        SparseTensor slat = new TrellisSlatSampler().Sample(Backend, _slatFlow, new SparseTensor(slatNoise, coords, 64), cond, negCond, steps, cfg);
        Backend.FreeWeights(_slatFlow.EnumerateWeights());
        cond.Dispose(); negCond.Dispose();
        using (Tensor mean = ConstantTensor(SlatMean))
        using (Tensor std = ConstantTensor(SlatStd))
        {
            TrellisSlatSampler.Denormalize(slat, mean, std);
        }

        Backend.PreloadWeights(_gsDecoder.EnumerateWeights());
        SparseTensor gs = _gsDecoder.Forward(Backend, slat);
        Backend.FreeWeights(_gsDecoder.EnumerateWeights());
        GaussianSplatCloud cloud = TrellisGaussianRepresentation.Build(gs);
        onProgress?.Invoke(new GenerationProgress(3, 3, 0));

        return new ThreeDResult { Splats = cloud, Seed = seed };
    }

    /// <inheritdoc/>
    protected override void DisposeCore()
    {
        foreach (IDisposable d in _ownedLoaders) d.Dispose();
        _ownedLoaders.Clear();
    }

    /// <summary>Builds a small host-resident F32 tensor from a constant array (the SLAT normalization stats).</summary>
    private static Tensor ConstantTensor(float[] values)
    {
        Tensor t = new(new TensorShape(values.Length), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < values.Length; i++) p[i] = values[i];
        return t;
    }

    /// <summary>Maps a dense occupancy grid <c>[1,1,D,H,W]</c> to active <c>(batch,z,y,x)</c> coordinates.</summary>
    private static int[] ActiveCoords(Tensor occ)
    {
        int d = (int)occ.Shape[2], h = (int)occ.Shape[3], w = (int)occ.Shape[4];
        float* p = (float*)occ.DataPointer;
        List<int> coords = [];
        for (int z = 0; z < d; z++)
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                    if (p[(z * h + y) * w + x] > 0f) { coords.Add(0); coords.Add(z); coords.Add(y); coords.Add(x); }
        return [.. coords];
    }
}
