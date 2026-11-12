using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Io;
using HartsyInference.ThreeD.Models.Hunyuan3D;
using HartsyInference.ThreeD.Models.Hunyuan3D.Part;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>What X-Part generates for one object: a mesh per part, its box and the whole as one GLB.</summary>
/// <param name="Parts">One generated mesh per part, in the input mesh's frame.</param>
/// <param name="Boxes">Per part, the P3-SAM box the part was generated in, <c>[minX, minY, minZ, maxX, maxY, maxZ]</c> in the input mesh's frame.</param>
/// <param name="Glb">The parts as one GLB, one coloured node per part (<see cref="PartGlbWriter"/>).</param>
public sealed record XPartResult(Mesh[] Parts, float[][] Boxes, byte[] Glb);

/// <summary>Settings of a run; the defaults are the release's (<c>partformer_pipeline.py</c>, <c>infer.yaml</c>) except the grid, where upstream extracts at octree resolution 512.</summary>
public sealed record XPartOptions
{
    /// <summary>Seed of the surface samples, the part ids, the farthest-point starts and the initial noise.</summary>
    public int Seed { get; init; } = 42;

    /// <summary>Flow-matching Euler steps.</summary>
    public int Steps { get; init; } = 50;

    /// <summary>Marching-cubes grid per axis of each part (upstream: 512).</summary>
    public int GridResolution { get; init; } = 256;

    /// <summary>Surface points sampled for the object and for each part.</summary>
    public int PointCount { get; init; } = 81920;

    /// <summary>Query points (and so conditioning tokens) of each encoder.</summary>
    public int QueryCount { get; init; } = 4096;

    /// <summary>Half-width of the decoded volume (upstream <c>box_v</c>).</summary>
    public float BoxExtent { get; init; } = 1.01f;

    /// <summary>The logit the surface is extracted at (upstream <c>mc_level</c>, -1/512).</summary>
    public float Level { get; init; } = -1f / 512;
}

/// <summary>X-Part (Hunyuan3D-Part): a mesh is split into parts by P3-SAM and every part is generated again as a complete shape. The mesh is normalised, P3-SAM gives the part boxes, the
/// object and each box are sampled (81,920 surface points), the conditioner encodes them (<see cref="XPartConditioner"/>), the DiT denoises one latent set per part together
/// (<see cref="XPartDit"/>, 50 Euler steps of the flow, no classifier-free guidance) and each part latent is decoded by the ShapeVAE and meshed.</summary>
/// <remarks>Each stage loads its checkpoint, runs and releases it, so the peak is the largest stage (the DiT) and not the sum. The folder holds <c>p3sam/p3sam.safetensors</c>,
/// <c>conditioner/conditioner.safetensors</c>, <c>model/model.safetensors</c> and <c>shapevae/shapevae.safetensors</c> of <c>tencent/Hunyuan3D-Part</c>.</remarks>
public sealed class XPartPipeline
{
    private const int LatentTokens = 1024, LatentChannels = 64;
    private const float VaeScaleFactor = 1.0039506158752403f;

    private readonly IBackend _backend;
    private readonly string _folder;

    /// <summary>Checks that the four checkpoints exist; nothing is loaded yet.</summary>
    public XPartPipeline(IBackend backend, string folder)
    {
        ArgumentNullException.ThrowIfNull(backend);
        _backend = backend;
        _folder = folder;
        foreach (string file in Files)
        {
            if (!File.Exists(Path.Combine(folder, file)))
                throw new FileNotFoundException($"The X-Part checkpoint '{Path.Combine(folder, file)}' was not found (tencent/Hunyuan3D-Part).", Path.Combine(folder, file));
        }
    }

    /// <summary>The checkpoint files the pipeline reads, relative to its folder.</summary>
    public static IReadOnlyList<string> Files { get; } =
        ["p3sam/p3sam.safetensors", "conditioner/conditioner.safetensors", "model/model.safetensors", "shapevae/shapevae.safetensors"];

    /// <summary>Generates the parts of <paramref name="mesh"/>.</summary>
    public XPartResult Generate(Mesh mesh, XPartOptions? options = null, Action<GenerationProgress>? onProgress = null, CancellationToken cancel = default)
    {
        options ??= new XPartOptions();
        P3SamPipeline.ValidateMesh(mesh);
        if (options.Steps < 1 || options.GridResolution < 8 || options.PointCount < 1 || options.QueryCount < 1 || options.QueryCount > options.PointCount)
            throw new ArgumentOutOfRangeException(nameof(options), "Steps, grid, point and query counts must be positive and the queries must not exceed the points.");
        Stopwatch clock = Stopwatch.StartNew();
        int total = options.Steps + 4, done = 0;
        void Advance() => onProgress?.Invoke(new GenerationProgress(Math.Min(++done, total), total, clock.Elapsed.TotalMilliseconds));

        (float[] vertices, double[] center, double scale) = XPartSampling.Normalize(mesh.Vertices);
        Mesh normalized = new() { Vertices = vertices, Indices = (int[])mesh.Indices.Clone() };

        // 1. The part boxes.
        float[][] boxes = PredictBoxes(normalized, options, cancel);
        Advance();
        cancel.ThrowIfCancellationRequested();

        // 2. Surface samples: the object, and the faces in each box.
        (float[] objectPoints, float[] objectFeatures) = XPartSampling.SurfaceSamples(normalized.Vertices, normalized.Indices, options.PointCount, options.Seed);
        List<XPartPartSample> samples = [];
        List<float[]> keptBoxes = [];
        for (int k = 0; k < boxes.Length; k++)
        {
            int[] faces = XPartSampling.FacesInBox(normalized.Vertices, normalized.Indices, boxes[k]);
            if (faces.Length == 0)
                continue;
            (float[] points, float[] features) = XPartSampling.SurfaceSamples(normalized.Vertices, XPartSampling.SubIndices(normalized.Indices, faces), options.PointCount, options.Seed + 1 + k);
            samples.Add(new XPartPartSample(points, features, SurfaceSampler.FarthestPointIndices(points, options.QueryCount, (options.Seed + k) % (points.Length / 3))));
            keptBoxes.Add(boxes[k]);
        }

        if (samples.Count == 0)
            throw new HartsyInferenceException("P3-SAM found no part with faces in its box.");
        int[] objectQueries = SurfaceSampler.FarthestPointIndices(objectPoints, options.QueryCount, options.Seed % (objectPoints.Length / 3));
        int parts = samples.Count;
        int[] ids = XPartSampling.PartIds(parts, 50, options.Seed);

        // 3. The conditioning tokens.
        Tensor objectContext, partContext;
        using (SafeTensorsLoader loader = new())
        {
            loader.Load(Path.Combine(_folder, "conditioner", "conditioner.safetensors"));
            using XPartConditioner conditioner = new(_backend, loader.GetAllTensors());
            XPartConditioning conditioning = conditioner.Condition(objectPoints, objectFeatures, objectQueries, samples);
            try
            {
                (objectContext, partContext) = Stack(conditioning, parts);
            }
            finally
            {
                conditioning.ObjectTokens.Dispose();
                foreach (Tensor token in conditioning.PartTokens)
                    token.Dispose();
            }
        }

        Advance();
        cancel.ThrowIfCancellationRequested();

        // 4. The flow.
        Tensor latents;
        try
        {
            latents = Denoise(parts, ids, objectContext, partContext, options, Advance, cancel);
        }
        finally
        {
            objectContext.Dispose();
            partContext.Dispose();
        }

        // 5. Decode and mesh each part, then back to the input frame.
        Mesh[] result = new Mesh[parts];
        try
        {
            using SafeTensorsLoader loader = new();
            loader.Load(Path.Combine(_folder, "shapevae", "shapevae.safetensors"));
            Hunyuan3DShapeVae vae = new(VaeConfig());
            Dictionary<string, Tensor> weights = loader.GetAllTensors();
            vae.LoadWeights(weights);
            for (int k = 0; k < parts; k++)
            {
                cancel.ThrowIfCancellationRequested();
                using Tensor one = new(new TensorShape(1, LatentTokens, LatentChannels), DType.F32);
                latents.AsSpan<float>().Slice(k * LatentTokens * LatentChannels, LatentTokens * LatentChannels).CopyTo(one.AsSpan<float>());
                using Tensor scaled = new(one.Shape, DType.F32);
                _backend.Scale(scaled, one, 1f / VaeScaleFactor);
                ScalarField3D field = vae.Decode(_backend, scaled, options.GridResolution, options.BoxExtent);
                Mesh part = Hunyuan3DSurface.Extract(field, options.Level);
                result[k] = new Mesh { Vertices = XPartSampling.Denormalize(part.Vertices, center, scale), Indices = part.Indices };
            }
        }
        finally
        {
            latents.Dispose();
        }

        Advance();
        float[][] outBoxes = [.. keptBoxes.Select(b => Denormalize(b, center, scale))];
        return new XPartResult(result, outBoxes, Combine(result, outBoxes));
    }

    // P3-SAM on the normalised mesh: the boxes of its parts.
    private float[][] PredictBoxes(Mesh normalized, XPartOptions options, CancellationToken cancel)
    {
        using P3SamPipeline p3 = new(_backend, Path.Combine(_folder, "p3sam", "p3sam.safetensors"));
        P3SamResult segmented = p3.Generate(normalized, options.Seed, cancel: cancel);
        if (segmented.Boxes.Length == 0)
            throw new HartsyInferenceException("P3-SAM found no parts in the mesh.");
        return segmented.Boxes;
    }

    // [K, Nc, 1024] contexts: the object tokens repeated per part, and every part's own.
    private static (Tensor Object, Tensor Part) Stack(XPartConditioning conditioning, int parts)
    {
        int rows = (int)conditioning.ObjectTokens.Shape[1];
        Tensor objectContext = new(new TensorShape(parts, rows, 1024), DType.F32), partContext = new(new TensorShape(parts, rows, 1024), DType.F32);
        for (int k = 0; k < parts; k++)
        {
            conditioning.ObjectTokens.AsSpan<float>().CopyTo(objectContext.AsSpan<float>().Slice(k * rows * 1024, rows * 1024));
            conditioning.PartTokens[k].AsSpan<float>().CopyTo(partContext.AsSpan<float>().Slice(k * rows * 1024, rows * 1024));
        }

        return (objectContext, partContext);
    }

    // The flow-matching Euler loop over all parts at once: sample += (sigma_next - sigma) * velocity, from sigma 0 (noise) to 1.
    private Tensor Denoise(int parts, int[] ids, Tensor objectContext, Tensor partContext, XPartOptions options, Action advance, CancellationToken cancel)
    {
        Tensor latents = new(new TensorShape(parts, LatentTokens, LatentChannels), DType.F32);
        XPartSampling.Gaussian(latents.ElementCount, options.Seed).CopyTo(latents.AsSpan<float>());
        float[] sigmas = Hunyuan3DFlowSchedule.Sigmas(options.Steps);
        using SafeTensorsLoader loader = new();
        loader.Load(Path.Combine(_folder, "model", "model.safetensors"));
        using XPartDit dit = new();
        dit.LoadWeights(loader.GetAllTensors(), halfWeights: true);
        try
        {
            for (int step = 0; step < options.Steps; step++)
            {
                cancel.ThrowIfCancellationRequested();
                float dt = sigmas[step + 1] - sigmas[step];
                using Tensor velocity = dit.Forward(_backend, latents, Hunyuan3DFlowSchedule.DitTimestep(sigmas[step]), objectContext, partContext, ids);
                using Tensor delta = new(latents.Shape, DType.F32);
                _backend.Scale(delta, velocity, dt);
                Tensor next = new(latents.Shape, DType.F32);
                _backend.Add(next, latents, delta);
                latents.Dispose();
                latents = next;
                advance();
            }
        }
        catch
        {
            latents.Dispose();
            throw;
        }

        return latents;
    }

    // The release's ShapeVAE: 16 decoder blocks of width 1024 and 16 heads, 8 Fourier frequencies, scale factor 1.0039506.
    private static Hunyuan3DConfig VaeConfig() => Hunyuan3DConfig.Hunyuan3D2 with
    {
        VaeWidth = 1024, VaeDepth = 16, VaeNumHeads = 16, FourierBands = 8, VaeScaleFactor = VaeScaleFactor, GeoDownsample = 1, GeoMlpRatio = 4, GeoLnPost = true, BoundingBox = 1.01f,
    };

    private static float[] Denormalize(float[] box, double[] center, double scale)
    {
        float[] result = new float[6];
        for (int c = 0; c < 3; c++)
        {
            result[c] = (float)(box[c] * scale + center[c]);
            result[3 + c] = (float)(box[3 + c] * scale + center[c]);
        }

        return result;
    }

    // One mesh of all the parts with a label per face for the GLB writer.
    private static byte[] Combine(Mesh[] parts, float[][] boxes)
    {
        List<float> vertices = [];
        List<int> indices = [], faceIds = [];
        for (int k = 0; k < parts.Length; k++)
        {
            int offset = vertices.Count / 3;
            vertices.AddRange(parts[k].Vertices);
            indices.AddRange(parts[k].Indices.Select(i => i + offset));
            faceIds.AddRange(Enumerable.Repeat(k, parts[k].TriangleCount));
        }

        Mesh all = new() { Vertices = [.. vertices], Indices = [.. indices] };
        return PartGlbWriter.Write(all, [.. faceIds], [.. Enumerable.Range(0, parts.Length)], boxes);
    }
}
