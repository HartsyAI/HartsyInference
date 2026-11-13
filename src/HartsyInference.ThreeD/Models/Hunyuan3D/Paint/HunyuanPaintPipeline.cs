using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Pipelines;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Paint;
using HartsyInference.ThreeD.Pipelines.Stages;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

/// <summary>Hunyuan3D Paint: textures a UV-mapped mesh from a reference image. It renders normal and position maps from six to twelve views, runs the
/// multiview diffusion over all views at once (generating UNet plus the reference stream), optionally upscales the views, back-projects them onto the
/// UV atlas and inpaints what no view saw. Version 2.0 yields an albedo; 2.1 yields albedo, metallic and roughness.</summary>
public sealed class HunyuanPaintPipeline
{
    private const int FixedViews = 6;
    private readonly IBackend _backend;
    private readonly HunyuanPaintConfig _config;
    private readonly HunyuanPaintUnet _unet, _reference;
    private readonly HunyuanPaintPromptTokens _prompts;
    private readonly IPaintLatentCodec _codec;
    private readonly IPaintConditioner? _conditioner;
    private readonly Func<byte[], int, (byte[] Rgb, int Size)>? _upscale;
    private readonly HunyuanPaintSchedule _schedule = new();

    /// <summary>Receives one line per stage with its wall time (view selection, geometry maps, VAE encode, reference stream, each step, VAE decode, bake).</summary>
    public Action<string>? StageLog { get; set; }

    /// <summary>A diagnostic hook: receives the generated views (material-major square RGB bytes, <c>materials * views</c> of them, at the view resolution) before the bake.</summary>
    public Action<byte[][], int>? GeneratedViewsLog { get; set; }

    private readonly Stopwatch _stage = new();

    private void Stage(string name)
    {
        StageLog?.Invoke($"{name}: {_stage.Elapsed.TotalSeconds:F2} s");
        _stage.Restart();
    }

    /// <summary>Assembles the pipeline from loaded parts. <paramref name="upscale"/> maps a square RGB view to a larger one (Real-ESRGAN x4 upstream; 2.0 skips it).</summary>
    public HunyuanPaintPipeline(IBackend backend, HunyuanPaintConfig config, HunyuanPaintUnet unet, HunyuanPaintUnet reference, HunyuanPaintPromptTokens prompts,
        IPaintLatentCodec codec, IPaintConditioner? conditioner = null, Func<byte[], int, (byte[] Rgb, int Size)>? upscale = null)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _unet = unet ?? throw new ArgumentNullException(nameof(unet));
        _reference = reference ?? throw new ArgumentNullException(nameof(reference));
        _prompts = prompts ?? throw new ArgumentNullException(nameof(prompts));
        _codec = codec ?? throw new ArgumentNullException(nameof(codec));
        if (config.UseDino && conditioner is null)
            throw new ArgumentException("This configuration needs a DINO conditioner.", nameof(conditioner));
        _conditioner = conditioner;
        _upscale = upscale;
    }

    /// <summary>Paints <paramref name="mesh"/> (unwrapped first if it has no UVs) from <paramref name="referenceRgb"/>, a square or any-size RGB image on white.</summary>
    public PbrPostProcessResult Generate(Mesh mesh, ReadOnlySpan<byte> referenceRgb, int referenceWidth, int referenceHeight, HunyuanPaintOptions? options = null,
        Action<GenerationProgress>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        options ??= new HunyuanPaintOptions();
        Validate(options);
        int size = options.ViewResolution;
        _stage.Restart();
        List<string> warnings = [];
        if (mesh.Uvs is null)
        {
            // Charts grow across shared vertices only: a mesh whose vertices are split along every face (a triangle soup, or a decimated mesh with duplicated seam vertices)
            // unwraps into thousands of one-face charts, an atlas that is mostly padding and a bake that is mostly inpainting. Merge them first.
            mesh = WeldForUnwrap(mesh);
            UvUnwrapResult unwrapped = UvUnwrapper.Unwrap(mesh, new UvUnwrapOptions { TextureSize = options.TextureSize }, cancellationToken);
            mesh = unwrapped.Mesh;
            Stage($"uv unwrap ({unwrapped.ChartCount} charts)");
            if (PaintInputAudit.TornMeshWarning(faceCount: mesh.TriangleCount, chartCount: unwrapped.ChartCount) is { } torn)
            {
                StageLog?.Invoke(torn);
                warnings.Add(torn);
            }
        }

        PaintMeshFrame frame = PaintMeshFrame.Create(mesh);
        PaintViewRenderer renderer = new(frame);
        IReadOnlyList<PaintViewCandidate> views = _config.UsePoseRope
            ? PaintViewSelector.Select(renderer, PaintViewCandidate.All, options.MaxViews, cancellationToken: cancellationToken)
            : PaintViewCandidate.Fixed;

        Stage($"frame and views ({views.Count})");
        (byte[][] normalMaps, byte[][] positionMaps) = RenderConditions(renderer, views, options, cancellationToken);
        Stage("geometry maps");
        byte[][] generated = Diffuse(views, normalMaps, positionMaps, referenceRgb, referenceWidth, referenceHeight, options, progress, cancellationToken);
        PbrPostProcessResult baked = Bake(mesh, frame, renderer, views, generated, options, cancellationToken);
        Stage("bake");
        return warnings.Count == 0 ? baked : baked with { Warnings = warnings };
    }

    /// <summary>Merges vertices at the same position (exact match) and drops degenerate and duplicate faces and unreferenced vertices, and does nothing else to the surface (no hole filling,
    /// no component removal, no reorientation), so that the unwrapper sees the mesh's real connectivity. Like every <c>MeshCleaner</c> result it holds positions and indices only: supplied
    /// normals and vertex colours are dropped (the bake replaces the colours, the renderer uses face normals). An edge shared by three or more faces stays; the unwrapper treats it as a boundary.</summary>
    internal static Mesh WeldForUnwrap(Mesh mesh) => MeshCleaner.Clean(mesh, new MeshCleanOptions
    {
        WeldVertices = true,
        RemoveDegenerateFaces = true,
        RemoveDuplicateFaces = true,
        RepairNonManifoldEdges = false,
        SmallComponentAreaRatio = 0,
        HoleFillMaxPerimeter = 0,
        UnifyOrientation = false,
        OrientOutward = false,
    });

    /// <summary>Encodes and decodes one image with the pipeline's VAE (a diagnostic for checking the VAE weights and timing): resizes to <paramref name="size"/>
    /// square, reports the latent statistics through <see cref="StageLog"/> and returns the decoded RGB bytes.</summary>
    public byte[] VaeRoundTrip(ReadOnlySpan<byte> rgb, int width, int height, int size)
    {
        byte[] resized = HunyuanPaintImages.Resize(rgb, width, height, size, size);
        using Tensor image = HunyuanPaintImages.ToTensor([resized], size);
        _stage.Restart();
        using Tensor latent = _codec.Encode(_backend, image);
        Stage($"vae encode {size} px, latent {latent.Shape}");
        float[] values = HunyuanPaintImages.ToArray(latent);
        double mean = values.Average(), std = Math.Sqrt(values.Average(v => (v - mean) * (v - mean)));
        StageLog?.Invoke($"latent mean {mean:F3} std {std:F3} finite {values.All(float.IsFinite)}");
        _stage.Restart();
        using Tensor decoded = _codec.Decode(_backend, latent);
        Stage("vae decode");
        return HunyuanPaintImages.ToBytes(decoded)[0];
    }

    /// <summary>The 2.0 camera class of a view (<c>camera_info</c> plus the five reference slots the UNet skips).</summary>
    public static int CameraClass(double elevation, double azimuth)
    {
        (int divisor, int offset) = (int)elevation switch
        {
            -20 => (1, 0), 0 => (1, 12), 20 => (1, 24), -90 => (3, 36), 90 => (3, 40),
            _ => throw new ArgumentOutOfRangeException(nameof(elevation), "Paint 2.0 knows elevations -90, -20, 0, 20 and 90."),
        };
        int slot = (int)Math.Floor(azimuth / 30.0);
        return (((slot + 9) % 12 + 12) % 12) / divisor + offset + 5;
    }

    /// <summary>The 2.1 guidance scale of a view by azimuth: stronger away from the front.</summary>
    public static float ViewScale(double azimuth) => azimuth >= 0 && azimuth < 90 ? (float)(azimuth / 90.0 + 1.0) : azimuth >= 90 && azimuth < 330 ? 2f : (float)(-azimuth / 90.0 + 5.0);

    private void Validate(HunyuanPaintOptions options)
    {
        if (options.ViewResolution < 64 || options.ViewResolution % 64 != 0)
            throw new ArgumentException("The view resolution must be a multiple of 64 (the checkpoints were trained at 512 and 768).", nameof(options));
        if (options.MaxViews < FixedViews || options.MaxViews > 12)
            throw new ArgumentException("MaxViews must be between 6 and 12.", nameof(options));
        if (options.RenderResolution < options.ViewResolution)
            throw new ArgumentException("Render resolution must be at least the view resolution.", nameof(options));
        if (options.TextureSize < 16)
            throw new ArgumentException("Texture size must be at least 16.", nameof(options));
    }

    private (byte[][] Normal, byte[][] Position) RenderConditions(PaintViewRenderer renderer, IReadOnlyList<PaintViewCandidate> views, HunyuanPaintOptions o,
        CancellationToken cancellationToken)
    {
        byte[][] normal = new byte[views.Count][], position = new byte[views.Count][];
        for (int v = 0; v < views.Count; v++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            normal[v] = HunyuanPaintImages.Resize(renderer.RenderNormalMap(views[v].Elevation, views[v].Azimuth, o.RenderResolution, cancellationToken),
                o.RenderResolution, o.RenderResolution, o.ViewResolution, o.ViewResolution);
            position[v] = HunyuanPaintImages.Resize(renderer.RenderPositionMap(views[v].Elevation, views[v].Azimuth, o.RenderResolution, cancellationToken),
                o.RenderResolution, o.RenderResolution, o.ViewResolution, o.ViewResolution);
        }
        return (normal, position);
    }

    // Returns the generated views, material-major: [material][view] as square RGB bytes at the view resolution.
    private byte[][] Diffuse(IReadOnlyList<PaintViewCandidate> views, byte[][] normalMaps, byte[][] positionMaps, ReadOnlySpan<byte> referenceRgb, int referenceWidth,
        int referenceHeight, HunyuanPaintOptions o, Action<GenerationProgress>? progress, CancellationToken cancellationToken)
    {
        int n = views.Count, size = o.ViewResolution, latent = size / 8, materials = _config.MaterialCount;
        byte[] reference = HunyuanPaintImages.Resize(referenceRgb, referenceWidth, referenceHeight, size, size);
        int steps = o.Steps ?? (_config.UsePoseRope ? 15 : 30);
        float guidance = o.GuidanceScale ?? (_config.UsePoseRope ? 3f : 2f);
        // The solver keeps multistep history, so every material walks its own copy (upstream steps one stacked tensor, which is the same thing).
        IHunyuanPaintSampler[] samplers = [.. Enumerable.Range(0, materials).Select(_ => _config.UsePoseRope
            ? (IHunyuanPaintSampler)new HunyuanPaintUniPcSampler(_schedule, steps) : new HunyuanPaintEulerAncestralSampler(_schedule, steps))];
        IHunyuanPaintSampler sampler = samplers[0];

        using Tensor normalImages = HunyuanPaintImages.ToTensor(normalMaps, size), positionImages = HunyuanPaintImages.ToTensor(positionMaps, size);
        using Tensor referenceImage = HunyuanPaintImages.ToTensor([reference], size);
        float[] normalLatent, positionLatent;
        using (Tensor t = _codec.Encode(_backend, normalImages))
            normalLatent = HunyuanPaintImages.ToArray(t);
        using (Tensor t = _codec.Encode(_backend, positionImages))
            positionLatent = HunyuanPaintImages.ToArray(t);
        using Tensor referenceLatent = _codec.Encode(_backend, referenceImage);
        Stage("vae encode (normal, position, reference)");

        HunyuanPaintBlockState write = new() { Write = true, Text = [_prompts.Reference] };
        Dictionary<int, (Tensor Cos, Tensor Sin)>? rotation = null;
        Tensor? dino = null, zeroDino = null;
        try
        {
            int[] referenceClass = _config.CameraClasses > 0 ? [0] : [];
            foreach (Tensor t in _reference.Forward(_backend, [referenceLatent], 0f, referenceClass, write))
                t.Dispose();
            Stage("reference stream");

            dino = _conditioner?.Tokens(_backend, reference, size, size);
            zeroDino = _conditioner?.UnconditionalTokens(_backend);
            rotation = _config.UsePoseRope ? BuildRotation(positionMaps, n, size, latent, _config.BlockOutChannels[0] / _config.Heads[0]) : null;
            // The reference stream, the VAE encoder and DINO are done: sweep their cached weights and activations so the sampling UNet does not start on a
            // saturated card (which runs several times slower without failing). What later steps still read is offloaded to the host first.
            DeviceSweep(write.Embeds.Values.Append(dino).Append(zeroDino).Where(t => t is not null).Select(t => t!));
            int[] classes = _config.CameraClasses > 0 ? [.. views.Select(v => CameraClass(v.Elevation, v.Azimuth))] : [];
            // 2.1 guides u + g s (r - u) + g s (f - r); with one s per view the reference-only prediction r cancels, so only (u, f) are run.
            (float Scale, bool Dino)[] branches = _config.UsePoseRope ? [(0f, false), (1f, true)] : [(0f, false), (1f, false)];

            Random random = o.Seed is int seed ? new Random(seed) : new Random();
            long perMaterial = (long)n * 4 * latent * latent;
            float[][] latents = new float[materials][];
            for (int m = 0; m < materials; m++)
            {
                latents[m] = new float[perMaterial];
                for (long i = 0; i < perMaterial; i++)
                    latents[m][i] = (float)(Gaussian(random) * sampler.InitialNoiseScale);
            }

            DateTime started = DateTime.UtcNow;
            for (int step = 0; step < steps; step++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                float[][][] predictions = new float[branches.Length][][];
                for (int b = 0; b < branches.Length; b++)
                {
                    Tensor?[] inputs = new Tensor?[materials];
                    try
                    {
                        for (int m = 0; m < materials; m++)
                            inputs[m] = Concatenate(samplers[m].ScaleModelInput(latents[m], step), normalLatent, positionLatent, n, latent);
                        HunyuanPaintBlockState state = new()
                        {
                            Embeds = write.Embeds, ReferenceScale = branches[b].Scale, Text = _prompts.PerMaterial, Dino = branches[b].Dino ? dino : zeroDino, Rotation = rotation,
                        };
                        Tensor[] outputs = _unet.Forward(_backend, [.. inputs!], sampler.Timesteps[step], classes, state);
                        try
                        {
                            predictions[b] = outputs.Select(HunyuanPaintImages.ToArray).ToArray();
                        }
                        finally
                        {
                            foreach (Tensor t in outputs)
                                t.Dispose();
                        }
                    }
                    finally
                    {
                        foreach (Tensor? t in inputs)
                            t?.Dispose();
                    }
                }
                for (int m = 0; m < materials; m++)
                {
                    float[] guided = Guide(predictions.Select(p => p[m]).ToArray(), views, guidance, n, latent);
                    latents[m] = samplers[m].Step(guided, latents[m], step, _config.UsePoseRope ? null : GaussianArray(random, perMaterial));
                }
                Stage($"step {step + 1}/{steps}");
                progress?.Invoke(new GenerationProgress(step + 1, steps, (DateTime.UtcNow - started).TotalMilliseconds));
            }

            DeviceSweep([]); // the UNets are done: free their cached weights before the VAE decodes
            byte[][] generated = new byte[materials * n][];
            for (int m = 0; m < materials; m++)
            {
                using Tensor latentTensor = HunyuanPaintImages.FromArray(latents[m], n, 4, latent, latent);
                using Tensor decoded = _codec.Decode(_backend, latentTensor);
                byte[][] images = HunyuanPaintImages.ToBytes(decoded);
                for (int v = 0; v < n; v++)
                    generated[m * n + v] = images[v];
            }
            Stage("vae decode");
            GeneratedViewsLog?.Invoke(generated, size);
            return generated;
        }
        finally
        {
            dino?.Dispose();
            zeroDino?.Dispose();
            foreach (Tensor t in write.Embeds.Values)
                t.Dispose();
            if (rotation is not null)
                foreach ((Tensor cos, Tensor sin) in rotation.Values)
                {
                    cos.Dispose();
                    sin.Dispose();
                }
        }
    }

    internal static Dictionary<int, (Tensor Cos, Tensor Sin)> BuildRotation(byte[][] positionMaps, int views, int size, int latent, int headDim)
    {
        Dictionary<int, (Tensor Cos, Tensor Sin)> tables = [];
        byte[] joined = new byte[positionMaps.Sum(p => p.Length)];
        int offset = 0;
        foreach (byte[] map in positionMaps)
        {
            map.CopyTo(joined, offset);
            offset += map.Length;
        }
        for (int level = 0; level < 4; level++)
        {
            int grid = latent >> level, voxels = latent * 8 >> level;
            int[] indices = HunyuanPaintRope.VoxelIndices(joined, views, size, grid, voxels);
            tables[views * grid * grid] = HunyuanPaintRope.TableTensors(indices, headDim);
        }
        return tables;
    }

    private Tensor Concatenate(float[] sample, float[] normal, float[] position, int views, int latent)
    {
        long plane = (long)latent * latent * 4;
        float[] merged = new float[views * plane * 3];
        for (int v = 0; v < views; v++)
        {
            Array.Copy(sample, v * plane, merged, v * plane * 3, plane);
            Array.Copy(normal, v * plane, merged, v * plane * 3 + plane, plane);
            Array.Copy(position, v * plane, merged, v * plane * 3 + plane * 2, plane);
        }
        return HunyuanPaintImages.FromArray(merged, views, 12, latent, latent);
    }

    // u + g s (b - u) per view with s from the azimuth (2.1: b is the full prediction, see Diffuse) and s = 1 for 2.0.
    private float[] Guide(float[][] branch, IReadOnlyList<PaintViewCandidate> views, float guidance, int n, int latent)
    {
        float[] scales = new float[n];
        for (int v = 0; v < n; v++)
            scales[v] = _config.UsePoseRope ? ViewScale(views[v].Azimuth) : 1f;
        return Guide(branch, scales, guidance, n, latent);
    }

    // Per view v, result = u + guidance * viewScales[v] * (b - u) over the view's latent plane.
    internal static float[] Guide(float[][] branch, float[] viewScales, float guidance, int n, int latent)
    {
        int plane = latent * latent * 4;
        float[] result = new float[branch[0].Length];
        for (int v = 0; v < n; v++)
        {
            float s = guidance * viewScales[v];
            for (int i = v * plane; i < (v + 1) * plane; i++)
                result[i] = branch[0][i] + s * (branch[1][i] - branch[0][i]);
        }
        return result;
    }

    private void DeviceSweep(IEnumerable<Tensor> live)
    {
        Trellis2DeviceMemory.ReleaseAll(_backend, [.. live]);
        Stage("device sweep");
    }

    // Texels more than GutterReach (a fixed texel count, measured as a square: up to about 1.4 times that on the diagonals) from any texel the mesh covers
    // never show on it, and the gradient fill runs away there, so they take the mean colour of the covered texels. Covered texels keep their inpainted values.
    private const int GutterReach = 12;

    internal static void FlattenFarGutter(float[] texture, PaintTexelSurface surface, int size)
    {
        bool[] covered = new bool[size * size];
        double[] mean = new double[3];
        foreach (int texel in surface.Texels)
        {
            covered[texel] = true;
            for (int c = 0; c < 3; c++)
                mean[c] += texture[texel * 3 + c];
        }
        if (surface.Count == 0)
            return;
        bool[] near = PaintImageFilters.Dilate(covered, size, size, GutterReach);
        for (int i = 0; i < near.Length; i++)
            if (!near[i])
                for (int c = 0; c < 3; c++)
                    texture[i * 3 + c] = (float)(mean[c] / surface.Count);
    }

    private PbrPostProcessResult Bake(Mesh mesh, PaintMeshFrame frame, PaintViewRenderer renderer, IReadOnlyList<PaintViewCandidate> views, byte[][] generated,
        HunyuanPaintOptions o, CancellationToken cancellationToken)
    {
        int n = views.Count, materials = _config.MaterialCount, size = o.TextureSize, render = o.RenderResolution;
        PaintTexelSurface surface = PaintTexelSurface.Create(frame, mesh.Uvs!, size, size, cancellationToken);
        float[][] textures = new float[materials][];
        int painted = 0;
        for (int m = 0; m < materials; m++)
        {
            PaintTextureMerger merger = new(surface, 3, o.Bake);
            for (int v = 0; v < n; v++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                byte[] view = generated[m * n + v];
                int side = o.ViewResolution;
                if (_upscale is not null)
                    (view, side) = _upscale(view, side);
                view = HunyuanPaintImages.Resize(view, side, side, render, render);
                float[] image = new float[view.Length];
                for (int i = 0; i < image.Length; i++)
                    image[i] = view[i] / 255f;
                PaintViewBake bake = PaintBackProjector.Project(renderer, surface, image, render, 3, views[v].Elevation, views[v].Azimuth, o.Bake, cancellationToken);
                merger.Add(bake, views[v].Weight);
            }
            (float[] texture, bool[] known) = merger.Finish();
            if (m == 0)
                painted = known.Count(k => k);
            textures[m] = Inpainter.Inpaint(texture, size, size, 3, known, o.InpaintRadius, clampToUnit: true);
            FlattenFarGutter(textures[m], surface, size);
        }

        TextureImage baseColor = TextureImage.FromFloats(textures[0], size, size, 3, 0, 3);
        TextureImage metallic = materials > 1 ? TextureImage.FromFloats(textures[1], size, size, 3, 0, 1) : new TextureImage(size, size, 1, new byte[size * size]);
        TextureImage roughness = materials > 1 ? TextureImage.FromFloats(textures[1], size, size, 3, 1, 1) : new TextureImage(size, size, 1, Enumerable.Repeat((byte)255, size * size).ToArray());
        PbrMaterial material = new() { Name = "paint", BaseColor = baseColor, Metallic = metallic, Roughness = roughness };
        return new PbrPostProcessResult(mesh, material, painted);
    }

    private static double Gaussian(Random random)
    {
        double u = 1.0 - random.NextDouble(), v = random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u)) * Math.Cos(2.0 * Math.PI * v);
    }

    private static float[] GaussianArray(Random random, long count)
    {
        float[] noise = new float[count];
        for (long i = 0; i < count; i++)
            noise[i] = (float)Gaussian(random);
        return noise;
    }
}
