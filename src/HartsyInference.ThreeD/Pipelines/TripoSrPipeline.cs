using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Requests;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Geometry.Ops;
using HartsyInference.ThreeD.Models.TripoSr;
using HartsyInference.ThreeD.Pipelines.Requests;
using HartsyInference.Vision.Dinov2;
using HartsyInference.Vision.DinoVit;

namespace HartsyInference.ThreeD.Pipelines;

/// <summary>TripoSR single-image → mesh pipeline: a DINO ViT tokenizes the image, the triplane transformer produces a <see cref="Triplane"/>, and the NeRF decoder yields a density field that marching cubes turns into a colored mesh, feed-forward and deterministic with no diffusion loop.</summary>
public sealed unsafe class TripoSrPipeline : ThreeDPipelineBase
{
    private readonly DinoViTEncoder _dino;
    private readonly Dinov2ImagePreprocessor _preprocessor;
    private readonly TripoSrTransformer _transformer;
    private readonly TriplaneNerfDecoder _decoder;
    private readonly TripoSrConfig _cfg;
    private readonly List<IDisposable> _ownedLoaders = [];

    public TripoSrPipeline(IBackend backend, DinoViTEncoder dino, TripoSrTransformer transformer, TriplaneNerfDecoder decoder, TripoSrConfig cfg)
        : base(backend)
    {
        _dino = dino;
        _transformer = transformer;
        _decoder = decoder;
        _cfg = cfg;
        _preprocessor = new Dinov2ImagePreprocessor(dino.Config.ImageSize);
    }

    /// <summary>Loads a TripoSR pipeline from a local checkpoint path, defaulting to <see cref="TripoSrConfig.TripoSr"/> + the <c>facebook/dino-vitb16</c> tokenizer at 512px.</summary>
    public static TripoSrPipeline LoadFromPath(IBackend backend, string modelPath, TripoSrConfig? cfg = null, DinoViTConfig? dinoCfg = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        TripoSrConfig config = cfg ?? TripoSrConfig.TripoSr;
        DinoViTConfig dp = dinoCfg ?? DinoViTConfig.DinoVitB16_512;

        string[] files = Directory.Exists(modelPath)
            ? Directory.GetFiles(modelPath, "*.safetensors", SearchOption.AllDirectories)
            : [modelPath];
        if (files.Length == 0) throw new FileNotFoundException($"No .safetensors found under '{modelPath}'.");

        List<IDisposable> loaders = [];
        Dictionary<string, Tensor> all = [];
        foreach (string f in files.OrderBy(s => s, StringComparer.Ordinal))
        {
            SafeTensorsLoader loader = new();
            loader.Load(f);
            loaders.Add(loader);
            foreach ((string k, Tensor t) in loader.GetAllTensors()) all[k] = t;
        }

        TripoSrCheckpointConverter.ConvertedWeights w = TripoSrCheckpointConverter.Convert(all);
        DinoViTEncoder dino = new(dp); dino.LoadWeights(w.Dino);
        TripoSrTransformer transformer = new(config); transformer.LoadWeights(w.Transformer);
        TriplaneNerfDecoder decoder = new(config); decoder.LoadWeights(w.Decoder);

        TripoSrPipeline pipeline = new(backend, dino, transformer, decoder, config);
        pipeline._ownedLoaders.AddRange(loaders);
        return pipeline;
    }

    /// <summary>Generates a colored mesh from a single conditioning image (deterministic).</summary>
    /// <remarks><b>Input contract:</b> <see cref="ImageTo3DRequest.ImageRgb"/> must already be a foreground-isolated
    /// image on a neutral gray (0.5) background, matching TripoSR's <c>run.py</c> (rembg background removal →
    /// <c>resize_foreground(0.85)</c> → composite <c>rgb·α + (1−α)·0.5</c>). Passing a raw photo with a real
    /// background produces a degenerate/noisy mesh. During e2e validation this compositing was done in Python.
    /// TODO(3D/no-python): implement a pure-C# foreground preprocessor so the app never shells out to Python —
    /// (1) a salient-object-segmentation model in HartsyInference.Vision (U²-Net / ISNet / BiRefNet → alpha mask),
    /// (2) a <c>ForegroundComposite</c> helper here (bbox-crop → pad-to-square → resize to ratio → gray-0.5
    /// composite). Then call it here when the request flags a raw/un-composited image. Tracked in
    /// docs/Checklists/PHASE_11_THREED.md §5.</remarks>
    public ThreeDResult Generate(ImageTo3DRequest request, Action<GenerationProgress>? onProgress = null)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(request);
        int gridRes = request.GridResolution > 0 ? request.GridResolution : _cfg.GridResolution;
        bool phase = Environment.GetEnvironmentVariable("HARTSY_3D_PHASE") == "1";
        System.Diagnostics.Stopwatch pw = System.Diagnostics.Stopwatch.StartNew();
        void Probe(string tag) { if (phase) { Backend.Sync(); Console.WriteLine($"[triposr-phase] {tag}: {pw.ElapsedMilliseconds} ms"); pw.Restart(); } }

        // 1. Image → DINO tokens. (request.ImageRgb is assumed foreground-on-gray; see the TODO in the doc comment.)
        Tensor pixels = _preprocessor.Preprocess(request.ImageRgb, request.Width, request.Height);
        Backend.PreloadWeights(_dino.EnumerateWeights());
        Tensor tokens = _dino.Encode(Backend, pixels);
        pixels.Dispose();
        Backend.FreeWeights(_dino.EnumerateWeights());
        Probe("dino-encode");
        onProgress?.Invoke(new GenerationProgress(1, 3, 0));

        // 2. Tokens → triplane.
        Backend.PreloadWeights(_transformer.EnumerateWeights());
        Triplane tri = _transformer.Forward(Backend, tokens);
        Backend.FreeWeights(_transformer.EnumerateWeights());
        tokens.Dispose();
        Probe("transformer");
        onProgress?.Invoke(new GenerationProgress(2, 3, 0));

        // 3. Triplane → density field → mesh.
        Backend.PreloadWeights(_decoder.EnumerateWeights());
        ScalarField3D density = _decoder.DecodeDensityField(Backend, tri, gridRes);
        Probe($"density-grid ({gridRes}^3)");
        float threshold = request.IsoLevel != 0f ? request.IsoLevel : _cfg.DensityThreshold;

        // Marching cubes treats "inside" as value < iso; the surface is density > threshold, so extract on
        // the negated field at -threshold (keeps outward-facing normals).
        float[] neg = new float[density.Values.Length];
        for (int i = 0; i < neg.Length; i++) neg[i] = -density.Values[i];
        ScalarField3D occ = new() { Values = neg, ResX = density.ResX, ResY = density.ResY, ResZ = density.ResZ, Min = density.Min, Max = density.Max };
        Mesh mesh = MeshOps.ComputeVertexNormals(MarchingCubes.Extract(occ, -threshold));
        Probe("marching-cubes");

        // 4. Per-vertex colors from the decoder.
        if (mesh.TriangleCount > 0)
            mesh.VertexColors = _decoder.DecodeColors(Backend, tri, mesh.Vertices, mesh.VertexCount);
        Backend.FreeWeights(_decoder.EnumerateWeights());
        Probe("vertex-colors");
        onProgress?.Invoke(new GenerationProgress(3, 3, 0));

        return new ThreeDResult { Mesh = mesh, Seed = request.Seed ?? 0 };
    }

    /// <inheritdoc/>
    protected override void DisposeCore()
    {
        foreach (IDisposable d in _ownedLoaders) d.Dispose();
        _ownedLoaders.Clear();
    }
}
