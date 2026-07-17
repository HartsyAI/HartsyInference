using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Segmentation.Sam2;

/// <summary>SAM 2's image encoder: a Hiera (hierarchical windowed-attention ViT) trunk followed by an FPN
/// neck. The trunk patch-embeds a <c>[1,3,S,S]</c> image (7×7 stride-4 conv → <c>S/4</c> tokens), runs four
/// stages of <see cref="HieraBlock"/> multi-scale blocks (each stage halves spatial resolution via a pooled
/// query and doubles the embed dim + head count), then the neck projects every stage to
/// <see cref="Sam2Config.PromptEmbedDim"/> channels and fuses them top-down. The returned embedding is the
/// stride-16 level (<c>S/16 × S/16</c>, 256-ch for a 1024 input) the mask decoder consumes.
///
/// <para><b>Parity note:</b> key names follow the Meta <c>sam2</c> checkpoint layout
/// (<c>trunk.patch_embed.proj.*</c>, <c>trunk.blocks.{i}.{norm1,attn.qkv,attn.proj,norm2,mlp.layers.0,mlp.layers.1}</c>,
/// <c>trunk.pos_embed[_window]</c>, <c>neck.convs.{j}.conv.*</c>). Two simplifications vs. the reference
/// are documented inline and are correctness-unverified against real weights: (1) the per-block
/// <c>global_att_blocks</c> override is not modeled — window size comes purely from the per-stage
/// <see cref="Sam2Config.EncoderWindowSizes"/>; (2) positional-embed interpolation uses bilinear where the
/// reference uses bicubic.</para></summary>
public sealed unsafe class HieraImageEncoder : ISamImageEncoder
{
    private const float LnEps = 1e-6f;
    private const int PatchKernel = 7;
    private const int PatchStride = 4;
    private const int PatchPad = 3;
    private const int MlpRatio = 4;

    private readonly int _embedDim;
    private readonly int _promptDim;
    private readonly int[] _stageEnds;
    private readonly HieraBlock[] _blocks;

    private Tensor? _patchProjW;   // [embedDim, 3, 7, 7]
    private Tensor? _patchProjB;   // [embedDim]
    private Tensor? _posEmbed;     // [1, embedDim, ph, pw]  (background grid, interpolated to token grid)
    private Tensor? _posEmbedWindow; // [1, embedDim, w, w]  (tiled over the token grid)

    // Neck 1×1 lateral convs, indexed 0=finest stage(0) .. 3=coarsest stage(3). The real checkpoint orders
    // neck.convs by descending resolution (channel list [8C,4C,2C,C]), so stage s maps to key index (3 - s).
    private readonly Tensor?[] _neckConvW = new Tensor?[4];
    private readonly Tensor?[] _neckConvB = new Tensor?[4];

    /// <inheritdoc/>
    public string EncoderName => "hiera";

    /// <inheritdoc/>
    public int FeatureStride => 16;

    /// <summary>Builds the block schedule (per-block dim/head/window/pool metadata) from the config; weights
    /// are loaded separately via <see cref="LoadWeights"/>.</summary>
    public HieraImageEncoder(Sam2Config config)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.EncoderStageDepths.Length != 4)
            throw new ArgumentException($"Hiera expects 4 stages; got {config.EncoderStageDepths.Length}.", nameof(config));

        _embedDim = config.EncoderEmbedDim;
        _promptDim = config.PromptEmbedDim;

        int[] depths = config.EncoderStageDepths;
        int depth = depths[0] + depths[1] + depths[2] + depths[3];
        _stageEnds = new int[4];
        int running = 0;
        for (int s = 0; s < 4; s++)
        {
            running += depths[s];
            _stageEnds[s] = running - 1;
        }

        // First block of stages 1..3 pools the query (q_stride = 2) — matches Hiera q_pool = 3.
        HashSet<int> qPoolBlocks = new() { _stageEnds[0] + 1, _stageEnds[1] + 1, _stageEnds[2] + 1 };

        _blocks = new HieraBlock[depth];
        int curStage = 1;
        int dim = _embedDim;
        int heads = config.EncoderNumHeads;
        for (int i = 0; i < depth; i++)
        {
            int dimOut = dim;
            // Window size uses the stage index BEFORE a stage transition increments it (Hiera ordering).
            int window = config.EncoderWindowSizes[curStage - 1];
            if (i - 1 >= 0 && Array.IndexOf(_stageEnds, i - 1) >= 0)
            {
                dimOut = dim * 2;
                heads *= 2;
                curStage++;
            }
            bool qPool = qPoolBlocks.Contains(i);
            _blocks[i] = new HieraBlock(dim, dimOut, heads, window, qPool, MlpRatio, LnEps);
            dim = dimOut;
        }
    }

    /// <inheritdoc/>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        ArgumentNullException.ThrowIfNull(weights);
        string p = prefix ?? string.Empty;

        _patchProjW = EnsureF32(Get(weights, $"{p}trunk.patch_embed.proj.weight"));
        _patchProjB = EnsureF32(Get(weights, $"{p}trunk.patch_embed.proj.bias"));

        // Positional embeds are optional (absent in some exported variants); skip cleanly when missing.
        if (weights.TryGetValue($"{p}trunk.pos_embed", out Tensor? pe)) _posEmbed = EnsureF32(pe);
        if (weights.TryGetValue($"{p}trunk.pos_embed_window", out Tensor? pw)) _posEmbedWindow = EnsureF32(pw);

        for (int i = 0; i < _blocks.Length; i++)
        {
            _blocks[i].LoadWeights(weights, $"{p}trunk.blocks.{i}.");
        }

        for (int s = 0; s < 4; s++)
        {
            _neckConvW[s] = EnsureF32(Get(weights, $"{p}neck.convs.{3 - s}.conv.weight"));
            _neckConvB[s] = EnsureF32(Get(weights, $"{p}neck.convs.{3 - s}.conv.bias"));
        }
    }

    /// <inheritdoc/>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_patchProjW is not null) yield return _patchProjW;
        if (_patchProjB is not null) yield return _patchProjB;
        if (_posEmbed is not null) yield return _posEmbed;
        if (_posEmbedWindow is not null) yield return _posEmbedWindow;
        foreach (HieraBlock b in _blocks)
            foreach (Tensor t in b.EnumerateWeights())
                yield return t;
        for (int s = 0; s < 4; s++)
        {
            if (_neckConvW[s] is not null) yield return _neckConvW[s]!;
            if (_neckConvB[s] is not null) yield return _neckConvB[s]!;
        }
    }

    /// <inheritdoc/>
    public Tensor Forward(IBackend backend, Tensor pixelValues)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (pixelValues.Shape.Rank != 4 || pixelValues.Shape[0] != 1 || pixelValues.Shape[1] != 3)
            throw new ArgumentException($"Hiera expects pixel values [1,3,H,W]; got {pixelValues.Shape}.", nameof(pixelValues));
        if (_patchProjW is null || _patchProjB is null)
            throw new InvalidOperationException("HieraImageEncoder weights not loaded.");

        int inH = (int)pixelValues.Shape[2];
        int inW = (int)pixelValues.Shape[3];
        int hp = (inH + 2 * PatchPad - PatchKernel) / PatchStride + 1;
        int wp = (inW + 2 * PatchPad - PatchKernel) / PatchStride + 1;

        Tensor patch = new Tensor(new TensorShape(1, _embedDim, hp, wp), DType.F32);
        backend.Conv2D(patch, pixelValues, _patchProjW!, _patchProjB, PatchStride, PatchStride, PatchPad, PatchPad);

        Tensor tokens = NchwToNhwc(backend, patch, 1, _embedDim, hp, wp);
        tokens = AddPosEmbed(backend, tokens, hp, wp);

        Tensor[] stageFeats = new Tensor[4];
        int[] stageH = new int[4];
        int[] stageW = new int[4];
        int stageIdx = 0;

        Tensor x = tokens;
        int curH = hp, curW = wp;
        for (int i = 0; i < _blocks.Length; i++)
        {
            (x, curH, curW) = _blocks[i].Forward(backend, x, curH, curW);
            if (i == _stageEnds[stageIdx])
            {
                stageFeats[stageIdx] = NhwcToNchw(backend, x, 1, curH, curW, _blocks[i].DimOut);
                stageH[stageIdx] = curH;
                stageW[stageIdx] = curW;
                stageIdx++;
            }
        }

        return NeckForward(backend, stageFeats, stageH, stageW);
    }

    /// <summary>FPN neck: 1×1 lateral convs to <see cref="_promptDim"/> channels, then top-down fusion
    /// (upsample the coarser level, add). Returns the stride-16 level (stage index 2), which is the SAM 2
    /// image embedding.</summary>
    private Tensor NeckForward(IBackend backend, Tensor[] stageFeats, int[] stageH, int[] stageW)
    {
        Tensor[] lateral = new Tensor[4];
        for (int s = 0; s < 4; s++)
        {
            lateral[s] = new Tensor(new TensorShape(1, _promptDim, stageH[s], stageW[s]), DType.F32);
            backend.Conv2D(lateral[s], stageFeats[s], _neckConvW[s]!, _neckConvB[s]!, 1, 1, 0, 0);
        }

        Tensor prev = lateral[3];
        Tensor[] fused = new Tensor[4];
        fused[3] = prev;
        for (int s = 2; s >= 0; s--)
        {
            Tensor up = new Tensor(new TensorShape(1, _promptDim, stageH[s], stageW[s]), DType.F32);
            backend.InterpolateBilinear2D(up, prev, alignCorners: false);
            Tensor sum = new Tensor(up.Shape, DType.F32);
            backend.Add(sum, lateral[s], up);
            prev = sum;
            fused[s] = sum;
        }

        return fused[2];
    }

    /// <summary>Interpolates the background pos-embed to the token grid, adds the tiled window pos-embed, and
    /// adds the result (as NHWC) to the tokens. No-op when pos-embeds were not loaded.</summary>
    private Tensor AddPosEmbed(IBackend backend, Tensor tokens, int hp, int wp)
    {
        if (_posEmbed is null || _posEmbedWindow is null)
            return tokens;

        Tensor interp = new Tensor(new TensorShape(1, _embedDim, hp, wp), DType.F32);
        backend.InterpolateBilinear2D(interp, _posEmbed, alignCorners: false);

        Tensor tiled = TileWindow(_posEmbedWindow, _embedDim, hp, wp);

        Tensor grid = new Tensor(new TensorShape(1, _embedDim, hp, wp), DType.F32);
        backend.Add(grid, interp, tiled);

        Tensor gridNhwc = NchwToNhwc(backend, grid, 1, _embedDim, hp, wp);
        Tensor outT = new Tensor(tokens.Shape, DType.F32);
        backend.Add(outT, tokens, gridNhwc);
        return outT;
    }

    /// <summary>Tiles a <c>[1,C,wh,ww]</c> window pos-embed over a <c>[1,C,H,W]</c> grid by wrapping indices —
    /// host data assembly of a loaded weight (no math), robust to non-divisible sizes.</summary>
    private static Tensor TileWindow(Tensor window, int c, int h, int w)
    {
        int wh = (int)window.Shape[2];
        int ww = (int)window.Shape[3];
        Tensor outT = new Tensor(new TensorShape(1, c, h, w), DType.F32);
        float* src = (float*)window.DataPointer;
        float* dst = (float*)outT.DataPointer;
        for (int ch = 0; ch < c; ch++)
        {
            for (int i = 0; i < h; i++)
            {
                for (int j = 0; j < w; j++)
                {
                    dst[((long)ch * h + i) * w + j] = src[((long)ch * wh + i % wh) * ww + j % ww];
                }
            }
        }
        return outT;
    }

    // ── Layout helpers (route the transpose through the backend; reshapes are zero-copy views) ──

    /// <summary>NCHW → NHWC via a batched transpose of the flattened spatial axis.</summary>
    internal static Tensor NchwToNhwc(IBackend backend, Tensor x, int b, int c, int h, int w)
    {
        Tensor v = x.Reshape(new TensorShape(b, c, h * w));
        Tensor t = new Tensor(new TensorShape(b, h * w, c), DType.F32);
        backend.Transpose2D(t, v, c, h * w);
        return t.Reshape(new TensorShape(b, h, w, c));
    }

    /// <summary>NHWC → NCHW via a batched transpose of the flattened spatial axis.</summary>
    internal static Tensor NhwcToNchw(IBackend backend, Tensor x, int b, int h, int w, int c)
    {
        Tensor v = x.Reshape(new TensorShape(b, h * w, c));
        Tensor t = new Tensor(new TensorShape(b, c, h * w), DType.F32);
        backend.Transpose2D(t, v, h * w, c);
        return t.Reshape(new TensorShape(b, c, h, w));
    }

    private static Tensor EnsureF32(Tensor t) => t.DType != DType.F32 ? t.CastTo(DType.F32) : t;

    private static Tensor Get(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        if (!weights.TryGetValue(key, out Tensor? t))
            throw new KeyNotFoundException($"HieraImageEncoder: missing weight '{key}'.");
        return t;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        // Weights are owned by the checkpoint loader (or the test that created them); nothing to free here.
    }
}
