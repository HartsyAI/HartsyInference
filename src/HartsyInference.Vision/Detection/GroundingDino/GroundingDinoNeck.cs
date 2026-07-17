using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection.GroundingDino;

/// <summary>Grounding DINO neck: projects the three Swin feature maps to <c>d_model</c> (1×1 conv + GroupNorm), adds a
/// fourth level via a 3×3 stride-2 conv on the last map, builds the DETR sinusoidal position embeddings, adds the
/// learned per-level embedding, and flattens/concatenates every level into the encoder input sequence
/// (<c>source_flatten</c> and <c>lvl_pos_embed_flatten</c>, both <c>[1, sum(h*w), d_model]</c>).</summary>
public sealed unsafe class GroundingDinoNeck : IDisposable
{
    private readonly GroundingDinoConfig _cfg;
    private readonly Tensor?[] _convW;
    private readonly Tensor?[] _convB;
    private readonly Tensor?[] _gnW;
    private readonly Tensor?[] _gnB;
    private Tensor? _levelEmbed;
    private int _disposed;

    public GroundingDinoNeck(GroundingDinoConfig cfg)
    {
        _cfg = cfg;
        _convW = new Tensor?[cfg.NumFeatureLevels];
        _convB = new Tensor?[cfg.NumFeatureLevels];
        _gnW = new Tensor?[cfg.NumFeatureLevels];
        _gnB = new Tensor?[cfg.NumFeatureLevels];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "model")
    {
        for (int l = 0; l < _cfg.NumFeatureLevels; l++)
        {
            _convW[l] = w[$"{prefix}.input_proj_vision.{l}.0.weight"];
            _convB[l] = w[$"{prefix}.input_proj_vision.{l}.0.bias"];
            _gnW[l] = w[$"{prefix}.input_proj_vision.{l}.1.weight"];
            _gnB[l] = w[$"{prefix}.input_proj_vision.{l}.1.bias"];
        }
        _levelEmbed = w[$"{prefix}.level_embed"];
    }

    /// <summary>Result of the neck: the flattened encoder inputs and the per-level spatial shapes.</summary>
    public sealed class Result
    {
        public required Tensor SourceFlatten { get; init; }
        public required Tensor PositionFlatten { get; init; }
        public required int[][] SpatialShapes { get; init; }
        public required int[] LevelStartIndex { get; init; }

        public int TotalTokens => SpatialShapes.Sum(s => s[0] * s[1]);
    }

    public Result Forward(IBackend backend, Tensor[] backboneFeats)
    {
        int d = _cfg.DModel;
        int levels = _cfg.NumFeatureLevels;
        Tensor[] proj = new Tensor[levels];

        for (int l = 0; l < backboneFeats.Length; l++)
            proj[l] = ProjectLevel(backend, backboneFeats[l], l, 1, 1, 0, 0);
        // Extra level: 3x3 stride-2 conv on the LAST backbone feature map.
        proj[backboneFeats.Length] = ProjectLevel(backend, backboneFeats[^1], backboneFeats.Length, 2, 2, 1, 1);

        int[][] shapes = new int[levels][];
        int total = 0;
        for (int l = 0; l < levels; l++)
        {
            shapes[l] = [(int)proj[l].Shape[2], (int)proj[l].Shape[3]];
            total += shapes[l][0] * shapes[l][1];
        }
        int[] levelStart = new int[levels];
        for (int l = 1; l < levels; l++)
            levelStart[l] = levelStart[l - 1] + shapes[l - 1][0] * shapes[l - 1][1];

        Tensor source = new(new TensorShape(1, total, d), DType.F32);
        Tensor vpos = new(new TensorShape(1, total, d), DType.F32);
        float* sp = (float*)source.DataPointer;
        float* pp = (float*)vpos.DataPointer;
        float* le = (float*)_levelEmbed!.DataPointer;

        for (int l = 0; l < levels; l++)
        {
            int h = shapes[l][0], wdt = shapes[l][1], hw = h * wdt;
            float* fp = (float*)proj[l].DataPointer;   // [1, d, h, w]
            float[] pos = SinePositionEmbedding(h, wdt);   // [hw, d]
            int baseTok = levelStart[l];
            for (int i = 0; i < h; i++)
                for (int j = 0; j < wdt; j++)
                {
                    int t = i * wdt + j;
                    long dst = (long)(baseTok + t) * d;
                    for (int c = 0; c < d; c++)
                    {
                        sp[dst + c] = fp[((long)c * h + i) * wdt + j];
                        pp[dst + c] = pos[(long)t * d + c] + le[(long)l * d + c];
                    }
                }
        }
        for (int l = 0; l < levels; l++) proj[l].Dispose();

        return new Result
        {
            SourceFlatten = source,
            PositionFlatten = vpos,
            SpatialShapes = shapes,
            LevelStartIndex = levelStart,
        };
    }

    private Tensor ProjectLevel(IBackend backend, Tensor feat, int level, int strideH, int strideW, int padH, int padW)
    {
        int inC = (int)feat.Shape[1], h = (int)feat.Shape[2], w = (int)feat.Shape[3];
        int outH = (h + 2 * padH - (int)_convW[level]!.Shape[2]) / strideH + 1;
        int outW = (w + 2 * padW - (int)_convW[level]!.Shape[3]) / strideW + 1;
        Tensor conv = new(new TensorShape(1, _cfg.DModel, outH, outW), DType.F32);
        backend.Conv2D(conv, feat, _convW[level]!, _convB[level], strideH, strideW, padH, padW);
        Tensor gn = new(conv.Shape, DType.F32);
        backend.GroupNorm(gn, conv, _gnW[level]!, _gnB[level]!, 32, 1e-5f);
        conv.Dispose();
        return gn;
    }

    /// <summary>DETR sine position embedding for an <paramref name="h"/>×<paramref name="w"/> map with a full-ones
    /// pixel mask; returns <c>[h*w, d_model]</c> with the <c>[pos_y | pos_x]</c> channel layout.</summary>
    private float[] SinePositionEmbedding(int h, int w)
    {
        int d = _cfg.DModel;
        int half = _cfg.SinePosDim;   // 128
        float scale = 2f * MathF.PI;
        float temp = _cfg.PositionalEmbeddingTemperature;
        float[] dimT = new float[half];
        for (int k = 0; k < half; k++)
            dimT[k] = MathF.Pow(temp, 2f * (k / 2) / half);

        float[] outp = new float[(long)h * w * d];
        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++)
            {
                float yEmbed = (float)(i + 1) / (h + 1e-6f) * scale;
                float xEmbed = (float)(j + 1) / (w + 1e-6f) * scale;
                long baseOff = (long)(i * w + j) * d;
                for (int m = 0; m < half; m += 2)
                {
                    float py0 = yEmbed / dimT[m], py1 = yEmbed / dimT[m + 1];
                    float px0 = xEmbed / dimT[m], px1 = xEmbed / dimT[m + 1];
                    outp[baseOff + m] = MathF.Sin(py0);
                    outp[baseOff + m + 1] = MathF.Cos(py1);
                    outp[baseOff + half + m] = MathF.Sin(px0);
                    outp[baseOff + half + m + 1] = MathF.Cos(px1);
                }
            }
        return outp;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
