using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection.Blocks;

/// <summary>YOLO segmentation Proto module — produces <c>nm</c> mask prototypes at half the
/// detection grid's stride (i.e. <c>2 × P3</c> spatial resolution). For 640×640 input that's
/// 160×160; for arbitrary input it's <c>2 × P3_height × 2 × P3_width</c>.
/// <para>Structure:</para>
/// <list type="number">
///   <item><c>cv1</c>: 3×3 ConvBnSilu — projects P3 features (<c>c1</c> channels) up to <c>npr</c>.</item>
///   <item><c>upsample</c>: 2×2 stride-2 ConvTranspose2d with bias — doubles spatial dims; weight shape <c>[npr, npr, 2, 2]</c>.</item>
///   <item><c>cv2</c>: 3×3 ConvBnSilu at the doubled resolution — same channel count.</item>
///   <item><c>cv3</c>: 1×1 ConvBnSilu — projects down to <c>nm</c> (mask count).</item>
/// </list>
/// <para>The output is <c>[B, nm, 2·P3_h, 2·P3_w]</c>. Each "channel" is a per-pixel mask
/// prototype; per-detection masks are then assembled as
/// <c>sigmoid(coefficients_per_detection @ prototypes_flattened)</c> by
/// <see cref="MaskAssembly.AssembleMasks"/>.</para></summary>
public sealed class Proto
{
    private readonly int _internalChannels; // npr in Ultralytics — typically 256 base, scaled by width
    private readonly int _numMasks;         // nm — typically 32

    private readonly ConvBnSilu _cv1;
    private readonly ConvBnSilu _cv2;
    private readonly ConvBnSilu _cv3;

    // Upsample is a 2×2 stride-2 ConvTranspose2d with bias. Loaded directly (no BN to fold).
    private Tensor? _upsampleWeight; // [npr, npr, 2, 2]
    private Tensor? _upsampleBias;   // [npr]

    /// <summary>Number of mask prototypes the module emits (matches the cv4 coefficient count in the segment head).</summary>
    public int NumMasks => _numMasks;

    /// <summary>Creates the Proto module. <paramref name="inChannels"/> is P3's channel count
    /// (Detect head's first input); <paramref name="internalChannels"/> is <c>npr</c> (Ultralytics' default 256, scaled by width); <paramref name="numMasks"/> is <c>nm</c> (default 32).</summary>
    public Proto(int inChannels, int internalChannels, int numMasks)
    {
        _internalChannels = internalChannels;
        _numMasks = numMasks;

        // cv1 is 3×3 — pad=1, stride=1 to preserve spatial dims.
        _cv1 = new ConvBnSilu(internalChannels, strideH: 1, strideW: 1, padH: 1, padW: 1, useSilu: true);
        // cv2 same.
        _cv2 = new ConvBnSilu(internalChannels, strideH: 1, strideW: 1, padH: 1, padW: 1, useSilu: true);
        // cv3 is 1×1 (kernel inferred from weight; pad=0).
        _cv3 = new ConvBnSilu(numMasks, strideH: 1, strideW: 1, padH: 0, padW: 0, useSilu: true);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _cv1.LoadWeights(weights, $"{prefix}.cv1.conv");
        _cv2.LoadWeights(weights, $"{prefix}.cv2.conv");
        _cv3.LoadWeights(weights, $"{prefix}.cv3.conv");
        // The upsample ConvTranspose2d carries weight+bias directly (no BN to fold; PyTorch's
        // nn.ConvTranspose2d has bias=True here per Ultralytics).
        _upsampleWeight = EnsureF32(weights[$"{prefix}.upsample.weight"]);
        _upsampleBias = EnsureF32(weights[$"{prefix}.upsample.bias"]);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _cv1.EnumerateWeights()) yield return t;
        foreach (Tensor t in _cv2.EnumerateWeights()) yield return t;
        foreach (Tensor t in _cv3.EnumerateWeights()) yield return t;
        if (_upsampleWeight is not null) yield return _upsampleWeight;
        if (_upsampleBias is not null) yield return _upsampleBias;
    }

    /// <summary>Forward: <c>cv1 → upsample(2×) → cv2 → cv3</c>. Returns <c>[B, nm, 2·H, 2·W]</c>
    /// where (H, W) is the input spatial size.</summary>
    public Tensor Forward(IBackend backend, Tensor input)
    {
        if (_upsampleWeight is null || _upsampleBias is null)
            throw new InvalidOperationException("Proto weights not loaded.");

        int batch = (int)input.Shape[0];
        int h = (int)input.Shape[2];
        int w = (int)input.Shape[3];

        Tensor a = _cv1.Forward(backend, input);
        // ConvTranspose2d with k=2, s=2, p=0 doubles spatial dims (no overlap, no padding).
        TensorShape upShape = new TensorShape(batch, _internalChannels, h * 2, w * 2);
        Tensor upsampled = new Tensor(upShape, DType.F32);
        backend.ConvTranspose2d(upsampled, a, _upsampleWeight, _upsampleBias,
            strideH: 2, strideW: 2, padH: 0, padW: 0);
        a.Dispose();

        // Apply SiLU after the ConvTranspose. (Ultralytics' Proto.forward applies the standard
        // Conv act on the upsample output; nn.ConvTranspose2d itself has no activation, but the
        // module's surrounding flow makes cv2's input the activated upsample output. Verified
        // by reading Proto.forward in Ultralytics: the upsample is NOT wrapped in a Conv block
        // so there is NO SiLU after it. We pass straight to cv2.)
        // ↑ Note: Ultralytics' `Proto.forward` literally writes
        //   `return self.cv3(self.cv2(self.upsample(self.cv1(x))))` — no SiLU between upsample
        //   and cv2. The convtranspose output goes straight into cv2.

        Tensor b = _cv2.Forward(backend, upsampled);
        upsampled.Dispose();
        Tensor c = _cv3.Forward(backend, b);
        b.Dispose();
        return c;
    }

    private static Tensor EnsureF32(Tensor t) =>
        t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
}
