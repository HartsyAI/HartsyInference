using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.Annotators;

namespace HartsyInference.Vision.Matting.BiRefNet;

/// <summary>Deformable ASPP (upstream <c>ASPPDeformable</c>, <c>parallel_block_sizes=[1,3,7]</c>): a 1×1 deformable
/// branch, three deformable branches of kernel 1/3/7, and a global-average-pool branch, each 256 channels after
/// BatchNorm and ReLU; concatenated to 1280 channels and projected back to the input width by a 1×1 conv. Eval-mode
/// dropout is the identity.</summary>
internal sealed unsafe class BiRefNetAspp
{
    private const int BranchChannels = 256;
    private const float BatchNormEps = 1e-5f;

    private readonly BiRefNetDeformConv _aspp1 = new(kernel: 1, pad: 0);
    private readonly BiRefNetDeformConv[] _deforms = [new(kernel: 1, pad: 0), new(kernel: 3, pad: 1), new(kernel: 7, pad: 3)];
    private Tensor? _poolW, _poolB, _projW, _projB;

    /// <summary>Loads the branches and folds their BatchNorms; <paramref name="prefix"/> is the module's own prefix (e.g. <c>...dec_att</c>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix, List<Tensor> owned)
    {
        _aspp1.LoadWeights(w, $"{prefix}.aspp1", owned);
        for (int i = 0; i < _deforms.Length; i++)
            _deforms[i].LoadWeights(w, $"{prefix}.aspp_deforms.{i}", owned);

        (Tensor poolW, Tensor poolB) = ConvBnFold.Fold(w, $"{prefix}.global_avg_pool.1", $"{prefix}.global_avg_pool.2", BatchNormEps);
        (Tensor projW, Tensor projB) = ConvBnFold.Fold(w, $"{prefix}.conv1", $"{prefix}.bn1", BatchNormEps);
        owned.Add(poolW); owned.Add(poolB); owned.Add(projW); owned.Add(projB);
        _poolW = poolW; _poolB = poolB; _projW = projW; _projB = projB;
    }

    public Tensor Forward(IBackend backend, Tensor x)
    {
        int h = (int)x.Shape[2], w = (int)x.Shape[3];
        // Each pre-activation map is released as soon as its ReLU has been written.
        Tensor a1 = ReluAndRelease(backend, _aspp1.Forward(backend, x));
        Tensor d0 = ReluAndRelease(backend, _deforms[0].Forward(backend, x));
        Tensor d1 = ReluAndRelease(backend, _deforms[1].Forward(backend, x));
        Tensor d2 = ReluAndRelease(backend, _deforms[2].Forward(backend, x));
        Tensor pooled = GlobalPoolBranch(backend, x, h, w);

        Tensor cat = BiRefNetOps.Concat(backend, a1, d0, d1, d2, pooled);
        a1.Dispose(); d0.Dispose(); d1.Dispose(); d2.Dispose(); pooled.Dispose();

        Tensor projected = BiRefNetOps.Conv(backend, cat, _projW!, _projB, pad: 0);
        cat.Dispose();
        Tensor output = BiRefNetOps.Relu(backend, projected);
        projected.Dispose();
        return output;
    }

    private static Tensor ReluAndRelease(IBackend backend, Tensor preActivation)
    {
        Tensor activated = BiRefNetOps.Relu(backend, preActivation);
        preActivation.Dispose();
        return activated;
    }

    /// <summary>Global-average-pool → 1×1 conv → BN → ReLU, broadcast back to <c>[1,256,H,W]</c>.
    /// Bilinear resize of a 1×1 map with align-corners is a constant, so the branch is computed once and filled.</summary>
    private Tensor GlobalPoolBranch(IBackend backend, Tensor x, int h, int w)
    {
        int channels = (int)x.Shape[1], hw = h * w;
        float* xp = (float*)x.DataPointer;
        float* wp = (float*)_poolW!.DataPointer, bp = (float*)_poolB!.DataPointer;
        float[] mean = new float[channels];
        for (int c = 0; c < channels; c++)
        {
            float acc = 0f;
            for (int i = 0; i < hw; i++) acc += xp[(long)c * hw + i];
            mean[c] = acc / hw;
        }
        Tensor pooled = new(new TensorShape(1, BranchChannels, h, w), DType.F32);
        float* op = (float*)pooled.DataPointer;
        for (int o = 0; o < BranchChannels; o++)
        {
            float acc = bp[o];
            for (int c = 0; c < channels; c++) acc += wp[(long)o * channels + c] * mean[c];
            float v = MathF.Max(acc, 0f);
            new Span<float>(op + (long)o * hw, hw).Fill(v);
        }
        return pooled;
    }
}
