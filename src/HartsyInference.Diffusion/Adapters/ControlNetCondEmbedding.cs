using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>The conditioning hint encoder that downsamples a 3-channel control image (canny edges, depth map, openpose skeleton, …) into the same spatial + channel dimensions as the base UNet's <c>conv_in</c> output, so the result can be added on top of the noisy latent before the down blocks. Architecture (matches diffusers' <c>ControlNetConditioningEmbedding</c>): <c>conv_in(3→16)</c> → SiLU → 6 blocks alternating <c>same-stride</c> and <c>stride-2</c> convs (16→16, 16→32↓, 32→32, 32→96↓, 96→96, 96→256↓) → <c>conv_out(256→modelChannels)</c>. Three stride-2 convs give 8× total downsampling — exactly the VAE compression ratio, so a <c>1024²</c> image lands at <c>128²</c> matching the latent grid. <c>conv_out</c> is zero-initialized at training time so an untrained ControlNet contributes zero residuals; we don't enforce that here, we just trust the checkpoint.</summary>
public sealed unsafe class ControlNetCondEmbedding : IDisposable
{
    private readonly int _condChannels;
    private readonly int _modelChannels;
    private readonly int[] _blockOut;
    private int _disposed;

    private Tensor? _convInWeight, _convInBias;
    private readonly Tensor?[] _blockWeights;
    private readonly Tensor?[] _blockBiases;
    private readonly bool[] _blockStride2;
    private Tensor? _convOutWeight, _convOutBias;

    /// <summary>Creates a new conditioning hint encoder.</summary>
    /// <param name="conditioningChannels">Channels of the input control image. <c>3</c> for RGB (canny, depth, openpose, …).</param>
    /// <param name="modelChannels">Output channel count — must match the base UNet's <c>conv_in</c> output (320 for SD1.5 / SDXL, 384 for SDXL refiner).</param>
    /// <param name="blockOutChannels">Per-block channel schedule. Default <c>[16, 32, 96, 256]</c> matches diffusers; gives 6 internal conv blocks (see class doc).</param>
    public ControlNetCondEmbedding(int conditioningChannels = 3, int modelChannels = 320, int[]? blockOutChannels = null)
    {
        _condChannels = conditioningChannels;
        _modelChannels = modelChannels;
        _blockOut = blockOutChannels ?? [16, 32, 96, 256];
        // 2 convs per (in→out) transition: same-stride then stride-2 → 6 total blocks for 4-tier schedule.
        int numBlocks = (_blockOut.Length - 1) * 2;
        _blockWeights = new Tensor?[numBlocks];
        _blockBiases = new Tensor?[numBlocks];
        _blockStride2 = new bool[numBlocks];
        for (int i = 0; i < _blockOut.Length - 1; i++)
        {
            _blockStride2[i * 2] = false; // same-stride
            _blockStride2[i * 2 + 1] = true; // stride-2 downsample
        }
    }

    /// <summary>Loads weights from named tensors. Expected key prefix is <c>controlnet_cond_embedding</c> (the standard diffusers layout for both SD1.5 and SDXL ControlNet checkpoints).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix = "controlnet_cond_embedding")
    {
        _convInWeight = weights[$"{prefix}.conv_in.weight"];
        _convInBias = weights[$"{prefix}.conv_in.bias"];
        for (int i = 0; i < _blockWeights.Length; i++)
        {
            _blockWeights[i] = weights[$"{prefix}.blocks.{i}.weight"];
            _blockBiases[i] = weights[$"{prefix}.blocks.{i}.bias"];
        }
        _convOutWeight = weights[$"{prefix}.conv_out.weight"];
        _convOutBias = weights[$"{prefix}.conv_out.bias"];
    }

    /// <summary>Forward pass on a <c>[B, condChannels, H, W]</c> control image. Returns a <c>[B, modelChannels, H/8, W/8]</c> tensor in the same dtype as the input.</summary>
    public Tensor Forward(IBackend backend, Tensor cond)
    {
        if (cond.Shape.Rank != 4 || cond.Shape[1] != _condChannels)
        {
            throw new ArgumentException(
                $"ControlNet conditioning input must be [B, {_condChannels}, H, W]; got {cond.Shape}.", nameof(cond));
        }
        int batch = (int)cond.Shape[0];
        int h = (int)cond.Shape[2];
        int w = (int)cond.Shape[3];
        DType dtype = cond.DType;

        // 1. conv_in: 3 → blockOut[0] (typically 3 → 16), same spatial dims.
        TensorShape convInShape = new TensorShape(batch, _blockOut[0], h, w);
        Tensor hidden = new Tensor(convInShape, dtype);
        backend.Conv2D(hidden, cond, _convInWeight!, _convInBias, 1, 1, 1, 1);
        backend.Silu(hidden, hidden);

        // 2. Six internal blocks: same-stride / stride-2 / same-stride / stride-2 / same-stride / stride-2.
        //    Channel transitions live in the stride-2 blocks (16→32, 32→96, 96→256) per the
        //    standard schedule; same-stride blocks preserve channels.
        for (int i = 0; i < _blockWeights.Length; i++)
        {
            // Decode in/out channels from the weight shape directly — matches the checkpoint
            // regardless of any block_out_channels override the user passed.
            int inCh = (int)_blockWeights[i]!.Shape[1];
            int outCh = (int)_blockWeights[i]!.Shape[0];
            int stride = _blockStride2[i] ? 2 : 1;
            int newH = stride == 2 ? h / 2 : h;
            int newW = stride == 2 ? w / 2 : w;
            TensorShape blockShape = new TensorShape(batch, outCh, newH, newW);
            Tensor next = new Tensor(blockShape, dtype);
            backend.Conv2D(next, hidden, _blockWeights[i]!, _blockBiases[i], stride, stride, 1, 1);
            backend.Silu(next, next);
            hidden.Dispose();
            hidden = next;
            h = newH;
            w = newW;
        }

        // 3. conv_out: blockOut[-1] (typically 256) → modelChannels (320 for SDXL/SD15), same spatial dims.
        //    No SiLU after — diffusers' ControlNetConditioningEmbedding ends here. The output
        //    is zero-initialized in pretrained checkpoints (zero conv) so the residual stack
        //    starts as a no-op until training pushes weights away from zero.
        TensorShape outShape = new TensorShape(batch, _modelChannels, h, w);
        Tensor output = new Tensor(outShape, dtype);
        backend.Conv2D(output, hidden, _convOutWeight!, _convOutBias, 1, 1, 1, 1);
        hidden.Dispose();
        return output;
    }

    /// <summary>Yields all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_convInWeight is not null) yield return _convInWeight;
        if (_convInBias is not null) yield return _convInBias;
        foreach (Tensor? w in _blockWeights) if (w is not null) yield return w;
        foreach (Tensor? b in _blockBiases) if (b is not null) yield return b;
        if (_convOutWeight is not null) yield return _convOutWeight;
        if (_convOutBias is not null) yield return _convOutBias;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _convInWeight = null; _convInBias = null;
            Array.Clear(_blockWeights);
            Array.Clear(_blockBiases);
            _convOutWeight = null; _convOutBias = null;
        }
        GC.SuppressFinalize(this);
    }
}
