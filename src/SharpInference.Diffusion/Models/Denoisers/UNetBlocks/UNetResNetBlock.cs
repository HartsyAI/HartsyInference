using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Denoisers.UNetBlocks;

/// <summary>ResNet block for UNet with timestep conditioning. GroupNorm→SiLU→Conv3x3→(temb projection)→GroupNorm→SiLU→Conv3x3 + skip.</summary>
public sealed class UNetResNetBlock
{
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int _timeDim;
    private readonly int _normGroups;
    private readonly float _normEps;
    private readonly bool _hasShortcut;

    // norm1 + conv1
    private Tensor? _norm1Weight;
    private Tensor? _norm1Bias;
    private Tensor? _conv1Weight;
    private Tensor? _conv1Bias;

    // timestep projection: Linear(timeDim, outChannels)
    private Tensor? _timeEmbProjWeight;
    private Tensor? _timeEmbProjBias;

    // norm2 + conv2
    private Tensor? _norm2Weight;
    private Tensor? _norm2Bias;
    private Tensor? _conv2Weight;
    private Tensor? _conv2Bias;

    // shortcut (1x1 conv when inChannels != outChannels)
    private Tensor? _shortcutWeight;
    private Tensor? _shortcutBias;

    /// <summary>Creates a UNet ResNet block.</summary>
    public UNetResNetBlock(int inChannels, int outChannels, int timeDim, int normGroups = 32, float normEps = 1e-5f)
    {
        _inChannels = inChannels;
        _outChannels = outChannels;
        _timeDim = timeDim;
        _normGroups = normGroups;
        _normEps = normEps;
        _hasShortcut = inChannels != outChannels;
    }

    /// <summary>Loads weights from named tensors using the given prefix.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _norm1Weight = weights[$"{prefix}.norm1.weight"];
        _norm1Bias = weights[$"{prefix}.norm1.bias"];
        _conv1Weight = weights[$"{prefix}.conv1.weight"];
        _conv1Bias = weights[$"{prefix}.conv1.bias"];

        _timeEmbProjWeight = weights[$"{prefix}.time_emb_proj.weight"];
        _timeEmbProjBias = weights[$"{prefix}.time_emb_proj.bias"];

        _norm2Weight = weights[$"{prefix}.norm2.weight"];
        _norm2Bias = weights[$"{prefix}.norm2.bias"];
        _conv2Weight = weights[$"{prefix}.conv2.weight"];
        _conv2Bias = weights[$"{prefix}.conv2.bias"];

        if (_hasShortcut)
        {
            _shortcutWeight = weights[$"{prefix}.conv_shortcut.weight"];
            _shortcutBias = weights[$"{prefix}.conv_shortcut.bias"];
        }
    }

    /// <summary>Enumerates all weight tensors held by this module.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_norm1Weight is not null) yield return _norm1Weight;
        if (_norm1Bias is not null) yield return _norm1Bias;
        if (_conv1Weight is not null) yield return _conv1Weight;
        if (_conv1Bias is not null) yield return _conv1Bias;
        if (_timeEmbProjWeight is not null) yield return _timeEmbProjWeight;
        if (_timeEmbProjBias is not null) yield return _timeEmbProjBias;
        if (_norm2Weight is not null) yield return _norm2Weight;
        if (_norm2Bias is not null) yield return _norm2Bias;
        if (_conv2Weight is not null) yield return _conv2Weight;
        if (_conv2Bias is not null) yield return _conv2Bias;
        if (_shortcutWeight is not null) yield return _shortcutWeight;
        if (_shortcutBias is not null) yield return _shortcutBias;
    }

    /// <summary>Forward pass: input [B, inCh, H, W] + temb [B, timeDim] → output [B, outCh, H, W].</summary>
    public Tensor Forward(IBackend backend, Tensor input, Tensor temb)
    {
        int batch = (int)input.Shape[0];
        int height = (int)input.Shape[2];
        int width = (int)input.Shape[3];

        // 1. norm1 → SiLU → conv1
        TensorShape inShape = new TensorShape(batch, _inChannels, height, width);
        Tensor norm1Out = new Tensor(inShape, DType.F32);
        backend.GroupNorm(norm1Out, input, _norm1Weight!, _norm1Bias!, _normGroups, _normEps);

        Tensor silu1Out = new Tensor(inShape, DType.F32);
        backend.Silu(silu1Out, norm1Out);
        norm1Out.Dispose();

        TensorShape outShape = new TensorShape(batch, _outChannels, height, width);
        Tensor conv1Out = new Tensor(outShape, DType.F32);
        backend.Conv2D(conv1Out, silu1Out, _conv1Weight!, _conv1Bias, 1, 1, 1, 1);
        silu1Out.Dispose();

        // 2. Project timestep embedding and add to hidden: temb [B, timeDim] → [B, outCh] → broadcast add
        Tensor tembProj = ProjectTimestepEmbedding(backend, temb, batch);
        backend.BroadcastAdd(conv1Out, tembProj, _outChannels, height * width);
        tembProj.Dispose();

        // 3. norm2 → SiLU → conv2
        Tensor norm2Out = new Tensor(outShape, DType.F32);
        backend.GroupNorm(norm2Out, conv1Out, _norm2Weight!, _norm2Bias!, _normGroups, _normEps);
        conv1Out.Dispose();

        Tensor silu2Out = new Tensor(outShape, DType.F32);
        backend.Silu(silu2Out, norm2Out);
        norm2Out.Dispose();

        Tensor conv2Out = new Tensor(outShape, DType.F32);
        backend.Conv2D(conv2Out, silu2Out, _conv2Weight!, _conv2Bias, 1, 1, 1, 1);
        silu2Out.Dispose();

        // 4. Skip connection
        Tensor skip;
        if (_hasShortcut)
        {
            skip = new Tensor(outShape, DType.F32);
            backend.Conv2D(skip, input, _shortcutWeight!, _shortcutBias, 1, 1, 0, 0);
        }
        else
        {
            skip = input;
        }

        // 5. Residual: output = conv2_out + skip
        Tensor output = new Tensor(outShape, DType.F32);
        backend.Add(output, conv2Out, skip);
        conv2Out.Dispose();

        if (_hasShortcut)
        {
            skip.Dispose();
        }

        return output;
    }

    /// <summary>Projects timestep embedding: SiLU(temb) → Linear → [B, outChannels].</summary>
    private Tensor ProjectTimestepEmbedding(IBackend backend, Tensor temb, int batch)
    {
        // SiLU on temb first (diffusers does this before the projection in the ResNet block)
        TensorShape tembShape = new TensorShape(batch, _timeDim);
        Tensor tembSilu = new Tensor(tembShape, DType.F32);
        backend.Silu(tembSilu, temb);

        // Linear: [B, timeDim] → [B, outChannels]
        TensorShape projShape = new TensorShape(batch, _outChannels);
        Tensor projected = new Tensor(projShape, DType.F32);
        backend.Linear(projected, tembSilu, _timeEmbProjWeight!, _timeEmbProjBias!);
        tembSilu.Dispose();

        return projected;
    }

}
