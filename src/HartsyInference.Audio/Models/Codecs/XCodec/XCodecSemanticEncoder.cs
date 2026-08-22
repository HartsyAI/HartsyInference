using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Codecs.XCodec;

/// <summary>The x-codec <c>encoder_semantic</c> — a RepCodec/AudioDec-lineage <c>Encoder(input_channels=768, encode_channels=768)</c> that conditions the mean-of-13 HuBERT feature before it is concatenated with the acoustic branch.</summary>
/// <remarks>
/// <code>
///   conv (k3 s1 p1, NO bias)                                          stem
///   for b in 0, 1:
///     res_units.{0,1}:  y = conv2(ELU(conv1(ELU(x)))); x = x + y      k3 p1 then k1, both bias-free
///     conv (k3 s1 p1, WITH bias)                                      block exit
/// </code>
///
/// <para>All strides are 1 (<c>channel_ratios=(1,1)</c>, <c>strides=(1,1)</c>) so the frame count is preserved
/// end to end and the block-exit kernel stays at RepCodec's stride-1 special case of 3 (not <c>2·stride</c>).
/// Activation is ELU(α=1), not DAC's Snake, and there is no weight-norm anywhere in this module.</para>
///
/// <para><b>Key-spelling trap:</b> <c>Conv1d1x1</c> subclasses <c>nn.Conv1d</c> directly, so the second res-unit
/// conv is <c>res_units.{u}.conv2.weight</c> — one nesting level shallower than its <c>conv1.conv.weight</c>
/// sibling. The res-unit convs and the stem carry no bias (<c>ResidualUnit.bias</c> defaults to false and
/// <c>EncoderBlock</c> never forwards its own <c>bias=True</c>); only the block-exit conv has one.</para></remarks>
internal sealed class XCodecSemanticEncoder
{
    private readonly string _prefix;
    private readonly int _dim;
    private readonly int _numBlocks;
    private readonly int _unitsPerBlock;
    private readonly int _kernel;

    private Tensor? _stemW;
    private readonly Tensor?[,] _unitConv1W;
    private readonly Tensor?[,] _unitConv2W;
    private readonly Tensor?[] _blockExitW;
    private readonly Tensor?[] _blockExitB;

    public XCodecSemanticEncoder(string prefix, int dim = 768, int numBlocks = 2, int unitsPerBlock = 2, int kernel = 3)
    {
        _prefix = prefix;
        _dim = dim;
        _numBlocks = numBlocks;
        _unitsPerBlock = unitsPerBlock;
        _kernel = kernel;
        _unitConv1W = new Tensor?[numBlocks, unitsPerBlock];
        _unitConv2W = new Tensor?[numBlocks, unitsPerBlock];
        _blockExitW = new Tensor?[numBlocks];
        _blockExitB = new Tensor?[numBlocks];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _stemW = WhisperOps.EnsureF32(w[$"{_prefix}.conv.conv.weight"]);
        for (int b = 0; b < _numBlocks; b++)
        {
            for (int u = 0; u < _unitsPerBlock; u++)
            {
                string p = $"{_prefix}.conv_blocks.{b}.res_units.{u}";
                _unitConv1W[b, u] = WhisperOps.EnsureF32(w[$"{p}.conv1.conv.weight"]);
                _unitConv2W[b, u] = WhisperOps.EnsureF32(w[$"{p}.conv2.weight"]);
            }
            _blockExitW[b] = WhisperOps.EnsureF32(w[$"{_prefix}.conv_blocks.{b}.conv.conv.weight"]);
            _blockExitB[b] = WhisperOps.EnsureF32(w[$"{_prefix}.conv_blocks.{b}.conv.conv.bias"]);
        }
    }

    /// <summary>Forward — channels-first <c>[B, 768, T]</c> in and out, T preserved. Input is not disposed.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int batch, int t)
    {
        if (_stemW is null) throw new InvalidOperationException("XCodecSemanticEncoder weights not loaded.");
        int pad = (_kernel - 1) / 2;

        Tensor cur = new(new TensorShape(batch, _dim, t), DType.F32);
        backend.Conv1d(cur, x, _stemW!, null, stride: 1, padLeft: pad, padRight: pad, dilation: 1, groups: 1);

        for (int b = 0; b < _numBlocks; b++)
        {
            for (int u = 0; u < _unitsPerBlock; u++)
            {
                Tensor a1 = new(cur.Shape, DType.F32);
                backend.Elu(a1, cur, 1f);
                Tensor mid = new(cur.Shape, DType.F32);
                backend.Conv1d(mid, a1, _unitConv1W[b, u]!, null, stride: 1, padLeft: pad, padRight: pad, dilation: 1, groups: 1);
                a1.Dispose();

                Tensor a2 = new(cur.Shape, DType.F32);
                backend.Elu(a2, mid, 1f);
                mid.Dispose();
                Tensor proj = new(cur.Shape, DType.F32);
                backend.Conv1d(proj, a2, _unitConv2W[b, u]!, null, stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
                a2.Dispose();

                Tensor sum = new(cur.Shape, DType.F32);
                backend.Add(sum, cur, proj);
                cur.Dispose(); proj.Dispose();
                cur = sum;
            }

            Tensor exit = new(cur.Shape, DType.F32);
            backend.Conv1d(exit, cur, _blockExitW[b]!, _blockExitB[b], stride: 1, padLeft: pad, padRight: pad, dilation: 1, groups: 1);
            cur.Dispose();
            cur = exit;
        }
        return cur;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_stemW is not null) yield return _stemW;
        for (int b = 0; b < _numBlocks; b++)
        {
            for (int u = 0; u < _unitsPerBlock; u++)
            {
                if (_unitConv1W[b, u] is not null) yield return _unitConv1W[b, u]!;
                if (_unitConv2W[b, u] is not null) yield return _unitConv2W[b, u]!;
            }
            if (_blockExitW[b] is not null) yield return _blockExitW[b]!;
            if (_blockExitB[b] is not null) yield return _blockExitB[b]!;
        }
    }
}
