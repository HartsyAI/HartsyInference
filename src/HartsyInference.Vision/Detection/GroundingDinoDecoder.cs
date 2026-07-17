using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>Grounding DINO cross-modality decoder: <c>numDecoderLayers</c> layers that iteratively refine the object
/// queries selected by language-guided query selection. Each layer runs query self-attention, cross-attention into
/// the enhanced image memory, cross-attention into the enhanced text memory, and a feed-forward block, each
/// post-normed.
///
/// <para><b>Parity note.</b> Grounding DINO's image cross-attention is deformable (samples a few points around each
/// query's reference box); here it is plain full cross-attention over all image tokens — same information flow,
/// shape-correct, not sparse. Reference-box refinement per layer is folded into the box head rather than iterated
/// here.</para></summary>
public sealed class GroundingDinoDecoder
{
    private readonly DecoderLayer[] _layers;

    public GroundingDinoDecoder(GroundingDinoConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _layers = new DecoderLayer[config.NumDecoderLayers];
        for (int i = 0; i < _layers.Length; i++)
            _layers[i] = new DecoderLayer(config);
    }

    public void Build(IGroundingWeightBag bag, string prefix)
    {
        for (int i = 0; i < _layers.Length; i++)
            _layers[i].Build(bag, $"{prefix}.layer.{i}");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (DecoderLayer layer in _layers)
            foreach (Tensor t in layer.EnumerateWeights())
                yield return t;
    }

    /// <summary>Refines <paramref name="queries"/> <c>[1, Nq, hidden]</c> against the enhanced
    /// <paramref name="imageMemory"/> <c>[1, Nimg, hidden]</c> and <paramref name="textMemory"/> <c>[1, T, textDim]</c>.
    /// Optional <paramref name="queryToTextMask"/> <c>[Nq, T]</c> masks padded text keys. Returns a new owned
    /// <c>[1, Nq, hidden]</c>; the caller's <paramref name="queries"/> is left intact.</summary>
    public Tensor Forward(IBackend backend, Tensor queries, Tensor imageMemory, Tensor textMemory, Tensor? queryToTextMask)
    {
        ArgumentNullException.ThrowIfNull(backend);
        Tensor q = queries;
        for (int i = 0; i < _layers.Length; i++)
        {
            Tensor next = _layers[i].Forward(backend, q, imageMemory, textMemory, queryToTextMask);
            if (i > 0)
                q.Dispose();
            q = next;
        }
        return q;
    }

    private sealed class DecoderLayer
    {
        private readonly GroundingDinoAttention _selfAttn;
        private readonly GroundingDinoAttention _imgCrossAttn;
        private readonly GroundingDinoAttention _textCrossAttn;
        private readonly GroundingDinoNorm _selfNorm;
        private readonly GroundingDinoNorm _imgCrossNorm;
        private readonly GroundingDinoNorm _textCrossNorm;
        private readonly GroundingDinoFeedForward _ffn;
        private readonly GroundingDinoNorm _ffnNorm;

        public DecoderLayer(GroundingDinoConfig config)
        {
            int h = config.ImageHiddenDim;
            int l = config.TextDim;
            int nh = config.NumHeads;
            _selfAttn = new GroundingDinoAttention(h, h, h, nh);
            _imgCrossAttn = new GroundingDinoAttention(h, h, h, nh);
            _textCrossAttn = new GroundingDinoAttention(h, l, h, nh);
            _selfNorm = new GroundingDinoNorm(h);
            _imgCrossNorm = new GroundingDinoNorm(h);
            _textCrossNorm = new GroundingDinoNorm(h);
            _ffn = new GroundingDinoFeedForward(h, config.FeedForwardDim);
            _ffnNorm = new GroundingDinoNorm(h);
        }

        public void Build(IGroundingWeightBag bag, string prefix)
        {
            _selfAttn.Build(bag, $"{prefix}.self_attn");
            _imgCrossAttn.Build(bag, $"{prefix}.img_cross_attn");
            _textCrossAttn.Build(bag, $"{prefix}.text_cross_attn");
            _selfNorm.Build(bag, $"{prefix}.self_norm");
            _imgCrossNorm.Build(bag, $"{prefix}.img_cross_norm");
            _textCrossNorm.Build(bag, $"{prefix}.text_cross_norm");
            _ffn.Build(bag, $"{prefix}.ffn");
            _ffnNorm.Build(bag, $"{prefix}.ffn_norm");
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            foreach (Tensor t in _selfAttn.EnumerateWeights()) yield return t;
            foreach (Tensor t in _imgCrossAttn.EnumerateWeights()) yield return t;
            foreach (Tensor t in _textCrossAttn.EnumerateWeights()) yield return t;
            foreach (Tensor t in _selfNorm.EnumerateWeights()) yield return t;
            foreach (Tensor t in _imgCrossNorm.EnumerateWeights()) yield return t;
            foreach (Tensor t in _textCrossNorm.EnumerateWeights()) yield return t;
            foreach (Tensor t in _ffn.EnumerateWeights()) yield return t;
            foreach (Tensor t in _ffnNorm.EnumerateWeights()) yield return t;
        }

        public Tensor Forward(IBackend backend, Tensor queries, Tensor imageMemory, Tensor textMemory, Tensor? queryToTextMask)
        {
            Tensor self = _selfAttn.Forward(backend, queries, queries, null);
            Tensor q1 = _selfNorm.AddNorm(backend, queries, self);
            Tensor imgCross = _imgCrossAttn.Forward(backend, q1, imageMemory, null);
            Tensor q2 = _imgCrossNorm.AddNorm(backend, q1, imgCross);
            q1.Dispose();
            Tensor textCross = _textCrossAttn.Forward(backend, q2, textMemory, queryToTextMask);
            Tensor q3 = _textCrossNorm.AddNorm(backend, q2, textCross);
            q2.Dispose();
            Tensor ff = _ffn.Forward(backend, q3);
            Tensor q4 = _ffnNorm.AddNorm(backend, q3, ff);
            q3.Dispose();
            return q4;
        }
    }
}
