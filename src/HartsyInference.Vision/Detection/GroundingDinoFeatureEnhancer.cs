using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vision.Detection;

/// <summary>Grounding DINO's cross-modality feature enhancer: <c>numEnhancerLayers</c> stacked layers that jointly
/// refine the flattened multi-scale image tokens (at the shared hidden width) and the caption text tokens (at the
/// text width). Each layer runs image self-attention, text self-attention, and <b>bidirectional</b> image↔text
/// cross-attention, every sub-layer post-normed with its own residual and followed by a feed-forward block.
///
/// <para><b>What is structurally faithful vs. simplified.</b> The bidirectional fusion is present and real
/// (image attends text and text attends image), but Grounding DINO's image self-attention is <i>deformable</i>
/// multi-scale attention; here it is plain full self-attention over the concatenated levels — correct in shape and
/// information flow, cheaper and not sparse. Grounding DINO also derives both cross directions from one shared
/// score matrix (BiMultiHeadAttention); here the two directions are independent attentions. These are the
/// documented parity gaps for this phase.</para></summary>
public sealed class GroundingDinoFeatureEnhancer
{
    private readonly GroundingDinoConfig _config;
    private readonly EnhancerLayer[] _layers;

    public GroundingDinoFeatureEnhancer(GroundingDinoConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
        _layers = new EnhancerLayer[config.NumEnhancerLayers];
        for (int i = 0; i < _layers.Length; i++)
            _layers[i] = new EnhancerLayer(config);
    }

    public void Build(IGroundingWeightBag bag, string prefix)
    {
        for (int i = 0; i < _layers.Length; i++)
            _layers[i].Build(bag, $"{prefix}.layer.{i}");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (EnhancerLayer layer in _layers)
            foreach (Tensor t in layer.EnumerateWeights())
                yield return t;
    }

    /// <summary>Refines <paramref name="imageTokens"/> <c>[1, Nimg, hidden]</c> and <paramref name="textTokens"/>
    /// <c>[1, T, textDim]</c>. Optional additive masks: <paramref name="textSelfMask"/> <c>[T, T]</c> and
    /// <paramref name="imageToTextMask"/> <c>[Nimg, T]</c> mask padded text keys (null = attend all). Returns new
    /// owned tensors of the same shapes; the caller's inputs are left intact.</summary>
    public (Tensor image, Tensor text) Forward(IBackend backend, Tensor imageTokens, Tensor textTokens, Tensor? textSelfMask, Tensor? imageToTextMask)
    {
        ArgumentNullException.ThrowIfNull(backend);
        Tensor image = imageTokens;
        Tensor text = textTokens;
        for (int i = 0; i < _layers.Length; i++)
        {
            (Tensor nextImage, Tensor nextText) = _layers[i].Forward(backend, image, text, textSelfMask, imageToTextMask);
            if (i > 0)
            {
                image.Dispose();
                text.Dispose();
            }
            image = nextImage;
            text = nextText;
        }
        return (image, text);
    }

    /// <summary>One enhancer layer: image/text self-attention, bidirectional cross-attention, and per-stream FFN,
    /// each post-normed.</summary>
    private sealed class EnhancerLayer
    {
        private readonly GroundingDinoAttention _imgSelfAttn;
        private readonly GroundingDinoAttention _textSelfAttn;
        private readonly GroundingDinoAttention _imgToText;
        private readonly GroundingDinoAttention _textToImg;
        private readonly GroundingDinoNorm _imgSelfNorm;
        private readonly GroundingDinoNorm _textSelfNorm;
        private readonly GroundingDinoNorm _imgCrossNorm;
        private readonly GroundingDinoNorm _textCrossNorm;
        private readonly GroundingDinoFeedForward _imgFfn;
        private readonly GroundingDinoFeedForward _textFfn;
        private readonly GroundingDinoNorm _imgFfnNorm;
        private readonly GroundingDinoNorm _textFfnNorm;

        public EnhancerLayer(GroundingDinoConfig config)
        {
            int h = config.ImageHiddenDim;
            int l = config.TextDim;
            int nh = config.NumHeads;
            _imgSelfAttn = new GroundingDinoAttention(h, h, h, nh);
            _textSelfAttn = new GroundingDinoAttention(l, l, l, nh);
            _imgToText = new GroundingDinoAttention(h, l, h, nh);   // image query, text key/value
            _textToImg = new GroundingDinoAttention(l, h, l, nh);   // text query, image key/value
            _imgSelfNorm = new GroundingDinoNorm(h);
            _textSelfNorm = new GroundingDinoNorm(l);
            _imgCrossNorm = new GroundingDinoNorm(h);
            _textCrossNorm = new GroundingDinoNorm(l);
            _imgFfn = new GroundingDinoFeedForward(h, config.FeedForwardDim);
            _textFfn = new GroundingDinoFeedForward(l, config.TextFeedForwardDim);
            _imgFfnNorm = new GroundingDinoNorm(h);
            _textFfnNorm = new GroundingDinoNorm(l);
        }

        public void Build(IGroundingWeightBag bag, string prefix)
        {
            _imgSelfAttn.Build(bag, $"{prefix}.img_self_attn");
            _textSelfAttn.Build(bag, $"{prefix}.text_self_attn");
            _imgToText.Build(bag, $"{prefix}.img2text_attn");
            _textToImg.Build(bag, $"{prefix}.text2img_attn");
            _imgSelfNorm.Build(bag, $"{prefix}.img_self_norm");
            _textSelfNorm.Build(bag, $"{prefix}.text_self_norm");
            _imgCrossNorm.Build(bag, $"{prefix}.img_cross_norm");
            _textCrossNorm.Build(bag, $"{prefix}.text_cross_norm");
            _imgFfn.Build(bag, $"{prefix}.img_ffn");
            _textFfn.Build(bag, $"{prefix}.text_ffn");
            _imgFfnNorm.Build(bag, $"{prefix}.img_ffn_norm");
            _textFfnNorm.Build(bag, $"{prefix}.text_ffn_norm");
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            foreach (Tensor t in _imgSelfAttn.EnumerateWeights()) yield return t;
            foreach (Tensor t in _textSelfAttn.EnumerateWeights()) yield return t;
            foreach (Tensor t in _imgToText.EnumerateWeights()) yield return t;
            foreach (Tensor t in _textToImg.EnumerateWeights()) yield return t;
            foreach (Tensor t in _imgSelfNorm.EnumerateWeights()) yield return t;
            foreach (Tensor t in _textSelfNorm.EnumerateWeights()) yield return t;
            foreach (Tensor t in _imgCrossNorm.EnumerateWeights()) yield return t;
            foreach (Tensor t in _textCrossNorm.EnumerateWeights()) yield return t;
            foreach (Tensor t in _imgFfn.EnumerateWeights()) yield return t;
            foreach (Tensor t in _textFfn.EnumerateWeights()) yield return t;
            foreach (Tensor t in _imgFfnNorm.EnumerateWeights()) yield return t;
            foreach (Tensor t in _textFfnNorm.EnumerateWeights()) yield return t;
        }

        public (Tensor image, Tensor text) Forward(IBackend backend, Tensor image, Tensor text, Tensor? textSelfMask, Tensor? imageToTextMask)
        {
            // Self-attention on each stream.
            Tensor imgSelf = _imgSelfAttn.Forward(backend, image, image, null);
            Tensor img1 = _imgSelfNorm.AddNorm(backend, image, imgSelf);
            Tensor textSelf = _textSelfAttn.Forward(backend, text, text, textSelfMask);
            Tensor text1 = _textSelfNorm.AddNorm(backend, text, textSelf);

            // Bidirectional cross-attention. Image is updated first; the updated image feeds the text update.
            Tensor crossI = _imgToText.Forward(backend, img1, text1, imageToTextMask);
            Tensor img2 = _imgCrossNorm.AddNorm(backend, img1, crossI);
            img1.Dispose();
            Tensor crossT = _textToImg.Forward(backend, text1, img2, null);
            Tensor text2 = _textCrossNorm.AddNorm(backend, text1, crossT);
            text1.Dispose();

            // Per-stream feed-forward.
            Tensor imgFf = _imgFfn.Forward(backend, img2);
            Tensor img3 = _imgFfnNorm.AddNorm(backend, img2, imgFf);
            img2.Dispose();
            Tensor textFf = _textFfn.Forward(backend, text2);
            Tensor text3 = _textFfnNorm.AddNorm(backend, text2, textFf);
            text2.Dispose();

            return (img3, text3);
        }
    }
}
