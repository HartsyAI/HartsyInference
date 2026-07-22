using HartsyInference.ModelAssets.TextEncoders.Bert;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Audio.Models.MeloTts;

/// <summary>Produces the per-phoneme BERT feature stream MeloTTS-English consumes: runs bert-base-uncased over the
/// orthographic text, takes the layer-<c>NumLayers-2</c> hidden state (<c>hidden_states[-3]</c>), and expands each
/// BERT token's vector to phoneme positions via <c>word2ph</c> (each token's feature repeated <c>word2ph[i]</c>
/// times). For English this fills the model's <c>ja_bert</c> (768-channel) slot; the 1024-channel <c>bert</c> slot
/// is left zero. Mirrors <c>melo/text/english_bert.py::get_bert_feature</c>.</summary>
public sealed unsafe class MeloTtsBertProvider
{
    private readonly BertModel _bert;
    private readonly BertWordPieceTokenizer _tokenizer;
    private readonly int _hidden;
    private readonly int _featureLayer;

    public MeloTtsBertProvider(BertWordPieceTokenizer tokenizer, BertConfig cfg, IReadOnlyDictionary<string, Tensor> weights,
        string bertPrefix = "bert")
    {
        _tokenizer = tokenizer;
        _hidden = cfg.Hidden;
        _featureLayer = cfg.NumLayers - 2;       // hidden_states[-3]
        _bert = new BertModel(cfg);
        _bert.LoadWeights(weights, bertPrefix);
    }

    /// <summary>Returns the phoneme-aligned BERT features <c>[1, hidden, T]</c> (channels-first). <paramref
    /// name="word2ph"/> must have one entry per orthographic BERT token (incl. [CLS]/[SEP]) and sum to <c>T</c>.</summary>
    public Tensor GetFeatures(IBackend backend, string text, int[] word2ph)
    {
        int[] ids = _tokenizer.EncodeWithSpecial(text).ToArray();
        if (ids.Length != word2ph.Length)
            throw new ArgumentException($"BERT token count {ids.Length} != word2ph length {word2ph.Length}.");

        using Tensor hidden = _bert.Forward(backend, ids, _featureLayer);     // [1, nTokens, hidden]
        float* hp = (float*)hidden.DataPointer;

        int t = 0;
        for (int i = 0; i < word2ph.Length; i++) t += word2ph[i];
        Tensor outT = new(new TensorShape(1, _hidden, t), DType.F32);         // channels-first [1, hidden, T]
        float* op = (float*)outT.DataPointer;

        int col = 0;
        for (int i = 0; i < word2ph.Length; i++)
            for (int r = 0; r < word2ph[i]; r++, col++)
                for (int c = 0; c < _hidden; c++)
                    op[(long)c * t + col] = hp[(long)i * _hidden + c];        // repeat token i's vector
        return outT;
    }
}
