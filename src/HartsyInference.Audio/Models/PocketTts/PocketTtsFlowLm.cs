using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.PocketTts;

/// <summary>Pocket-TTS FlowLM: the autoregressive flow language model over continuous audio latents
/// (kyutai-labs/pocket-tts). Composes the verified components: <c>input_linear</c> (32->1024) + a learned
/// <c>bos_emb</c>, the <see cref="PocketTtsStreamingTransformer"/> backbone (6 layers, full causal), an
/// <c>out_norm</c> LayerNorm, and the <see cref="PocketTtsFlowNet"/> per-frame flow head.
///
/// <para>Generation is autoregressive at 12.5 Hz: at each step the prefix
/// <c>[text_embeddings ++ input_linear([bos, latent_0..latent_{f-1}])]</c> runs through the causal backbone;
/// the last hidden conditions <c>flow_net</c>, and an <c>lsd_decode</c> step turns a noise sample into the next
/// 32-d latent. Latents then decode to audio via <see cref="PocketTtsMimiDecoder"/>. The per-step math
/// (backbone + flow_net) is verified bit-exact; full generation differs only by the noise draw, so
/// <see cref="GenerateLatents"/> takes the noise explicitly for deterministic, parity-testable use.</para></summary>
internal sealed unsafe class PocketTtsFlowLm
{
    private const int Ldim = 32, Dim = 1024;
    private readonly string _p;

    private Tensor? _inputLinW;   // [1024,32]
    private Tensor? _bos;         // [32]
    private Tensor? _outNormW, _outNormB;
    private readonly PocketTtsStreamingTransformer _backbone;
    private readonly PocketTtsFlowNet _flowNet;

    public PocketTtsFlowLm(string prefix = "flow_lm")
    {
        _p = prefix;
        _backbone = new PocketTtsStreamingTransformer($"{prefix}.transformer", dim: Dim, heads: 16, layers: 6, ffn: 4096, context: null, layerScale: false);
        _flowNet = new PocketTtsFlowNet($"{prefix}.flow_net");
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _inputLinW = WhisperOps.EnsureF32(w[$"{_p}.input_linear.weight"]);
        _bos = WhisperOps.EnsureF32(w[$"{_p}.bos_emb"]);
        _outNormW = WhisperOps.EnsureF32(w[$"{_p}.out_norm.weight"]);
        _outNormB = WhisperOps.EnsureF32(w[$"{_p}.out_norm.bias"]);
        _backbone.LoadWeights(w);
        _flowNet.LoadWeights(w);
    }

    /// <summary>Deterministic AR decode. <paramref name="textEmb"/> = <c>[1, T, 1024]</c> conditioner output;
    /// <paramref name="noises"/> = <c>[N, 32]</c> per-frame noise. Returns latents <c>[1, 32, N]</c>.</summary>
    public Tensor GenerateLatents(IBackend backend, Tensor textEmb, float[][] noises)
    {
        int tText = (int)textEmb.Shape[1];
        int n = noises.Length;
        float* tep = (float*)textEmb.DataPointer;
        float* bos = (float*)_bos!.DataPointer;
        float* ilW = (float*)_inputLinW!.DataPointer;   // [1024,32]

        float[][] latents = new float[n][];
        for (int f = 0; f < n; f++)
        {
            int seqLen = f + 1;               // [bos, latent_0..latent_{f-1}]
            int total = tText + seqLen;
            Tensor full = new(new TensorShape(1, total, Dim), DType.F32);
            float* fp = (float*)full.DataPointer;
            // text prefix
            for (int i = 0; i < tText * Dim; i++) fp[i] = tep[i];
            // input_linear([bos, latents...])
            for (int j = 0; j < seqLen; j++)
            {
                float[] lat = j == 0 ? null! : latents[j - 1];
                long rowBase = (long)(tText + j) * Dim;
                for (int o = 0; o < Dim; o++)
                {
                    double acc = 0;
                    long wb = (long)o * Ldim;
                    for (int i = 0; i < Ldim; i++)
                    {
                        float v = j == 0 ? bos[i] : lat[i];
                        acc += (double)v * ilW[wb + i];
                    }
                    fp[rowBase + o] = (float)acc;
                }
            }

            Tensor tout = _backbone.Forward(backend, full, 1, total);
            full.Dispose();
            Tensor normed = new(tout.Shape, DType.F32);
            backend.LayerNorm(normed, tout, _outNormW!, _outNormB!, 1e-5f);
            tout.Dispose();

            // last position hidden
            float[] hidden = new float[Dim];
            float* np = (float*)normed.DataPointer;
            long lastBase = (long)(total - 1) * Dim;
            for (int i = 0; i < Dim; i++) hidden[i] = np[lastBase + i];
            normed.Dispose();

            // lsd_decode, 1 step: latent = noise + flow_net(hidden, s=0, t=1, noise)
            float[] noise = noises[f];
            float[] flowDir = _flowNet.Forward(hidden, 0f, 1f, noise);
            float[] latent = new float[Ldim];
            for (int i = 0; i < Ldim; i++) latent[i] = noise[i] + flowDir[i];
            latents[f] = latent;
        }

        // assemble [1, 32, N]
        Tensor outLat = new(new TensorShape(1, Ldim, n), DType.F32);
        float* op = (float*)outLat.DataPointer;
        for (int c = 0; c < Ldim; c++)
            for (int f = 0; f < n; f++)
                op[(long)c * n + f] = latents[f][c];
        return outLat;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_inputLinW is not null) yield return _inputLinW;
        if (_bos is not null) yield return _bos;
        if (_outNormW is not null) yield return _outNormW;
        if (_outNormB is not null) yield return _outNormB;
        foreach (Tensor t in _backbone.EnumerateWeights()) yield return t;
        foreach (Tensor t in _flowNet.EnumerateWeights()) yield return t;
    }
}
