using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.OpenVoice;

/// <summary>OpenVoice tone-color extractor — the <c>ReferenceEncoder</c> that turns a reference mel into the
/// speaker conditioning vector <c>g [1, gin, 1]</c> (gin 256) the OpenVoice tone converter is conditioned on.
/// It is a stack of stride-2 Conv1d layers (channels-first over the mel time axis) with GroupNorm + ReLU,
/// followed by an attention pool over time that collapses the variable-length sequence to a single token, then
/// a Linear projection to <c>gin</c>.
///
/// <para>Reuses the shared <see cref="IBackend"/> Conv1d / GroupNorm / Linear ops and
/// <see cref="WhisperOps"/> helpers. Input is a mel <c>[1, nMels, T]</c> (channels-first); output is
/// <c>[1, gin, 1]</c> so it drops straight into the tone converter's <c>g</c> slot.</para>
///
/// <para>Structural scaffold (validation-pending): the layer count / channel schedule and the attention-pool
/// formulation are captured from <c>docs/Research/CHATTERBOX_ARCHITECTURE.md</c>'s sibling tone-color notes;
/// the precise checkpoint key spelling is a reconcile item against the downloaded <c>config.json</c>/
/// <c>checkpoint.pth</c>.</para></summary>
public sealed unsafe class OpenVoiceSpeakerEncoder : IDisposable
{
    private readonly OpenVoiceSpeakerConfig _cfg;
    private readonly ConvLayer[] _convs;

    private Tensor? _attnW, _attnB;     // attention pool Linear(lastC → 1)
    private Tensor? _projW, _projB;     // Linear(lastC → gin)
    private int _disposed;

    public OpenVoiceSpeakerConfig Config => _cfg;

    public OpenVoiceSpeakerEncoder(OpenVoiceSpeakerConfig cfg)
    {
        _cfg = cfg;
        _convs = new ConvLayer[cfg.Channels.Count];
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "ref_enc")
    {
        int inC = _cfg.NumMels;
        for (int i = 0; i < _convs.Length; i++)
        {
            int outC = _cfg.Channels[i];
            string cp = $"{prefix}.convs.{i}";
            _convs[i] = new ConvLayer
            {
                ConvW = WhisperOps.EnsureF32(w[$"{cp}.conv.weight"]),
                ConvB = TryGet(w, $"{cp}.conv.bias"),
                NormW = WhisperOps.EnsureF32(w[$"{cp}.norm.weight"]),
                NormB = WhisperOps.EnsureF32(w[$"{cp}.norm.bias"]),
                OutC = outC,
            };
            inC = outC;
        }
        _attnW = WhisperOps.EnsureF32(w[$"{prefix}.attn.weight"]);
        _attnB = TryGet(w, $"{prefix}.attn.bias");
        _projW = WhisperOps.EnsureF32(w[$"{prefix}.proj.weight"]);
        _projB = TryGet(w, $"{prefix}.proj.bias");
    }

    /// <summary>Extracts the tone-color vector <c>g [1, gin, 1]</c> from a mel <c>[1, nMels, t]</c>.</summary>
    public Tensor Extract(IBackend backend, Tensor mel, int t)
    {
        if (_attnW is null) throw new InvalidOperationException("OpenVoiceSpeakerEncoder weights not loaded.");

        Tensor x = new(mel.Shape, DType.F32);
        backend.CopyTo(x, mel);
        int curT = t, curC = _cfg.NumMels;

        foreach (ConvLayer layer in _convs)
        {
            int k = _cfg.KernelSize;
            int stride = _cfg.Stride;
            int pad = k / 2;
            int outT = (curT + 2 * pad - (k - 1) - 1) / stride + 1;
            if (outT < 1) outT = 1;
            Tensor conv = new(new TensorShape(1, layer.OutC, outT), DType.F32);
            backend.Conv1d(conv, x, layer.ConvW!, layer.ConvB, stride, pad, pad, 1, 1);
            x.Dispose();
            Tensor n = new(conv.Shape, DType.F32);
            backend.GroupNorm(n, conv, layer.NormW!, layer.NormB!, _cfg.NormGroups, 1e-5f);
            conv.Dispose();
            backend.LeakyRelu(n, n, 0f);
            x = n;
            curT = outT;
            curC = layer.OutC;
        }

        // Attention pool over time. x is [1, curC, curT]; to [1, curT, curC] for the scoring Linear.
        Tensor xCl = new(new TensorShape(1, curT, curC), DType.F32);
        backend.Transpose2D(xCl, x, curC, curT);
        Tensor score = WhisperOps.ProjectLinear(backend, xCl, _attnW!, _attnB, 1, curT, curC, 1);

        // Softmax over time.
        float* scp = (float*)score.DataPointer;
        float maxV = float.NegativeInfinity;
        for (int ti = 0; ti < curT; ti++) if (scp[ti] > maxV) maxV = scp[ti];
        float sum = 0f;
        for (int ti = 0; ti < curT; ti++) { float e = MathF.Exp(scp[ti] - maxV); scp[ti] = e; sum += e; }
        float inv = sum > 0 ? 1f / sum : 0f;
        for (int ti = 0; ti < curT; ti++) scp[ti] *= inv;

        // Weighted sum over time → [1, 1, curC].
        Tensor pooled = new(new TensorShape(1, 1, curC), DType.F32);
        float* pp = (float*)pooled.DataPointer;
        float* xclp = (float*)xCl.DataPointer;
        for (int c = 0; c < curC; c++)
        {
            float acc = 0f;
            for (int ti = 0; ti < curT; ti++) acc += scp[ti] * xclp[(long)ti * curC + c];
            pp[c] = acc;
        }
        score.Dispose();
        xCl.Dispose();
        x.Dispose();

        // Project to gin → [1, 1, gin] → [1, gin, 1].
        Tensor proj = WhisperOps.ProjectLinear(backend, pooled, _projW!, _projB, 1, 1, curC, _cfg.Gin);
        pooled.Dispose();
        return proj.Reshape(new TensorShape(1, _cfg.Gin, 1));
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] top = [_attnW, _attnB, _projW, _projB];
        foreach (Tensor? x in top) if (x is not null) yield return x;
        foreach (ConvLayer c in _convs) foreach (Tensor t in c.All()) yield return t;
    }

    private static Tensor? TryGet(IReadOnlyDictionary<string, Tensor> w, string key)
        => w.TryGetValue(key, out Tensor? t) ? WhisperOps.EnsureF32(t) : null;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }

    private sealed class ConvLayer
    {
        public Tensor? ConvW, ConvB, NormW, NormB;
        public int OutC;
        public IEnumerable<Tensor> All()
        {
            Tensor?[] a = [ConvW, ConvB, NormW, NormB];
            foreach (Tensor? t in a) if (t is not null) yield return t;
        }
    }
}
