using SharpInference.Audio.Models.Whisper;
using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Audio.Models.VibeVoice;

/// <summary>Projects VAE latents into the LM hidden space. Used twice on the multi-speaker
/// variants — once for acoustic (64 → lm_hidden) and once for semantic (128 → lm_hidden) —
/// and once on the streaming variant (acoustic only).
///
/// <para>Architecture: <c>fc1 → LlamaRMSNorm(eps=1e-6) → fc2</c>. Both linears carry bias.
/// Identical math on the Python side (<c>vibevoice/modular/modeling_vibevoice.py::SpeechConnector</c>).
/// Inputs and outputs are channels-last <c>[B, T, D]</c>; <c>T</c> on the inference hot
/// path is typically 1 (single VAE frame per LM step).</para></summary>
internal sealed unsafe class SpeechConnector
{
    private readonly int _inputDim;
    private readonly int _outputDim;

    private Tensor? _fc1W, _fc1B;
    private Tensor? _normW;
    private Tensor? _fc2W, _fc2B;

    public SpeechConnector(int inputDim, int outputDim)
    {
        if (inputDim <= 0) throw new ArgumentException("inputDim must be positive.", nameof(inputDim));
        if (outputDim <= 0) throw new ArgumentException("outputDim must be positive.", nameof(outputDim));
        _inputDim = inputDim;
        _outputDim = outputDim;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        _fc1W = WhisperOps.EnsureF32(w[$"{prefix}.fc1.weight"]);
        _fc1B = WhisperOps.EnsureF32(w[$"{prefix}.fc1.bias"]);
        _normW = WhisperOps.EnsureF32(w[$"{prefix}.norm.weight"]);
        _fc2W = WhisperOps.EnsureF32(w[$"{prefix}.fc2.weight"]);
        _fc2B = WhisperOps.EnsureF32(w[$"{prefix}.fc2.bias"]);
    }

    /// <summary>Forward: <paramref name="x"/> <c>[B, T, in_dim]</c> → <c>[B, T, out_dim]</c>.
    /// Allocates and returns a fresh tensor; caller owns disposal.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int batch, int seqLen)
    {
        if (_fc1W is null) throw new InvalidOperationException("SpeechConnector weights not loaded.");

        Tensor y1 = WhisperOps.ProjectLinear(backend, x, _fc1W!, _fc1B, batch, seqLen, _inputDim, _outputDim);
        Tensor normed = new(y1.Shape, DType.F32);
        backend.RmsNorm(normed, y1, _normW!, 1e-6f);
        y1.Dispose();
        Tensor y2 = WhisperOps.ProjectLinear(backend, normed, _fc2W!, _fc2B, batch, seqLen, _outputDim, _outputDim);
        normed.Dispose();
        return y2;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_fc1W, _fc1B, _normW, _fc2W, _fc2B];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }
}
