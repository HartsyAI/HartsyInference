using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.HeartMula;

/// <summary>HeartCodec ResidualVQ decode (from <c>vector_quantize_pytorch.ResidualVQ.get_output_from_indices</c>).
/// The 8 EMA codebooks each hold <c>embed [8192, 32]</c> (squeezed from the stored <c>[1, 8192, 32]</c>). For a
/// code grid <c>[Q=8, T]</c> the decode is: per layer look up <c>embed[q][code]</c> (a 32-vector), <b>sum</b> all
/// layers (residual reconstruction), then the shared <c>project_out</c> Linear <c>[512, 32]</c> (+bias) lifts the
/// 32-D summed code to the 512-D codec conditioning. (<c>project_in</c> is encode-only and unused here.)
///
/// <para>Keys: <c>{prefix}.layers.{q}._codebook.embed [1,8192,32]</c>, <c>{prefix}.project_out.{weight,bias}</c>.</para></summary>
public sealed unsafe class HeartCodecRvq
{
    private readonly int _nq, _codebookSize, _codebookDim, _dim;
    private readonly Tensor?[] _embed;     // [q] → [codebook_size, codebook_dim]
    private Tensor? _projOutW, _projOutB;  // Linear [dim, codebook_dim]

    public HeartCodecRvq(int numQuantizers, int codebookSize, int codebookDim, int dim)
    {
        _nq = numQuantizers;
        _codebookSize = codebookSize;
        _codebookDim = codebookDim;
        _dim = dim;
        _embed = new Tensor?[_nq];
    }

    public int Dim => _dim;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        for (int q = 0; q < _nq; q++)
        {
            // stored [1, codebook_size, codebook_dim]; squeeze dim 0 (the row layout is identical).
            _embed[q] = WhisperOps.EnsureF32(w[$"{prefix}.layers.{q}._codebook.embed"]);
        }
        _projOutW = WhisperOps.EnsureF32(w[$"{prefix}.project_out.weight"]);
        _projOutB = WhisperOps.EnsureF32(w[$"{prefix}.project_out.bias"]);
    }

    /// <summary>Decodes a code grid <c>[Q, T]</c> into the codec conditioning <c>[1, T, dim]</c>.</summary>
    public Tensor Decode(IBackend backend, int[,] codes, int t)
    {
        if (_embed[0] is null) throw new InvalidOperationException("HeartCodecRvq weights not loaded.");

        // Summed code vectors [1, T, codebook_dim].
        Tensor summed = new(new TensorShape(1, t, _codebookDim), DType.F32);
        float* sp = (float*)summed.DataPointer;
        for (long i = 0; i < (long)t * _codebookDim; i++) sp[i] = 0f;

        for (int q = 0; q < _nq; q++)
        {
            float* eb = (float*)_embed[q]!.DataPointer;   // [codebook_size, codebook_dim]
            for (int ti = 0; ti < t; ti++)
            {
                int idx = codes[q, ti];
                if ((uint)idx >= (uint)_codebookSize)
                    throw new ArgumentOutOfRangeException(nameof(codes), idx,
                        $"code at book {q}, t {ti} out of range [0,{_codebookSize}).");
                long rowBase = (long)idx * _codebookDim;
                long dst = (long)ti * _codebookDim;
                for (int d = 0; d < _codebookDim; d++) sp[dst + d] += eb[rowBase + d];
            }
        }

        // project_out: Linear [dim, codebook_dim] → [1, T, dim].
        Tensor outp = WhisperOps.ProjectLinear(backend, summed, _projOutW!, _projOutB, 1, t, _codebookDim, _dim);
        summed.Dispose();
        return outp;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        for (int q = 0; q < _nq; q++) if (_embed[q] is not null) yield return _embed[q]!;
        if (_projOutW is not null) yield return _projOutW;
        if (_projOutB is not null) yield return _projOutB;
    }
}
