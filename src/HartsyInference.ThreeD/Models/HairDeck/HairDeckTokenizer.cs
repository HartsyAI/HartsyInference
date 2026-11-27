using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.HairDeck;

/// <summary>The 40-channel card tokens of the head frame (<c>hairdeck/models/tokenizer.py</c>, <c>cards/tokens.py</c>):
/// root 0-2 (standardised), latent 3-34 (standardised), root width direction 35-37, shape class 38 (+-1), existence 39 (+-1).
/// Decoding undoes the standardisation; the clean and sign rules put generated tokens in the format of the training tokens.</summary>
public sealed unsafe class HairDeckTokenizer
{
    /// <summary>Channels of one card token.</summary>
    public const int TokenDim = 40;

    /// <summary>Channels of the card latent.</summary>
    public const int LatentDim = 32;

    private const int RootAt = 0;
    private const int LatentAt = 3;
    private const int NormalAt = 35;
    private const int ClassAt = 38;
    private const int ExistAt = 39;
    private const float MinNormal = 1e-6f;

    private readonly float[] _rootMu;
    private readonly float[] _rootSd;
    private readonly float[] _latentMu;
    private readonly float[] _latentSd;

    /// <summary>Builds the tokenizer from the head-frame root and latent statistics.</summary>
    public HairDeckTokenizer(ReadOnlySpan<float> rootMu, ReadOnlySpan<float> rootSd, ReadOnlySpan<float> latentMu, ReadOnlySpan<float> latentSd)
    {
        if (rootMu.Length != 3 || rootSd.Length != 3) throw new ArgumentException("Root statistics have three channels.");
        if (latentMu.Length != LatentDim || latentSd.Length != LatentDim) throw new ArgumentException($"Latent statistics have {LatentDim} channels.");
        _rootMu = rootMu.ToArray();
        _rootSd = rootSd.ToArray();
        _latentMu = latentMu.ToArray();
        _latentSd = latentSd.ToArray();
    }

    /// <summary>Reads slot <paramref name="slot"/> of tokens <c>[N, 40]</c>: the root in world units, the latent, the root width
    /// direction (as stored), the shape class (1 for a triangle) and the existence flag.</summary>
    public void DecodeSlot(Tensor tokens, int slot, Span<float> root, Span<float> latent, Span<float> normal, out int shapeClass, out bool exists)
    {
        ValidateTokens(tokens);
        float* t = (float*)tokens.DataPointer + (long)slot * TokenDim;
        for (int k = 0; k < 3; k++)
        {
            root[k] = t[RootAt + k] * _rootSd[k] + _rootMu[k];
            normal[k] = t[NormalAt + k];
        }
        for (int k = 0; k < LatentDim; k++) latent[k] = t[LatentAt + k] * _latentSd[k] + _latentMu[k];
        shapeClass = t[ClassAt] > 0f ? 1 : 0;
        exists = t[ExistAt] > 0f;
    }

    /// <summary>Clean tokens (the training-time format): non-existing slots zeroed with existence -1, existing root widths of
    /// unit length, shape class snapped to +-1 (0 counts as +1). The caller disposes the result.</summary>
    public Tensor Clean(Tensor tokens)
    {
        ValidateTokens(tokens);
        int n = (int)tokens.Shape[0];
        Tensor y = HairDeckNumerics.Create(n, TokenDim);
        float* x = (float*)tokens.DataPointer;
        float* o = (float*)y.DataPointer;
        for (int i = 0; i < n; i++)
        {
            float* src = x + (long)i * TokenDim;
            float* dst = o + (long)i * TokenDim;
            if (src[ExistAt] <= 0f)
            {
                for (int k = 0; k < TokenDim; k++) dst[k] = 0f;
                dst[ExistAt] = -1f;
                continue;
            }
            for (int k = 0; k < TokenDim; k++) dst[k] = src[k];
            float norm = MathF.Sqrt(src[NormalAt] * src[NormalAt] + src[NormalAt + 1] * src[NormalAt + 1] + src[NormalAt + 2] * src[NormalAt + 2]);
            float inv = 1f / MathF.Max(norm, MinNormal);
            for (int k = 0; k < 3; k++) dst[NormalAt + k] = src[NormalAt + k] * inv;
            dst[ClassAt] = src[ClassAt] > 0f ? 1f : src[ClassAt] < 0f ? -1f : 1f;
            dst[ExistAt] = 1f;
        }
        return y;
    }

    /// <summary>Clean tokens with the root width direction of every existing card signed as the training tokens are: flipped
    /// where it points against the root's offset from the mean root of the hairstyle. The caller disposes the result.</summary>
    public Tensor SignNormals(Tensor clean)
    {
        ValidateTokens(clean);
        int n = (int)clean.Shape[0];
        Tensor y = HairDeckNumerics.Clone(clean);
        float* p = (float*)y.DataPointer;
        double[] mean = new double[3];
        int count = 0;
        for (int i = 0; i < n; i++)
        {
            if (p[(long)i * TokenDim + ExistAt] <= 0f) continue;
            count++;
            for (int k = 0; k < 3; k++) mean[k] += p[(long)i * TokenDim + RootAt + k] * _rootSd[k] + _rootMu[k];
        }
        if (count == 0) return y;
        for (int k = 0; k < 3; k++) mean[k] /= count;

        for (int i = 0; i < n; i++)
        {
            float* row = p + (long)i * TokenDim;
            if (row[ExistAt] <= 0f) continue;
            double dot = 0;
            for (int k = 0; k < 3; k++)
            {
                double root = row[RootAt + k] * _rootSd[k] + _rootMu[k];
                dot += row[NormalAt + k] * (root - mean[k]);
            }
            if (dot < 0)
                for (int k = 0; k < 3; k++) row[NormalAt + k] = -row[NormalAt + k];
        }
        return y;
    }

    private static void ValidateTokens(Tensor tokens)
    {
        ArgumentNullException.ThrowIfNull(tokens);
        if (tokens.Shape.Rank != 2 || tokens.Shape[1] != TokenDim)
            throw new ArgumentException($"HairDeck card tokens must be [N, {TokenDim}], got {tokens.Shape}.", nameof(tokens));
        if (tokens.DType != DType.F32)
            throw new ArgumentException($"HairDeck card tokens must be F32, got {tokens.DType}.", nameof(tokens));
    }
}
