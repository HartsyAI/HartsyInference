using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Core.Tensors;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>Exact tests for the S3 FSQ tokenizer packing (<c>tanh → round → shift → base-3 pack</c>,
/// D=8 / L=3 → 6561 codes). The encoder forward needs <c>speech_tokenizer_v2.onnx</c>; the packing math
/// is deterministic and validated here against hand-computed codes.</summary>
public sealed unsafe class CosyVoiceTokenizerTests
{
    private static Tensor Latent(int t, ReadOnlySpan<float> frame0)
    {
        Tensor z = new(new TensorShape(1, t, 8), DType.F32);
        float* p = (float*)z.DataPointer;
        for (int j = 0; j < 8; j++) p[j] = frame0[j];
        return z;
    }

    [Fact]
    public void PackFsq_MixedPattern_MatchesHandComputed()
    {
        // shifts per channel: [+1,-1,0,+1,0,-1,+1,0] → [2,0,1,2,1,0,2,1]
        // token = 2·1 + 0·3 + 1·9 + 2·27 + 1·81 + 0·243 + 2·729 + 1·2187 = 2+9+54+81+1458+2187 = 3791.
        using Tensor z = Latent(1, [9, -9, 0, 9, 0, -9, 9, 0]);
        int[] tok = S3Tokenizer.PackFsqTokens(z, 8, 3);
        Assert.Equal(3791, tok[0]);
    }

}
