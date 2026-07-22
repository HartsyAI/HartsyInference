using HartsyInference.Core.Tensors;

namespace HartsyInference.World.Camera;

/// <summary>Builds the per-token Plücker input Matrix-Game 3.0's <c>patch_embedding_wancamctrl</c> consumes: 6-channel
/// pixel-resolution ray maps grouped into transformer-token blocks (one token covers
/// <c>vae_spatial · patch = 32×32</c> pixels), flattened channel-major <c>(c, py, px)</c> per the reference rearrange.
/// Memory/conditioning frames without a camera trajectory get zero rows. Exact flatten order and the injection point
/// are validation-gated against the checkpoint.</summary>
public static unsafe class MatrixGame3PluckerTokens
{
    /// <summary>Builds <c>[T_total · gh · gw, 6 · block²]</c> tokens. <paramref name="framePoses"/> holds one
    /// camera-to-world pose per latent frame for the trailing frames (memory/past slots earlier than
    /// <paramref name="framePoses"/>.Count from the end are zero-filled).</summary>
    public static Tensor Build(IReadOnlyList<float[]?> framePoses, float fx, float fy, float cx, float cy,
        int gh, int gw, int pixelBlock)
    {
        int tokensPerFrame = gh * gw;
        int tokenDim = PluckerEmbedding.Channels * pixelBlock * pixelBlock;
        int frames = framePoses.Count;
        int width = gw * pixelBlock, height = gh * pixelBlock;

        Tensor tokens = new Tensor(new TensorShape(frames * tokensPerFrame, tokenDim), DType.F32);
        float* tp = (float*)tokens.DataPointer;
        float[] rays = new float[height * width * PluckerEmbedding.Channels];

        for (int f = 0; f < frames; f++)
        {
            float[]? pose = framePoses[f];
            if (pose is null) continue;   // zero rows (memory slots / unknown camera)
            PluckerEmbedding.Compute(pose, fx, fy, cx, cy, height, width, rays);
            for (int ty = 0; ty < gh; ty++)
                for (int tx = 0; tx < gw; tx++)
                {
                    long tokenBase = ((long)f * tokensPerFrame + (long)ty * gw + tx) * tokenDim;
                    int d = 0;
                    for (int c = 0; c < PluckerEmbedding.Channels; c++)
                        for (int py = 0; py < pixelBlock; py++)
                            for (int px = 0; px < pixelBlock; px++)
                            {
                                int yPix = ty * pixelBlock + py, xPix = tx * pixelBlock + px;
                                tp[tokenBase + d++] = rays[(yPix * width + xPix) * PluckerEmbedding.Channels + c];
                            }
                }
        }
        return tokens;
    }
}
