using HartsyInference.Core.Tensors;

namespace HartsyInference.World.Pipelines.SanaWm;

/// <summary>The two conditioning encodes every SANA-WM run starts with, shared by the clip pipeline and the stream: each opens
/// its component for exactly one stage and releases it before returning. Values cross the stage boundary rounded to BF16.</summary>
internal static class SanaWmEncoding
{
    private static readonly SanaWmGuidanceBranch[] Branches = [SanaWmGuidanceBranch.Unconditional, SanaWmGuidanceBranch.Conditional];

    /// <summary>Encodes the prompt and the negative prompt. The caller owns the result.</summary>
    internal static SanaWmTextConditioning EncodeText(Func<SanaWmStage<ISanaWmTextEncoder>> open, string prompt, string negativePrompt)
    {
        using SanaWmStage<ISanaWmTextEncoder> stage = open();
        SanaWmTextConditioning text = stage.Component.Encode(prompt, negativePrompt);
        bool accepted = false;
        try
        {
            foreach (SanaWmGuidanceBranch branch in Branches)
            {
                RequireShape(text.Features(branch), "caption features", 4);
                RequireShape(text.Mask(branch), "caption mask", 2);
                SanaWmPipeline.RoundToBf16(text.Features(branch).AsSpan<float>());
            }
            accepted = true;
            return text;
        }
        finally
        {
            if (!accepted)
                text.Dispose();
        }
    }

    /// <summary>Cover-crops the seed image and encodes it to the normalized latent <c>[1, 128, 1, 22, 40]</c>. The caller owns
    /// the result.</summary>
    internal static Tensor EncodeFirstFrame(Func<SanaWmStage<ISanaWmImageEncoder>> open, byte[] imageRgb, int imageWidth, int imageHeight,
        Action<string, Tensor>? tap)
    {
        using Tensor video = SanaWmPipeline.PrepareImage(imageRgb, imageWidth, imageHeight);
        SanaWmPipeline.RoundToBf16(video.AsSpan<float>());
        Tensor first;
        using (SanaWmStage<ISanaWmImageEncoder> stage = open())
            first = stage.Component.Encode(video);
        bool accepted = false;
        try
        {
            TensorShape expected = FirstLatentShape;
            if (first.DType != DType.F32 || first.Shape != expected)
                throw new InvalidOperationException($"The image encoder returned {first.DType} {first.Shape}; expected F32 {expected}.");
            SanaWmPipeline.RoundToBf16(first.AsSpan<float>());
            tap?.Invoke("first_latent", first);
            accepted = true;
            return first;
        }
        finally
        {
            if (!accepted)
                first.Dispose();
        }
    }

    /// <summary>Shape of the seed frame's latent.</summary>
    internal static TensorShape FirstLatentShape => new([1, SanaWmPipeline.LatentChannels, 1, SanaWmPipeline.LatentHeight,
        SanaWmPipeline.LatentWidth]);

    private static void RequireShape(Tensor tensor, string name, int rank)
    {
        if (tensor.DType != DType.F32 || tensor.Shape.Rank != rank)
            throw new InvalidOperationException($"The text encoder returned {name} as {tensor.DType} {tensor.Shape}; expected F32 rank {rank}.");
    }
}
