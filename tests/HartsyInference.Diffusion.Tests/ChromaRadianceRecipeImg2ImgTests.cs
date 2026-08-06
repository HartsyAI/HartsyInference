using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Recipe-layer img2img for Chroma Radiance — the <see cref="ImageRequest"/> → <c>ImageToImageRequest</c>
/// binding Phase 1 added, as opposed to <see cref="ChromaRadianceImg2ImgTests"/> which drives the diffusion pipeline
/// directly with an already-built request.
///
/// <para>Radiance is the family where the dimension contract is non-trivial: the pipeline pads the latent up to the
/// patch grid but validates the img2img source against the <b>unpadded</b> request size, so a recipe that resolved the
/// init image at padded dimensions would throw at generation time. A strength-0 pass-through exercises that whole chain
/// — resolve at the right size, promote the request, reach the pipeline — and needs no trained weights, because the
/// pass-through short-circuits before any forward pass.</para></summary>
public sealed class ChromaRadianceRecipeImg2ImgTests
{
    private const int Width = 64;
    private const int Height = 64;

    private static ImageData Gradient(int width, int height)
    {
        byte[] rgb = new byte[width * height * 3];
        for (int i = 0; i < rgb.Length; i++)
        {
            rgb[i] = (byte)((i * 19) & 0xFF);
        }
        return new ImageData { Rgb = rgb, Width = width, Height = height };
    }

    /// <summary>Weights stay unloaded on purpose: reaching a forward pass would prove the pass-through never fired.</summary>
    private static ChromaRadianceRecipePipeline MakeRecipePipeline(CpuBackend backend)
    {
        T5TextEncoder t5 = new T5TextEncoder(T5TextEncoderConfig.Xxl);
        ChromaRadianceTransformer transformer = new ChromaRadianceTransformer(ChromaRadianceConfig.V1);
        ChromaRadiancePipeline pipeline = new ChromaRadiancePipeline(backend, t5, transformer, ChromaRadianceConfig.V1);
        return new ChromaRadianceRecipePipeline(
            pipeline, ChromaRadianceConfig.V1, new T5Tokenizer(maxLength: 256),
            new SafeTensorsLoader(), new SafeTensorsLoader());
    }

    private static ImageRequest RequestWith(ImageData init, double creativity) => new ImageRequest
    {
        Prompt = "ignored — the pass-through returns before any text encode",
        Width = Width,
        Height = Height,
        Steps = 4,
        Seed = 42,
        Img2Img = new Img2Img { InitImage = init, Creativity = creativity },
    };

    [Fact]
    public void Generate_WithCreativityZero_ReturnsTheInitImageUnchanged()
    {
        using CpuBackend backend = new CpuBackend();
        using ChromaRadianceRecipePipeline recipe = MakeRecipePipeline(backend);
        ImageData init = Gradient(Width, Height);

        ImageResult result = recipe.Generate(RequestWith(init, creativity: 0.0), progress: null, cancel: default);

        Assert.Equal(Width, result.Width);
        Assert.Equal(Height, result.Height);
        Assert.Equal(init.Rgb, result.Rgb);
    }

    /// <summary>The init image is resized to the request's size, so a differently-sized source must still satisfy the
    /// pipeline's exact-shape check rather than throwing.</summary>
    [Fact]
    public void Generate_ResizesAnOffSizeInitImageToTheRequestedSize()
    {
        using CpuBackend backend = new CpuBackend();
        using ChromaRadianceRecipePipeline recipe = MakeRecipePipeline(backend);

        ImageResult result = recipe.Generate(
            RequestWith(Gradient(37, 21), creativity: 0.0), progress: null, cancel: default);

        Assert.Equal(Width, result.Width);
        Assert.Equal(Height, result.Height);
        Assert.Equal(Width * Height * 3, result.Rgb.Length);
    }

    /// <summary>Without an init image the recipe must still build a plain text-to-image request — i.e. the binder is
    /// inert rather than always promoting.</summary>
    [Fact]
    public void Generate_WithoutAnInitImage_DoesNotTakeTheImg2ImgPath()
    {
        using CpuBackend backend = new CpuBackend();
        using ChromaRadianceRecipePipeline recipe = MakeRecipePipeline(backend);
        ImageRequest request = new ImageRequest { Prompt = "test", Width = Width, Height = Height, Steps = 4, Seed = 42 };

        // Unloaded weights: a text-to-image run reaches the T5 forward and fails there. Reaching that point is the
        // assertion — it proves no pass-through happened, which is only possible if no init image was bound.
        Assert.ThrowsAny<Exception>(() => recipe.Generate(request, progress: null, cancel: default));
    }
}
