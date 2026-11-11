using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

namespace HartsyInference.ThreeD.Tests.Paint;

/// <summary>Integration lane: the released Paint 2.0 checkpoint against the UNet's key map and shapes. Set <c>HARTSY_PAINT20_UNET</c> to the
/// <c>unet/diffusion_pytorch_model.safetensors</c> of <c>tencent/Hunyuan3D-2</c> <c>hunyuan3d-paint-v2-0</c>.</summary>
[Trait("Category", "Integration")]
public sealed class HunyuanPaintRealWeightsTests
{
    [RealWeightsFact("HARTSY_PAINT20_UNET")]
    public void Paint20Unet_LoadsEveryTensorIntoBothStreamsWithoutAMissingKey()
    {
        using SafeTensorsLoader loader = new();
        loader.Load(Environment.GetEnvironmentVariable("HARTSY_PAINT20_UNET")!);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();

        HunyuanPaintConfig config = HunyuanPaintLoader.Detect(weights);
        HunyuanPaintUnet unet = new(config, generating: true), reference = new(config, generating: false);
        unet.LoadWeights(weights, "unet");
        reference.LoadWeights(weights, "unet_dual");
        HunyuanPaintPromptTokens tokens = HunyuanPaintPromptTokens.FromWeights(weights, config, "unet");

        Assert.Equal(49, config.CameraClasses);
        Assert.Single(tokens.PerMaterial);
        Assert.Equal(new TensorShape(1, 77, 1024), tokens.Reference.Shape);
    }
}
