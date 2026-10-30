using HartsyInference.Cli.Commands;
using HartsyInference.Cli.Infra;
using HartsyInference.Engine;
using Xunit;

namespace HartsyInference.Cli.Tests;

/// <summary>Every reference option the image command declares reaches the parameter bag. A declared option that is not bound is dropped
/// without an error, and the engine then applies its own default, so the run looks successful with the wrong settings.</summary>
public sealed class ImageReferenceBindingTests
{
    [Fact]
    public void Every_Reference_Option_Reaches_The_Parameter_Bag()
    {
        ImageCommand.Settings settings = new()
        {
            ReferenceAdapter = "custom.safetensors",
            ReferenceAdapterStrength = 0.5f,
            ReferenceGuidance = 1f,
            ReferenceCorrection = "none",
            ReferenceRescalePhi = 0.25f,
            ReferenceStart = 0.1f,
            ReferenceEnd = 0.6f,
            ReferenceOutside = "drop",
        };
        ParamState parameters = new(Modality.Image);

        ImageCommand.BindReference(parameters, settings);

        Assert.Equal("custom.safetensors", parameters.GetStringOrNull("reference-adapter"));
        Assert.Equal(0.5f, parameters.GetFloatOrNull("reference-strength"));
        Assert.Equal(1f, parameters.GetFloatOrNull("reference-guidance"));
        Assert.Equal("none", parameters.GetStringOrNull("reference-correction"));
        Assert.Equal(0.25f, parameters.GetFloatOrNull("reference-rescale-phi"));
        Assert.Equal(0.1f, parameters.GetFloatOrNull("reference-start"));
        Assert.Equal(0.6f, parameters.GetFloatOrNull("reference-end"));
        Assert.Equal("drop", parameters.GetStringOrNull("reference-outside"));
    }

    [Fact]
    public void Unset_Reference_Options_Stay_Out_Of_The_Parameter_Bag()
    {
        ParamState parameters = new(Modality.Image);

        ImageCommand.BindReference(parameters, new ImageCommand.Settings());

        Assert.Null(parameters.GetFloatOrNull("reference-guidance"));
        Assert.Null(parameters.GetStringOrNull("reference-outside"));
    }
}
