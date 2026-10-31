using HartsyInference.Cli.Commands;
using HartsyInference.Cli.Dispatch;
using HartsyInference.Cli.Infra;
using HartsyInference.Engine;
using HartsyInference.Engine.Features;
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
    public void Bound_Reference_Options_Reach_The_Engine_Extras()
    {
        // The second half of the path: the bag to the request extras the engine reads. A key typo here drops an option just as silently.
        ParamState parameters = new(Modality.Image);
        ImageCommand.BindReference(parameters, new ImageCommand.Settings
        {
            ReferenceAdapter = "custom.safetensors",
            ReferenceAdapterStrength = 0.5f,
            ReferenceGuidance = 1f,
            ReferenceCorrection = "none",
            ReferenceRescalePhi = 0.25f,
            ReferenceStart = 0.1f,
            ReferenceEnd = 0.6f,
            ReferenceOutside = "drop",
        });

        IReadOnlyDictionary<string, object> extra = GenerationDispatch.BuildImageExtra(parameters);

        Assert.Equal("custom.safetensors", extra[RequestExtras.ReferenceAdapter]);
        Assert.Equal(0.5f, (float)extra[RequestExtras.ReferenceStrength]);
        Assert.Equal(1f, (float)extra[RequestExtras.ReferenceGuidance]);
        Assert.Equal("none", extra[RequestExtras.ReferenceCorrection]);
        Assert.Equal(0.25f, (float)extra[RequestExtras.ReferenceRescalePhi]);
        Assert.Equal(0.1f, (float)extra[RequestExtras.ReferenceStart]);
        Assert.Equal(0.6f, (float)extra[RequestExtras.ReferenceEnd]);
        Assert.Equal("drop", extra[RequestExtras.ReferenceOutside]);
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
