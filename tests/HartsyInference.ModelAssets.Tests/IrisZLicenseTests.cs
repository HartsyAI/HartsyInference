using HartsyInference.ModelAssets.Metadata;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Iris-Z carries the FLUX non-commercial licence under the exact label HartsyWeb's licence list offers for it. A stored
/// value that is not that label maps to "Other" on upload, so the file would not show the licence it ships under.</summary>
public sealed class IrisZLicenseTests
{
    [Theory]
    [InlineData("iris-z-small")]
    [InlineData("iris-z-big")]
    public void IrisZCarriesTheSitesFluxNonCommercialLabel(string engineId)
    {
        Assert.Equal("FLUX Non-Commercial License", ModelIdentityCatalog.Find(engineId)!.License);
    }
}
