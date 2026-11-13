using System.Globalization;
using HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

namespace HartsyInference.ThreeD.Tests.Paint;

/// <summary>The torn-input warning: the unwrap that came out as confetti on the 38,613-face bust (10,453 charts) warns, a clean mesh and a tiny one do not.</summary>
public sealed class PaintInputAuditTests
{
    [Fact]
    public void TheTornBustFromTheGpuRunWarns_WithTheNumbersAndTheFix()
    {
        string? warning = PaintInputAudit.TornMeshWarning(38_613, 10_453);

        Assert.NotNull(warning);
        Assert.Contains("10,453 charts", warning);
        Assert.Contains("3.7 faces", warning);
        Assert.Contains("remove-small-components", warning);
        Assert.Contains("fill-holes", warning);
        Assert.Contains("decimate", warning);
    }

    [Theory]
    [InlineData(38_613, 400)]   // a clean mesh: a few hundred charts of about 100 faces
    [InlineData(38_613, 3_000)] // 12.9 faces per chart: busy but plausible
    [InlineData(150, 99)]       // fewer than 100 charts is never a problem, whatever their size
    [InlineData(10, 5)]
    [InlineData(0, 0)]
    public void PlausibleUnwrapsDoNotWarn(int faces, int charts) => Assert.Null(PaintInputAudit.TornMeshWarning(faces, charts));

    [Theory]
    [InlineData(1_199, 100)] // just under 12 faces per chart
    [InlineData(300, 300)]
    public void ImplausibleUnwrapsWarn(int faces, int charts) => Assert.NotNull(PaintInputAudit.TornMeshWarning(faces, charts));

    [Fact]
    public void TheTextDoesNotDependOnTheCulture()
    {
        CultureInfo before = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = new CultureInfo("de-DE");
            string? warning = PaintInputAudit.TornMeshWarning(38_613, 10_453);

            Assert.Contains("10,453 charts of 3.7 faces", warning);
        }
        finally
        {
            CultureInfo.CurrentCulture = before;
        }
    }

    [Fact]
    public void TheBoundaryIs12FacesPerChart() => Assert.Null(PaintInputAudit.TornMeshWarning(1_200, 100));
}
