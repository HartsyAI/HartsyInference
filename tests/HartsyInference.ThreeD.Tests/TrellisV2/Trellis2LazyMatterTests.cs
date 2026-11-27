using HartsyInference.ThreeD.Models.TrellisV2;
using HartsyInference.ThreeD.Pipelines.Stages;
using Xunit;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

[Collection("Trellis2ModelFolders")]
public sealed class Trellis2LazyMatterTests
{
    [Fact]
    public void IsAvailable_FollowsTheDiskAfterTheMatterWasBuilt()
    {
        string root = Path.Combine(Path.GetTempPath(), "trellis2-matter-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            using Trellis2LazyMatter matter = new(root);
            Assert.False(matter.IsAvailable);

            string checkpoint = Path.Combine(root, "Vision", Trellis2ModelFiles.MatterFolder, Trellis2ModelFiles.MatterFile);
            Directory.CreateDirectory(Path.GetDirectoryName(checkpoint)!);
            File.WriteAllBytes(checkpoint, [0]);

            // A checkpoint installed after the pipeline loaded is seen; the same matter would otherwise have been fixed at "none".
            Assert.True(matter.IsAvailable);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void Alpha_WithoutACheckpointNamesTheMissingFileInsteadOfLoadingAnything()
    {
        string root = Path.Combine(Path.GetTempPath(), "trellis2-matter-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            using Trellis2LazyMatter matter = new(root);

            FileNotFoundException error = Assert.Throws<FileNotFoundException>(
                () => matter.Alpha(null!, new byte[12], 2, 2));

            Assert.Contains("BiRefNet", error.Message, StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }
}
