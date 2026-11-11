using HartsyInference.Core.Configuration;
using HartsyInference.ThreeD.Models.TrellisV2;
using Xunit;

namespace HartsyInference.ThreeD.Tests.TrellisV2;

[Collection("Trellis2ModelFolders")]
public sealed class Trellis2ModelFilesTests
{
    private static string NewRoot()
    {
        string root = Path.Combine(Path.GetTempPath(), "trellis2-files-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(root, "3D", "trellis2"));
        return root;
    }

    private static void Touch(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, [0]);
    }

    [Fact]
    public void Resolve_FindsTheSharedFilesInTheSiblingFoldersFromAnyFileInTheFlowFolder()
    {
        string root = NewRoot();
        try
        {
            string flows = Path.Combine(root, "3D", "trellis2");
            foreach (string name in new[]
            {
                Trellis2ModelFiles.PipelineFile, Trellis2ModelFiles.StructureFlowFile, Trellis2ModelFiles.StructureDecoderFile,
                Trellis2ModelFiles.ShapeFlow512File, Trellis2ModelFiles.ShapeFlow1024File,
            })
                Touch(Path.Combine(flows, name));
            Touch(Path.Combine(root, "TRELLIS2-ShapeVAE", Trellis2ModelFiles.ShapeDecoderFile));
            Touch(Path.Combine(root, "DINOv3", Trellis2ModelFiles.DinoFile));

            Trellis2ModelFiles viaDirectory = Trellis2ModelFiles.Resolve(flows);
            Trellis2ModelFiles viaFile = Trellis2ModelFiles.Resolve(Path.Combine(flows, Trellis2ModelFiles.PipelineFile));

            Assert.Equal(viaDirectory, viaFile);
            Assert.EndsWith(Trellis2ModelFiles.ShapeDecoderFile, viaDirectory.ShapeDecoder, StringComparison.Ordinal);
            Assert.Contains("DINOv3", viaDirectory.Dino, StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void Resolve_PrefersTheCatalogLocationsAndAcceptsTheShapeDecoderBesideTheFlows()
    {
        string root = NewRoot();
        try
        {
            string flows = Path.Combine(root, "3D", "trellis2");
            foreach (string name in new[]
            {
                Trellis2ModelFiles.PipelineFile, Trellis2ModelFiles.StructureFlowFile, Trellis2ModelFiles.StructureDecoderFile,
                Trellis2ModelFiles.ShapeFlow512File, Trellis2ModelFiles.ShapeFlow1024File, Trellis2ModelFiles.ShapeDecoderFile,
            })
                Touch(Path.Combine(flows, name));
            Touch(Path.Combine(root, "Vision", "Dinov3", Trellis2ModelFiles.DinoFile));
            Touch(Path.Combine(root, "DINOv3", Trellis2ModelFiles.DinoFile));

            Trellis2ModelFiles files = Trellis2ModelFiles.Resolve(flows);

            Assert.Equal(Path.Combine(flows, Trellis2ModelFiles.ShapeDecoderFile), files.ShapeDecoder);
            Assert.Equal(Path.Combine(root, "Vision", "Dinov3", Trellis2ModelFiles.DinoFile), files.Dino);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void Resolve_NamesEveryMissingFileAtOnce()
    {
        string root = NewRoot();
        try
        {
            Touch(Path.Combine(root, "3D", "trellis2", Trellis2ModelFiles.PipelineFile));

            string flows = Path.Combine(root, "3D", "trellis2");
            FileNotFoundException error = Assert.Throws<FileNotFoundException>(() => Trellis2ModelFiles.Resolve(flows));

            Assert.Contains(Trellis2ModelFiles.StructureFlowFile, error.Message, StringComparison.Ordinal);
            Assert.Contains(Trellis2ModelFiles.ShapeFlow1024File, error.Message, StringComparison.Ordinal);
            Assert.Contains(Trellis2ModelFiles.ShapeDecoderFile, error.Message, StringComparison.Ordinal);
            Assert.Contains("DINOv3", error.Message, StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void Resolve_RejectsAMissingFolder()
    {
        string missing = Path.Combine(Path.GetTempPath(), "no-such-" + Guid.NewGuid().ToString("N"));
        Assert.Throws<DirectoryNotFoundException>(() => Trellis2ModelFiles.Resolve(missing));
    }

    [Fact]
    public void Resolve_FindsTheDinoEncoderInAnOverriddenVisionFolder()
    {
        string root = NewRoot();
        KnobStore.Set(EngineKnobs.ModelFolderVision, "vision-models");
        try
        {
            string flows = Path.Combine(root, "3D", "trellis2");
            foreach (string name in new[]
            {
                Trellis2ModelFiles.PipelineFile, Trellis2ModelFiles.StructureFlowFile, Trellis2ModelFiles.StructureDecoderFile,
                Trellis2ModelFiles.ShapeFlow512File, Trellis2ModelFiles.ShapeFlow1024File, Trellis2ModelFiles.ShapeDecoderFile,
            })
                Touch(Path.Combine(flows, name));
            string dino = Path.Combine(root, "vision-models", "Dinov3", Trellis2ModelFiles.DinoFile);
            Touch(dino);

            Assert.Equal(dino, Trellis2ModelFiles.Resolve(flows).Dino);
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.ModelFolderVision);
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void FindMatterFile_HonoursTheVisionFolderSetting()
    {
        string root = NewRoot();
        try
        {
            Assert.Null(Trellis2ModelFiles.FindMatterFile(root));

            string defaultPlace = Path.Combine(root, "Vision", Trellis2ModelFiles.MatterFolder, Trellis2ModelFiles.MatterFile);
            Touch(defaultPlace);
            Assert.Equal(defaultPlace, Trellis2ModelFiles.FindMatterFile(root));

            File.Delete(defaultPlace);
            KnobStore.Set(EngineKnobs.ModelFolderVision, "vision-models");
            Assert.Null(Trellis2ModelFiles.FindMatterFile(root));
            string moved = Path.Combine(root, "vision-models", Trellis2ModelFiles.MatterFolder, Trellis2ModelFiles.MatterFile);
            Touch(moved);
            Assert.Equal(moved, Trellis2ModelFiles.FindMatterFile(root));
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.ModelFolderVision);
            Directory.Delete(root, recursive: true);
        }
    }
}
