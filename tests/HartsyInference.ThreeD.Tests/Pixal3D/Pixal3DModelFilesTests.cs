using HartsyInference.ThreeD.Models.Pixal3D;
using HartsyInference.ThreeD.Models.TrellisV2;
using Xunit;

namespace HartsyInference.ThreeD.Tests.Pixal3D;

/// <summary>Where the Pixal3D install looks for its files: its own flows and NAF in the folder, the TRELLIS.2 decoders and DINOv3 beside it.</summary>
public sealed class Pixal3DModelFilesTests
{
    private static string Touch(string root, params string[] parts)
    {
        string path = Path.Combine([root, .. parts]);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, "x");
        return path;
    }

    [Fact]
    public void Resolve_FindsTheFlowsAndNafInTheFolderAndTheSharedFilesBesideIt()
    {
        string models = Directory.CreateTempSubdirectory("pixal3d-files-").FullName;
        try
        {
            string folder = Path.Combine(models, "3D", "pixal3d");
            string[] own =
            [
                Trellis2ModelFiles.PipelineFile, Trellis2ModelFiles.StructureFlowFile, Trellis2ModelFiles.ShapeFlow512File,
                Trellis2ModelFiles.ShapeFlow1024File, Pixal3DModelFiles.NafFile,
            ];
            foreach (string name in own)
                Touch(folder, name);
            string structureDecoder = Touch(models, "3D", "trellis2", Trellis2ModelFiles.StructureDecoderFile);
            string shapeDecoder = Touch(models, "TRELLIS2-ShapeVAE", Trellis2ModelFiles.ShapeDecoderFile);
            string dino = Touch(models, "DINOv3", Trellis2ModelFiles.DinoFile);

            Pixal3DModelFiles files = Pixal3DModelFiles.Resolve(folder);

            Assert.Equal(Path.Combine(folder, Pixal3DModelFiles.NafFile), files.Naf);
            Assert.Equal(Path.Combine(folder, Trellis2ModelFiles.ShapeFlow1024File), files.ShapeFlow1024);
            Assert.Equal(structureDecoder, files.StructureDecoder);
            Assert.Equal(shapeDecoder, files.ShapeDecoder);
            Assert.Equal(dino, files.Dino);
            Assert.Equal(files, Pixal3DModelFiles.Resolve(files.StructureFlow));
        }
        finally
        {
            Directory.Delete(models, recursive: true);
        }
    }

    [Fact]
    public void Resolve_ReadsNafFromItsSafetensorsRepackInPreferenceToThePickleAndWithoutIt()
    {
        string models = Directory.CreateTempSubdirectory("pixal3d-files-").FullName;
        try
        {
            string folder = Path.Combine(models, "3D", "pixal3d");
            foreach (string name in new[]
            {
                Trellis2ModelFiles.PipelineFile, Trellis2ModelFiles.StructureFlowFile, Trellis2ModelFiles.ShapeFlow512File,
                Trellis2ModelFiles.ShapeFlow1024File, Trellis2ModelFiles.StructureDecoderFile, Trellis2ModelFiles.ShapeDecoderFile,
            })
                Touch(folder, name);
            Touch(models, "DINOv3", Trellis2ModelFiles.DinoFile);
            string repack = Touch(folder, "naf_release.safetensors");

            Assert.Equal(repack, Pixal3DModelFiles.Resolve(folder).Naf);
            Touch(folder, Pixal3DModelFiles.NafFile);
            Assert.Equal(repack, Pixal3DModelFiles.Resolve(folder).Naf);
            File.Delete(repack);
            Assert.Equal(Path.Combine(folder, Pixal3DModelFiles.NafFile), Pixal3DModelFiles.Resolve(folder).Naf);
        }
        finally
        {
            Directory.Delete(models, recursive: true);
        }
    }

    [Fact]
    public void Resolve_PrefersADecoderInsideThePixal3DFolder()
    {
        string models = Directory.CreateTempSubdirectory("pixal3d-files-").FullName;
        try
        {
            string folder = Path.Combine(models, "3D", "pixal3d");
            foreach (string name in new[]
            {
                Trellis2ModelFiles.PipelineFile, Trellis2ModelFiles.StructureFlowFile, Trellis2ModelFiles.ShapeFlow512File,
                Trellis2ModelFiles.ShapeFlow1024File, Pixal3DModelFiles.NafFile, Trellis2ModelFiles.ShapeDecoderFile,
            })
                Touch(folder, name);
            string own = Touch(folder, Trellis2ModelFiles.StructureDecoderFile);
            Touch(models, "3D", "trellis2", Trellis2ModelFiles.StructureDecoderFile);
            Touch(models, "DINOv3", Trellis2ModelFiles.DinoFile);

            Assert.Equal(own, Pixal3DModelFiles.Resolve(folder).StructureDecoder);
        }
        finally
        {
            Directory.Delete(models, recursive: true);
        }
    }

    [Fact]
    public void Resolve_NamesEveryMissingFile()
    {
        string models = Directory.CreateTempSubdirectory("pixal3d-files-").FullName;
        try
        {
            string folder = Path.Combine(models, "3D", "pixal3d");
            Touch(folder, Trellis2ModelFiles.PipelineFile);

            FileNotFoundException error = Assert.Throws<FileNotFoundException>(() => Pixal3DModelFiles.Resolve(folder));

            Assert.Contains(Pixal3DModelFiles.NafFile, error.Message, StringComparison.Ordinal);
            Assert.Contains(Trellis2ModelFiles.StructureFlowFile, error.Message, StringComparison.Ordinal);
            Assert.Contains("DINOv3", error.Message, StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(models, recursive: true);
        }
    }

    [Fact]
    public void Resolve_RejectsAMissingFolder()
        => Assert.Throws<DirectoryNotFoundException>(() => Pixal3DModelFiles.Resolve(Path.Combine(Path.GetTempPath(), "no-such-pixal3d-folder")));
}
