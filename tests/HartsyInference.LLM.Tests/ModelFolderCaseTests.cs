using HartsyInference.Core.Configuration;
using HartsyInference.Engine;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Vision;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests;

/// <summary>Model lookup against a store whose folders are spelled differently from the engine's names. SwarmUI's
/// models root has <c>llm/</c> where the catalog says <c>LLM/</c>, so on a case-sensitive filesystem <c>qwen3</c>
/// resolved to nothing and a download would have created a second <c>LLM/</c> beside it. Matching other cases must
/// only fill in former misses: every lookup searches all exact spellings before any case variant, so whatever
/// resolved before still resolves to the same file. Points the models root at a temp tree; tests that need a folder
/// spelled apart from the engine's name skip on a case-insensitive filesystem.</summary>
[Collection("ModelsRootKnob")]
public sealed class ModelFolderCaseTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _root = Directory.CreateTempSubdirectory("casefix-models-").FullName;

    // The root to put back afterwards. (A Set now loads the settings file itself, so no read is needed before it.)
    private readonly string? _previousRoot = EngineKnobs.ModelsRoot.Value;

    public ModelFolderCaseTests(ITestOutputHelper output)
    {
        _output = output;
        KnobStore.Set(EngineKnobs.ModelsRoot, _root);
    }

    public void Dispose()
    {
        if (_previousRoot is null)
            KnobStore.Clear(EngineKnobs.ModelsRoot);
        else
            KnobStore.Set(EngineKnobs.ModelsRoot, _previousRoot);
        Directory.Delete(_root, recursive: true);
    }

    private static CatalogEntry Qwen3 => ModelCatalog.Find("qwen3")!;

    private static string Krea2File => ModelDownloader.PrimaryAsset(ModelCatalog.Find("krea2")!)!.FileName;

    [Fact]
    public void Qwen3_ResolvesTheCatalogFile_InALowercaseFolderHoldingTwoGgufs()
    {
        if (!CaseSensitive())
            return;
        // Two GGUFs make the folder itself ambiguous, so only the catalog's named file can resolve it.
        string file = Place("llm", "qwen3", Qwen3.Assets[0].FileName);
        Place("llm", "qwen3", "Qwen3-32B-Q4_K_M.gguf");

        ModelSpec spec = ModelResolver.Resolve("qwen3", modelPathArg: null, Modality.Text);

        Assert.Equal(file, spec.LocalPath);
    }

    [Fact]
    public void Qwen3_DownloadTarget_LandsInTheExistingLowercaseFolder()
    {
        if (!CaseSensitive())
            return;
        Directory.CreateDirectory(Path.Combine(_root, "llm"));

        Assert.Equal(Path.Combine(_root, "llm", "qwen3", Qwen3.Assets[0].FileName),
            ModelDownloader.TargetPath(Qwen3.Assets[0]));
        Assert.Equal(Qwen3.Assets[0], Assert.Single(ModelDownloader.MissingAssets(Qwen3)));
        Assert.Null(ModelResolver.Resolve("qwen3", modelPathArg: null, Modality.Text).LocalPath);
    }

    [Fact]
    public void AnExactCaseFolder_StillWins_OverALowercaseTwin()
    {
        if (!CaseSensitive())
            return;
        string exact = Place("LLM", "qwen3", Qwen3.Assets[0].FileName);
        Place("llm", "qwen3", Qwen3.Assets[0].FileName);

        Assert.Equal(exact, ModelResolver.Resolve("qwen3", modelPathArg: null, Modality.Text).LocalPath);
        Assert.Equal(exact, ModelDownloader.TargetPath(Qwen3.Assets[0]));
    }

    [Fact]
    public void ACatalogLessId_ResolvesThroughTheLowercaseModalityFolder()
    {
        if (!CaseSensitive())
            return;
        string gguf = Place("llm", "some-model", "model-Q4.gguf");

        Assert.Equal(gguf, ModelResolver.Resolve("some-model", modelPathArg: null, Modality.Text).LocalPath);
    }

    [Fact]
    public void AnExactModalityGuess_StillWins_OverTheCatalogFileInACaseVariantFolder()
    {
        if (!CaseSensitive())
            return;
        // Before case matching, the catalog path (Stable-Diffusion/Krea2/Turbo/...) missed and the guess
        // (Image/krea2) won.
        Place("stable-diffusion", "Krea2", "Turbo", Krea2File);
        string guess = Directory.CreateDirectory(Path.Combine(_root, "Image", "krea2")).FullName;

        Assert.Equal(guess, ModelResolver.Resolve("krea2", modelPathArg: null, Modality.Image).LocalPath);
    }

    [Fact]
    public void ACaseVariantCatalogFolderWithoutTheFile_FallsThroughToTheModalityGuess()
    {
        if (!CaseSensitive())
            return;
        Directory.CreateDirectory(Path.Combine(_root, "stable-diffusion", "Krea2", "Turbo"));
        string guess = Directory.CreateDirectory(Path.Combine(_root, "image", "krea2")).FullName;

        Assert.Equal(guess, ModelResolver.Resolve("krea2", modelPathArg: null, Modality.Image).LocalPath);
    }

    [Fact]
    public void AnExactLegacyName_StillWins_OverTheCanonicalNameInAnotherCase()
    {
        if (!CaseSensitive())
            return;
        ModelAsset asset = SideModels.Qwen3VL_4B;
        string legacy = Place(asset.TargetSubdir, asset.LegacyTargetNames[0]);
        Place(asset.TargetSubdir, asset.FileName.ToUpperInvariant());

        Assert.Equal(legacy, ModelDownloader.TargetPath(asset));
    }

    [Fact]
    public void YueCheckpointFolder_IsTheFolderItsDownloadLandsIn()
    {
        if (!CaseSensitive())
            return;
        // YuE is written through ModelDownloader and read through MusicCatalog: both must pick AudioLab's YuE/ folder.
        Directory.CreateDirectory(Path.Combine(_root, "Audio", "music", "YuE"));
        ModelAsset transformer = AudioWeightsCatalog.AssetsFor(AudioWeightsCatalog.YueId, "en-cot")[0];
        AudioModelSelector selector = new(AudioWeightsCatalog.YueId, "en-cot", null);

        string checkpoint = MusicCatalog.ResolveLocalCheckpoint(AudioWeightsCatalog.YueId, selector);

        Assert.Equal(Path.Combine(_root, "Audio", "music", "YuE", "en-cot"), checkpoint);
        Assert.Equal(checkpoint, Path.GetDirectoryName(ModelDownloader.TargetPath(transformer)));
        // Below the audio root nothing else is matched in another case, so other audio lookups keep their paths.
        Assert.Equal(Path.Combine(_root, "Audio", "music", "yue"), AudioModelRoot.WeightsDirectory("music", "yue"));
    }

    [Fact]
    public void SideModelLookup_SearchesAFolderSpelledInAnotherCase()
    {
        if (!CaseSensitive())
            return;
        // The CAM++ speaker encoder lookup: its folders are named audio/speaker and audio.
        string campplus = Place("Audio", "Speaker", "campplus.safetensors");

        Assert.Equal(campplus, ModelFileLocator.Find("campplus", Path.Combine("audio", "speaker"), "audio"));
    }

    [Fact]
    public void SideModelLookup_KeepsTheExactFolderMatch_WhenACaseVariantFolderAlsoHasIt()
    {
        if (!CaseSensitive())
            return;
        Place("audio", "Speaker", "campplus.safetensors");
        string exact = Place("audio", "campplus.safetensors");

        Assert.Equal(exact, ModelFileLocator.Find("campplus", Path.Combine("audio", "speaker"), "audio"));
    }

    [Fact]
    public void SideModelLookup_FallsThroughACaseVariantFolderWithoutTheFile()
    {
        if (!CaseSensitive())
            return;
        Directory.CreateDirectory(Path.Combine(_root, "Audio", "Speaker"));
        string campplus = Place("Audio", "Voices", "campplus.safetensors");

        Assert.Equal(campplus,
            ModelFileLocator.Find("campplus", Path.Combine("audio", "speaker"), Path.Combine("audio", "voices")));
    }

    [Fact]
    public void YoloByName_StillWins_OverAFolderScanOfACaseVariantYolov8()
    {
        if (!CaseSensitive())
            return;
        Place("YOLOv8", "a-other.safetensors");
        string named = Place("yolo", "yolov8n.safetensors");

        Assert.Equal(named, VisionModelPaths.FindYolo("yolov8n", explicitPath: null));
    }

    /// <summary>Creates an empty file at the joined segments under the models root and returns its path.</summary>
    private string Place(params string[] segments)
    {
        string path = Path.Combine([_root, .. segments]);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, [0x00]);
        return path;
    }

    private bool CaseSensitive()
    {
        bool sensitive = FileSystemCase.IsCaseSensitive(_root);
        if (!sensitive)
            _output.WriteLine("SKIPPED: the temp filesystem is case-insensitive, so no folder can differ from its lookup by case.");
        return sensitive;
    }
}
