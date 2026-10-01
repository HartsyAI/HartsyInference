using System.Collections.Concurrent;
using HartsyInference.Core.IO;
using HartsyInference.Core.Logging;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests;

/// <summary>Covers <see cref="CaseInsensitivePath"/> on a temp tree. The store it exists for has SwarmUI's <c>llm/</c>
/// and <c>audio/</c> where the engine spells <c>LLM/</c> and <c>Audio/</c>, and also holds case variants side by side
/// (<c>music/yue</c> and <c>music/YuE</c>), so exact case must keep winning wherever it exists. Tests that need two
/// spellings of one name skip on a case-insensitive filesystem, where they cannot be created.</summary>
public sealed class CaseInsensitivePathTests : IDisposable
{
    private const string Gguf = "Qwen3-4B-Q4_K_M.gguf";

    private readonly ITestOutputHelper _output;
    private readonly string _root = Directory.CreateTempSubdirectory("casefix-path-").FullName;

    public CaseInsensitivePathTests(ITestOutputHelper output) => _output = output;

    public void Dispose() => Directory.Delete(_root, recursive: true);

    [Fact]
    public void ExistingExactPath_IsReturnedExactlyAsPathCombineBuildsIt()
    {
        string file = Place("LLM", "qwen3", Gguf);
        string relative = "LLM/qwen3/" + Gguf;

        string resolved = CaseInsensitivePath.ResolveFile(_root, relative);

        Assert.Equal(Path.Combine(_root, relative), resolved);
        Assert.True(File.Exists(resolved));
        Assert.Equal(Path.GetFullPath(file), Path.GetFullPath(resolved));
    }

    [Fact]
    public void NothingOnDisk_ReturnsThePathAsSpelled()
    {
        string relative = "LLM/qwen3/" + Gguf;

        Assert.Equal(Path.Combine(_root, relative), CaseInsensitivePath.ResolveFile(_root, relative));
        Assert.Equal(Path.Combine(_root, "LLM"), CaseInsensitivePath.ResolveDirectory(_root, "LLM"));
    }

    [Fact]
    public void RootedRelativePath_IsReturnedAsPathCombineWould()
    {
        string rooted = Path.Combine(_root, "elsewhere", Gguf);

        Assert.Equal(rooted, CaseInsensitivePath.ResolveFile(Path.Combine(_root, "models"), rooted));
    }

    [Fact]
    public void UniqueLowercaseFolder_ResolvesTheSpelledPath()
    {
        if (!CaseSensitive())
            return;
        string file = Place("llm", "qwen3", Gguf);

        Assert.Equal(file, CaseInsensitivePath.ResolveFile(_root, "LLM/qwen3/" + Gguf));
        Assert.Equal(Path.Combine(_root, "llm"), CaseInsensitivePath.ResolveDirectory(_root, "LLM"));
        Assert.Equal(Path.Combine(_root, "llm", "qwen3"),
            CaseInsensitivePath.ResolveEntry(_root, Path.Combine("LLM", "qwen3")));
    }

    [Fact]
    public void ExactCase_WinsOverACaseVariantBesideIt()
    {
        if (!CaseSensitive())
            return;
        string lower = Place("music", "yue", "xcodec.safetensors");
        string mixed = Place("music", "YuE", "xcodec.safetensors");

        Assert.Equal(lower, CaseInsensitivePath.ResolveFile(_root, "music/yue/xcodec.safetensors"));
        Assert.Equal(mixed, CaseInsensitivePath.ResolveFile(_root, "music/YuE/xcodec.safetensors"));
        Assert.Equal(Path.Combine(_root, "music", "YuE"), CaseInsensitivePath.ResolveDirectory(_root, "music/YuE"));
    }

    [Fact]
    public void TwoCaseVariantsAndNoExactMatch_AreAmbiguous_AndResolveToThePathAsSpelled()
    {
        if (!CaseSensitive())
            return;
        Place("LLM", "qwen3", Gguf);
        Place("llm", "qwen3", Gguf);
        ConcurrentQueue<string> warnings = new();
        Logs.SetLogger((level, message) =>
        {
            if (level == LogLevel.Warning && message.Contains(_root, StringComparison.Ordinal))
                warnings.Enqueue(message);
        });
        try
        {
            string relative = "Llm/qwen3/" + Gguf;

            Assert.Equal(Path.Combine(_root, relative), CaseInsensitivePath.ResolveFile(_root, relative));
            Assert.Equal(Path.Combine(_root, relative), CaseInsensitivePath.ResolveFile(_root, relative));
            Assert.Equal(Path.Combine(_root, "Llm"), CaseInsensitivePath.ResolveDirectory(_root, "Llm"));
        }
        finally
        {
            Logs.SetLogger(null!);
        }
        string warning = Assert.Single(warnings);
        Assert.Contains(Path.Combine(_root, "LLM"), warning, StringComparison.Ordinal);
        Assert.Contains(Path.Combine(_root, "llm"), warning, StringComparison.Ordinal);
    }

    [Fact]
    public void OnlyEntriesOfTheWantedKind_Match()
    {
        if (!CaseSensitive())
            return;
        Directory.CreateDirectory(Path.Combine(_root, "weights"));
        File.WriteAllBytes(Path.Combine(_root, "WEIGHTS"), [0x01]);

        Assert.Equal(Path.Combine(_root, "weights"), CaseInsensitivePath.ResolveDirectory(_root, "Weights"));
        Assert.Equal(Path.Combine(_root, "WEIGHTS"), CaseInsensitivePath.ResolveFile(_root, "Weights"));
        Assert.Equal(Path.Combine(_root, "Weights"), CaseInsensitivePath.ResolveEntry(_root, "Weights"));
    }

    [Fact]
    public void AFileNeverStandsInForAnIntermediateFolder()
    {
        if (!CaseSensitive())
            return;
        File.WriteAllBytes(Path.Combine(_root, "llm"), [0x01]);
        string relative = "LLM/qwen3/" + Gguf;

        Assert.Equal(Path.Combine(_root, relative), CaseInsensitivePath.ResolveFile(_root, relative));
    }

    [Fact]
    public void EverySegment_IsMatchedOnItsOwn()
    {
        if (!CaseSensitive())
            return;
        string file = Place("llm", "Qwen3", Gguf);

        Assert.Equal(file, CaseInsensitivePath.ResolveFile(_root, "LLM/qwen3/qwen3-4b-q4_k_m.GGUF"));
    }

    [Fact]
    public void AMissingFile_LandsInTheFoldersThatExist()
    {
        if (!CaseSensitive())
            return;
        Directory.CreateDirectory(Path.Combine(_root, "llm"));

        Assert.Equal(Path.Combine(_root, "llm", "qwen3", Gguf),
            CaseInsensitivePath.ResolveFile(_root, "LLM/qwen3/" + Gguf));

        Directory.CreateDirectory(Path.Combine(_root, "llm", "Qwen3"));
        Assert.Equal(Path.Combine(_root, "llm", "Qwen3", Gguf),
            CaseInsensitivePath.ResolveFile(_root, "LLM/qwen3/" + Gguf));
    }

    [Fact]
    public void ASymlinkedFolder_Matches()
    {
        if (!CaseSensitive())
            return;
        Place("real-yue", Gguf);
        Directory.CreateSymbolicLink(Path.Combine(_root, "yue"), Path.Combine(_root, "real-yue"));

        Assert.Equal(Path.Combine(_root, "yue", Gguf), CaseInsensitivePath.ResolveFile(_root, "YUE/" + Gguf));
    }

    /// <summary>Creates an empty file at the joined segments under the root and returns its path.</summary>
    private string Place(params string[] segments)
    {
        string path = Path.Combine([_root, .. segments]);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, [0x00]);
        return path;
    }

    /// <summary>True on a case-sensitive filesystem; logs a skip otherwise.</summary>
    private bool CaseSensitive()
    {
        bool sensitive = FileSystemCase.IsCaseSensitive(_root);
        if (!sensitive)
            _output.WriteLine("SKIPPED: the temp filesystem is case-insensitive, so two spellings of one name cannot coexist.");
        return sensitive;
    }
}
