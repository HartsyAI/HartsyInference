using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tests.MemoryManagement;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Core.Tests.Configuration;

/// <summary>A host's <see cref="KnobStore.Set{T}"/> must beat the settings file even when it races the file's load on
/// another thread. The load is held open on a named pipe, so the race is deterministic. While the file is half-read, a
/// second thread sets the models root, after reading it or without. That thread has to wait for the load, so its value
/// lands last. A read used to be let through as soon as the load started, and a Set never waited at all, so the file's
/// value landed afterwards and replaced the host's. Serialized with the other classes that change process-wide
/// knobs.</summary>
[Collection(EnvironmentSensitiveCollection.Name)]
public sealed partial class KnobFileConcurrentLoadTests
{
    private readonly ITestOutputHelper _output;

    public KnobFileConcurrentLoadTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void AHostSetMadeWhileTheFileIsLoading_IsNotOverwrittenByTheFile(bool readFirst)
    {
        if (OperatingSystem.IsWindows())
        {
            // Holding the load open needs a POSIX named pipe; the code under test is the same on every OS.
            _output.WriteLine("SKIPPED: needs mkfifo");
            return;
        }
        string dir = Directory.CreateTempSubdirectory("hartsy-knob-race-").FullName;
        string pipe = Path.Combine(dir, "settings.json");
        string? previous = KnobFile.ExplicitPath;
        Thread? load = null;
        try
        {
            Assert.True(MakeFifo(pipe, 0x180) == 0, $"mkfifo failed with errno {Marshal.GetLastPInvokeError()}");
            KnobFile.ExplicitPath = pipe;

            Exception? loadFailure = null;
            load = new(() =>
            {
                try
                {
                    KnobFile.Reload();
                }
                catch (Exception ex)
                {
                    loadFailure = ex;
                }
            }) { IsBackground = true };
            load.Start();

            Thread host = new(() =>
            {
                if (readFirst)
                {
                    _ = EngineKnobs.ModelsRoot.Value;
                }
                KnobStore.Set(EngineKnobs.ModelsRoot, "/from-host");
            }) { IsBackground = true };
            // Opening the write end waits for the load to open the read end, so once this returns the loader is inside
            // the file's load and blocked on its contents.
            using (FileStream writer = new(pipe, FileMode.Open, FileAccess.Write))
            {
                host.Start();
                // Long enough for the host thread to finish if nothing holds it back, which is the bug: it got past a
                // load that had started and not yet applied anything.
                bool hostFinishedEarly = host.Join(TimeSpan.FromMilliseconds(300));
                _output.WriteLine($"host thread finished before the file was applied: {hostFinishedEarly}");
                writer.Write("""{"settings":{"paths.modelsRoot":"/from-file"}}"""u8);
            }

            Assert.True(load.Join(TimeSpan.FromSeconds(30)), "the settings load never finished");
            Assert.True(host.Join(TimeSpan.FromSeconds(30)), "the host thread never finished");
            Assert.Null(loadFailure);
            Assert.Equal("/from-host", EngineKnobs.ModelsRoot.Value);
            Assert.Equal("host", KnobStore.SourceOf("paths.modelsRoot"));
        }
        finally
        {
            if (load is { IsAlive: true })
            {
                // A failure before the pipe was written leaves the load waiting on it, holding the settings lock;
                // an empty write ends it, so the reload below cannot hang.
                using FileStream release = new(pipe, FileMode.Open, FileAccess.Write);
            }
            KnobFile.ExplicitPath = previous;
            KnobFile.Reload();
            Directory.Delete(dir, recursive: true);
        }
    }

    [LibraryImport("libc", EntryPoint = "mkfifo", StringMarshalling = StringMarshalling.Utf8, SetLastError = true)]
    private static partial int MakeFifo(string path, uint mode);
}
