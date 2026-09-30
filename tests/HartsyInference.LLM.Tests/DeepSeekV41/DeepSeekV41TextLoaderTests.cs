using HartsyInference.Core.Exceptions;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.Checkpoints;
using Xunit;

namespace HartsyInference.LLM.Tests.DeepSeekV41;

/// <summary>The Text path's Hugging Face directory hook: it validates the checkpoint, then refuses because no model class is wired.</summary>
public sealed class DeepSeekV41TextLoaderTests : IDisposable
{
    private readonly string _directory = Directory.CreateTempSubdirectory("dsv41-loader-").FullName;

    public void Dispose() => Directory.Delete(_directory, recursive: true);

    [Fact]
    public void ValidDirectory_OpensThenSaysTheModelClassIsNotWired()
    {
        TinyDeepSeekV41Checkpoint.Write(_directory);
        HfCheckpointInfo info = HfCheckpointDirectory.TryProbe(_directory)!;

        NotSupportedException error = Assert.Throws<NotSupportedException>(() => HfTextDirectoryLoader.Load(info));

        Assert.Contains("opened and validated", error.Message);
        Assert.Contains("Official", error.Message);
        Assert.Contains("3 shards", error.Message);
        Assert.Contains("not wired", error.Message);
    }

    [Fact]
    public void CorruptCheckpoint_FailsValidationBeforeTheNotWiredMessage()
    {
        TinyDeepSeekV41Checkpoint.Write(_directory);
        HfCheckpointInfo info = HfCheckpointDirectory.TryProbe(_directory)!;
        File.Delete(Path.Combine(_directory, "model-00002-of-00003.safetensors"));

        Exception error = Assert.ThrowsAny<Exception>(() => HfTextDirectoryLoader.Load(info));

        Assert.IsNotType<NotSupportedException>(error);
    }

    [Fact]
    public void OtherModelType_IsRefusedByName()
    {
        File.WriteAllText(Path.Combine(_directory, "config.json"), "{\"model_type\":\"llama\"}");
        File.WriteAllBytes(Path.Combine(_directory, "model.safetensors"), new byte[8]);
        HfCheckpointInfo info = HfCheckpointDirectory.TryProbe(_directory)!;

        HartsyInferenceException error = Assert.Throws<HartsyInferenceException>(() => HfTextDirectoryLoader.Load(info));

        Assert.Contains("llama", error.Message);
        Assert.Contains(".gguf", error.Message);
    }
}
