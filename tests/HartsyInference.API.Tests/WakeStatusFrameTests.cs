using System.Text;
using HartsyInference.Engine.Audio.Wake;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>The bytes a status frame puts on the wire.
///
/// <para>Pinned rather than trusted, because the reader is a C++ parser in another repository on a
/// microcontroller, and it cannot be refactored alongside this. A field renamed here is a light that stops
/// changing colour over there, with nothing failing anywhere in between.</para></summary>
public sealed class WakeStatusFrameTests
{
    [Fact]
    public async Task AStatusFrame_IsOneLineTheDeviceCanParse()
    {
        using MemoryStream stream = new();
        WakeFrameCodec codec = new(stream);

        await codec.WriteAsync("status", WakeStatus.Data(WakeStatus.Thinking), CancellationToken.None);

        string written = Encoding.UTF8.GetString(stream.ToArray());
        Assert.Equal("{\"type\":\"status\",\"data\":{\"state\":\"thinking\"}}\n", written);
    }

    [Fact]
    public async Task DetailRidesAlongWhenThereIsOne()
    {
        using MemoryStream stream = new();
        WakeFrameCodec codec = new(stream);

        await codec.WriteAsync("status", WakeStatus.Data(WakeStatus.Error, "transcription failed"),
            CancellationToken.None);

        Assert.Equal("{\"type\":\"status\",\"data\":{\"state\":\"error\",\"detail\":\"transcription failed\"}}\n",
            Encoding.UTF8.GetString(stream.ToArray()));
    }

    [Fact]
    public void DetailIsEscaped_SoAModelSMistakeCannotBreakTheFrame()
    {
        // Detail can carry an exception message, and an exception message can carry a quote or a newline. The
        // device's parser reads one line at a time, so an unescaped newline would truncate the frame and leave
        // the rest of it being read as the start of the next one.
        string data = WakeStatus.Data(WakeStatus.Error, "he said \"no\"\nand left");

        Assert.DoesNotContain("\n", data);
        Assert.Contains("\\n", data);
        Assert.Contains("\\u0022", data.Replace("\\\"", "\\u0022"));
    }

    [Theory]
    [InlineData("captured")]
    [InlineData("transcribing")]
    [InlineData("thinking")]
    [InlineData("speaking")]
    [InlineData("done")]
    [InlineData("error")]
    public void EveryStateTheDeviceHandles_IsOneThisClassNames(string state) => Assert.True(WakeStatus.IsKnown(state));

    [Fact]
    public void AnythingElseIsNot() => Assert.False(WakeStatus.IsKnown("listening"));
}
