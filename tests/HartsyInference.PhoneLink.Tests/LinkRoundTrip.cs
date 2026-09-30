namespace HartsyInference.PhoneLink.Tests;

/// <summary>Writes frames into memory and reads them back, returning frames whose payloads are copied out of the reader's buffer.</summary>
internal static class LinkRoundTrip
{
    public static async Task<byte[]> EncodeAsync(Func<LinkFrameWriter, ValueTask> write)
    {
        using MemoryStream stream = new();
        using LinkFrameWriter writer = new(stream);
        await write(writer);
        return stream.ToArray();
    }

    public static async Task<List<LinkFrame>> DecodeAsync(Stream stream)
    {
        using LinkFrameReader reader = new(stream);
        List<LinkFrame> frames = new();
        while (await reader.ReadAsync(CancellationToken.None) is LinkFrame frame)
            frames.Add(new LinkFrame(frame.Header, frame.Payload.ToArray()));
        return frames;
    }

    public static async Task<LinkFrame> OneAsync(Func<LinkFrameWriter, ValueTask> write)
    {
        byte[] bytes = await EncodeAsync(write);
        List<LinkFrame> frames = await DecodeAsync(new MemoryStream(bytes));
        return Xunit.Assert.Single(frames);
    }
}
