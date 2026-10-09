using System.Buffers.Binary;
using HartsyInference.API.Endpoints;
using HartsyInference.Core.Exceptions;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>The pixel guard on chat images: a PNG header that declares more decoded RGB than the 32 MiB limit is refused before the decoder allocates it.</summary>
public sealed class ChatImageLimitTests
{
    [Fact]
    public void A_PNG_Whose_Header_Declares_A_Canvas_Over_The_Limit_Is_Refused_Before_Decoding()
    {
        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(() => CompatEndpoints.RefuseOversizedPng(PngHeader(6000, 6000)));

        Assert.Contains("6000x6000 pixels", ex.Message);
    }

    [Fact]
    public void A_PNG_Header_Within_The_Limit_Is_Left_To_The_Decoder()
    {
        CompatEndpoints.RefuseOversizedPng(PngHeader(3000, 3000)); // 27 MB of RGB, under 32 MiB
        CompatEndpoints.RefuseOversizedPng(PngHeader(1, 1));
    }

    [Fact]
    public void Bytes_Without_A_PNG_Header_Are_Left_To_The_Codec()
    {
        CompatEndpoints.RefuseOversizedPng("GIF89a and then some more bytes"u8);
    }

    [Fact]
    public void Images_Together_Past_The_Request_Budget_Are_Refused()
    {
        // Four images' worth fits the budget; the fifth, however small, takes the request past it.
        CompatEndpoints.ImageBudget budget = new();
        HartsyInference.Engine.Requests.ImageData image = new() { Rgb = new byte[CompatEndpoints.MaxImageBytes], Width = 1, Height = 1 };
        for (int i = 0; i < 4; i++)
            budget.Take(image);

        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(() => budget.Take(image));
        Assert.Contains("128 MiB", ex.Message);
    }

    /// <summary>A PNG header alone: the signature and the IHDR chunk that carries the canvas size. The decoder would allocate width x height x 3 bytes before it reads
    /// any pixel data.</summary>
    private static byte[] PngHeader(uint width, uint height)
    {
        byte[] header = new byte[24];
        new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A }.CopyTo(header, 0);
        BinaryPrimitives.WriteUInt32BigEndian(header.AsSpan(8), 13);
        "IHDR"u8.CopyTo(header.AsSpan(12));
        BinaryPrimitives.WriteUInt32BigEndian(header.AsSpan(16), width);
        BinaryPrimitives.WriteUInt32BigEndian(header.AsSpan(20), height);
        return header;
    }
}
