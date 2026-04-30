using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Utilities;

/// <summary>Converts between VAE-style image tensors [B, 3, H, W] (float, range ~[-1, 1]) and standard RGB byte arrays [H, W, 3] in [0, 255]. Used at the boundary between user image I/O and the VAE encoder/decoder.</summary>
public static unsafe class ImagePostProcessor
{
    /// <summary>Converts RGB byte data [H, W, 3] in [0, 255] (HWC interleaved) to a tensor [1, 3, H, W] in [-1, 1] (NCHW planar). Mirrors the inverse mapping in <see cref="TensorToRgbBytes"/>.</summary>
    public static Tensor RgbBytesToTensor(ReadOnlySpan<byte> rgbData, int width, int height)
    {
        if (rgbData.Length != width * height * 3)
            throw new ArgumentException($"rgbData length {rgbData.Length} != width*height*3 ({width * height * 3})", nameof(rgbData));

        Tensor image = new Tensor(new TensorShape(1, 3, height, width), DType.F32);
        float* imgPtr = (float*)image.DataPointer;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int pixelOffset = (y * width + x) * 3;
                for (int c = 0; c < 3; c++)
                {
                    // Inverse of TensorToRgbBytes: byte/255 → [0,1] → [-1,1]
                    float val = rgbData[pixelOffset + c] / 255.0f;
                    val = val * 2.0f - 1.0f;
                    imgPtr[c * height * width + y * width + x] = val;
                }
            }
        }

        return image;
    }

    /// <summary>Converts the first image in a batch from tensor [B, 3, H, W] to RGB byte array [H, W, 3]. Clamps to [0, 255].</summary>
    public static byte[] TensorToRgbBytes(Tensor image)
    {
        int channels = (int)image.Shape[1];
        int height = (int)image.Shape[2];
        int width = (int)image.Shape[3];

        byte[] rgb = new byte[height * width * 3];
        float* imgPtr = (float*)image.DataPointer;

        // Tensor is NCHW, we output HWC
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int pixelOffset = (y * width + x) * 3;

                for (int c = 0; c < 3; c++)
                {
                    // Normalize from [-1, 1] to [0, 255]
                    float val = imgPtr[c * height * width + y * width + x];
                    val = (val + 1.0f) * 0.5f; // [-1,1] → [0,1]
                    val = Math.Clamp(val, 0.0f, 1.0f);
                    rgb[pixelOffset + c] = (byte)(val * 255.0f + 0.5f);
                }
            }
        }

        return rgb;
    }

    /// <summary>Writes a raw RGB byte array as a BMP file (simple, no external dependencies).</summary>
    public static void SaveBmp(string path, byte[] rgbData, int width, int height)
    {
        // BMP format: header (14) + DIB header (40) + pixel data (BGR, padded rows)
        int rowSize = ((width * 3 + 3) / 4) * 4; // Row size padded to 4 bytes
        int dataSize = rowSize * height;
        int fileSize = 54 + dataSize;

        byte[] bmp = new byte[fileSize];

        // BMP header
        bmp[0] = (byte)'B';
        bmp[1] = (byte)'M';
        WriteInt32(bmp, 2, fileSize);
        WriteInt32(bmp, 10, 54); // Pixel data offset

        // DIB header (BITMAPINFOHEADER)
        WriteInt32(bmp, 14, 40); // DIB header size
        WriteInt32(bmp, 18, width);
        WriteInt32(bmp, 22, -height); // Negative = top-down
        WriteInt16(bmp, 26, 1); // Color planes
        WriteInt16(bmp, 28, 24); // Bits per pixel
        WriteInt32(bmp, 34, dataSize);

        // Pixel data (BMP uses BGR, top-down due to negative height)
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int srcOffset = (y * width + x) * 3;
                int dstOffset = 54 + y * rowSize + x * 3;

                bmp[dstOffset + 0] = rgbData[srcOffset + 2]; // B
                bmp[dstOffset + 1] = rgbData[srcOffset + 1]; // G
                bmp[dstOffset + 2] = rgbData[srcOffset + 0]; // R
            }
        }

        File.WriteAllBytes(path, bmp);
    }

    private static void WriteInt32(byte[] buffer, int offset, int value)
    {
        buffer[offset + 0] = (byte)(value & 0xFF);
        buffer[offset + 1] = (byte)((value >> 8) & 0xFF);
        buffer[offset + 2] = (byte)((value >> 16) & 0xFF);
        buffer[offset + 3] = (byte)((value >> 24) & 0xFF);
    }

    private static void WriteInt16(byte[] buffer, int offset, short value)
    {
        buffer[offset + 0] = (byte)(value & 0xFF);
        buffer[offset + 1] = (byte)((value >> 8) & 0xFF);
    }
}
