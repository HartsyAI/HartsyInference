using System.Buffers.Binary;

namespace SharpInference.Audio.Io;

/// <summary>RIFF/WAVE file reader and writer for the subset of PCM formats that audio
/// inference cares about: 8 / 16 / 24 / 32-bit signed PCM and 32-bit float, mono or
/// multi-channel.
///
/// <para>WAV is intentionally the only format we read natively — everything else
/// (MP3 / FLAC / OGG / OPUS) is handled by asking the user to pipe through ffmpeg or
/// convert at upload time. This keeps SharpInference.Audio pure C# with zero
/// native-decoder dependencies.</para>
///
/// <para>All samples are returned as normalized <see cref="float"/> in [-1, 1] with
/// channels deinterleaved (mono is the typical STT input).</para></summary>
public static class WavFile
{
    /// <summary>Decoded WAV contents: deinterleaved float samples per channel plus the
    /// header sample rate. Use <see cref="ToMono"/> to downmix.</summary>
    public readonly record struct DecodedAudio(float[][] Channels, int SampleRate)
    {
        /// <summary>Number of samples per channel.</summary>
        public int Length => Channels.Length == 0 ? 0 : Channels[0].Length;

        /// <summary>Downmixes to mono by averaging channels. Returns the existing
        /// mono buffer unchanged if already mono.</summary>
        public float[] ToMono()
        {
            if (Channels.Length == 1) return Channels[0];
            int n = Length;
            float[] mono = new float[n];
            float inv = 1f / Channels.Length;
            for (int c = 0; c < Channels.Length; c++)
            {
                float[] src = Channels[c];
                for (int i = 0; i < n; i++) mono[i] += src[i] * inv;
            }
            return mono;
        }
    }

    /// <summary>Reads a WAV file from disk. Throws <see cref="InvalidDataException"/>
    /// if the file is not a recognized PCM/IEEE-float WAV.</summary>
    public static DecodedAudio Read(string path)
    {
        using FileStream fs = File.OpenRead(path);
        return Read(fs);
    }

    /// <summary>Reads WAV from any seekable stream. The stream is consumed up to the
    /// end of the <c>data</c> chunk; any trailing chunks are ignored.</summary>
    public static DecodedAudio Read(Stream stream)
    {
        Span<byte> header = stackalloc byte[12];
        ReadExact(stream, header);
        if (header[0] != 'R' || header[1] != 'I' || header[2] != 'F' || header[3] != 'F')
            throw new InvalidDataException("Not a RIFF file.");
        if (header[8] != 'W' || header[9] != 'A' || header[10] != 'V' || header[11] != 'E')
            throw new InvalidDataException("RIFF file is not WAVE.");

        WaveFormat? fmt = null;
        Span<byte> chunkHdr = stackalloc byte[8];
        // The RIFF spec puts chunks in any order, so walk them until we find the
        // mandatory `fmt ` and `data` chunks (in that order — `data` always trails).
        while (true)
        {
            int got = stream.Read(chunkHdr);
            if (got < 8) throw new InvalidDataException("Unexpected end of file before data chunk.");

            uint chunkSize = BinaryPrimitives.ReadUInt32LittleEndian(chunkHdr[4..]);
            string chunkId = System.Text.Encoding.ASCII.GetString(chunkHdr[..4]);

            if (chunkId == "fmt ")
            {
                byte[] fmtBytes = new byte[chunkSize];
                ReadExact(stream, fmtBytes);
                fmt = WaveFormat.Parse(fmtBytes);
                // fmt chunks are padded to even length per RIFF; skip the pad byte.
                if ((chunkSize & 1) == 1) stream.ReadByte();
            }
            else if (chunkId == "data")
            {
                if (fmt is null) throw new InvalidDataException("data chunk preceded fmt chunk.");
                return ReadDataChunk(stream, (int)chunkSize, fmt.Value);
            }
            else
            {
                // Skip unrelated chunks (LIST/INFO/junk/cue/etc.)
                if (stream.CanSeek) stream.Seek(chunkSize + (chunkSize & 1), SeekOrigin.Current);
                else { byte[] discard = new byte[chunkSize + (chunkSize & 1)]; ReadExact(stream, discard); }
            }
        }
    }

    private static DecodedAudio ReadDataChunk(Stream s, int dataBytes, WaveFormat fmt)
    {
        int channels = fmt.NumChannels;
        int bytesPerSample = fmt.BitsPerSample / 8;
        int frameBytes = bytesPerSample * channels;
        int frames = dataBytes / frameBytes;

        float[][] channelData = new float[channels][];
        for (int c = 0; c < channels; c++) channelData[c] = new float[frames];

        byte[] buf = new byte[Math.Min(dataBytes, 1 << 16)];
        int frameIdx = 0;
        int bytesLeft = frames * frameBytes;
        while (bytesLeft > 0)
        {
            int want = Math.Min(buf.Length, bytesLeft);
            // Round to a whole number of frames so the converter loop never straddles one.
            want -= want % frameBytes;
            int read = s.Read(buf, 0, want);
            if (read == 0) break;
            bytesLeft -= read;

            int chunkFrames = read / frameBytes;
            ConvertFrames(buf.AsSpan(0, read), channelData, frameIdx, chunkFrames, fmt);
            frameIdx += chunkFrames;
        }

        return new DecodedAudio(channelData, fmt.SampleRate);
    }

    private static void ConvertFrames(ReadOnlySpan<byte> bytes, float[][] outCh, int frameIdx, int frames, WaveFormat fmt)
    {
        int channels = fmt.NumChannels;
        int sb = fmt.BitsPerSample / 8;

        if (fmt.FormatTag == FormatTag.IeeeFloat && fmt.BitsPerSample == 32)
        {
            for (int f = 0; f < frames; f++)
                for (int c = 0; c < channels; c++)
                    outCh[c][frameIdx + f] = BinaryPrimitives.ReadSingleLittleEndian(bytes.Slice((f * channels + c) * sb));
        }
        else if (fmt.FormatTag == FormatTag.Pcm)
        {
            switch (fmt.BitsPerSample)
            {
                case 8:
                    // Per the WAV spec, 8-bit PCM is unsigned (0..255 centered at 128).
                    for (int f = 0; f < frames; f++)
                        for (int c = 0; c < channels; c++)
                            outCh[c][frameIdx + f] = (bytes[(f * channels + c)] - 128) / 128f;
                    break;
                case 16:
                    for (int f = 0; f < frames; f++)
                        for (int c = 0; c < channels; c++)
                            outCh[c][frameIdx + f] = BinaryPrimitives.ReadInt16LittleEndian(bytes.Slice((f * channels + c) * 2)) / 32768f;
                    break;
                case 24:
                    for (int f = 0; f < frames; f++)
                        for (int c = 0; c < channels; c++)
                        {
                            int off = (f * channels + c) * 3;
                            int v = bytes[off] | (bytes[off + 1] << 8) | (bytes[off + 2] << 16);
                            if ((v & 0x800000) != 0) v |= unchecked((int)0xFF000000);
                            outCh[c][frameIdx + f] = v / 8388608f;
                        }
                    break;
                case 32:
                    for (int f = 0; f < frames; f++)
                        for (int c = 0; c < channels; c++)
                            outCh[c][frameIdx + f] = BinaryPrimitives.ReadInt32LittleEndian(bytes.Slice((f * channels + c) * 4)) / 2147483648f;
                    break;
                default:
                    throw new NotSupportedException($"PCM bit depth {fmt.BitsPerSample} not supported.");
            }
        }
        else
        {
            throw new NotSupportedException($"WAV format tag {fmt.FormatTag} not supported.");
        }
    }

    /// <summary>Writes a mono 16-bit PCM WAV file. Used by TTS pipelines that produce
    /// a float[] waveform in [-1, 1].</summary>
    public static void WriteMono16(string path, ReadOnlySpan<float> samples, int sampleRate)
    {
        using FileStream fs = File.Create(path);
        WriteMono16(fs, samples, sampleRate);
    }

    /// <summary>Writes a mono 16-bit PCM WAV stream.</summary>
    public static void WriteMono16(Stream stream, ReadOnlySpan<float> samples, int sampleRate)
    {
        int dataBytes = samples.Length * 2;
        Span<byte> hdr = stackalloc byte[44];
        // RIFF
        hdr[0] = (byte)'R'; hdr[1] = (byte)'I'; hdr[2] = (byte)'F'; hdr[3] = (byte)'F';
        BinaryPrimitives.WriteUInt32LittleEndian(hdr[4..], (uint)(36 + dataBytes));
        hdr[8] = (byte)'W'; hdr[9] = (byte)'A'; hdr[10] = (byte)'V'; hdr[11] = (byte)'E';
        // fmt
        hdr[12] = (byte)'f'; hdr[13] = (byte)'m'; hdr[14] = (byte)'t'; hdr[15] = (byte)' ';
        BinaryPrimitives.WriteUInt32LittleEndian(hdr[16..], 16);
        BinaryPrimitives.WriteUInt16LittleEndian(hdr[20..], 1);              // PCM
        BinaryPrimitives.WriteUInt16LittleEndian(hdr[22..], 1);              // channels
        BinaryPrimitives.WriteUInt32LittleEndian(hdr[24..], (uint)sampleRate);
        BinaryPrimitives.WriteUInt32LittleEndian(hdr[28..], (uint)(sampleRate * 2)); // byte rate
        BinaryPrimitives.WriteUInt16LittleEndian(hdr[32..], 2);              // block align
        BinaryPrimitives.WriteUInt16LittleEndian(hdr[34..], 16);             // bits
        // data
        hdr[36] = (byte)'d'; hdr[37] = (byte)'a'; hdr[38] = (byte)'t'; hdr[39] = (byte)'a';
        BinaryPrimitives.WriteUInt32LittleEndian(hdr[40..], (uint)dataBytes);

        stream.Write(hdr);

        byte[] buf = new byte[Math.Min(dataBytes, 1 << 16) & ~1];
        int i = 0;
        while (i < samples.Length)
        {
            int n = Math.Min(samples.Length - i, buf.Length / 2);
            for (int k = 0; k < n; k++)
            {
                float v = samples[i + k];
                int s = (int)Math.Clamp(MathF.Round(v * 32767f), -32768f, 32767f);
                BinaryPrimitives.WriteInt16LittleEndian(buf.AsSpan(k * 2), (short)s);
            }
            stream.Write(buf, 0, n * 2);
            i += n;
        }
    }

    private static void ReadExact(Stream s, Span<byte> dst)
    {
        int read = 0;
        while (read < dst.Length)
        {
            int got = s.Read(dst[read..]);
            if (got == 0) throw new EndOfStreamException();
            read += got;
        }
    }

    private static void ReadExact(Stream s, byte[] dst) => ReadExact(s, dst.AsSpan());

    private enum FormatTag : ushort { Pcm = 1, IeeeFloat = 3, Extensible = 0xFFFE }

    private readonly record struct WaveFormat(FormatTag FormatTag, int NumChannels, int SampleRate, int BitsPerSample)
    {
        public static WaveFormat Parse(ReadOnlySpan<byte> bytes)
        {
            ushort tag = BinaryPrimitives.ReadUInt16LittleEndian(bytes);
            ushort ch = BinaryPrimitives.ReadUInt16LittleEndian(bytes[2..]);
            uint sr = BinaryPrimitives.ReadUInt32LittleEndian(bytes[4..]);
            ushort bps = BinaryPrimitives.ReadUInt16LittleEndian(bytes[14..]);

            // WAVE_FORMAT_EXTENSIBLE wraps a subformat GUID at offset 24 (cbSize+22).
            // The first 2 bytes of the GUID are the real format tag.
            if (tag == (ushort)FormatTag.Extensible && bytes.Length >= 26)
            {
                tag = BinaryPrimitives.ReadUInt16LittleEndian(bytes[24..]);
            }

            return new WaveFormat((FormatTag)tag, ch, (int)sr, bps);
        }
    }
}
