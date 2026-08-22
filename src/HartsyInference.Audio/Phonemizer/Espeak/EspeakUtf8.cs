namespace HartsyInference.Audio.Phonemizer.Espeak;

/// <summary>UTF-8 codepoint readers ported from espeak-ng <c>utf8_in</c>/<c>utf8_in2</c> (translate.c); the rule interpreter walks the word buffer one codepoint at a time, forwards and backwards, and relies on these returning the exact same byte counts espeak uses so multi-byte letters consume the right number of input bytes.</summary>
internal static class EspeakUtf8
{
    private static ReadOnlySpan<byte> LeadMask => [0xff, 0x1f, 0x0f, 0x07];

    /// <summary>Reads the codepoint starting at <paramref name="pos"/> in <paramref name="buf"/> moving forward; returns the number of bytes the codepoint occupies (1 for ASCII).</summary>
    public static int Read(byte[] buf, int pos, out int c) => Read2(buf, pos, false, out c);

    /// <summary>Reads a codepoint at <paramref name="pos"/>, skipping over non-initial bytes in the <paramref name="backwards"/> direction first (mirror of <c>utf8_in2</c>); returns bytes consumed.</summary>
    public static int Read2(byte[] buf, int pos, bool backwards, out int c)
    {
        while (pos >= 0 && pos < buf.Length && (buf[pos] & 0xc0) == 0x80)
            pos += backwards ? -1 : 1;

        int nBytes = 0;
        int c1 = pos >= 0 && pos < buf.Length ? buf[pos] : 0;
        pos++;
        if ((c1 & 0x80) != 0)
        {
            if ((c1 & 0xe0) == 0xc0) nBytes = 1;
            else if ((c1 & 0xf0) == 0xe0) nBytes = 2;
            else if ((c1 & 0xf8) == 0xf0) nBytes = 3;

            c1 &= LeadMask[nBytes];
            for (int ix = 0; ix < nBytes; ix++)
            {
                int next = pos < buf.Length ? buf[pos] : 0;
                pos++;
                c1 = (c1 << 6) + (next & 0x3f);
            }
        }
        c = c1;
        return nBytes + 1;
    }
}
