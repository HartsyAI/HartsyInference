namespace HartsyInference.Vision.Codec;

/// <summary>Decodes a baseline (sequential, Huffman-coded) JPEG to RGB24: 8-bit samples, one or three components, any sampling factors, and restart
/// intervals. Chroma is upsampled by nearest neighbour, and the IDCT is a float separable transform, so pixels can differ by a few levels from libjpeg.
/// Progressive, arithmetic-coded, 12-bit, lossless and CMYK JPEGs are refused with the reason, rather than decoded wrongly.</summary>
public static class JpegDecoder
{
    private static readonly int[] ZigZag =
    [
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    ];

    /// <summary>The largest frame decoded: each plane is allocated up front, so a header cannot ask for more than this.</summary>
    public const long MaxPixels = 64L * 1024 * 1024;

    /// <summary>Cosine basis for the 8-point IDCT: <c>Basis[x * 8 + u] = C(u) / 2 · cos((2x + 1) u π / 16)</c>.</summary>
    private static readonly float[] Basis = BuildBasis();

    /// <summary>True when the bytes start with the JPEG SOI marker and the next marker byte, which is what a JPEG file begins with.</summary>
    public static bool IsJpeg(ReadOnlySpan<byte> data) => data.Length >= 3 && data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF;

    /// <summary>Decodes a baseline JPEG to RGB24 (row-major, 3 bytes per pixel). Grayscale is replicated to RGB. Applies the EXIF orientation of an APP1 block, and reads an Adobe APP14 transform flag.</summary>
    public static (byte[] Rgb, int Width, int Height) Decode(ReadOnlySpan<byte> encoded)
    {
        if (!IsJpeg(encoded))
            throw new ArgumentException("Not a JPEG file: the SOI marker is missing.", nameof(encoded));
        try
        {
            return new Decoder(encoded.ToArray()).Run();
        }
        catch (IndexOutOfRangeException ex)
        {
            throw new ArgumentException("Not a decodable JPEG: the file is truncated or its headers are malformed.", nameof(encoded), ex);
        }
    }

    private static float[] BuildBasis()
    {
        float[] b = new float[64];
        for (int x = 0; x < 8; x++)
        for (int u = 0; u < 8; u++)
        {
            double cu = u == 0 ? Math.Sqrt(0.5) : 1.0;
            b[x * 8 + u] = (float)(cu / 2.0 * Math.Cos((2 * x + 1) * u * Math.PI / 16.0));
        }
        return b;
    }

    private sealed class Huffman
    {
        public readonly int[] MaxCode = new int[17];
        public readonly int[] MinCode = new int[17];
        public readonly int[] ValPtr = new int[17];
        public byte[] Values = [];

        public static Huffman Build(byte[] counts, byte[] values)
        {
            Huffman h = new() { Values = values };
            int code = 0, k = 0;
            for (int len = 1; len <= 16; len++)
            {
                int n = counts[len - 1];
                if (n > 0)
                {
                    h.ValPtr[len] = k;
                    h.MinCode[len] = code;
                    code += n;
                    k += n;
                    h.MaxCode[len] = code - 1;
                }
                else
                {
                    h.MaxCode[len] = -1;
                }
                code <<= 1;
            }
            return h;
        }
    }

    private sealed class Component
    {
        public int Id, H, V, Tq;
        public int BlocksPerLine, BlocksPerColumn;
        public int PlaneWidth;
        public byte[] Plane = [];
        public int Pred;
        public int DcTable, AcTable;
    }

    private sealed class Decoder
    {
        private readonly byte[] _d;
        private int _pos = 2;
        private readonly int[][] _qt = [new int[64], new int[64], new int[64], new int[64]];   // natural order
        private readonly Huffman?[] _dc = new Huffman?[4];
        private readonly Huffman?[] _ac = new Huffman?[4];
        private int _width, _height, _restart;
        private Component[] _comps = [];
        private int _hMax, _vMax, _mcusX, _mcusY;
        private bool _frame;
        private byte[]? _exif;
        private int _adobeTransform = -1;

        // Bit reader state.
        private int _bitBuf, _bitCnt;
        private bool _hitMarker;   // the bit reader met a marker where it needed data

        public Decoder(byte[] d) => _d = d;

        public (byte[] Rgb, int Width, int Height) Run()
        {
            while (true)
            {
                int m = NextMarker();
                switch (m)
                {
                    case 0xD9:
                        throw Bad("the file ends before any scan");
                    case 0xE1:
                        if (SegmentLength(out int exifEnd) >= 8 && IsExif(_pos + 2, exifEnd)) _exif = _d[(_pos + 8)..exifEnd];
                        Skip();
                        break;
                    case 0xEE:
                        if (SegmentLength(out _) >= 14 && IsAdobe(_pos + 2)) _adobeTransform = _d[_pos + 2 + 11];
                        Skip();
                        break;
                    case 0xDB:
                        ReadQuant();
                        break;
                    case 0xC4:
                        ReadHuffman();
                        break;
                    case 0xDD:
                        _restart = U16(_pos + 2);
                        Skip();
                        break;
                    case 0xC0:
                    case 0xC1:
                        ReadFrame();
                        break;
                    case 0xDA:
                        if (!_frame) throw Bad("a scan appears before the frame header");
                        DecodeScan();
                        return ApplyExif(ToRgb());
                    case 0xC2:
                    case 0xC3:
                    case 0xC5:
                    case 0xC6:
                    case 0xC7:
                    case 0xC9:
                    case 0xCA:
                    case 0xCB:
                    case 0xCD:
                    case 0xCE:
                    case 0xCF:
                        throw new NotSupportedException(
                            $"JPEG marker 0x{m:X2} is progressive, arithmetic, lossless or hierarchical; only baseline (SOF0/SOF1) is decoded. Re-encode the image as a baseline JPEG or as a PNG.");
                    default:
                        Skip();
                        break;
                }
            }
        }

        private int NextMarker()
        {
            while (true)
            {
                if (_pos + 1 >= _d.Length) throw Bad("the file ends inside the headers");
                if (_d[_pos] != 0xFF) { _pos++; continue; }
                int m = _d[_pos + 1];
                _pos += 2;
                if (m == 0xFF || m == 0x00) { _pos--; continue; }   // fill byte or stuffed zero
                if (m == 0x01 || (m >= 0xD0 && m <= 0xD7)) continue;   // standalone markers carry no segment
                return m;
            }
        }

        private void Skip()
        {
            _pos += SegmentLength(out _);
        }

        /// <summary>The declared length of the segment at <see cref="_pos"/>, checked to lie inside the file.</summary>
        private int SegmentLength(out int end)
        {
            if (_pos + 2 > _d.Length) throw Bad("a segment header runs past the end of the file");
            int len = U16(_pos);
            end = _pos + len;
            if (len < 2 || end > _d.Length) throw Bad("a segment runs past the end of the file");
            return len;
        }

        private bool IsExif(int at, int end) =>
            end - at >= 6 && _d[at] == 'E' && _d[at + 1] == 'x' && _d[at + 2] == 'i' && _d[at + 3] == 'f' && _d[at + 4] == 0 && _d[at + 5] == 0;

        private bool IsAdobe(int at) =>
            _d[at] == 'A' && _d[at + 1] == 'd' && _d[at + 2] == 'o' && _d[at + 3] == 'b' && _d[at + 4] == 'e';

        /// <summary>Applies the EXIF orientation of an APP1 block (a TIFF block after the <c>Exif  </c> header), as the PNG path does.</summary>
        private (byte[] Rgb, int Width, int Height) ApplyExif((byte[] Rgb, int Width, int Height) img)
        {
            int orientation = ExifOrientation.ReadTiff(_exif);
            if (orientation == ExifOrientation.Upright) return img;
            (byte[] rgb, _, int w, int h) = ExifOrientation.Apply(img.Rgb, null, img.Width, img.Height, orientation);
            return (rgb, w, h);
        }

        private int U16(int at) => (_d[at] << 8) | _d[at + 1];

        private void ReadQuant()
        {
            int len = U16(_pos), end = _pos + len;
            int p = _pos + 2;
            while (p < end)
            {
                int pq = _d[p] >> 4, tq = _d[p] & 15;
                p++;
                if (tq > 3) throw Bad("a quantization table index above 3");
                for (int k = 0; k < 64; k++)
                {
                    int v = pq == 0 ? _d[p++] : U16(p);
                    if (pq != 0) p += 2;
                    _qt[tq][ZigZag[k]] = v;
                }
            }
            _pos = end;
        }

        private void ReadHuffman()
        {
            int len = U16(_pos), end = _pos + len;
            int p = _pos + 2;
            while (p < end)
            {
                int tc = _d[p] >> 4, th = _d[p] & 15;
                p++;
                if (th > 3) throw Bad("a Huffman table index above 3");
                byte[] counts = new byte[16];
                Array.Copy(_d, p, counts, 0, 16);
                p += 16;
                int total = 0;
                foreach (byte c in counts) total += c;
                byte[] values = new byte[total];
                Array.Copy(_d, p, values, 0, total);
                p += total;
                Huffman h = Huffman.Build(counts, values);
                if (tc == 0) _dc[th] = h; else _ac[th] = h;
            }
            _pos = end;
        }

        private void ReadFrame()
        {
            int len = U16(_pos);
            int p = _pos + 2;
            if (_d[p] != 8) throw new NotSupportedException($"JPEG sample precision {_d[p]} is not 8 bits; only 8-bit baseline is decoded.");
            _height = U16(p + 1);
            _width = U16(p + 3);
            int n = _d[p + 5];
            if (n != 1 && n != 3) throw new NotSupportedException($"JPEG has {n} components; only grayscale and YCbCr (1 or 3) are decoded, not CMYK.");
            if (_width <= 0 || _height <= 0) throw Bad("the frame has no size");
            // The PNG rule (the RGB bytes must fit an int), and a pixel ceiling because every plane is allocated up front.
            if ((long)_width * _height * 3 > int.MaxValue || (long)_width * _height > MaxPixels)
                throw Bad($"the frame {_width}x{_height} is larger than the decoder's {MaxPixels}-pixel limit");
            _comps = new Component[n];
            _hMax = 1;
            _vMax = 1;
            for (int i = 0; i < n; i++)
            {
                int q = p + 6 + i * 3;
                Component c = new() { Id = _d[q], H = _d[q + 1] >> 4, V = _d[q + 1] & 15, Tq = _d[q + 2] & 3 };
                if (c.H < 1 || c.H > 4 || c.V < 1 || c.V > 4) throw Bad("a sampling factor outside 1..4");
                _comps[i] = c;
                _hMax = Math.Max(_hMax, c.H);
                _vMax = Math.Max(_vMax, c.V);
            }
            _mcusX = (_width + 8 * _hMax - 1) / (8 * _hMax);
            _mcusY = (_height + 8 * _vMax - 1) / (8 * _vMax);
            foreach (Component c in _comps)
            {
                c.BlocksPerLine = _mcusX * c.H;
                c.BlocksPerColumn = _mcusY * c.V;
                c.PlaneWidth = c.BlocksPerLine * 8;
                c.Plane = new byte[c.PlaneWidth * c.BlocksPerColumn * 8];
            }
            _frame = true;
            _pos += len;
        }

        private void DecodeScan()
        {
            int len = U16(_pos);
            int p = _pos + 2;
            int ns = _d[p];
            if (ns != _comps.Length) throw new NotSupportedException("A non-interleaved JPEG scan is not decoded; re-encode it as a single interleaved scan.");
            for (int i = 0; i < ns; i++)
            {
                int id = _d[p + 1 + i * 2], tables = _d[p + 2 + i * 2];
                Component c = _comps.FirstOrDefault(x => x.Id == id) ?? throw Bad($"the scan names component {id}, which the frame does not have");
                c.DcTable = tables >> 4;
                c.AcTable = tables & 15;
                if (c.DcTable > 3 || c.AcTable > 3) throw Bad("a scan names a Huffman table above 3");
            }
            int ss = _d[p + 1 + ns * 2], se = _d[p + 2 + ns * 2];
            if (ss != 0 || se != 63) throw new NotSupportedException("A spectral-selection scan is progressive; only baseline is decoded.");
            _pos += len;

            _bitBuf = 0;
            _bitCnt = 0;
            foreach (Component c in _comps) c.Pred = 0;
            int total = _mcusX * _mcusY;
            int[] block = new int[64];
            for (int mcu = 0; mcu < total; mcu++)
            {
                if (_restart > 0 && mcu > 0 && mcu % _restart == 0)
                {
                    SyncRestart();
                    foreach (Component c in _comps) c.Pred = 0;
                }
                int mcuRow = mcu / _mcusX, mcuCol = mcu % _mcusX;
                foreach (Component c in _comps)
                {
                    for (int v = 0; v < c.V; v++)
                    for (int h = 0; h < c.H; h++)
                    {
                        DecodeBlock(c, block);
                        int by = mcuRow * c.V + v, bx = mcuCol * c.H + h;
                        WriteBlock(c, block, bx, by);
                    }
                }
            }
            if (_hitMarker) throw Bad("the scan ends before all its MCUs are decoded");
        }

        private void DecodeBlock(Component c, int[] block)
        {
            Array.Clear(block);
            Huffman dc = _dc[c.DcTable] ?? throw Bad("a scan uses an undefined DC table");
            Huffman ac = _ac[c.AcTable] ?? throw Bad("a scan uses an undefined AC table");
            int[] q = _qt[c.Tq];
            int t = Decode(dc);
            int diff = t == 0 ? 0 : Extend(Receive(t), t);
            c.Pred += diff;
            block[0] = c.Pred * q[0];
            for (int k = 1; k < 64;)
            {
                int rs = Decode(ac);
                int s = rs & 15, r = rs >> 4;
                if (s == 0)
                {
                    if (r == 15) { k += 16; continue; }
                    break;
                }
                k += r;
                if (k > 63) break;
                int z = Extend(Receive(s), s);
                int nat = ZigZag[k];
                block[nat] = z * q[nat];
                k++;
            }
        }

        private void WriteBlock(Component c, int[] coef, int bx, int by)
        {
            // Separable IDCT: rows of frequencies first (tmp[y][u]), then columns.
            Span<float> tmp = stackalloc float[64];
            for (int y = 0; y < 8; y++)
            for (int u = 0; u < 8; u++)
            {
                float s = 0;
                for (int v = 0; v < 8; v++) s += Basis[y * 8 + v] * coef[v * 8 + u];
                tmp[y * 8 + u] = s;
            }
            int baseX = bx * 8, baseY = by * 8;
            for (int y = 0; y < 8; y++)
            for (int x = 0; x < 8; x++)
            {
                float s = 0;
                for (int u = 0; u < 8; u++) s += Basis[x * 8 + u] * tmp[y * 8 + u];
                int val = (int)MathF.Round(s + 128f);
                c.Plane[(baseY + y) * c.PlaneWidth + baseX + x] = (byte)(val < 0 ? 0 : val > 255 ? 255 : val);
            }
        }

        private int Decode(Huffman h)
        {
            int code = 0;
            for (int len = 1; len <= 16; len++)
            {
                code = (code << 1) | ReadBit();
                if (code <= h.MaxCode[len])
                    return h.Values[h.ValPtr[len] + code - h.MinCode[len]];
            }
            throw Bad("a Huffman code matches no table entry");
        }

        private int Receive(int s)
        {
            int v = 0;
            for (int i = 0; i < s; i++) v = (v << 1) | ReadBit();
            return v;
        }

        private static int Extend(int v, int s) => v < (1 << (s - 1)) ? v + (-1 << s) + 1 : v;

        private int ReadBit()
        {
            if (_bitCnt == 0)
            {
                _bitBuf = NextByte();
                _bitCnt = 8;
            }
            _bitCnt--;
            return (_bitBuf >> _bitCnt) & 1;
        }

        private int NextByte()
        {
            if (_pos >= _d.Length) throw Bad("the scan data ends early");
            byte b = _d[_pos];
            if (b != 0xFF) { _pos++; return b; }
            byte next = _pos + 1 < _d.Length ? _d[_pos + 1] : (byte)0xD9;
            if (next == 0x00) { _pos += 2; return 0xFF; }
            // A marker inside the scan (a restart or the end of the scan): feed zeros and leave the marker for SyncRestart.
            _hitMarker = true;
            return 0;
        }

        private void SyncRestart()
        {
            _bitCnt = 0;
            _hitMarker = false;   // the marker met here is the restart, which is expected
            while (true)
            {
                if (_pos + 1 >= _d.Length) throw Bad("a restart marker is missing");
                if (_d[_pos] == 0xFF && _d[_pos + 1] != 0x00 && _d[_pos + 1] != 0xFF) break;
                _pos++;
            }
            int m = _d[_pos + 1];
            if (m < 0xD0 || m > 0xD7) throw Bad("the scan ends before all its MCUs are decoded");
            _pos += 2;
        }

        private (byte[] Rgb, int Width, int Height) ToRgb()
        {
            byte[] rgb = new byte[_width * _height * 3];
            if (_comps.Length == 1)
            {
                Component g = _comps[0];
                for (int y = 0; y < _height; y++)
                for (int x = 0; x < _width; x++)
                {
                    byte v = g.Plane[y * g.PlaneWidth + x];
                    int o = (y * _width + x) * 3;
                    rgb[o] = v; rgb[o + 1] = v; rgb[o + 2] = v;
                }
                return (rgb, _width, _height);
            }
            Component yc = _comps[0], cb = _comps[1], cr = _comps[2];
            if (_adobeTransform == 0)
            {
                // Adobe transform 0: the three planes are RGB already, so no YCbCr conversion.
                for (int y = 0; y < _height; y++)
                for (int x = 0; x < _width; x++)
                {
                    int o = (y * _width + x) * 3;
                    rgb[o] = yc.Plane[(y * yc.V / _vMax) * yc.PlaneWidth + x * yc.H / _hMax];
                    rgb[o + 1] = cb.Plane[(y * cb.V / _vMax) * cb.PlaneWidth + x * cb.H / _hMax];
                    rgb[o + 2] = cr.Plane[(y * cr.V / _vMax) * cr.PlaneWidth + x * cr.H / _hMax];
                }
                return (rgb, _width, _height);
            }
            for (int y = 0; y < _height; y++)
            for (int x = 0; x < _width; x++)
            {
                int yv = yc.Plane[(y * yc.V / _vMax) * yc.PlaneWidth + x * yc.H / _hMax];
                int cbv = cb.Plane[(y * cb.V / _vMax) * cb.PlaneWidth + x * cb.H / _hMax] - 128;
                int crv = cr.Plane[(y * cr.V / _vMax) * cr.PlaneWidth + x * cr.H / _hMax] - 128;
                int o = (y * _width + x) * 3;
                rgb[o] = Clamp(yv + 1.402f * crv);
                rgb[o + 1] = Clamp(yv - 0.344136f * cbv - 0.714136f * crv);
                rgb[o + 2] = Clamp(yv + 1.772f * cbv);
            }
            return (rgb, _width, _height);
        }

        private static byte Clamp(float v)
        {
            int r = (int)MathF.Round(v);
            return (byte)(r < 0 ? 0 : r > 255 ? 255 : r);
        }

        private ArgumentException Bad(string why) => new($"Not a decodable JPEG: {why}.", "encoded");
    }
}
