using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Vision.CrossFrame;
using HartsyInference.Vision.Dinov3;

namespace HartsyInference.Vision.DepthPro2;

/// <summary>
/// Depth Pro 2 over one video (upstream <c>DepthStream</c>): <see cref="Step"/> takes each frame and <see cref="Flush"/> ends the video; both
/// return the maps they finish, in frame order, each exactly once and never revised. Buffered mode predicts clips of <see cref="Window"/> frames
/// that overlap by <see cref="Overlap"/> and cross-fades the inverse depth of the overlapping frames linearly with no scale fit, so a frame is
/// returned once its clip is done and has also seen later frames. Unbuffered mode predicts each frame as it arrives, its patches attending to the
/// <c>Window - 1</c> frames before it through a <see cref="CrossFrameKvRing"/>. The focal length is fixed for the stream (upstream
/// <c>infer_video</c> takes the first window's estimate up front, see <see cref="DepthPro2Model.EstimateFocal"/>).
/// </summary>
public sealed class DepthPro2VideoStream : IDisposable
{
    /// <summary>Upstream's default clip length (buffered) or cache size (unbuffered).</summary>
    public const int DefaultWindow = 16;

    /// <summary>Upstream's default overlap between consecutive buffered clips.</summary>
    public const int DefaultOverlap = 4;

    private readonly DepthPro2Network _network;
    private readonly IBackend _backend;
    private readonly int _width, _height, _clip, _overlap;
    private readonly double _focal;
    private readonly CrossFrameKvRing? _ring;
    private readonly Dinov3RopeCache _rope = new();   // every clip of one length has the same tables
    private readonly Queue<byte[]> _frames = new();   // the last _clip frames; the last _held.Count + _new are not yet returned
    private readonly List<float[]> _held = [];        // predicted, not yet returned
    private int _new;
    private bool _flushed, _faulted;

    /// <param name="window">Frames attended together: the clip length (buffered) or the cache size with the current frame (unbuffered); 1 to
    /// <see cref="DepthPro2Model.MaxJointFrames"/>.</param>
    /// <param name="overlap">Frames shared by consecutive buffered clips, 0 to <paramref name="window"/> - 1; must be 0 for unbuffered.</param>
    /// <exception cref="ArgumentException">The frame is over <see cref="DepthPro2Model.MaxPatchTokens"/>, or an unbuffered stream's cache and
    /// the weights are larger than the device's memory.</exception>
    public DepthPro2VideoStream(DepthPro2Network network, IBackend backend, int width, int height, double focalLengthPx,
        int window = DefaultWindow, int overlap = DefaultOverlap, DepthPro2VideoMode mode = DepthPro2VideoMode.Buffered)
    {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
        DepthPro2Model.CheckSize(width, height);
        ArgumentOutOfRangeException.ThrowIfLessThan(window, 1);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(window, DepthPro2Model.MaxJointFrames);
        if (!double.IsFinite(focalLengthPx) || focalLengthPx <= 0)
            throw new ArgumentOutOfRangeException(nameof(focalLengthPx), $"The focal length must be positive pixels; got {focalLengthPx}.");
        if (mode == DepthPro2VideoMode.Buffered && (overlap < 0 || overlap >= window))
            throw new ArgumentOutOfRangeException(nameof(overlap), $"The overlap must be from 0 to {window - 1}; got {overlap}.");
        if (mode == DepthPro2VideoMode.Unbuffered && overlap != 0)
            throw new ArgumentOutOfRangeException(nameof(overlap), "Unbuffered frames do not overlap; give 0.");
        if (!Enum.IsDefined(mode)) throw new ArgumentOutOfRangeException(nameof(mode), mode, "Unknown video mode.");
        if (mode == DepthPro2VideoMode.Unbuffered)
        {
            long weights = network.EnumerateWeights().Sum(w => (long)w.DType.ComputeByteCount(w.ElementCount));
            CheckCacheFits(network.Preset, width, height, window, weights, backend.GetVramInfo().TotalBytes);
        }

        (_network, _backend, _width, _height, _focal) = (network, backend, width, height, focalLengthPx);
        Window = window;
        Mode = mode;
        _ring = mode == DepthPro2VideoMode.Unbuffered ? new CrossFrameKvRing(window) : null;
        _clip = _ring is null ? window : 1;
        _overlap = _ring is null ? overlap : 0;
    }

    public int Window { get; }

    public int Overlap => _overlap;

    public DepthPro2VideoMode Mode { get; }

    public double FocalLengthPx => _focal;

    /// <summary>Adds the next frame (RGB24 at the stream's size); returns the maps it finishes.</summary>
    /// <exception cref="InvalidOperationException">The stream was flushed, or an earlier step failed part-way (its cache is then
    /// inconsistent).</exception>
    public IReadOnlyList<DepthPro2Map> Step(byte[] rgb)
    {
        ArgumentNullException.ThrowIfNull(rgb);
        if (rgb.Length != (long)_width * _height * 3)
            throw new ArgumentException($"Every frame must be {_width}x{_height}x3 RGB bytes; got {rgb.Length}.", nameof(rgb));
        CheckUsable();
        _frames.Enqueue(rgb);
        if (_frames.Count > _clip) _frames.Dequeue();
        _new++;
        return _held.Count + _new < _clip ? [] : Guarded(() => Predict(_frames.Count, _overlap));
    }

    /// <summary>Ends the video; returns the maps not yet returned. With overlap, a short last clip is pulled back to a full window over frames
    /// already returned; without, the last frames run as a shorter clip of their own. The stream takes no frame after this.</summary>
    public IReadOnlyList<DepthPro2Map> Flush()
    {
        CheckUsable();
        _flushed = true;
        if (_new == 0) return Emit(_held.Count);
        return Guarded(() => Predict(_overlap > 0 ? _frames.Count : _new, keep: 0));
    }

    public void Dispose()
    {
        _ring?.Dispose();
        _rope.Dispose();
    }

    /// <summary>The bytes an unbuffered stream's attention cache needs: each cross-frame block's full window of keys, values and prefix tokens,
    /// and the one block's window of keys and values a frame reads into new tensors. The frame must be within
    /// <see cref="DepthPro2Model.MaxPatchTokens"/>.</summary>
    public static long UnbufferedCacheBytes(DepthPro2Preset preset, int width, int height, int window)
    {
        ArgumentNullException.ThrowIfNull(preset);
        DepthPro2Model.CheckSize(width, height);
        int prefix = 1 + preset.Backbone.NumRegisterTokens, hidden = preset.Backbone.HiddenSize;
        int blocks = preset.CrossFrameBlocks.Length;
        long rows = prefix + window * DepthPro2Model.PatchTokens(width, height);
        long keysAndValues = KvSegmentStore.EstimateBytes(blocks + 1, rows, hidden, DType.F32);
        return checked(keysAndValues + DType.F32.ComputeByteCount((long)blocks * window * prefix * hidden));
    }

    // The cache is pinned on the device, so one that cannot fit beside the weights would fail part-way through a video. Cache plus weights is a lower
    // bound of what a frame needs (its activations are not counted), measured against all of the device's memory. A host backend reports no device
    // memory and is not checked.
    internal static void CheckCacheFits(DepthPro2Preset preset, int width, int height, int window, long weightBytes, long deviceBytes)
    {
        if (deviceBytes <= 0) return;
        long cache = UnbufferedCacheBytes(preset, width, height, window);
        if (cache + weightBytes <= deviceBytes) return;
        throw new ArgumentException(string.Create(CultureInfo.InvariantCulture,
            $"An unbuffered {width}x{height} video with a window of {window} caches {cache / 1e9:F1} GB beside {weightBytes / 1e9:F1} GB of "
            + $"weights; the device has {deviceBytes / 1e9:F1} GB. Use a smaller window or frame size, or buffered mode."), nameof(window));
    }

    /// <summary>Upstream <c>blend_ramp</c>: <c>torch.linspace(1, 0, length + 2)[1:-1]</c> in float32, weights from 1 to 0 with both endpoints
    /// dropped.</summary>
    public static float[] BlendRamp(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        int steps = length + 2;
        float step = (0f - 1f) / (steps - 1);
        int halfway = steps / 2;
        float[] ramp = new float[length];
        for (int i = 1; i <= length; i++)
            ramp[i - 1] = i < halfway ? 1f + step * i : 0f - step * (steps - i - 1);   // torch's two-sided linspace
        return ramp;
    }

    private void CheckUsable()
    {
        if (_faulted) throw new InvalidOperationException("An earlier frame failed part-way; open a new stream.");
        if (_flushed) throw new InvalidOperationException("The stream was flushed; open a new stream for another video.");
    }

    // A prediction that throws can leave the cache with some blocks a frame ahead of others, so the stream refuses every later call.
    private List<DepthPro2Map> Guarded(Func<List<DepthPro2Map>> predict)
    {
        try
        {
            return predict();
        }
        catch
        {
            _faulted = true;
            throw;
        }
    }

    // Predicts the last `length` frames and returns those finished, all but the last `keep`.
    private List<DepthPro2Map> Predict(int length, int keep)
    {
        byte[][] clip = [.. _frames.Skip(_frames.Count - length)];
        float[][] inverse;
        using (Tensor frames = DepthPro2Preprocessor.NormalizeFrames(clip, _width, _height))
        using (Tensor maps = _network.PredictInverseDepthFrames(_backend, frames, _ring, _rope))
            inverse = DepthPro2Model.SplitFrames(maps, length);

        int skip = length - _held.Count - _new, held = _held.Count;
        _new = 0;
        if (held > 0)
        {
            float[] ramp = BlendRamp(held);
            for (int i = 0; i < held; i++)
            {
                float[] previous = _held[i], next = inverse[skip + i];
                float r = ramp[i], s = 1f - r;
                for (int p = 0; p < previous.Length; p++) previous[p] = r * previous[p] + s * next[p];
            }
        }

        for (int i = skip + held; i < length; i++) _held.Add(inverse[i]);
        return Emit(_held.Count - keep);
    }

    private List<DepthPro2Map> Emit(int count)
    {
        List<DepthPro2Map> done = new(count);
        for (int i = 0; i < count; i++) done.Add(DepthPro2Model.ToMap(_held[i], _width, _focal));
        _held.RemoveRange(0, count);
        return done;
    }
}
