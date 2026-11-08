using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using HartsyInference.Audio.Io;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Vision.Codec;

// hartsy-gencapture manifest --dir <capture dir> [--backend sel] [--commit sha] [--note text]
//   Records what a capture directory was produced with: commit, backend and device, kernel hashes, settings file,
//   and a hash of every file in the directory. Run it after the CLI generations that fill the directory.
// hartsy-gencapture compare --baseline <dir> --candidate <dir> [--max-abs 1e-4] [--max-rel 1e-3] [--pcm-lsb 1] [--png-levels 0] [--logprob-tol 1e-3]
//   Compares every file the baseline holds against the candidate: safetensors and raw F32 dumps (.bin from the
//   model dump sinks) by tolerance, PNG by decoded 8-bit levels, WAV by PCM samples, --scores-out TSVs by forced
//   token ids, top-token agreement and log-probability tolerance. Any other file is compared by hash and reported.

Dictionary<string, string> opts = ParseArgs(args);
if (!opts.TryGetValue("command", out string? command))
{
    Console.Error.WriteLine("usage: hartsy-gencapture manifest --dir <dir> | compare --baseline <dir> --candidate <dir>");
    return 2;
}
return command switch
{
    "manifest" => WriteManifest(opts),
    "compare" => Compare(opts),
    _ => Usage(),
};

static int Usage()
{
    Console.Error.WriteLine("usage: hartsy-gencapture manifest --dir <dir> | compare --baseline <dir> --candidate <dir>");
    return 2;
}

static int WriteManifest(Dictionary<string, string> opts)
{
    string dir = opts["dir"];
    string selector = opts.GetValueOrDefault("backend", "auto");
    string device;
    string kind = BackendFactory.Kind(BackendFactory.Resolve(selector));
    using (IBackend backend = BackendFactory.Create(selector))
        device = string.IsNullOrEmpty(backend.Capabilities.DeviceName) ? backend.Capabilities.Name : backend.Capabilities.DeviceName;
    Dictionary<string, string> kernels = kind switch
    {
        "cuda" => HashDir(BackendFactory.KernelDir(BackendFactory.PtxDirName)),
        "vulkan" => HashDir(BackendFactory.KernelDir(BackendFactory.SpirvDirName)),
        _ => new(StringComparer.Ordinal),
    };
    Dictionary<string, string> files = new(StringComparer.Ordinal);
    foreach (string f in Directory.EnumerateFiles(dir, "*", SearchOption.AllDirectories).Order(StringComparer.Ordinal))
    {
        string rel = Path.GetRelativePath(dir, f).Replace('\\', '/');
        if (rel == "manifest.json") continue;
        files[rel] = Hash(f);
    }
    Dictionary<string, object?> manifest = new(StringComparer.Ordinal)
    {
        ["commit"] = opts.GetValueOrDefault("commit") ?? GitHead(),
        ["recordedUtc"] = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture),
        ["backend"] = selector,
        ["backendKind"] = kind,
        ["device"] = device,
        ["settingsFile"] = KnobFile.LoadedFrom,
        ["note"] = opts.GetValueOrDefault("note"),
        ["kernelHashes"] = kernels,
        ["fileHashes"] = files,
    };
    File.WriteAllText(Path.Combine(dir, "manifest.json"), JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true }));
    Console.WriteLine($"manifest: {files.Count} file(s), {kind} ({device})");
    return 0;
}

static int Compare(Dictionary<string, string> opts)
{
    string baseline = opts["baseline"], candidate = opts["candidate"];
    double maxAbs = Parse(opts, "max-abs", 1e-4), maxRel = Parse(opts, "max-rel", 1e-3), logprobTol = Parse(opts, "logprob-tol", 1e-3);
    int pcmLsb = (int)Parse(opts, "pcm-lsb", 1), pngLevels = (int)Parse(opts, "png-levels", 0);
    int failures = 0, compared = 0;
    foreach (string b in Directory.EnumerateFiles(baseline, "*", SearchOption.AllDirectories).Order(StringComparer.Ordinal))
    {
        string rel = Path.GetRelativePath(baseline, b).Replace('\\', '/');
        if (rel == "manifest.json") continue;
        string c = Path.Combine(candidate, rel);
        compared++;
        if (!File.Exists(c))
        {
            Console.WriteLine($"  {rel}: FAIL missing in candidate");
            failures++;
            continue;
        }
        (bool pass, string detail) = Path.GetExtension(rel).ToLowerInvariant() switch
        {
            ".safetensors" => CompareSafeTensors(b, c, maxAbs, maxRel),
            ".bin" => CompareRawF32(b, c, maxAbs, maxRel),
            ".png" => ComparePng(b, c, pngLevels),
            ".wav" => CompareWav(b, c, pcmLsb),
            ".tsv" => CompareScores(b, c, logprobTol),
            _ => CompareHash(b, c),
        };
        if (!pass) failures++;
        Console.WriteLine($"  {rel}: {(pass ? "ok" : "FAIL")} {detail}");
    }
    // An empty or mismatched baseline compares nothing; that is a failed gate, not a pass.
    if (compared == 0)
    {
        Console.WriteLine("FAIL: no capture files were compared (empty baseline or no matching files)");
        return 1;
    }
    Console.WriteLine(failures == 0 ? $"PASS ({compared} file(s))" : $"FAIL: {failures} of {compared} file(s)");
    return failures == 0 ? 0 : 1;
}

static (bool, string) CompareSafeTensors(string b, string c, double maxAbs, double maxRel)
{
    using SafeTensorsLoader lb = new();
    using SafeTensorsLoader lc = new();
    lb.Load(b);
    lc.Load(c);
    Dictionary<string, Tensor> tb = lb.GetAllTensors();
    Dictionary<string, Tensor> tc = lc.GetAllTensors();
    List<string> problems = [];
    double worstAbs = 0, worstRel = 0;
    foreach ((string name, Tensor eb) in tb)
    {
        if (!tc.TryGetValue(name, out Tensor? ec)) { problems.Add($"{name} missing"); continue; }
        if (eb.Shape.ToString() != ec.Shape.ToString() || eb.DType != ec.DType) { problems.Add($"{name} shape/dtype {ec.Shape} {ec.DType} vs {eb.Shape} {eb.DType}"); continue; }
        using Tensor fb = eb.DType == DType.F32 ? eb.To(eb.Device) : eb.CastTo(DType.F32);
        using Tensor fc = ec.DType == DType.F32 ? ec.To(ec.Device) : ec.CastTo(DType.F32);
        (double a, double r, string? bad) = Diff(fb.AsReadOnlySpan<float>(), fc.AsReadOnlySpan<float>());
        if (bad is not null) problems.Add($"{name} {bad}");
        else if (a > maxAbs && r > maxRel) problems.Add($"{name} maxAbs {a:E2} maxRel {r:E2}");
        worstAbs = Math.Max(worstAbs, a);
        worstRel = Math.Max(worstRel, r);
    }
    foreach (Tensor t in tb.Values) t.Dispose();
    foreach (Tensor t in tc.Values) t.Dispose();
    return (problems.Count == 0, problems.Count == 0 ? $"{tb.Count} tensor(s), worst maxAbs {worstAbs:E2} maxRel {worstRel:E2}" : string.Join("; ", problems));
}

static (bool, string) CompareRawF32(string b, string c, double maxAbs, double maxRel)
{
    byte[] bb = File.ReadAllBytes(b), cb = File.ReadAllBytes(c);
    if (bb.Length != cb.Length) return (false, $"length {cb.Length} vs {bb.Length} bytes");
    if (bb.Length % 4 != 0) return (bb.AsSpan().SequenceEqual(cb), "not a multiple of 4 bytes; compared as bytes");
    (double a, double r, string? bad) = Diff(System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(bb), System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(cb));
    if (bad is not null) return (false, bad);
    bool pass = a <= maxAbs || r <= maxRel;
    return (pass, $"{bb.Length / 4} floats, maxAbs {a:E2} maxRel {r:E2}");
}

static (double, double, string?) Diff(ReadOnlySpan<float> e, ReadOnlySpan<float> a)
{
    double maxAbs = 0, maxRel = 0;
    for (int i = 0; i < e.Length; i++)
    {
        float ev = e[i], av = a[i];
        if (!float.IsFinite(av) || !float.IsFinite(ev))
        {
            if (float.IsNaN(ev) && float.IsNaN(av)) continue;
            if (ev == av) continue;
            return (double.NaN, double.NaN, $"non-finite mismatch at [{i}]: {av} vs {ev}");
        }
        double d = Math.Abs((double)av - ev);
        maxAbs = Math.Max(maxAbs, d);
        maxRel = Math.Max(maxRel, d / Math.Max(Math.Abs((double)ev), 1e-6));
    }
    return (maxAbs, maxRel, null);
}

static (bool, string) ComparePng(string b, string c, int levels)
{
    (byte[] rb, int wb, int hb) = PngDecoder.DecodeFromFile(b);
    (byte[] rc, int wc, int hc) = PngDecoder.DecodeFromFile(c);
    if (wb != wc || hb != hc) return (false, $"size {wc}x{hc} vs {wb}x{hb}");
    int worst = 0; long differing = 0;
    for (int i = 0; i < rb.Length; i++)
    {
        int d = Math.Abs(rb[i] - rc[i]);
        if (d > 0) differing++;
        worst = Math.Max(worst, d);
    }
    return (worst <= levels, $"{wb}x{hb}, max level diff {worst}, {differing} of {rb.Length} channel values differ");
}

static (bool, string) CompareWav(string b, string c, int lsb)
{
    WavFile.DecodedAudio ab = WavFile.Read(b), ac = WavFile.Read(c);
    if (ab.SampleRate != ac.SampleRate || ab.Channels.Length != ac.Channels.Length || ab.Length != ac.Length)
        return (false, $"format {ac.SampleRate}Hz x{ac.Channels.Length} {ac.Length} samples vs {ab.SampleRate}Hz x{ab.Channels.Length} {ab.Length}");
    double worst = 0;
    for (int ch = 0; ch < ab.Channels.Length; ch++)
        for (int i = 0; i < ab.Length; i++)
            worst = Math.Max(worst, Math.Abs(ab.Channels[ch][i] - ac.Channels[ch][i]) * 32768.0);
    return (worst <= lsb, $"{ab.Length} samples, max {worst:F2} LSB (16-bit)");
}

static (bool, string) CompareScores(string b, string c, double tol)
{
    string[] lb = File.ReadAllLines(b), lc = File.ReadAllLines(c);
    if (lb.Length == 0 || !lb[0].StartsWith("index\ttoken_id", StringComparison.Ordinal)) return CompareHash(b, c);
    if (lb.Length != lc.Length) return (false, $"{lc.Length - 1} rows vs {lb.Length - 1}");
    int idMismatch = 0, topMismatch = 0; double worst = 0;
    for (int i = 1; i < lb.Length; i++)
    {
        string[] fb = lb[i].Split('\t'), fc = lc[i].Split('\t');
        if (fb.Length < 5 || fc.Length < 5) return (false, $"row {i} malformed");
        if (fb[1] != fc[1]) idMismatch++;
        if (fb[3] != fc[3]) topMismatch++;
        // Column 2 is the forced token's log-probability, column 4 the top token's: both are under the tolerance.
        double forced = Math.Abs(double.Parse(fb[2], CultureInfo.InvariantCulture) - double.Parse(fc[2], CultureInfo.InvariantCulture));
        double top = Math.Abs(double.Parse(fb[4], CultureInfo.InvariantCulture) - double.Parse(fc[4], CultureInfo.InvariantCulture));
        worst = Math.Max(worst, Math.Max(forced, top));
    }
    bool pass = idMismatch == 0 && topMismatch == 0 && worst <= tol;
    return (pass, $"{lb.Length - 1} tokens, forced-id mismatches {idMismatch}, top-id mismatches {topMismatch}, max |Δlogprob| {worst:E2} (tol {tol:E1})");
}

static (bool, string) CompareHash(string b, string c)
{
    string hb = Hash(b), hc = Hash(c);
    return (hb == hc, hb == hc ? "identical bytes" : "bytes differ (no typed comparison for this file)");
}

static string Hash(string path)
{
    using FileStream s = File.OpenRead(path);
    return Convert.ToHexString(SHA256.HashData(s)).ToLowerInvariant();
}

static Dictionary<string, string> HashDir(string dir)
{
    Dictionary<string, string> h = new(StringComparer.Ordinal);
    if (!Directory.Exists(dir)) return h;
    foreach (string f in Directory.EnumerateFiles(dir).Order(StringComparer.Ordinal)) h[Path.GetFileName(f)] = Hash(f);
    return h;
}

static double Parse(Dictionary<string, string> opts, string key, double fallback)
    => opts.TryGetValue(key, out string? v) ? double.Parse(v, CultureInfo.InvariantCulture) : fallback;

static string GitHead()
{
    try
    {
        using Process p = Process.Start(new ProcessStartInfo("git", "rev-parse HEAD") { RedirectStandardOutput = true, UseShellExecute = false })!;
        string head = p.StandardOutput.ReadToEnd().Trim();
        p.WaitForExit();
        return p.ExitCode == 0 && head.Length > 0 ? head : "unknown";
    }
    catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or InvalidOperationException)
    {
        return "unknown";
    }
}

static Dictionary<string, string> ParseArgs(string[] argv)
{
    Dictionary<string, string> parsed = new(StringComparer.Ordinal);
    for (int i = 0; i < argv.Length; i++)
    {
        if (argv[i].StartsWith("--", StringComparison.Ordinal))
        {
            string key = argv[i][2..];
            parsed[key] = i + 1 < argv.Length && !argv[i + 1].StartsWith("--", StringComparison.Ordinal) ? argv[++i] : "true";
        }
        else if (!parsed.ContainsKey("command"))
        {
            parsed["command"] = argv[i];
        }
    }
    return parsed;
}
