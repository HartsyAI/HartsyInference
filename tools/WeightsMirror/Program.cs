using System.Reflection;
using System.Security.Cryptography;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tools.WeightsMirror;

const string HelpText = """
WeightsMirror - repack pickle checkpoints to safetensors for hosting, and prove the repack equals its source.

  The pickle is read only by the safe-subset reader (nothing in it is executed), one tensor at a time. The output keeps the
  source's tensor names, order, shapes, dtypes and bytes. Sources are only read; nothing is written next to them.

USAGE
  WeightsMirror inspect <checkpoint> [--recursive]
      Lists tensor count, bytes, dtypes and how often a tensor-by-tensor read would have to restart a storage entry.
  WeightsMirror run --plan <mirror-plan.json> --models-root <dir> --staging <dir> [--only id,id] [--force]
      For each plan item: hash the local pickle against the upstream pin, repack it into <staging>/<output>, verify the repack
      tensor for tensor, hash it, and record everything in <staging>/manifest.json. The staging folder must be outside the
      models root and outside the repository.
  WeightsMirror check --staging <dir>
      Re-hash every file the manifest lists and compare sizes and sha256.
""";

if (args.Length == 0 || args[0] is "-h" or "--help")
{
    Console.WriteLine(HelpText);
    return 0;
}
Dictionary<string, string> options = new(StringComparer.Ordinal);
List<string> positionals = [];
for (int i = 1; i < args.Length; i++)
{
    if (!args[i].StartsWith("--", StringComparison.Ordinal))
    {
        positionals.Add(args[i]);
    }
    else if (args[i] is "--recursive" or "--force")
    {
        options[args[i]] = "true";
    }
    else if (i + 1 < args.Length)
    {
        options[args[i]] = args[++i];
    }
    else
    {
        return Fail($"{args[i]} needs a value.");
    }
}
return args[0] switch
{
    "inspect" => positionals.Count == 1 ? Inspect(positionals[0], options.ContainsKey("--recursive")) : Fail("inspect takes one checkpoint path."),
    "run" => Run(options),
    "check" => Check(options),
    _ => Fail($"Unknown command '{args[0]}'."),
};

static int Fail(string message)
{
    Console.Error.WriteLine(message);
    return 1;
}

static string Sha256Of(string path)
{
    using FileStream stream = new(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20);
    return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
}

static int Inspect(string path, bool recursive)
{
    using PytorchPickleLoader loader = new();
    loader.OpenLazy(path, recursive);
    Dictionary<string, (int Count, long Bytes)> dtypes = new(StringComparer.Ordinal);
    foreach (PytorchTensorDescriptor d in loader.Descriptors.Values)
    {
        (int count, long bytes) = dtypes.GetValueOrDefault(d.DType.Name);
        dtypes[d.DType.Name] = (count + 1, bytes + d.DType.ComputeByteCount(d.Shape.ElementCount));
    }
    Console.WriteLine($"{path}: {loader.Descriptors.Count} tensors, {loader.Descriptors.Values.Select(d => d.StorageKey).Distinct().Count()} storage entries");
    foreach ((string name, (int count, long bytes)) in dtypes)
    {
        Console.WriteLine($"  {name}: {count} tensors, {bytes:N0} bytes");
    }
    foreach (PytorchTensorDescriptor d in loader.Descriptors.Values.Take(6))
    {
        Console.WriteLine($"  {d.Name} {d.DType.Name} {d.Shape} storage {d.StorageKey}+{d.StorageByteOffset}");
    }
    string[] roots = [.. loader.Descriptors.Keys.Select(k => k.Split('.')[0]).Distinct().Take(12)];
    Console.WriteLine($"  key roots: {string.Join(", ", roots)}");
    return 0;
}

static bool IsUnder(string path, string root)
{
    string relative = Path.GetRelativePath(Path.GetFullPath(root), Path.GetFullPath(path));
    return relative != ".." && !relative.StartsWith(".." + Path.DirectorySeparatorChar, StringComparison.Ordinal) && !Path.IsPathRooted(relative);
}

static string? RepoRootOf(string start)
{
    for (DirectoryInfo? dir = new DirectoryInfo(Path.GetFullPath(start)); dir is not null; dir = dir.Parent)
    {
        if (Directory.Exists(Path.Combine(dir.FullName, ".git")) || File.Exists(Path.Combine(dir.FullName, ".git")))
        {
            return dir.FullName;
        }
    }
    return null;
}

static int Run(Dictionary<string, string> options)
{
    if (!options.TryGetValue("--plan", out string? planPath) || !options.TryGetValue("--models-root", out string? modelsRoot)
        || !options.TryGetValue("--staging", out string? staging))
    {
        return Fail("run needs --plan, --models-root and --staging.");
    }
    string staged = Path.GetFullPath(staging);
    if (IsUnder(staged, modelsRoot) || IsUnder(modelsRoot, staged))
    {
        return Fail($"The staging folder '{staged}' must be outside the models root '{modelsRoot}'.");
    }
    foreach (string start in new[] { Directory.GetCurrentDirectory(), Path.GetDirectoryName(Path.GetFullPath(planPath)) ?? "." })
    {
        if (RepoRootOf(start) is string repo && IsUnder(staged, repo))
        {
            return Fail($"The staging folder '{staged}' must be outside the repository '{repo}'.");
        }
    }
    MirrorPlan plan = MirrorPlan.Load(planPath);
    HashSet<string>? only = options.TryGetValue("--only", out string? ids) ? [.. ids.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)] : null;
    if (only is not null && only.Except(plan.Items.Select(i => i.Id)).FirstOrDefault() is string unknown)
    {
        return Fail($"The plan has no item '{unknown}'.");
    }
    Directory.CreateDirectory(staged);
    string manifestPath = Path.Combine(staged, "manifest.json");
    MirrorManifest manifest = MirrorManifest.LoadOrNew(manifestPath);
    string tool = "WeightsMirror " + (Assembly.GetExecutingAssembly().GetName().Version?.ToString() ?? "0");
    int failures = 0;
    foreach (MirrorItem item in plan.Items.Where(i => only is null || only.Contains(i.Id)))
    {
        try
        {
            MirrorManifestItem done = RunItem(plan, item, modelsRoot, staged, tool, options.ContainsKey("--force"));
            manifest.Items = [.. manifest.Items.Where(m => m.Id != item.Id), done];
            manifest.Save(manifestPath);
            Console.WriteLine($"[{item.Id}] ok: {done.Repack.Size:N0} bytes, sha256 {done.Repack.Sha256}");
        }
        catch (Exception ex) when (ex is IOException or InvalidDataException or InvalidOperationException or NotSupportedException or UnauthorizedAccessException)
        {
            failures++;
            Console.Error.WriteLine($"[{item.Id}] FAILED: {ex.Message}");
        }
    }
    return failures == 0 ? 0 : 1;
}

static MirrorManifestItem RunItem(MirrorPlan plan, MirrorItem item, string modelsRoot, string staged, string tool, bool force)
{
    string source = Path.GetFullPath(Path.Combine(modelsRoot, item.LocalSource));
    if (!File.Exists(source))
    {
        throw new FileNotFoundException($"The local source '{item.LocalSource}' is not under the models root.");
    }
    long size = new FileInfo(source).Length;
    if (size != item.SourceSize)
    {
        throw new InvalidDataException($"The local source is {size:N0} bytes; upstream lists {item.SourceSize:N0}.");
    }
    string output = Path.GetFullPath(Path.Combine(staged, item.Output));
    if (!IsUnder(output, staged))
    {
        throw new InvalidDataException($"The output '{item.Output}' leaves the staging folder.");
    }
    if (File.Exists(output) && !force)
    {
        throw new IOException($"'{item.Output}' already exists in the staging folder; pass --force to replace it.");
    }
    Console.WriteLine($"[{item.Id}] hashing the local source ({size:N0} bytes)");
    string sourceSha = Sha256Of(source);
    if (!string.Equals(sourceSha, item.SourceSha256, StringComparison.OrdinalIgnoreCase))
    {
        throw new InvalidDataException($"The local source hashes to {sourceSha}; upstream pins {item.SourceSha256}.");
    }
    Dictionary<string, string> metadata = new(StringComparer.Ordinal)
    {
        ["format"] = "pt",
        ["hartsy.converter"] = "HartsyInference.StreamingPickleRepack",
        ["hartsy.source_repo"] = item.SourceRepo,
        ["hartsy.source_path"] = item.SourcePath,
        ["hartsy.source_revision"] = item.Revision,
        ["hartsy.source_sha256"] = item.SourceSha256,
    };
    Console.WriteLine($"[{item.Id}] repacking");
    PickleRepackSummary summary = StreamingPickleRepack.Repack(source, output, item.Recursive, metadata);
    Console.WriteLine($"[{item.Id}] verifying {summary.TensorCount} tensors against the source");
    PickleRepackVerification verification = StreamingPickleRepack.Verify(source, output, item.Recursive);
    if (!verification.Equal)
    {
        File.Delete(output);
        throw new InvalidDataException($"The repack differs from its source: {string.Join("; ", verification.Differences)}");
    }
    using SafeTensorsLoader loader = new();
    loader.Load(output);
    Dictionary<string, int> dtypes = loader.Descriptors.Values.GroupBy(d => d.DType.Name).OrderBy(g => g.Key, StringComparer.Ordinal).ToDictionary(g => g.Key, g => g.Count());
    MirrorLicense license = plan.LicenseOf(item);
    return new MirrorManifestItem
    {
        Id = item.Id,
        Model = item.Model,
        DateUtc = DateTime.UtcNow.ToString("yyyy-MM-dd'T'HH:mm:ss'Z'"),
        Tool = tool,
        Source = new ManifestSource
        {
            Repo = item.SourceRepo, Path = item.SourcePath, Revision = item.Revision, Url = item.SourceUrl, Size = item.SourceSize, Sha256 = sourceSha,
            LocalCopyMatchesUpstream = true,
        },
        Repack = new ManifestRepack
        {
            Path = item.Output.Replace('\\', '/'), Size = new FileInfo(output).Length, Sha256 = Sha256Of(output), TensorCount = summary.TensorCount,
            TensorBytes = summary.TensorBytes, Dtypes = dtypes, Streamed = summary.Streamed, StorageRestarts = summary.StorageRestarts,
        },
        Verification = new ManifestVerification
        {
            Equal = true, TensorsCompared = verification.TensorCount,
            Method = "names and order, shapes, dtypes and bytes: the pickle through the safe-subset reader, the repack through the safetensors loader",
        },
        License = new ManifestLicense
        {
            Id = license.Id, Name = license.Name, Spdx = license.Spdx, HostObligations = license.HostObligations, TextFiles = license.TextFiles,
        },
        Notes = item.Notes,
    };
}

static int Check(Dictionary<string, string> options)
{
    if (!options.TryGetValue("--staging", out string? staging))
    {
        return Fail("check needs --staging.");
    }
    MirrorManifest manifest = MirrorManifest.LoadOrNew(Path.Combine(staging, "manifest.json"));
    int bad = 0;
    foreach (MirrorManifestItem item in manifest.Items)
    {
        string path = Path.Combine(staging, item.Repack.Path);
        bool ok = File.Exists(path) && new FileInfo(path).Length == item.Repack.Size && Sha256Of(path) == item.Repack.Sha256;
        Console.WriteLine($"[{item.Id}] {(ok ? "ok" : "MISMATCH")} {item.Repack.Path}");
        bad += ok ? 0 : 1;
    }
    Console.WriteLine($"{manifest.Items.Count - bad} of {manifest.Items.Count} files match the manifest.");
    return bad == 0 ? 0 : 1;
}
