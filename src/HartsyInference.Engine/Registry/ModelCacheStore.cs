using System.Text.Json;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Registry;

/// <summary>Disk-based model cache at ~/.hartsyinference/models/. Tracks downloaded models and their local file paths for fast re-loading without re-downloading.</summary>
public sealed class ModelCacheStore
{
    private readonly string _cacheDir;
    private readonly string _indexPath;
    private Dictionary<string, ModelInfo> _index = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>Root directory of the model cache.</summary>
    public string CacheDirectory => _cacheDir;

    /// <summary>All cached model IDs.</summary>
    public IReadOnlyCollection<string> CachedModelIds => _index.Keys;

    /// <summary>Creates a model cache store at the default location (~/.hartsyinference/models/).</summary>
    public ModelCacheStore() : this(DefaultCacheDirectory())
    {
    }

    /// <summary>Creates a model cache store at the specified directory.</summary>
    public ModelCacheStore(string cacheDirectory)
    {
        _cacheDir = cacheDirectory;
        _indexPath = Path.Combine(_cacheDir, "index.json");

        Directory.CreateDirectory(_cacheDir);
        LoadIndex();
    }

    /// <summary>Registers a model in the disk cache index.</summary>
    public void Add(ModelInfo info)
    {
        _index[info.Id] = info;
        SaveIndex();
        Logs.Debug($"Cached model '{info.Name}' at {info.LocalPath}.");
    }

    /// <summary>Looks up a cached model by ID.</summary>
    public ModelInfo? Get(string modelId)
    {
        if (_index.TryGetValue(modelId, out ModelInfo? info))
        {
            if (File.Exists(info.LocalPath))
                return info;

            // File was deleted externally — remove stale entry
            _index.Remove(modelId);
            SaveIndex();
        }
        return null;
    }

    /// <summary>Removes a model from the cache index (does not delete the file).</summary>
    public bool Remove(string modelId)
    {
        if (_index.Remove(modelId))
        {
            SaveIndex();
            return true;
        }
        return false;
    }

    /// <summary>Returns the path where a model file should be stored in the cache.</summary>
    public string GetCachePath(string modelId, string fileName)
    {
        string modelDir = Path.Combine(_cacheDir, SanitizeId(modelId));
        Directory.CreateDirectory(modelDir);
        return Path.Combine(modelDir, fileName);
    }

    private void LoadIndex()
    {
        if (!File.Exists(_indexPath))
            return;

        try
        {
            string json = File.ReadAllText(_indexPath);
            List<ModelInfo>? entries = JsonSerializer.Deserialize<List<ModelInfo>>(json, _jsonOptions);
            if (entries is not null)
            {
                foreach (ModelInfo entry in entries)
                    _index[entry.Id] = entry;
            }
        }
        catch (Exception ex)
        {
            Logs.Warning($"Failed to load model cache index: {ex.Message}");
        }
    }

    private void SaveIndex()
    {
        try
        {
            string json = JsonSerializer.Serialize(_index.Values.ToList(), _jsonOptions);
            File.WriteAllText(_indexPath, json);
        }
        catch (Exception ex)
        {
            Logs.Warning($"Failed to save model cache index: {ex.Message}");
        }
    }

    private static string SanitizeId(string id) =>
        string.Concat(id.Select(c => char.IsLetterOrDigit(c) || c == '-' || c == '_' ? c : '_'));

    private static string DefaultCacheDirectory()
    {
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        return Path.Combine(home, ".hartsyinference", "models");
    }

    private static readonly JsonSerializerOptions _jsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    };
}
