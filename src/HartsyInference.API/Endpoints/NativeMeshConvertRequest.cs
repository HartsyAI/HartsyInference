namespace HartsyInference.API.Endpoints;

/// <summary>Body of <c>POST /v1/native/mesh/convert</c>.</summary>
public sealed class NativeMeshConvertRequest
{
    /// <summary>The model file, base64.</summary>
    public string? Data { get; set; }

    /// <summary>The file's format: <c>glb</c> or <c>obj</c>.</summary>
    public string? Format { get; set; }

    /// <summary>The format to write: <c>glb</c>, <c>obj</c> (with its MTL and images), <c>ply</c> or <c>stl</c>.</summary>
    public string? Target { get; set; }

    /// <summary>The base name of the output files; <c>model</c> when absent.</summary>
    public string? Name { get; set; }

    /// <summary>Return the files as one zip archive, base64, in <c>Archive</c>, instead of each file's bytes in <c>Files</c> (which then list names and types only).</summary>
    public bool Archive { get; set; }
}
