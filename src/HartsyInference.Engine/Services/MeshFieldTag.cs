namespace HartsyInference.Engine.Services;

/// <summary>Names the request field a validation exception is about, in <see cref="Exception.Data"/>, so planning can point a host at the right control without changing the exception's type.</summary>
internal static class MeshFieldTag
{
    private const string Key = "mesh.field";

    /// <summary>Records <paramref name="field"/> on <paramref name="error"/> and returns it, for <c>throw new X(...).ForField(...)</c>.</summary>
    public static TException ForField<TException>(this TException error, string field) where TException : Exception
    {
        error.Data[Key] = field;
        return error;
    }

    /// <summary>The field recorded on <paramref name="error"/>, or null.</summary>
    public static string? FieldOf(Exception error) => error.Data[Key] as string;
}
