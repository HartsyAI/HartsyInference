using HartsyInference.ThreeD.Document;

namespace HartsyInference.ThreeD.Operations;

/// <summary>The operations the engine ships, and the runner that validates and applies a list of them. To add an operation, implement <see cref="IMeshOperation"/> and add one instance to <see cref="All"/>; the service, its HTTP routes and the operation listing pick it up, and no model profile changes.</summary>
public static class MeshOperations
{
    /// <summary>The most steps one request may hold.</summary>
    public const int MaxSteps = 16;

    /// <summary>Every operation, in the order they are listed.</summary>
    public static IReadOnlyList<IMeshOperation> All { get; } =
    [
        new RemoveSmallComponentsOperation(), new FillHolesOperation(), new DecimateOperation(), new SmoothOperation(),
        new RecomputeNormalsOperation(), new RecenterOperation(), new RescaleOperation(),
    ];

    /// <summary>The operation for <paramref name="id"/> (case-insensitive).</summary>
    /// <exception cref="NotSupportedException">No operation has that id.</exception>
    public static IMeshOperation Get(string id) =>
        All.FirstOrDefault(o => string.Equals(o.Id, id, StringComparison.OrdinalIgnoreCase))
        ?? throw new NotSupportedException($"Operation '{id}' is not supported; expected one of {string.Join(", ", All.Select(o => o.Id))}.");

    /// <summary>Checks a step's numbers against the operation's parameters and fills in the defaults.</summary>
    /// <exception cref="ArgumentException">A name is not a parameter of the operation, or a value is not a finite whole number where one is needed.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A value is outside its parameter's range.</exception>
    public static MeshOperationArguments Validate(IMeshOperation operation, IReadOnlyDictionary<string, double>? given)
    {
        Dictionary<string, double> values = [];
        HashSet<string> seen = new(StringComparer.OrdinalIgnoreCase);
        foreach (MeshOperationParameter parameter in operation.Parameters.Where(p => p.Default is not null))
            values[parameter.Name] = parameter.Default!.Value;
        foreach ((string name, double value) in given ?? new Dictionary<string, double>())
        {
            MeshOperationParameter parameter = operation.Parameters.FirstOrDefault(p => string.Equals(p.Name, name, StringComparison.OrdinalIgnoreCase))
                ?? throw new ArgumentException($"'{operation.Id}' has no parameter '{name}'; it takes {(operation.Parameters.Count == 0 ? "none" : string.Join(", ", operation.Parameters.Select(p => p.Name)))}.");
            if (!seen.Add(parameter.Name))
                throw new ArgumentException($"Parameter '{parameter.Name}' of '{operation.Id}' is given more than once (names are not case-sensitive).");
            if (!double.IsFinite(value))
                throw new ArgumentException($"Parameter '{parameter.Name}' of '{operation.Id}' must be a finite number.");
            if (parameter.Integer && value != Math.Floor(value))
                throw new ArgumentException($"Parameter '{parameter.Name}' of '{operation.Id}' must be a whole number.");
            if (value < parameter.Min || value > parameter.Max)
                throw new ArgumentOutOfRangeException(parameter.Name, value, $"Parameter '{parameter.Name}' of '{operation.Id}' must be between {parameter.Min} and {parameter.Max}.");
            values[parameter.Name] = value;
        }
        return new MeshOperationArguments(values);
    }

    /// <summary>Validates every step up front (so a bad last step fails before any work is done), then applies them in order, each to the result of the one before.</summary>
    /// <exception cref="ArgumentException">There are no steps or more than <see cref="MaxSteps"/>, or a step is invalid.</exception>
    /// <exception cref="NotSupportedException">A step names an operation that does not exist.</exception>
    public static MeshOperationRun Run(MeshDocument document, IReadOnlyList<MeshOperationStep> steps, CancellationToken cancel = default)
    {
        ArgumentNullException.ThrowIfNull(document);
        ArgumentNullException.ThrowIfNull(steps);
        if (steps.Count is 0 or > MaxSteps)
            throw new ArgumentException($"Send between 1 and {MaxSteps} operations.", nameof(steps));
        List<(IMeshOperation Operation, MeshOperationArguments Arguments)> prepared = [];
        foreach (MeshOperationStep step in steps)
        {
            IMeshOperation operation = Get(step.Operation);
            MeshOperationArguments arguments = Validate(operation, step.Parameters);
            operation.Check(arguments);
            prepared.Add((operation, arguments));
        }

        List<MeshOperationReport> reports = [];
        MeshDocument current = document;
        foreach ((IMeshOperation operation, MeshOperationArguments arguments) in prepared)
        {
            cancel.ThrowIfCancellationRequested();
            MeshOperationOutcome outcome = operation.Apply(current, arguments, cancel);
            reports.Add(new MeshOperationReport(operation.Id, current.VertexCount, outcome.Document.VertexCount, current.TriangleCount, outcome.Document.TriangleCount, outcome.Notes));
            current = outcome.Document;
        }
        return new MeshOperationRun(current, reports);
    }
}
