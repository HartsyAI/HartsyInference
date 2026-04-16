namespace SharpInference.Server;

public class Program
{
    public static void Main(string[] args)
    {
        WebApplicationBuilder builder = WebApplication.CreateBuilder(args);

        // TODO: builder.Services.AddSharpInference(options => { ... });

        WebApplication app = builder.Build();

        app.MapGet("/health", () => Results.Ok(new { status = "healthy" }));
        app.MapGet("/ready", () => Results.Ok(new { status = "ready" }));

        // TODO: app.MapSharpInferenceEndpoints();

        app.Run();
    }
}
