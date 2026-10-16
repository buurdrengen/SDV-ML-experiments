using System.Collections.Concurrent;
using System.Net;
using System.Net.Sockets;
using System.Text.Json.Nodes;
using StardewModdingAPI;

namespace SdvBridge;

/// <summary>One request from the agent, answered on the game thread.</summary>
internal sealed record Request(JsonObject Body, TaskCompletionSource<JsonObject> Reply)
{
    public string Cmd => Body["cmd"]?.GetValue<string>() ?? "";
}

/// <summary>Accepts one client at a time over newline-delimited JSON; requests are queued for the game thread.</summary>
internal sealed class BridgeServer
{
    private readonly int port;
    private readonly IMonitor monitor;
    private readonly BlockingCollection<Request> queue = new();

    public BridgeServer(int port, IMonitor monitor)
    {
        this.port = port;
        this.monitor = monitor;
    }

    public void Start() => new Thread(Serve) { IsBackground = true, Name = "SdvBridge" }.Start();

    public bool TryTake(out Request? request, int timeoutMs) => queue.TryTake(out request, timeoutMs);

    public static JsonObject Error(string message) => new() { ["ok"] = false, ["error"] = message };

    private void Serve()
    {
        var listener = new TcpListener(IPAddress.Loopback, port);
        listener.Start();
        monitor.Log($"Bridge listening on 127.0.0.1:{port}", LogLevel.Info);

        while (true)
        {
            using TcpClient client = listener.AcceptTcpClient();
            client.NoDelay = true;
            monitor.Log("Agent connected", LogLevel.Info);
            try
            {
                using NetworkStream stream = client.GetStream();
                using var reader = new StreamReader(stream);
                using var writer = new StreamWriter(stream) { AutoFlush = true, NewLine = "\n" };
                while (reader.ReadLine() is { } line)
                    writer.WriteLine(Handle(line).ToJsonString());
            }
            catch (IOException) { }
            monitor.Log("Agent disconnected", LogLevel.Info);
        }
    }

    private JsonObject Handle(string line)
    {
        try
        {
            var request = new Request(JsonNode.Parse(line)!.AsObject(), new(TaskCreationOptions.RunContinuationsAsynchronously));
            queue.Add(request);
            return request.Reply.Task.Result;
        }
        catch (Exception ex)
        {
            return Error(ex.GetBaseException().Message);
        }
    }
}
