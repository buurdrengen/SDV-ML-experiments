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

    /// <summary>Set when the agent disconnected before the reply was ready; the game thread drops the request.</summary>
    public volatile bool Abandoned;
}

/// <summary>Accepts one client at a time over newline-delimited JSON; requests are queued for the game thread.</summary>
internal sealed class BridgeServer
{
    /// <summary>The Bridge's port, overridable with <c>SDV_BRIDGE_PORT</c>. The Python client mirrors both.</summary>
    public const int DefaultPort = 47800;
    public const string PortVariable = "SDV_BRIDGE_PORT";

    private const int DisconnectPollMs = 100;

    private readonly int port;
    private readonly IMonitor monitor;
    private readonly BlockingCollection<Request> queue = new();

    public BridgeServer(IMonitor monitor)
    {
        port = int.TryParse(Environment.GetEnvironmentVariable(PortVariable), out int p) ? p : DefaultPort;
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
                while (reader.ReadLine() is { } line && Handle(line, client.Client) is { } reply)
                    writer.WriteLine(reply.ToJsonString());
            }
            catch (IOException) { }
            monitor.Log("Agent disconnected", LogLevel.Info);
        }
    }

    /// <summary>Queues the request and waits for its reply; returns null if the agent disconnects meanwhile.</summary>
    private JsonObject? Handle(string line, Socket socket)
    {
        Request request;
        try
        {
            request = new Request(JsonNode.Parse(line)!.AsObject(), new(TaskCreationOptions.RunContinuationsAsynchronously));
        }
        catch (Exception ex)
        {
            return Error(ex.GetBaseException().Message);
        }

        queue.Add(request);
        while (!request.Reply.Task.Wait(DisconnectPollMs))
        {
            if (socket.Poll(0, SelectMode.SelectRead) && socket.Available == 0)
            {
                request.Abandoned = true;
                return null;
            }
        }
        return request.Reply.Task.Result;
    }
}
