using System.Diagnostics;
using System.Text.Json.Nodes;
using HarmonyLib;
using Microsoft.Xna.Framework;
using StardewModdingAPI;
using StardewValley;
using StardewValley.Menus;

namespace SdvBridge;

/// <summary>
/// Replaces MonoGame's <c>Game.Tick</c> once an agent sends its first request. From then on the game only updates
/// while a Step (or load) is running, always with a fixed 1/60 s timestep, so the game sees normal speed no
/// matter how fast it really runs. Fast mode runs updates back to back and never draws; render mode runs one
/// update per frame at 60 FPS and draws.
/// </summary>
internal static class LockstepLoop
{
    private static readonly TimeSpan Dt = TimeSpan.FromTicks(TimeSpan.TicksPerSecond / 60);
    private const int FastSliceMs = 100;

    private static BridgeServer server = null!;
    private static IMonitor monitor = null!;
    private static bool engaged;
    private static bool fast;
    private static Action<GameTime> doUpdate = null!;
    private static Action<GameTime> doDraw = null!;
    private static TimeSpan totalTime;
    private static long frameStart;

    private static IJob? job;
    private static Request? jobRequest;
    private static long updates;

    public static void Init(BridgeServer bridgeServer, IMonitor modMonitor)
    {
        server = bridgeServer;
        monitor = modMonitor;
    }

    public static bool TickPrefix(Game __instance)
    {
        if (Traverse.Create(__instance).Field<bool>("_shouldExit").Value)
            return true;

        if (!engaged)
        {
            if (!server.TryTake(out Request? first, 0))
                return true;
            Engage(__instance);
            Accept(first!);
        }

        try
        {
            if (fast)
                RunFastSlice();
            else
                RunRenderedFrame();
        }
        catch (Exception ex)
        {
            monitor.Log($"Lockstep loop failed: {ex}", LogLevel.Error);
            Finish(BridgeServer.Error(ex.GetBaseException().Message));
        }
        return false;
    }

    /// <summary>Keep the game behaving as focused (no pause, input read) while an agent drives it.</summary>
    public static bool IsActivePrefix(ref bool __result)
    {
        if (!engaged)
            return true;
        __result = true;
        return false;
    }

    public static JsonObject State()
    {
        var reply = new JsonObject { ["ok"] = true, ["updates"] = updates, ["time"] = Game1.timeOfDay, ["day"] = Game1.dayOfMonth };
        if (Context.IsWorldReady)
        {
            Point tile = Game1.player.TilePoint;
            reply["location"] = Game1.player.currentLocation?.NameOrUniqueName;
            reply["x"] = tile.X;
            reply["y"] = tile.Y;
        }
        return reply;
    }

    private static void Engage(Game game)
    {
        engaged = true;
        doUpdate = AccessTools.Method(typeof(Game), "DoUpdate").CreateDelegate<Action<GameTime>>(game);
        doDraw = AccessTools.Method(typeof(Game), "DoDraw").CreateDelegate<Action<GameTime>>(game);
        totalTime = Traverse.Create(game).Field<GameTime>("_gameTime").Value.TotalGameTime;
        frameStart = Stopwatch.GetTimestamp();
        monitor.Log("Agent took control of the game loop (lockstep)", LogLevel.Info);
    }

    private static void RunFastSlice()
    {
        var slice = Stopwatch.StartNew();
        while (fast && slice.ElapsedMilliseconds < FastSliceMs)
        {
            if (job is null)
            {
                if (server.TryTake(out Request? request, 5))
                    Accept(request!);
                continue;
            }
            UpdateOnce();
        }
    }

    private static void RunRenderedFrame()
    {
        if (job is null && server.TryTake(out Request? request, 0))
            Accept(request!);
        if (job is not null)
            UpdateOnce();

        doDraw(new GameTime(totalTime, Dt));

        // Pace to 60 FPS ourselves rather than relying on vsync.
        long frameTicks = Stopwatch.Frequency / 60;
        while (Stopwatch.GetTimestamp() - frameStart < frameTicks)
            Thread.Sleep(0);
        frameStart = Stopwatch.GetTimestamp();
    }

    private static void Accept(Request request)
    {
        try
        {
            switch (request.Cmd)
            {
                case "mode":
                    fast = request.Body["fast"]?.GetValue<bool>() ?? false;
                    request.Reply.SetResult(State());
                    return;
                case "state":
                    request.Reply.SetResult(State());
                    return;
                case "load":
                    Start(request, new LoadJob(request.Body["save"]?.GetValue<string>() ?? throw new ArgumentException("load needs 'save'")));
                    return;
                case "move":
                    if (!Context.IsWorldReady)
                        throw new InvalidOperationException("no save loaded");
                    Start(request, new MoveJob(request.Body["dir"]?.GetValue<string>() ?? ""));
                    return;
                default:
                    throw new ArgumentException($"unknown cmd '{request.Cmd}'");
            }
        }
        catch (Exception ex)
        {
            request.Reply.SetResult(BridgeServer.Error(ex.Message));
        }
    }

    private static void Start(Request request, IJob newJob)
    {
        job = newJob;
        jobRequest = request;
    }

    private static void UpdateOnce()
    {
        job!.BeforeUpdate();
        totalTime += Dt;
        doUpdate(new GameTime(totalTime, Dt));
        updates++;
        if (Context.IsWorldReady)
            AutoDismiss();
        if (job.AfterUpdate() is { } reply)
            Finish(reply);
    }

    private static void Finish(JsonObject reply)
    {
        StepInput.Release();
        job = null;
        jobRequest?.Reply.TrySetResult(reply);
        jobRequest = null;
    }

    /// <summary>The agent never sees menus, dialogue or skippable cutscenes.</summary>
    private static void AutoDismiss()
    {
        switch (Game1.activeClickableMenu)
        {
            case null:
                break;
            // Close dialogue the way the game does, so it restores the player's state (e.g. stepping off the bed).
            case DialogueBox { isQuestion: true } question:
                Response decline = question.responses.FirstOrDefault(r => r.responseKey == "No") ?? question.responses[^1];
                Game1.currentLocation.answerDialogue(decline);
                question.closeDialogue();
                break;
            case DialogueBox dialogue:
                dialogue.closeDialogue();
                break;
            default:
                Game1.exitActiveMenu();
                break;
        }
        if (Game1.CurrentEvent is { skippable: true, skipped: false } ev)
            ev.skipEvent();
    }
}
