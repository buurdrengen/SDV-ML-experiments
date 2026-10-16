using HarmonyLib;
using Microsoft.Xna.Framework;
using StardewModdingAPI;

namespace SdvBridge;

/// <summary>Starts the socket server and takes over MonoGame's frame loop so the game only advances during a Step.</summary>
internal sealed class ModEntry : Mod
{
    private const int DefaultPort = 47800;

    public override void Entry(IModHelper helper)
    {
        int port = int.TryParse(Environment.GetEnvironmentVariable("SDV_BRIDGE_PORT"), out int p) ? p : DefaultPort;
        var server = new BridgeServer(port, Monitor);
        LockstepLoop.Init(server, Monitor);

        var harmony = new Harmony(ModManifest.UniqueID);
        harmony.Patch(
            AccessTools.Method(typeof(Game), "Tick"),
            prefix: new HarmonyMethod(typeof(LockstepLoop), nameof(LockstepLoop.TickPrefix)));
        harmony.Patch(
            AccessTools.PropertyGetter(typeof(Game), nameof(Game.IsActive)),
            prefix: new HarmonyMethod(typeof(LockstepLoop), nameof(LockstepLoop.IsActivePrefix)));

        server.Start();
    }
}
