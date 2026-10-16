using HarmonyLib;
using Microsoft.Xna.Framework;
using StardewModdingAPI;

namespace SdvBridge;

/// <summary>Starts the socket server and takes over MonoGame's frame loop so the game only advances during a Step.</summary>
internal sealed class ModEntry : Mod
{
    public override void Entry(IModHelper helper)
    {
        var server = new BridgeServer(Monitor);
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
