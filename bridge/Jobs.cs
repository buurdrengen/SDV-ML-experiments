using System.Text.Json.Nodes;
using Microsoft.Xna.Framework;
using StardewModdingAPI;
using StardewValley;
using StardewValley.Menus;

namespace SdvBridge;

/// <summary>Multi-update work started by one request; returns the reply once done.</summary>
internal interface IJob
{
    void BeforeUpdate();
    JsonObject? AfterUpdate();
}

/// <summary>Moves the player one Tile, ending at the next Tile's centre, on a Location change, or when blocked.</summary>
internal sealed class MoveJob : IJob
{
    private const int TileSize = Game1.tileSize;
    private const int StillTicksForBlocked = 4;
    private const int MaxTicks = 180;

    private readonly int dx, dy;
    private readonly string? startLocation;
    private readonly Point target;
    private int bestProgress;
    private int ticks, stillTicks;

    public MoveJob(string dir)
    {
        (dx, dy) = dir switch
        {
            "up" => (0, -1),
            "down" => (0, 1),
            "left" => (-1, 0),
            "right" => (1, 0),
            _ => throw new ArgumentException($"unknown dir '{dir}'"),
        };
        Farmer player = Game1.player;
        startLocation = player.currentLocation?.NameOrUniqueName;
        target = player.TilePoint + new Point(dx, dy);
        bestProgress = Progress(player.StandingPixel);
    }

    public void BeforeUpdate() => StepInput.Hold(dx, dy);

    public JsonObject? AfterUpdate()
    {
        ticks++;
        Farmer player = Game1.player;
        if (player.currentLocation?.NameOrUniqueName != startLocation)
            return Finish("warped");

        Point px = player.StandingPixel;
        int centreX = target.X * TileSize + TileSize / 2, centreY = target.Y * TileSize + TileSize / 2;
        bool reached = dx > 0 ? px.X >= centreX : dx < 0 ? px.X <= centreX : dy > 0 ? px.Y >= centreY : px.Y <= centreY;
        if (reached)
            return Finish("moved");

        // Blocked = no progress along the move direction; sliding sideways along a wall doesn't count as progress.
        // A fade or warp in progress freezes the player; that's not being blocked.
        bool transitioning = Game1.isWarping || Game1.globalFade || Game1.fadeToBlack || !player.CanMove;
        int progress = Progress(px);
        stillTicks = progress <= bestProgress && !transitioning ? stillTicks + 1 : 0;
        bestProgress = Math.Max(bestProgress, progress);
        if (stillTicks >= StillTicksForBlocked)
            return Finish("blocked");
        if (ticks >= MaxTicks)
        {
            JsonObject reply = Finish("timeout");
            reply["debug"] = $"standingPixel={px} target={target} canMove={player.CanMove} fade={Game1.globalFade}/{Game1.fadeToBlack} event={Game1.eventUp}";
            return reply;
        }
        return null;
    }

    private int Progress(Point px) => dx * px.X + dy * px.Y;

    private JsonObject Finish(string outcome)
    {
        StepInput.Release();
        Game1.player.Halt();
        JsonObject reply = LockstepLoop.State();
        reply["outcome"] = outcome;
        reply["ticks"] = ticks;
        return reply;
    }
}

/// <summary>Returns to the title screen if needed, loads a save, and waits until the player can move.</summary>
internal sealed class LoadJob : IJob
{
    private const int MaxTicks = 60 * 120;
    private const int SettleTicks = 10;

    private readonly string save;
    private bool loadStarted;
    private int ticks, readyTicks;

    public LoadJob(string save)
    {
        this.save = save;
        if (Context.IsWorldReady)
            Game1.ExitToTitle();
    }

    public void BeforeUpdate() { }

    public JsonObject? AfterUpdate()
    {
        if (++ticks >= MaxTicks)
            return BridgeServer.Error($"loading '{save}' timed out");

        if (!loadStarted)
        {
            if (Context.IsWorldReady || Game1.activeClickableMenu is not TitleMenu)
                return null;
            Game1.multiplayerMode = 0;
            SaveGame.Load(save);
            Game1.exitActiveMenu();
            loadStarted = true;
            return null;
        }

        bool ready = Context.IsWorldReady && Game1.player.CanMove && !Game1.globalFade && !Game1.isWarping;
        readyTicks = ready ? readyTicks + 1 : 0;
        if (readyTicks < SettleTicks)
            return null;

        Game1.options.pauseWhenOutOfFocus = false;
        JsonObject reply = LockstepLoop.State();
        reply["ticks"] = ticks;
        return reply;
    }
}
