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

internal static class Player
{
    /// <summary>The player is free to act: no warp, fade, menu, dialogue or cutscene, and nothing freezing them.</summary>
    public static bool IsSettled() =>
        Context.IsWorldReady
        && Game1.player.CanMove
        && !IsTransitioning()
        && Game1.activeClickableMenu is null && !Game1.dialogueUp && !Game1.eventUp;

    /// <summary>A warp or screen fade is under way, which freezes the player without them being blocked.</summary>
    public static bool IsTransitioning() =>
        Game1.isWarping || Game1.globalFade || Game1.fadeToBlack || Game1.locationRequest is not null;

    public static Point TileCentre(Point tile) =>
        new(tile.X * Game1.tileSize + Game1.tileSize / 2, tile.Y * Game1.tileSize + Game1.tileSize / 2);

    /// <summary>Moves the player onto <paramref name="tile"/>'s centre if their bounding box fits there.</summary>
    public static bool TrySnapTo(Point tile)
    {
        Farmer player = Game1.player;
        Point px = player.StandingPixel;
        Point offset = TileCentre(tile) - px;
        Rectangle box = player.GetBoundingBox();
        box.Offset(offset);
        if (player.currentLocation.isCollidingPosition(box, Game1.viewport, isFarmer: true, 0, glider: false, player))
            return false;
        player.Position += offset.ToVector2();
        return true;
    }
}

/// <summary>
/// One Step: act (walk one Tile, interact with the faced Tile, or idle), then let the game run on until the player
/// is settled again, so the next Step starts from a still, controllable player. Ends early when the Episode does.
/// </summary>
internal sealed class StepJob : IJob
{
    /// <summary>Interact and do-nothing Steps last about as long as a move (#4 measured ~11.8 updates), so every Step advances the clock alike.</summary>
    private const int IdleUpdates = 12;
    /// <summary>Generous cap for warps, fades and unskippable cutscenes.</summary>
    private const int MaxUpdates = 60 * 60;

    private readonly StepAction action;
    private readonly IMonitor monitor;
    private readonly string? startLocation;
    private readonly MoveTracker? move;
    private Outcome? moveOutcome;
    private bool acting = true;
    private int updates;

    public StepJob(StepAction action, IMonitor monitor)
    {
        this.action = action;
        this.monitor = monitor;
        startLocation = Game1.player.currentLocation?.NameOrUniqueName;
        move = action switch
        {
            StepAction.Up => new MoveTracker(0, -1),
            StepAction.Down => new MoveTracker(0, 1),
            StepAction.Left => new MoveTracker(-1, 0),
            StepAction.Right => new MoveTracker(1, 0),
            _ => null,
        };
    }

    public void BeforeUpdate()
    {
        if (!acting)
            return;
        if (move is not null)
            StepInput.Hold(move.Dx, move.Dy);
        else if (action == StepAction.Interact && updates == 0)
            // Check the faced Tile, like the action button does, but never the Tile under the mouse. SDV's own
            // GetGrabTile() only reaches 5 px past the bounding box, which from a Tile's centre is the player's own Tile.
            Game1.tryToCheckAt(Game1.player.Tile + Utility.DirectionsTileVectors[Game1.player.FacingDirection], Game1.player);
    }

    public JsonObject? AfterUpdate()
    {
        updates++;
        if (GameState.CurrentEpisodeEnd() is not null)
            return Finish();

        if (acting)
        {
            if (move is not null)
            {
                moveOutcome = move.Check();
                if (moveOutcome is not null)
                    EndMove();
            }
            else if (updates >= IdleUpdates)
                acting = false;
        }

        if (!acting && Player.IsSettled())
            return Finish();
        if (updates >= MaxUpdates)
        {
            monitor.Log($"Step '{Wire.Name(action)}' still not settled after {updates} updates (menu={Game1.activeClickableMenu}, event={Game1.eventUp}, canMove={Game1.player.CanMove})", LogLevel.Warn);
            return Finish();
        }
        return null;
    }

    private void EndMove()
    {
        acting = false;
        StepInput.Release();
        Game1.player.Halt();
        // SDV nudges the player around corners, which can leave them off-centre or on a diagonal Tile; re-centre on the Tile the move aimed for.
        if (moveOutcome == Outcome.Moved && !Player.TrySnapTo(move!.Target))
            Player.TrySnapTo(Game1.player.TilePoint);
    }

    private JsonObject Finish()
    {
        StepInput.Release();
        bool warped = Game1.player.currentLocation?.NameOrUniqueName != startLocation;
        Outcome outcome = warped ? Outcome.Warped : moveOutcome ?? (move is null ? Outcome.Stayed : Outcome.Blocked);
        JsonObject reply = GameState.Read(LockstepLoop.TotalUpdates).ToJson();
        reply["outcome"] = Wire.Name(outcome);
        reply["updates"] = updates;
        return reply;
    }
}

/// <summary>Watches a one-Tile walk: done at the next Tile's centre, or blocked once the player stops making progress.</summary>
internal sealed class MoveTracker
{
    private const int StillUpdatesForBlocked = 4;
    private const int MaxUpdates = 180;

    public readonly int Dx, Dy;
    public readonly Point Target;
    private int bestProgress;
    private int updates, stillUpdates;

    public MoveTracker(int dx, int dy)
    {
        (Dx, Dy) = (dx, dy);
        Target = Game1.player.TilePoint + new Point(dx, dy);
        bestProgress = Progress(Game1.player.StandingPixel);
    }

    /// <summary>Called after each update; null while the move is still under way.</summary>
    public Outcome? Check()
    {
        updates++;
        Point px = Game1.player.StandingPixel;
        Point centre = Player.TileCentre(Target);
        bool reached = Dx > 0 ? px.X >= centre.X : Dx < 0 ? px.X <= centre.X : Dy > 0 ? px.Y >= centre.Y : px.Y <= centre.Y;
        if (reached)
            return Outcome.Moved;

        // No progress along the move direction is being blocked (sliding sideways along a wall doesn't count),
        // and so is being frozen, unless a warp or fade is under way.
        int progress = Progress(px);
        stillUpdates = progress <= bestProgress && !Player.IsTransitioning() ? stillUpdates + 1 : 0;
        bestProgress = Math.Max(bestProgress, progress);
        if (stillUpdates >= StillUpdatesForBlocked || updates >= MaxUpdates)
            return Outcome.Blocked;
        return null;
    }

    private int Progress(Point px) => Dx * px.X + Dy * px.Y;
}

/// <summary>Copies the Starting save into SDV's saves folder, returns to the title screen, loads it and waits until the player is settled.</summary>
internal sealed class ResetJob : IJob
{
    private const int MaxUpdates = 60 * 120;

    private readonly string saveName;
    private bool loadStarted;
    private int updates;

    public ResetJob(string saveFolder)
    {
        string source = Path.TrimEndingDirectorySeparator(Path.GetFullPath(saveFolder));
        if (!Directory.Exists(source))
            throw new ArgumentException($"no save folder at '{source}'");
        saveName = Path.GetFileName(source);
        if (!File.Exists(Path.Combine(source, saveName)))
            throw new ArgumentException($"'{source}' is not an SDV save folder (no '{saveName}' file inside)");

        // SDV only loads from its own saves folder, so overwrite its copy with the Starting save every time;
        // that also undoes anything the game wrote there.
        string target = Path.Combine(Constants.SavesPath, saveName);
        Directory.CreateDirectory(target);
        foreach (string file in Directory.GetFiles(source))
            File.Copy(file, Path.Combine(target, Path.GetFileName(file)), overwrite: true);

        if (Context.IsWorldReady)
            Game1.ExitToTitle();
    }

    public void BeforeUpdate() { }

    public JsonObject? AfterUpdate()
    {
        if (++updates >= MaxUpdates)
            return BridgeServer.Error($"loading '{saveName}' timed out");

        if (!loadStarted)
        {
            if (Context.IsWorldReady || Game1.activeClickableMenu is not TitleMenu)
                return null;
            Game1.multiplayerMode = 0;
            SaveGame.Load(saveName);
            Game1.exitActiveMenu();
            loadStarted = true;
            return null;
        }

        if (!Player.IsSettled())
            return null;
        Game1.options.pauseWhenOutOfFocus = false;
        JsonObject reply = GameState.Read(LockstepLoop.TotalUpdates).ToJson();
        reply["updates"] = updates;
        return reply;
    }
}

/// <summary>Testing aid: warps the player to a Tile (optionally setting the clock) and waits until they are settled there.</summary>
internal sealed class TeleportJob : IJob
{
    private const int MaxUpdates = 60 * 10;

    private readonly string location;
    private int updates;

    public TeleportJob(string location, Point tile, Direction facing, int? time)
    {
        this.location = location;
        if (time is { } t)
            Game1.timeOfDay = t;
        Game1.warpFarmer(location, tile.X, tile.Y, (int)facing);
    }

    public void BeforeUpdate() { }

    public JsonObject? AfterUpdate()
    {
        if (++updates >= MaxUpdates)
            return BridgeServer.Error($"teleport to '{location}' timed out");
        if (Game1.player.currentLocation?.NameOrUniqueName != location || !Player.IsSettled())
            return null;
        return GameState.Read(LockstepLoop.TotalUpdates).ToJson();
    }
}
