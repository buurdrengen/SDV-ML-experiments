using System.Text.Json.Nodes;
using StardewModdingAPI;
using StardewValley;

namespace SdvBridge;

/// <summary>The Game state sent back with every reply. Location fields are null until a save is loaded.</summary>
internal sealed record GameState(
    long TotalUpdates, int Time, int Day,
    string? Location, int? X, int? Y, Direction? Facing,
    string? Menu, EpisodeEnd? EpisodeEnd)
{
    /// <summary>2:00 AM, as SDV writes it in <c>Game1.timeOfDay</c>.</summary>
    private const int TwoAm = 2600;

    public static GameState Read(long totalUpdates)
    {
        if (!Context.IsWorldReady)
            return new(totalUpdates, Game1.timeOfDay, Game1.dayOfMonth, null, null, null, null, Game1.activeClickableMenu?.GetType().Name, null);

        Farmer player = Game1.player;
        return new(
            totalUpdates, Game1.timeOfDay, Game1.dayOfMonth,
            player.currentLocation?.NameOrUniqueName, player.TilePoint.X, player.TilePoint.Y, (Direction)player.FacingDirection,
            Game1.activeClickableMenu?.GetType().Name,
            CurrentEpisodeEnd());
    }

    public static EpisodeEnd? CurrentEpisodeEnd()
    {
        if (!Context.IsWorldReady)
            return null;
        if (Game1.timeOfDay >= TwoAm)
            return SdvBridge.EpisodeEnd.TwoAm;
        if (Game1.player.passedOut || Game1.player.FarmerSprite.isPassingOut())
            return SdvBridge.EpisodeEnd.PassedOut;
        return null;
    }

    public JsonObject ToJson() => new()
    {
        ["ok"] = true,
        ["total_updates"] = TotalUpdates,
        ["time"] = Time,
        ["day"] = Day,
        ["location"] = Location,
        ["x"] = X,
        ["y"] = Y,
        ["facing"] = Facing is { } facing ? Wire.Name(facing) : null,
        ["menu"] = Menu,
        ["episode_end"] = EpisodeEnd is { } end ? Wire.Name(end) : null,
    };
}
