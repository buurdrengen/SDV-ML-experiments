using System.Text.Json.Nodes;
using Microsoft.Xna.Framework;
using StardewValley;
using StardewValley.Buildings;

namespace SdvBridge;

/// <summary>Per-Tile flags in the View, OR-ed together on the wire.</summary>
[Flags]
internal enum TileFlags
{
    None = 0,
    /// <summary>Inside the Location's bounds, and the player would not collide with anything standing on it.</summary>
    Walkable = 1,
    /// <summary>Leads to another Location: a warp the player walks onto, a door they interact with, or a farm building's door.</summary>
    Exit = 2,
    /// <summary>Holds an object, a piece of furniture or an NPC.</summary>
    Occupied = 4,
}

/// <summary>The View: the square of Tiles centred on the player, as rows of <see cref="TileFlags"/>.</summary>
internal static class View
{
    /// <summary>Tiles from the player to the View's edge; the View is <c>2 * Radius + 1</c> Tiles wide.</summary>
    public const int Radius = 7;

    /// <summary>Rows top to bottom (y - Radius .. y + Radius), each left to right (x - Radius .. x + Radius).</summary>
    public static TileFlags[][] Read(GameLocation location, Point centre)
    {
        HashSet<Point> exits = Exits(location);
        var rows = new TileFlags[2 * Radius + 1][];
        for (int row = 0; row < rows.Length; row++)
        {
            rows[row] = new TileFlags[2 * Radius + 1];
            for (int col = 0; col < rows[row].Length; col++)
            {
                var tile = new Point(centre.X - Radius + col, centre.Y - Radius + row);
                rows[row][col] = Flags(location, tile, exits);
            }
        }
        return rows;
    }

    public static JsonArray ToJson(TileFlags[][] rows) =>
        new(rows.Select(row => (JsonNode)new JsonArray(row.Select(flags => (JsonNode)(int)flags).ToArray())).ToArray());

    private static TileFlags Flags(GameLocation location, Point tile, HashSet<Point> exits)
    {
        TileFlags flags = TileFlags.None;
        // Warps on a Location's edge sit just outside its bounds, so exits are flagged even there.
        if (exits.Contains(tile))
            flags |= TileFlags.Exit;
        if (!location.isTileOnMap(tile))
            return flags;
        if (IsWalkable(location, tile))
            flags |= TileFlags.Walkable;
        if (location.isObjectAtTile(tile.X, tile.Y) || location.isCharacterAtTile(tile.ToVector2()) is not null)
            flags |= TileFlags.Occupied;
        return flags;
    }

    /// <summary>Whether the player's bounding box, placed on <paramref name="tile"/>, collides with anything (walls, objects, NPCs, buildings…).</summary>
    private static bool IsWalkable(GameLocation location, Point tile)
    {
        // The farmer's bounding box is 48x32 pixels; centre it in the 64x64 Tile.
        var box = new Rectangle(tile.X * Game1.tileSize + 8, tile.Y * Game1.tileSize + 16, 48, 32);
        return !location.isCollidingPosition(box, Game1.viewport, isFarmer: true, damagesFarmer: 0, glider: false, Game1.player,
            pathfinding: false, skipCollisionEffects: true);
    }

    private static HashSet<Point> Exits(GameLocation location)
    {
        var exits = new HashSet<Point>(location.warps.Select(warp => new Point(warp.X, warp.Y)));
        exits.UnionWith(location.doors.Keys);
        foreach (Building building in location.buildings)
            if (building.HasIndoors() && building.humanDoor.X >= 0)
                exits.Add(new Point(building.tileX.Value + building.humanDoor.X, building.tileY.Value + building.humanDoor.Y));
        return exits;
    }
}
