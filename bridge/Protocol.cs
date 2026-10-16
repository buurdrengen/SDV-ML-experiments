using System.Text;

namespace SdvBridge;

/// <summary>The agent's Discrete(6) action space.</summary>
internal enum StepAction { Up, Down, Left, Right, Interact, Noop }

/// <summary>Whether a Step changed the player's Tile or Location.</summary>
internal enum Outcome { Moved, Blocked, Warped, Stayed }

/// <summary>Why the Bridge ended the Episode.</summary>
internal enum EpisodeEnd { TwoAm, PassedOut }

/// <summary>Facing directions, numbered the way SDV numbers them (<c>Character.FacingDirection</c>).</summary>
internal enum Direction { Up = 0, Right = 1, Down = 2, Left = 3 }

/// <summary>Enums travel as snake_case strings: <c>EpisodeEnd.TwoAm</c> is <c>"two_am"</c>.</summary>
internal static class Wire
{
    public static string Name<T>(T value) where T : struct, Enum
    {
        string pascal = value.ToString();
        var snake = new StringBuilder();
        foreach (char c in pascal)
        {
            if (char.IsUpper(c) && snake.Length > 0)
                snake.Append('_');
            snake.Append(char.ToLowerInvariant(c));
        }
        return snake.ToString();
    }

    public static T Parse<T>(string? name, string field) where T : struct, Enum
    {
        foreach (T value in Enum.GetValues<T>())
            if (Name(value) == name)
                return value;
        throw new ArgumentException($"unknown {field} '{name}'; expected one of {string.Join(", ", Enum.GetValues<T>().Select(Name))}");
    }
}
