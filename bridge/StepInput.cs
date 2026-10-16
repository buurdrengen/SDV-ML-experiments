using System.Reflection;
using HarmonyLib;
using StardewValley;
using StardewValley.Util;

namespace SdvBridge;

/// <summary>Feeds the agent's held direction into SDV's built-in input simulation hook.</summary>
internal sealed class StepInput : IInputSimulator
{
    private static readonly StepInput Instance = new();
    private static readonly FieldInfo SimulatorField = AccessTools.Field(typeof(Game1), "inputSimulator");
    private int dx, dy;
    private bool firstTick;

    public static void Hold(int dx, int dy)
    {
        SimulatorField.SetValue(SimulatorField.IsStatic ? null : Game1.game1, Instance);
        if (Instance.dx != dx || Instance.dy != dy)
            Instance.firstTick = true;
        Instance.dx = dx;
        Instance.dy = dy;
    }

    public static void Release()
    {
        Instance.dx = Instance.dy = 0;
        Instance.firstTick = false;
    }

    public void SimulateInput(
        ref bool actionButtonPressed, ref bool switchToolButtonPressed, ref bool useToolButtonPressed,
        ref bool useToolButtonReleased, ref bool addItemToInventoryButtonPressed, ref bool cancelButtonPressed,
        ref bool moveUpPressed, ref bool moveRightPressed, ref bool moveLeftPressed, ref bool moveDownPressed,
        ref bool moveUpReleased, ref bool moveRightReleased, ref bool moveLeftReleased, ref bool moveDownReleased,
        ref bool moveUpHeld, ref bool moveRightHeld, ref bool moveLeftHeld, ref bool moveDownHeld)
    {
        bool pressed = firstTick;
        firstTick = false;
        if (dy < 0) { moveUpHeld = true; moveUpPressed |= pressed; }
        if (dy > 0) { moveDownHeld = true; moveDownPressed |= pressed; }
        if (dx < 0) { moveLeftHeld = true; moveLeftPressed |= pressed; }
        if (dx > 0) { moveRightHeld = true; moveRightPressed |= pressed; }
    }
}
