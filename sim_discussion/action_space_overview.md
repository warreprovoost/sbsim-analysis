# Action Space Design for Multi-Zone HVAC RL

## System overview
- 1 global AirHandler (supply_air_setpoint)
- 1 global Boiler (boiler_setpoint)
- N zones, each with: damper + reheat valve
- Total raw action space: `2 + 2*N`

---

## Why reducing the action space matters
- Global actions (boiler, supply_air) interact with **all zones simultaneously**.
- Per-zone damper + reheat are **partially redundant** (both heat the zone).
- 22 continuous actions for 10 zones is very hard for standard RL (PPO/SAC) to explore efficiently.

---

## Recommended progression

### Stage 1: Fix globals, learn reheat only ✅ (start here)
```
fixed : supply_air = 0.0  (22°C)
fixed : boiler     = 1.0  (65°C, always hot water available)
fixed : dampers    = 1.0  (always open)
learn : 10x reheat valve  → 10 actions
```
**Pros**
- Smallest action space, easiest to train.
- Reheat alone can control per-zone comfort well.
- Clean baseline to compare against later stages.

**Cons**
- No AH or damper optimization.
- Boiler always running at max (not energy optimal).

---

### Stage 2: Add damper control ✅ (intermediate)
```
fixed : supply_air = 0.0  (22°C)
fixed : boiler     = 1.0  (65°C)
learn : 10x damper + 10x reheat → 20 actions
```
**Pros**
- Agent can trade off airflow vs reheat per zone.
- More energy efficient than Stage 1.

**Cons**
- 20 actions is harder to train.
- Damper and reheat still partially redundant per zone.

---

### Stage 3: Merged per-zone action (most elegant) ✅ (final)
Combine damper and reheat into a **single heating demand signal** per zone:

```
action ∈ [-1, 1] per zone
  negative → cooling demand → open damper, close reheat
  positive → heating demand → close damper, open reheat

fixed : supply_air, boiler
learn : 10 actions (one per zone)
```
**Pros**
- Only 10 actions for 10 zones.
- Mimics real thermostat control.
- Physically interpretable.
- Much easier to train than Stage 2.

**Cons**
- Cannot simultaneously use damper + reheat (by design).
- Requires a wrapper to split single action into damper/reheat commands.

---

## Summary table

| Stage | Actions | Globals fixed | Description | Difficulty |
|---|---|---|---|---|
| 1 | 10 | supply_air, boiler, damper | reheat only | ⭐ easy |
| 2 | 20 | supply_air, boiler | damper + reheat | ⭐⭐⭐ hard |
| 3 | 10 | supply_air, boiler | merged per-zone | ⭐⭐ medium |

---

## Todos
1. **Start with Stage 1** to get a working trained agent and baseline results.
2. **Move to Stage 3** to show a smarter action space design.
3. **Compare Stage 1 vs Stage 3** in terms of comfort, energy use and training stability.
4. Mention Stage 2 as a possible extension but note the dimensionality problem.
