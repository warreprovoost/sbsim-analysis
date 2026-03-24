## Heating Paths

There are two independent heating paths in the simulator, both controlled by different actions.

### Path 1: Air Handler (AH) — supply air heating
```
outdoor_air + recirculated_air
        ↓
   mixed_air_temp
        ↓
   AirHandler heats/cools to supply_air_setpoint
        ↓
   VAV damper controls airflow into zone
        ↓
   zone temperature changes
```
**Controlled by**: actions `[0]` (supply_air_setpoint) and `[2]` (damper).
**Energy cost**: `ah_conditioning_rate` + `blower_rate`.

---

### Path 2: VAV Reheat Coil — hot water reheating
```
Boiler heats water to reheat_water_setpoint
        ↓
   VAV.output() uses boiler.reheat_water_setpoint
        ↓
   compute_reheat_energy_rate(supply_air_temp, boiler.reheat_water_setpoint)
        ↓
   heat delivered to zone air
```
**Controlled by**: actions `[1]` (boiler_setpoint) and `[3]` (reheat valve).
**Energy cost**: `boiler_gas_rate` / `reheat_coil_rate`.

---

### Key interaction
Both paths heat the same zone, which is why they interfere:

| Situation | Effect |
|---|---|
| High AH supply + high reheat | Zone overheats rapidly |
| High AH supply + low reheat | AH does all the work |
| Low AH supply + high reheat | Reheat compensates cold supply air |
| Low AH supply + low reheat | Zone cools toward outdoor temp |

---

### Default configuration (`get_base_params`)
- `recirculation=0.8` → 80% recirculated air, AH supply temp stays close to zone temp.
- `vav_reheat_flow=0.4` → relatively strong reheat coil.
- Reheat dominates heating in the default config.
