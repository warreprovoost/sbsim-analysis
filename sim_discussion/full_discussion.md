# RL HVAC Simulator Overview

## 1. What this simulator is

This project trains an RL controller for a **simplified HVAC building simulator**.

### High-level structure
- **Building**: thermal model of the room
- **Weather controller**: replayed outdoor weather from CSV
- **Air handler (AHU)**: sets central supply-air temperature
- **Boiler**: provides hot water for reheat
- **VAV terminal(s)**: apply damper + reheat locally
- **Gym wrapper**: exposes the simulator as an RL environment

---

## 2. Important simplification: single-room setup

Although the code is written to support multiple zones, the current factory builds a **single room**.

### In `building_factory.py`
- A floorplan is created with:
  - `room_width = 10`
  - `room_height = 10`

### Practical meaning
The current experiments are effectively:
- **1 room**
- **1 VAV**
- **1 average room temperature**

So the setup is a **single-zone HVAC control problem**.

---

## 3. The two heating systems

The simulator has **two ways to heat the room**.

## 3.1 Air handler path: central air heating + damper airflow

The air handler controls the **temperature of the supply air** sent into the duct system.

### RL-controlled AHU quantities
- supply air heating setpoint
- supply air cooling setpoint
  - these are set around one shared target supply temperature
- shared damper command

### How it works
1. The RL agent chooses a **supply air temperature**
2. The RL agent chooses a **damper opening**
3. More open damper = more airflow into the room
4. Warm supply air + airflow heats the room

### Interpretation
This is the **central heating path**:
- **AHU temperature**
- **damper / airflow**

---

## 3.2 Boiler path: hot-water reheat at the VAV

The boiler provides hot water to the VAV reheat coil.

### RL-controlled boiler/VAV quantities
- boiler supply water setpoint
- reheat valve command per zone

### How it works
1. The RL agent chooses a **boiler water temperature**
2. The RL agent chooses a **reheat valve opening**
3. The VAV reheat coil uses hot water to further warm the supply air locally

### Interpretation
This is the **local terminal heating path**:
- **boiler hot water**
- **reheat valve**

---

## 4. HVAC control picture

## Current control concept
There are two heating mechanisms:

### A. Central air-side heating
- **AHU supply air setpoint**
- **shared damper**

### B. Local water-side heating
- **boiler setpoint**
- **reheat valve**

### In the current single-room setup
This becomes:

- one room temperature
- one damper
- one reheat valve
- one shared AHU
- one boiler

---

## 5. Action space

In `gym_wrapper.py`, the action space is:

- `[0]` supply air temperature action
- `[1]` boiler setpoint action
- `[2]` shared damper action
- `[3..3+n_zones]` per-zone reheat valve actions

### Size
Action space shape:
- `3 + n_zones`

## Possible simplification:
1 shared reheat valve

### Size
Constant action space shape:
- `4`

### Current single-room case
Since `n_zones = 1`, the current action vector is:

- `action[0]` = AHU supply temperature
- `action[1]` = boiler setpoint
- `action[2]` = damper
- `action[3]` = reheat valve

So the current policy outputs **4 actions**.

---

## 6. Action scaling inside the environment

The policy outputs actions in `[-1, 1]`.

These are mapped to physical controls as follows.

### Supply air target

Approximately:
- `[-1, 1] -> [12°C, 32°C]`

The wrapper then sets:
- heating setpoint = supply target - 0.5°C
- cooling setpoint = supply target + 0.5°C

---

### Boiler setpoint

Approximately:
- `[-1, 1] -> [21°C, 66°C]`

It is also clamped to be at least slightly above outdoor temperature.

---

## 8. Observation space

The observation is defined in `gym_wrapper.py`.

## Current observation contents
The observation includes:

### Per zone
- zone temperature in °C
- occupancy estimate
- last reheat command

### Time features
- `sin(hour_of_day)`
- `cos(hour_of_day)`
- `sin(day_of_week)`
- `cos(day_of_week)`
- weekend flag

### Global HVAC/weather features
- current AHU supply air setpoint in °C
- current boiler setpoint in °C
- current outside air temperature in °C
- last shared damper command

---

## 9. Observation size

Observation shape:
```text
3 * n_zones + 9
```

### Current single-room case
With `n_zones = 1`:

```text
3 * 1 + 9 = 12
```

So the current observation vector has **12 values**.

---

## 10. Reward function

The reward is defined in `gym_wrapper.py`.

## 10.1 Comfort penalty
The main comfort term is based on room temperature violation outside the comfort band.

### Comfort band
Default:
- `21°C to 22°C`

### Zone penalty
For each zone:
```text
violation = max(low - T, 0) + max(T - high, 0)
```

This means:
- no penalty inside the band
- penalty grows when temperature goes below or above the band

### Occupancy weighting
The penalty is weighted by occupancy:

```text
weight = sqrt(occupancy)
```

Then a weighted average is taken across zones.

### Meaning
Prioritize comfort where people are present.

---

## 10.2 Energy term
The environment computes:

```text
energy_rate = blower_rate + ah_conditioning_rate + boiler_gas_rate
```

### Included in reward
- blower electrical power
- air-handler conditioning electrical power
- boiler gas heating power

---

## 10.3 Total reward
The final scalar reward is:

```text
reward = -(comfort_penalty + energy_weight * energy_rate)
```

### Energy weight
- `energy_weight = 1e-4` for `"full"` mode
- `energy_weight = 0.0` for `"step"` occupancy mode

### Meaning
- higher comfort penalty -> worse reward
- higher energy use -> worse reward
- RL tries to maximize reward, so it tries to reduce both

---

## 11. Occupancy modes

Two occupancy modes are used.

### `comfort_only`
- uses deterministic step occupancy
- mainly focuses on comfort
- energy term is effectively disabled

### `full`
- uses randomized occupancy
- comfort + energy tradeoff
- this is the more realistic mode

---

## 12. Weather

The building uses **replayed weather from CSV based in LA**.

### In `building_factory.py`
- `weather_source = "replay"`
- `ReplayWeatherController` loads the weather file

### Meaning
Outdoor temperature is not synthetic during the main experiments.
It is taken from recorded weather data.

---

## 13. Training setup used for the full SAC run

From `train_rl.py`, the preset for the full run is:

### `full`
- `total_timesteps = 100_000`
- `chunk_timesteps = 5_000`
- `episode_days = 7`
- `n_eval_episodes = 5`
- `training_mode = "full"`
- `eval_training_mode = "full"`

### Command
```bash
python scripts/train_rl.py --mode full --algo sac --seed 42 --unique_run
```

---

## 14. What `train_rl.py` does

The training script:

1. selects a preset
2. loads the replay weather CSV
3. creates an output directory
4. calls `run_rl_2024_setup(...)`
5. trains the model
6. evaluates on validation/test periods
7. compares RL against a thermostat baseline
8. saves:
   - model
   - CSV summaries
   - episode traces
   - plots

---

## 15. What `run_rl_2024_setup(...)` does

In `rl_trainer.py`, training is done in chunks.

### Time splits
- **train**: `2024-01-16` to `2024-08-01`
- **val**: `2024-08-01` to `2024-10-01`
- **test**: `2024-10-01` to `2024-12-01`

### Episode length
- 7 days per episode

### Training loop
- sample a random start time within the train period
- create environment
- train SAC for one chunk
- repeat until total timesteps are reached

### Evaluation
After training:
- evaluate on random starts in validation period
- evaluate on random starts in test period
- optionally compare against thermostat baseline

---

## 16. What the SAC agent actually controls in the current experiment

Because the factory creates a single room, the agent currently controls:

- AHU supply temperature
- boiler water temperature
- damper opening
- one reheat valve

and observes:

- room temperature
- occupancy
- previous reheat command
- time features
- current AHU setpoint
- current boiler setpoint
- outdoor temperature
- previous damper command

---

## 18. Possible future improvements

- make reheat a **shared action**
- include more physical state in observations
- add action smoothing / rate penalties
- constrain unrealistic boiler spikes
- scale from single-room to multi-zone experiments
