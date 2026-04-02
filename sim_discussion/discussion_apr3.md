# Discussion — Week of 31 March – 3 April 2026

## Features added


### Weather data
- Cold Weather replay (Oslo), can be everywhere
- Only trained on winter, training/val set 2019 - 22; test set 2023

### Rewards
- Positive comfort inside middle of the zone (during the day)
- Pricing schemes added. Took a high gas price year.
- Energy weight explored with 4 runs.

### New plot
- Comfort meassure: °C * h / episode
- Box plots RL vs baseline on comofort and energy cost

### Floorplan
- New floorplans
- Multi zone support on plotting
- Multi zone training

### Action space design (`action_design`)

| Mode | Actions | Best for |
|---|---|---|
| `reheat_per_zone` (default) | `[supply, boiler, shared_damper, reheat_0..n]` | Cold climate |
| `damper_per_zone` | `[supply, boiler, shared_reheat, damper_0..n]` | Warm climate |
| `full_per_zone` | `[supply, boiler, reheat_0..n, damper_0..n]` | Both |




## Notes

- Shared reheater overheats the building faster.
- Why are some models underperforming? Unlucky with test set they cannot handle?
- How can I handle the multi zone sensitivity better?

## Next objectives
- I let the test start in the night. Is this not beneficial
- Carbon cost is also a meassure availble in sbsim. Train on carbon cost/comfort. Something nice to add.
- Increase gas prices more to make it more relevant.
