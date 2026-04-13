# Thesis Structure

**Working Title:** Reinforcement Learning for Home Energy Management:
Optimization and Evaluation in the (HomeLab)

---

## 1. Introduction

- Context: energy consumption in buildings, HVAC as dominant contributor
- Problem: traditional rule-based HVAC controllers are rigid; optimal control is hard due to complex thermodynamics, and fluctuating energy prices
- Opportunity: reinforcement learning as a data-driven approach to adaptive building control
- Research questions:
  - **RQ1:** How do off-policy deep RL algorithms (SAC and TD3) compare in optimizing the trade-off between energy cost and occupant thermal comfort in a simulated office building, and can they outperform a rule-based thermostat baseline?
  - **RQ2:** How does the inclusion of real-time Belgian electricity and gas spot prices in the observation and reward signal affect the learned control policy, and does the agent exhibit price-aware demand-response behaviour?
- Contribution overview and thesis outline

> **[DRAFT NOTE]** Should we add DDPG as third algorithm (time issue) historical baseline that TD3 directly improves upon


## 2. Background & Literature Review

### 2.1 Building Energy and HVAC Systems
- Building energy consumption and its environmental impact
- HVAC system components: air handling units, boilers, VAV boxes, reheat coils, dampers
- Two heating paths: air-side (supply air temperature + damper) vs water-side (boiler + reheat coil)
- Comfort standards and thermal comfort metrics (comfort bands, degree-hours, percentage outside band, wordt case violation)

### 2.2 Classical HVAC Control
- Rule-based control (thermostat hysteresis)
- Limitations of traditional approaches: rigidity, need for system models, inability to adapt
> **[DRAFT NOTE]** Should I do more research into making the baseline rule based better. For better comparrison? e.g. PID controller

### 2.3 Reinforcement Learning Fundamentals
- Markov decision processes: states, actions, rewards, transitions
- Policy gradient methods vs value-based methods
- Actor-critic architectures
- Continuous action spaces and their challenges
> **[DRAFT NOTE]** How much of explaining is expected here?

### 2.4 Deep RL Algorithms for Continuous Control
- TD3: twin critics, delayed policy updates, target policy smoothing — fixing DDPG's overestimation bias
- SAC: maximum entropy RL, automatic temperature tuning, stochastic policies
- Comparison of theoretical properties (exploration, stability, sample efficiency)

> **[DRAFT NOTE — DDPG]** If DDPG is included: add a section on DDPG as the predecessor to TD3, motivating why TD3 was developed.

> **[DRAFT NOTE — Future algorithms]** TQC (Truncated Quantile Critics) extends SAC with distributional critics and truncation to reduce overestimation bias (Kuznetsov et al., "Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics", ICML 2020, arXiv:2005.04269). Available in sb3-contrib. Could be added as a "potential upgrade" in the discussion/future work section without requiring additional experiments.

### 2.5 RL for Building Control
- Survey of prior work applying RL to HVAC
- Simulation-based approaches vs real-building deployment
- Google SBSim and its position in the landscape
- Gaps in existing comparisons (limited algorithms, simplified environments, lack of real energy pricing)

## 3. The Google Sustainable Buildings Simulator (SBSim)

### 3.1 Simulator Architecture
- Finite difference method (FDM) for thermal simulation
- Building representation: cells, materials, zones
- Weather models and data sources
- Fixed working-hours occupancy schedule (comfort band active during office hours only)

> **[DRAFT NOTE — Occupancy]** Currently using a fixed schedule (office hours). SBSim supports randomized occupancy, should we add that?

### 3.2 Available Building Configurations
- Floorplan descriptions: small office (4 zones, 72 m²) and large office floor (9 zones, 450 m²)
- Zone connectivity and multi-zone thermodynamics
- Scalability characteristics

> **[DRAFT NOTE]** If time permits, the headquarters_floor (largest configuration) could be tested as an upper bound on complexity.

### 3.3 Simulator Performance and Optimization
- Bottleneck analysis: FDM solver as computational hotspot
- Numba JIT-compiled Gauss-Seidel solver (FastCPUSimulator): design and speedup (15-30x)
- Safe boiler wrapper: preventing unphysical states during RL exploration
- Fast weather replay controller
- Benchmarking results: steps/second across building sizes

## 4. Environment Design

### 4.1 Gymnasium Wrapper (BuildingGymEnv)
- Bridging SBSim's API to the Gymnasium interface
- Episode structure: 7-day episodes with configurable start times
- Time step: 600s (10 minutes) and its implications

### 4.2 Observation Space
- Design choices and rationale for each feature group (30 features total for 4-zone building):
  - Zone temperatures and comfort errors (normalized around comfort midpoint)
  - Previous actions (action echo for temporal context)
  - Temporal features (cyclic sin/cos encoding of hour and day-of-week, weekend flag)
  - Supply air and boiler setpoints
  - Ambient temperature and trend (rate of change)
  - Weather forecast (1h, 3h, 6h ahead)
  - Real-time energy prices (Belpex electricity, ZTP gas) — normalized to [-1, 1]
- Observation normalization via VecNormalize (running statistics, clip at 10.0)

### 4.3 Action Space Design
- Normalized action range [-1, 1] and physical mappings
- Three action space designs and their trade-offs:
  - reheat_per_zone: shared damper, per-zone reheat (cold-climate focus)
  - damper_per_zone: shared reheat, per-zone damper
  - full_per_zone: full per-zone control (highest dimensionality)
- Action space dimensionality scaling with number of zones
- Physical constraints and clamping (e.g., boiler >= outdoor temp + 1K)
- Action space comparison conducted on small office (4-zone) building

### 4.4 Reward Function
- Design philosophy: balancing thermal comfort, energy efficiency, and control smoothness
- Comfort penalty: linear violation from comfort band, mean over zones — zero inside band, linear outside
- Energy penalty: normalized by physics-based peak estimate; active during working hours only
- Smoothness penalty: penalizing abrupt action changes to discourage bang-bang oscillation
- Night setback: no comfort penalty outside working hours — agent free to let building cool
- Energy weight parameter as comfort-vs-efficiency trade-off knob

### 4.5 Energy Weight Tuning and Signal Balance
- The energy weight controls the relative importance of energy vs comfort in the reward signal
- Challenge: at low weights (ew=1) the agent learns excellent comfort but ignores energy cost; at high weights (ew≥4) the energy signal dominates and destroys the comfort policy
- Root cause: when inside the comfort band, comfort penalty is zero — energy is the only gradient signal. When outside the band, comfort penalty is large. This creates a non-stationary gradient landscape as the policy improves.
- Empirical finding: with a well-trained comfort policy, energy/comfort ratio at ew=1 is already ~20x in favor of energy (because comfort ≈ 0). Higher weights amplify this imbalance further.
- Curriculum learning attempt: ramping energy weight from 1→15 over training — policy learned comfort in phase 1 but collapsed when energy weight increased
- Final design: constant energy weight with smoothness penalty (0.5) to prevent bang-bang heating, which naturally reduces energy use without sacrificing comfort
- Key insight: a policy that learns smooth proportional control is inherently more energy-efficient than one that learns on/off heating

### 4.5 Real Energy Price Integration
- Belpex hourly electricity spot prices (Belgian market, EUR/MWh → USD/Ws)
- EEX ZTP monthly gas prices (EUR/MWh → USD/1000 ft³, using sbsim energy content constant)
- Boiler thermal efficiency correction (0.88, condensing gas boiler per EN 15316 / JRC BAT reference)
- Prices included in observation space enabling price-aware control
- Rationale: This allows the agent to optimize for actual energy cost and learn to shift heating load away from expensive hours.

## 5. Experimental Setup

### 5.1 Algorithms and Hyperparameters
- SAC configuration: learning rate, replay buffer size, batch size, network architecture, entropy tuning
- TD3 configuration: twin critics, delayed updates, target policy noise
- Common settings: MlpPolicy, VecNormalize, training frequency
- Justification of hyperparameter choices

> **[DRAFT NOTE — DDPG]** If included: add DDPG configuration here as well.

### 5.2 Baseline Controller
- Thermostat hysteresis controller design
  - Working-hours logic: boiler on/off with 0.6K deadband around comfort midpoint
  - Anti-chatter: 4-step minimum dwell time
  - Per-zone reheat: proportional to zone coldness
  - Per-zone damper: open for cold zones, closed for warm
  - Off-hours: all actuators off
- Why this baseline is representative of conventional practice

 > **[DRAFT NOTE]** Should the baseline be better?

### 5.3 Training Protocol
- Data splits: train (Oct 2019 – Mar 2023), test (Oct 2023 – Mar 2024)
- Heating season focus and rationale (Belgian climate, winter-dominant energy use)
- Chunked training: random episode start times sampled within training period
- Total training budget: ~5M timesteps (~15h overnight run), We are at 1M now.
- Reward and observation normalization strategy
- Seed management and reproducibility

> **[DRAFT NOTE — Data split]** The validation period (Oct 2022 – Mar 2023) was merged into the training set since no hyperparameter tuning is performed on it. If hyperparameter tuning is added later, a separate validation split should be reintroduced.

> **[DRAFT NOTE — Data expansion]** If results show poor generalization, extending training data beyond 2023 (adding 2023-2024 winter to training, shifting test to 2024-2025) is an option. Requires extending Belpex and ZTP price data.

### 5.4 Evaluation Protocol
- Deterministic policy evaluation on test period only
- Metrics collected per episode:
  - Cumulative reward
  - Comfort penalty and discomfort degree-hours
  - Percentage of steps outside comfort band and maximum temperature deviation
  - Total energy consumption and energy cost (USD)
- Head-to-head RL vs baseline comparison on identical episodes
- Statistical reporting: mean/std across multiple episode starts (≥30 episodes)

### 5.5 Computational Infrastructure
- Hardware: HPC cluster (Slurm/donphan), GPU-accelerated training (NVIDIA A2, 16GB)
- Training time and resource requirements
- Experiment tracking via Weights & Biases

## 6. Results

### 6.1 Training Dynamics
- Learning curves per algorithm (episode reward over timesteps)
- Convergence speed comparison
- Training stability (reward variance)

### 6.2 Algorithm Comparison (SAC vs TD3)
- Test-period performance
  - Thermal comfort metrics (discomfort degree-hours, % time outside band, max deviation)
  - Energy efficiency metrics (total consumption, cost in USD)
  - Reward breakdown (comfort vs energy vs smoothness contributions)
- Cost vs comfort scatter plot: showing which algorithm achieves the best trade-off between energy savings and thermal comfort

### 6.3 RL vs Baseline Controller
- Head-to-head comparison per algorithm against thermostat baseline
- Energy savings (%) while maintaining or improving comfort
- Cost savings with real energy prices
- Behavioral analysis: how RL policies differ from rule-based control

### 6.4 Behavioral Analysis
- Episode trace analysis: temperature profiles, action patterns, energy consumption over time
- Proactive behavior: do agents pre-heat before working hours or price spikes?
- Night setback exploitation: how agents handle unoccupied periods
- Multi-zone coordination: per-zone reheat/damper strategies

### 6.5 Sensitivity Analysis
- Effect of energy weight parameter on comfort-energy trade-off:
  - ew=1: excellent comfort, energy use higher than baseline (bang-bang heating)
  - ew=4: comfort degrades, energy marginally better
  - ew≥8: both comfort and energy worsen — gradient signal too noisy for SAC to learn
  - Smoothness penalty as an alternative lever: reduces bang-bang without destabilizing training
- Action space design comparison (reheat_per_zone vs damper_per_zone vs full_per_zone) on 4-zone building
- Building complexity: 4-zone small office vs 9-zone large office floor (450 m²)

## 7. Discussion

### 7.1 Interpretation of Results
- Why SAC/TD3 perform as they do (connecting to algorithmic properties)
- Role of the maximum entropy objective (SAC) vs deterministic policies (TD3) — SAC: Haarnoja et al., ICML 2018 (arXiv:1801.01290, Section 3.2); TD3: Fujimoto et al., ICML 2018 (arXiv:1802.09477, Sections 4.1–4.2)
- Sample efficiency vs final performance trade-offs

### 7.2 Design Decisions and Their Impact
- Reward function design: how shaping choices influenced learned behavior
- Observation space: which features matter most (prices, forecast, action echo)
- Action space: expressiveness vs learnability
- Comfort band width and night setback: effect on training signal

### 7.3 Practical Considerations
- Sim-to-real gap: what would change for real building deployment
- Computational cost of training vs potential energy savings
- Safety constraints and deployment guardrails
- Generalization across buildings, climates, and seasons

> **[DRAFT NOTE]** The points above are not studied in this thesis. Should this section be kept as a theoretical discussion, or into limitations/future work?

### 7.4 Limitations
- Single weather location (Brussels) and heating-season focus
- Fixed deterministic occupancy schedule (no stochastic arrival/departure)
- No cooling-dominated scenarios
- Limited hyperparameter tuning: defaults from literature used, no systematic grid search or Bayesian optimization
- Simulation fidelity vs real-world complexity

> **[DRAFT NOTE]** Should a basic hyperparameter search be added (e.g. learning rate, network size)? Requires significant extra compute.

## 8. Conclusion

- Summary of key findings
- Recommendation: which algorithm(s) are most suitable for HVAC control and why
- Contributions: wrapper framework, simulator optimizations, real Belgian energy price integration, systematic comparison
- Future work:
  - TQC or other improved off-policy algorithms
  - Transfer learning across buildings
  - Multi-objective optimization (comfort, energy, cost simultaneously)
  - Real-building validation
  - Cooling season and mixed climate evaluation
  - Multi-agent approaches for large buildings

---

## Appendices

### A. Simulator Technical Details
- FDM discretization and convergence parameters
- Material properties and building construction parameters
- Weather data preprocessing

### B. Full Hyperparameter Tables
- Complete configuration for each algorithm and experiment
- Network architecture details

### C. Additional Results
- Extended episode traces
- Per-zone temperature analysis for multi-zone buildings
- Full parameter sensitivity sweep results

### D. Code and Reproducibility
- Repository structure overview
- Instructions for reproducing experiments
- Dependency versions
