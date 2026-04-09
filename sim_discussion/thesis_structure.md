# Thesis Structure

**Working Title:** Comparing Reinforcement Learning Algorithms for HVAC Control Using the Google Smart Buildings Simulator

---

## 1. Introduction

- Context: energy consumption in buildings, HVAC as dominant contributor
- Problem: traditional rule-based HVAC controllers are rigid; optimal control is hard due to complex thermodynamics, variable occupancy, and fluctuating energy prices
- Opportunity: reinforcement learning as a data-driven approach to adaptive building control
- Research question: how do different continuous-control RL algorithms (SAC, TD3, DDPG) compare for HVAC optimization, and can they outperform a well-tuned rule-based baseline?
- Contribution overview and thesis outline

## 2. Background & Literature Review

### 2.1 Building Energy and HVAC Systems
- Building energy consumption and its environmental impact
- HVAC system components: air handling units, boilers, VAV boxes, reheat coils, dampers
- Two heating paths: air-side (supply air temperature + damper) vs water-side (boiler + reheat coil)
- Comfort standards and thermal comfort metrics (comfort bands, degree-hours)

### 2.2 Classical HVAC Control
- Rule-based control (thermostat hysteresis, PID)
- Model predictive control (MPC) and its limitations
- Limitations of traditional approaches: rigidity, need for system models, inability to adapt

### 2.3 Reinforcement Learning Fundamentals
- Markov decision processes: states, actions, rewards, transitions
- Policy gradient methods vs value-based methods
- Actor-critic architectures
- Continuous action spaces and their challenges

### 2.4 Deep RL Algorithms for Continuous Control
- DDPG: deterministic policy gradients with replay buffer and target networks
- TD3: twin critics, delayed policy updates, target policy smoothing
- SAC: maximum entropy RL, automatic temperature tuning, stochastic policies
- Comparison of theoretical properties (exploration, stability, sample efficiency)

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
- Occupancy models (deterministic step, constant, randomized)

### 3.2 Available Building Configurations
- Floorplan descriptions: single_room, office_4room, corporate_floor, headquarters_floor
- Zone connectivity and multi-zone thermodynamics
- Scalability characteristics (1 to 16 zones)

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
- Time step selection (600s) and its implications

### 4.2 Observation Space
- Design choices and rationale for each feature group:
  - Zone temperatures and comfort errors (normalized around comfort midpoint)
  - Previous actions (action echo for temporal context)
  - Temporal features (cyclic sin/cos encoding of hour and day-of-week, weekend flag)
  - Supply air and boiler setpoints
  - Ambient temperature and trend (rate of change)
  - Weather forecast (1h, 3h, 6h ahead) for anticipatory control
  - Real-time energy prices (Belpex electricity, ZTP gas)
- Observation normalization via VecNormalize (running statistics, clip at 10.0)

### 4.3 Action Space Design
- Normalized action range [-1, 1] and physical mappings
- Three action space designs and their trade-offs:
  - reheat_per_zone: shared damper, per-zone reheat (cold-climate focus)
  - damper_per_zone: shared reheat, per-zone damper
  - full_per_zone: full per-zone control (highest dimensionality)
- Action space dimensionality scaling with number of zones
- Physical constraints and clamping (e.g., boiler >= outdoor temp + 1K)

### 4.4 Reward Function
- Design philosophy: balancing thermal comfort, energy efficiency, and control smoothness
- Comfort penalty: quadratic violation from comfort band (zones averaged)
- Energy penalty: weighted sum of blower, air handler, boiler, and pump power
- Smoothness penalty: penalizing abrupt action changes to discourage oscillation
- Night setback: reducing comfort floor outside working hours (2-4K reduction)
- Center bonus: small incentive for temperatures near band midpoint during occupied hours
- Monetary cost mode: using real Belpex/ZTP prices instead of raw Watts
- Energy weight parameter as comfort-vs-efficiency trade-off knob

### 4.5 Real Energy Price Integration
- Belpex hourly electricity spot prices (Belgium market)
- EEX ZTP monthly gas prices
- Unit conversions and normalization to align with reward scale
- Rationale: enabling economically meaningful optimization

## 5. Experimental Setup

### 5.1 Algorithms and Hyperparameters
- SAC configuration: learning rate, replay buffer size, batch size, network architecture, entropy tuning
- TD3 configuration: twin critics, delayed updates, target policy noise
- DDPG configuration: as baseline continuous-control algorithm
- Common settings: MlpPolicy, VecNormalize, training frequency
- Justification of hyperparameter choices

### 5.2 Baseline Controller
- Thermostat hysteresis controller design
  - Working-hours logic: boiler on/off with 0.6K deadband around comfort midpoint
  - Anti-chatter: 4-step minimum dwell time
  - Per-zone reheat: proportional to zone coldness
  - Per-zone damper: open for cold zones, closed for warm
  - Off-hours: all actuators off
- Why this baseline is representative of conventional practice

### 5.3 Training Protocol
- Data splits: train (2019-10 to 2022-03), validation (2022-10 to 2023-03), test (2023-10 to 2024-03)
- Heating season focus and rationale
- Chunked training: random episode start times within training period
- Total training budget (1M-10M timesteps)
- Reward and observation normalization strategy
- Seed management and reproducibility

### 5.4 Evaluation Protocol
- Deterministic policy evaluation on validation/test periods
- Metrics collected per episode:
  - Cumulative reward
  - Comfort penalty and discomfort degree-hours
  - Percentage of steps outside comfort band and maximum temperature deviation
  - Total energy consumption and energy cost (USD)
- Head-to-head RL vs baseline comparison on identical episodes
- Statistical reporting: mean/std across multiple episode starts

### 5.5 Computational Infrastructure
- Hardware: HPC cluster (PBS), CPU-focused training
- Training time and resource requirements
- Experiment tracking via Weights & Biases

## 6. Results

### 6.1 Training Dynamics
- Learning curves per algorithm (episode reward over timesteps)
- Convergence speed comparison
- Training stability (reward variance, catastrophic forgetting)

### 6.2 Algorithm Comparison
- Validation-period performance: SAC vs TD3 vs DDPG
  - Thermal comfort metrics (discomfort degree-hours, % time outside band)
  - Energy efficiency metrics (total consumption, cost in USD)
  - Reward breakdown (comfort vs energy vs smoothness contributions)
- Statistical significance of differences
- Pareto analysis: comfort-energy trade-off across algorithms

### 6.3 RL vs Baseline Controller
- Head-to-head comparison per algorithm against thermostat baseline
- Energy savings (%) while maintaining or improving comfort
- Cost savings with real energy prices
- Behavioral analysis: how RL policies differ from rule-based control

### 6.4 Behavioral Analysis
- Episode trace analysis: temperature profiles, action patterns, energy consumption over time
- Anticipatory behavior: do agents pre-heat before occupancy or price spikes?
- Night setback exploitation: how agents handle unoccupied periods
- Multi-zone coordination: per-zone reheat/damper strategies
- Action smoothness: oscillation and bang-bang behavior

### 6.5 Sensitivity Analysis
- Effect of energy weight parameter on comfort-energy trade-off
- Action space design comparison (reheat_per_zone vs damper_per_zone vs full_per_zone)
- Building complexity: single room vs multi-zone performance scaling

## 7. Discussion

### 7.1 Interpretation of Results
- Why SAC/TD3/DDPG perform as they do (connecting to algorithmic properties)
- Role of entropy regularization (SAC) vs deterministic policies (TD3/DDPG)
- Sample efficiency vs final performance trade-offs

### 7.2 Design Decisions and Their Impact
- Reward function design: how shaping choices influenced learned behavior
- Observation space: which features matter most for control quality
- Action space: expressiveness vs learnability
- Comfort band width and night setback: effect on training signal

### 7.3 Practical Considerations
- Sim-to-real gap: what would change for real building deployment
- Computational cost of training vs potential energy savings
- Safety constraints and deployment guardrails
- Generalization across buildings, climates, and seasons

### 7.4 Limitations
- Single weather location (Brussels) and heating-season focus
- Simplified occupancy models
- No cooling-dominated scenarios
- Limited hyperparameter tuning budget
- Simulation fidelity vs real-world complexity

## 8. Conclusion

- Summary of key findings
- Recommendation: which algorithm(s) are most suitable for HVAC control and why
- Contributions: wrapper framework, simulator optimizations, systematic comparison
- Future work:
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
- Dependency versions (Poetry lockfile summary)
