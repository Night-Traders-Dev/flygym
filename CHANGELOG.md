# Changelog

All notable changes to our fork of [NeLy-EPFL/flygym](https://github.com/NeLy-EPFL/flygym) are documented here.

## [Unreleased]

### Added — Multi-Biome World System

- **`biome.py`** — Biome data model, world generator, and 3D environment objects
  - `BiomeParams` dataclass: name, ground colors, reflectance, wind vector, temperature, humidity, friction, food density, elevation parameters
  - 5 preset biomes tuned for Drosophila ecology:
    - `FOREST_FLOOR`: dark soil, leaves/twigs, 20C, 75% humidity, no wind, normal friction
    - `MEADOW`: green grass, seed heads, 26C, 40% humidity, light breeze, lower friction
    - `WETLAND`: dark mud, puddles/reeds, 18C, 95% humidity, still air, very low friction, high reflectance
    - `SANDY_ARID`: tan sand, pebbles/rocks, 34C, 12% humidity, strong wind, high friction, low food
    - `FRUIT_GARDEN`: rich soil, colorful fallen fruit, 24C, 55% humidity, high food density
  - `generate_biome_texture()`: 1024x1024 procedural textures using multi-octave value noise (no sine banding), with per-biome detail features (soil patches, water pools, wind ripples)
  - `_add_scatter_objects()`: generates real 3D MuJoCo geometry per biome zone (~935 objects across 25 zones):
    - Forest: ellipsoid leaves (5 color variants), capsule twigs, sphere pebbles
    - Meadow: thin capsule grass blades (30-60 per zone), sphere seed heads
    - Wetland: cylinder puddle discs with reflective rims, tall capsule reeds
    - Sandy: ellipsoid rocks, capsule dried sticks
    - Fruit garden: colorful sphere fruits (red/yellow/orange/green/plum), ellipsoid green leaves
  - `BiomeWorld(FlatGroundWorld)`: 2D grid of biome zones with 3D objects, per-zone textures/materials/friction, neutral dirt fill between zones
  - `get_biome_at(x, y)`: position-to-biome lookup for runtime effects

- **`biome_effects.py`** — Runtime environmental effects engine
  - `BiomeEffectsEngine`: applies position-dependent effects each timestep
  - Wind: per-biome force vectors applied via `mj_data.xfrc_applied` on fly root bodies
  - Temperature: modulates walking speed factor (Gaussian curve centered at 25C optimal)
  - Humidity: modulates tarsal adhesion gain (wet surfaces reduce grip)
  - Food density: biases food spawn probability toward richer biomes (fruit garden gets 2.5x food)
  - `get_biome_summary()`: one-line status string for terminal metrics

- **`fly_vitals.py`** — Fly biological needs simulation
  - `FlyVitals` class: tracks hunger (0-100), thirst (0-100), energy (0-100), health (0-100)
  - Hunger: drains at base rate, faster in heat and while walking, restored by eating food (+25)
  - Thirst: drains faster in hot/dry biomes, slower in humid biomes, slowly rehydrates in wetlands
  - Energy: drains while walking, drains faster in extreme temperatures, recovers while resting
  - Health: degrades when hunger/thirst/energy critically low, degrades in extreme heat/cold, slowly recovers when well-fed/hydrated
  - Death tracking: cause of death (starvation, dehydration, exhaustion, exposure)
  - `get_status_bar()`: visual bar chart display for terminal (HNG/THR/NRG/HP with block chars)
  - `get_oneliner()`: compact single-line summary
  - `VitalsManager`: manages vitals for all flies, bulk update per tick

- **`fly_autonomous.py`** updated to use biome + vitals systems
  - 5x5 biome grid (100x100mm world) with natural biome arrangement (sand→meadow→forest→garden→wetland diagonal)
  - Walking speed adapts to temperature (slower in hot/cold biomes)
  - Adhesion adapts to humidity (slippery in wetland)
  - Wind pushes flies in sandy/arid zones
  - Food spawns weighted by biome (most food in fruit garden, least in desert)
  - Terminal metrics now show per-fly: position, biome, hunger/thirst/energy/health bars, food eaten, distance traveled

### Upstream

- Rebased onto upstream `main` — flygym updated from **v1.2.1 to v2.0.0**
- Reinstalled package (`pip install -e .`) pulling in new dependencies: `dm_control 1.0.38`, `mujoco 3.6.0`, `jaxtyping`, `loguru`, `tabulate`, `wadler-lindig`
- The v2.0.0 API is a complete rewrite:
  - `Fly` moved from `flygym.Fly` to `flygym.compose.Fly` (now a composable MJCF builder, not a monolithic class)
  - `SingleFlySimulation` / `Simulation` replaced by `flygym.Simulation` wrapping `BaseWorld`
  - Arenas (`OdorArena`, `MixedTerrain`) replaced by `FlatGroundWorld` / `TetheredWorld`
  - `HybridTurningController`, `CPGNetwork`, `PreprogrammedSteps` removed (no more `flygym.examples.locomotion`)
  - Walking kinematics now provided via `flygym_demo.spotlight_data.MotionSnippet` (real Spotlight motion capture recordings)
  - Rendering decoupled: `flygym.rendering.Renderer`, `launch_interactive_viewer`, `preview_model`
  - Per-fly observation getters: `sim.get_body_positions(fly_name)`, `sim.get_joint_angles(fly_name)`, `sim.get_ground_contact_info(fly_name)`, etc.

### Added

- **`fly_autonomous.py`** — Live multi-fly autonomous simulation viewer
  - 3 flies walking on a procedural forest floor environment
  - Uses real recorded walking kinematics from Spotlight motion capture (`MotionSnippet`), looped continuously for indefinite runtime
  - `WalkingController` class: wraps `MotionSnippet.get_joint_angles()` with automatic looping and per-timestep turning modulation via coxa-pitch amplitude bias
  - `ForestFloorWorld` class: custom `FlatGroundWorld` subclass with:
    - Procedural forest floor texture (`_generate_forest_floor_texture`): earthy browns, dark soil patches, leaf splotches (5 color variants), pebble specks, twig lines
    - Gradient skybox (canopy green to pale sky)
    - Warm directional sun light + canopy fill light
    - Atmospheric fog (green-tinted, fogstart=20mm, fogend=80mm)
    - Pre-allocated mocap food bodies (up to 12 slots) for dynamic spawning/despawning
    - Multi-fly safe `_set_ground_contact` and `_add_ground_contact_sensors` overrides
  - `FoodManager` class: manages dynamic food spawning
    - Red sphere markers at random positions within configurable range
    - Auto-despawn when any fly walks within 1.5mm
    - Maintains 4-6 active food sources, spawning replacements as needed
  - Food-seeking behavior: flies steer toward nearest food via cross-product heading error
  - Live terminal metrics every 2s: position, legs on ground, ground reaction force, distance to nearest food
  - MuJoCo `launch_passive` interactive viewer with full UI panels
  - Configurable `realtime_factor` (default 5x) for visible locomotion at fruit-fly scale
  - Actuator gain `kp=150.0` (matching official tutorial recommendations)

- **`audit_sim.py`** — Headless simulation auditing tool
  - Runs multi-fly simulation offscreen via EGL backend
  - Saves PNG screenshots at configurable time intervals using `mujoco.Renderer`
  - Auto-centering camera tracks mean fly position
  - Prints per-fly metrics (position, leg contact count) at each snapshot
  - Used for automated visual verification without a display server

- **`CHANGELOG.md`** — This file

### Fixed (upstream v2.0.0 bugs)

- **Multi-fly spawn position bug** — `BaseWorld._rebuild_neutral_keyframe` used `jnt_dofadr` (velocity-state address) instead of `jnt_qposadr` (position-state address) when writing freejoint neutral positions into the keyframe. Freejoints have 7 qpos elements (xyz + quaternion) at `jnt_qposadr` but only 6 dof elements at `jnt_dofadr`. Result: all flies spawned at origin `(0,0,0)`, overlapped, and were catapulted by collision forces. Fixed via runtime monkey-patch using `jnt_qposadr`.

- **Multi-fly contact pair name collision** — `FlatGroundWorld._set_ground_contact` creates contact pairs named `{body_segment}-ground` (e.g., `c_thorax-ground`). When multiple flies are added, the second fly's body segments produce duplicate MJCF identifiers, raising `ValueError: Duplicated identifier`. Fixed by prefixing all contact pair names with `{fly.name}_`.

- **Multi-fly contact sensor name collision** — Same issue in `_add_ground_contact_sensors`: sensor names like `ground_contact_lf_leg` collide across flies. Fixed by prefixing with `{fly.name}_`.

- **Multi-fly neutral keyframe conflict assertion** — `_rebuild_neutral_keyframe` is called after each `add_fly()`. On the second call, it rebuilds from scratch (zeros) then iterates all flies, but the conflict check `np.any(~np.isclose(neutral_qpos[indices_to_fill], 0))` fails because the world's freejoint positions (set in step 1) are non-zero at addresses that don't actually overlap with fly joint addresses — the check is overly broad. Fixed by removing the conflict assertion in the monkey-patch.

### Changed (pre-existing custom scripts)

- **`fly_live.py`** — Pre-existing v1.x live viewer script (GLFW + kinematic replay). Not updated for v2 API; kept for reference.
- **`fly.py`** — Pre-existing v1.x headless kinematic replay (OSMesa). Not updated for v2 API.
- **`fly_plume.py`** — Pre-existing v1.x odor plume navigation controller. Not updated for v2 API.
- **`controller.py`** — Pre-existing v1.x `SimplePlumeNavigationController`. Not updated for v2 API.
- **`first_nmf_run.py`** — Pre-existing v1.x quick test. Not updated for v2 API.

### Environment / Platform Setup

- **Wayland/GLFW crash fix** — `libdecor-gtk.so` plugin segfaults when GLFW initializes on Wayland. Fixed two ways:
  1. Installed `libdecor-0-plugin-1-cairo` (`apt install libdecor-0-plugin-1-cairo`) as a working alternative decoration plugin
  2. Set `LIBDECOR_PLUGIN_DIR=/tmp/libdecor_cairo_only` pointing to a directory containing only the cairo plugin symlink, bypassing the broken GTK plugin
  3. Unset `WAYLAND_DISPLAY` at script startup to force GLFW onto XWayland
- **Window decorations** — The empty `LIBDECOR_PLUGIN_DIR` trick (used initially) caused windows to render without title bars, preventing maximize/minimize/close. Switching to the cairo plugin restored proper client-side decorations.
- **`networkx`** — Installed via `pip install networkx` (required by `flygym.examples.locomotion.rule_based_controller` in v1.x examples, pulled transitively)
- **`xdotool`** — Installed via `apt install xdotool` (used during debugging for window management)

### Development iterations

This simulation went through several major iterations to reach a working state:

1. **v1.x CPG attempt** — Initial script used `HybridTurningController` + `OdorArena` from v1.x API. Worked but v1.x was superseded by upstream v2.0.0 update.
2. **v2.0 custom CPG** — Rewrote for v2 API with a hand-tuned `SimpleCPG` class. Flies were stable but barely moved — the CPG was driving the wrong DOF indices (coxa-yaw instead of coxa-pitch) and using zero-based offsets instead of neutral-pose-based targets.
3. **DOF mapping fix** — Identified correct per-leg DOF layout: `[0]=yaw, [1]=pitch, [2]=roll, [3]=femur-pitch, [4]=femur-roll, [5]=tibia-pitch, [6]=tarsus1-pitch`. Fixed CPG to drive index 1 (pitch) for forward swing.
4. **Neutral ctrl offset fix** — Discovered `set_actuator_inputs` takes absolute target angles, not offsets from zero. Added neutral ctrl retrieval via `sim._intern_actuatorids_by_type_by_fly` and added CPG offsets to neutral values.
5. **Spawn position fix** — Found `jnt_dofadr` vs `jnt_qposadr` bug causing all flies to spawn at origin. This was the root cause of the "flies die immediately" behavior.
6. **Real kinematics** — Replaced custom CPG entirely with `MotionSnippet` recorded data (660 frames at 330Hz from real fly Spotlight capture). This produces natural, visible leg articulation matching real Drosophila walking.
