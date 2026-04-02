"""
biome_effects.py — Runtime environmental effects for multi-biome world

Applies per-biome wind, temperature-based speed scaling, and humidity-based
adhesion modulation to flies based on their position in the world.
"""

import numpy as np
import mujoco as mj

from flygym.compose import ActuatorType


class BiomeEffectsEngine:
    """Applies position-dependent environmental effects each timestep."""

    def __init__(self, biome_world, sim):
        self.world = biome_world
        self.sim = sim

        # Cache fly root body IDs for xfrc_applied
        self._fly_root_bodyids = {}
        for fly_name, fly in biome_world.fly_lookup.items():
            root_body = fly.bodyseg_to_mjcfbody[fly.root_segment]
            self._fly_root_bodyids[fly_name] = mj.mj_name2id(
                sim.mj_model, mj.mjtObj.mjOBJ_BODY, root_body.full_identifier
            )

        # Cache for current biome per fly (avoid repeated lookups)
        self._current_biomes = {}

    def update_biomes(self, fly_positions: dict):
        """Update the current biome for each fly based on position."""
        for fly_name, pos in fly_positions.items():
            self._current_biomes[fly_name] = self.world.get_biome_at(pos[0], pos[1])

    def get_current_biome(self, fly_name: str):
        return self._current_biomes.get(fly_name)

    def apply_wind(self):
        """Apply per-biome wind force to each fly's root body."""
        for fly_name, biome in self._current_biomes.items():
            body_id = self._fly_root_bodyids[fly_name]
            # xfrc_applied is (nbody, 6): [fx, fy, fz, tx, ty, tz]
            self.sim.mj_data.xfrc_applied[body_id, :3] = biome.wind

    def clear_forces(self):
        """Zero out applied forces (call before apply_wind each step)."""
        for fly_name in self._fly_root_bodyids:
            body_id = self._fly_root_bodyids[fly_name]
            self.sim.mj_data.xfrc_applied[body_id, :] = 0.0

    def get_speed_factor(self, fly_name: str) -> float:
        """Temperature-based locomotion speed factor.

        Drosophila are most active at ~25C. Activity drops at temperature
        extremes. Returns a multiplier in [0.3, 1.2].
        """
        biome = self._current_biomes.get(fly_name)
        if biome is None:
            return 1.0
        temp = biome.temperature
        # Gaussian-ish curve centered at 25C
        return float(np.clip(1.2 - 0.003 * (temp - 25.0) ** 2, 0.3, 1.2))

    def get_adhesion_modifier(self, fly_name: str) -> float:
        """Humidity-based adhesion modifier.

        High humidity makes surfaces slippery for flies (wet tarsal pads).
        Returns a multiplier in [0.3, 1.0].
        """
        biome = self._current_biomes.get(fly_name)
        if biome is None:
            return 1.0
        return float(np.clip(1.0 - 0.6 * biome.humidity, 0.3, 1.0))

    def get_food_spawn_weight(self, x: float, y: float) -> float:
        """Get relative food spawn probability at world coordinates."""
        biome = self.world.get_biome_at(x, y)
        return biome.food_density

    def get_biome_summary(self, fly_name: str) -> str:
        """One-line summary of the fly's current biome conditions."""
        biome = self._current_biomes.get(fly_name)
        if biome is None:
            return "unknown"
        wind_speed = np.linalg.norm(biome.wind) * 1e6  # convert to readable
        return (
            f"{biome.name:12s} "
            f"T={biome.temperature:4.0f}C "
            f"H={biome.humidity:.0%} "
            f"W={wind_speed:3.0f}"
        )
