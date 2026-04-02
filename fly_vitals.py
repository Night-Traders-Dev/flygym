"""
fly_vitals.py — Fly vitals/needs simulation

Tracks hunger, thirst, energy, and health for each fly. These are affected
by biome conditions (temperature, humidity) and food consumption.
"""

import numpy as np


class FlyVitals:
    """Tracks biological needs for a single fly.

    All vitals are 0-100 scale:
      100 = fully satisfied / maximum
        0 = critical / depleted

    Rates are per-second of simulation time.
    """

    def __init__(self, name: str):
        self.name = name
        self.hunger = 80.0      # 100=full, 0=starving
        self.thirst = 85.0      # 100=hydrated, 0=dehydrated
        self.energy = 90.0      # 100=fully rested, 0=exhausted
        self.health = 100.0     # 100=perfect, 0=dead

        # Base drain rates (per second of sim time)
        self._hunger_drain = 0.8    # ~2 min to starve from full
        self._thirst_drain = 1.0    # ~1.5 min to dehydrate
        self._energy_drain = 0.3    # ~5 min to exhaust

        # Tracking
        self.food_eaten = 0
        self.distance_traveled = 0.0
        self._last_pos = None
        self.time_alive = 0.0
        self.cause_of_death = None

    @property
    def alive(self):
        return self.health > 0

    def update(self, dt: float, pos: np.ndarray, biome_temp: float,
               biome_humidity: float, ate_food: bool, is_walking: bool):
        """Update vitals for one tick.

        Args:
            dt: time step in seconds
            pos: current (x, y, z) position
            biome_temp: current biome temperature (C)
            biome_humidity: current biome humidity (0-1)
            ate_food: whether the fly ate food this tick
            is_walking: whether the fly is actively walking
        """
        if not self.alive:
            return

        self.time_alive += dt

        # Track distance
        if self._last_pos is not None:
            self.distance_traveled += np.linalg.norm(pos[:2] - self._last_pos[:2])
        self._last_pos = pos.copy()

        # --- Hunger ---
        hunger_rate = self._hunger_drain
        # Hot temperatures increase metabolism
        if biome_temp > 28:
            hunger_rate *= 1.0 + (biome_temp - 28) * 0.05
        # Walking burns more calories
        if is_walking:
            hunger_rate *= 1.3
        self.hunger -= hunger_rate * dt
        if ate_food:
            self.hunger = min(100, self.hunger + 25)  # food restores 25 hunger
            self.food_eaten += 1

        # --- Thirst ---
        thirst_rate = self._thirst_drain
        # Hot + dry = faster dehydration
        if biome_temp > 25:
            thirst_rate *= 1.0 + (biome_temp - 25) * 0.03
        if biome_humidity < 0.3:
            thirst_rate *= 1.5  # dry air dehydrates faster
        elif biome_humidity > 0.7:
            thirst_rate *= 0.6  # humid air helps
            # In wetland, slowly rehydrate
            if biome_humidity > 0.85:
                self.thirst = min(100, self.thirst + 0.5 * dt)
        self.thirst -= thirst_rate * dt

        # --- Energy ---
        energy_rate = self._energy_drain
        if is_walking:
            energy_rate *= 2.0
        # Extreme temperatures drain energy
        temp_stress = abs(biome_temp - 25) / 10
        energy_rate *= (1.0 + temp_stress * 0.3)
        self.energy -= energy_rate * dt
        # Resting (not walking) slowly recovers energy
        if not is_walking:
            self.energy = min(100, self.energy + 0.8 * dt)
        # Eating gives a small energy boost
        if ate_food:
            self.energy = min(100, self.energy + 10)

        # Clamp
        self.hunger = max(0, min(100, self.hunger))
        self.thirst = max(0, min(100, self.thirst))
        self.energy = max(0, min(100, self.energy))

        # --- Health ---
        # Health degrades when vitals are critically low
        damage = 0.0
        if self.hunger < 10:
            damage += (10 - self.hunger) * 0.15 * dt
        if self.thirst < 10:
            damage += (10 - self.thirst) * 0.2 * dt
        if self.energy < 5:
            damage += (5 - self.energy) * 0.1 * dt
        # Extreme heat/cold damages health directly
        if biome_temp > 38:
            damage += (biome_temp - 38) * 0.3 * dt
        if biome_temp < 5:
            damage += (5 - biome_temp) * 0.3 * dt

        # Health recovers slowly when well-fed and hydrated
        if self.hunger > 50 and self.thirst > 50 and self.energy > 20:
            self.health = min(100, self.health + 0.2 * dt)

        self.health -= damage
        self.health = max(0, min(100, self.health))

        if self.health <= 0:
            if self.hunger <= 0:
                self.cause_of_death = "starvation"
            elif self.thirst <= 0:
                self.cause_of_death = "dehydration"
            elif self.energy <= 0:
                self.cause_of_death = "exhaustion"
            else:
                self.cause_of_death = "exposure"

    def get_status_bar(self, width=20):
        """Return a compact multi-line status string."""
        def bar(val, w=width):
            filled = int(val / 100 * w)
            return f"[{'█' * filled}{'░' * (w - filled)}]"

        return (
            f"  HNG {bar(self.hunger)} {self.hunger:5.1f}%\n"
            f"  THR {bar(self.thirst)} {self.thirst:5.1f}%\n"
            f"  NRG {bar(self.energy)} {self.energy:5.1f}%\n"
            f"  HP  {bar(self.health)} {self.health:5.1f}%"
        )

    def get_oneliner(self):
        """Compact single-line vitals summary."""
        status = "ALIVE" if self.alive else f"DEAD({self.cause_of_death})"
        return (
            f"H:{self.hunger:4.0f} T:{self.thirst:4.0f} "
            f"E:{self.energy:4.0f} HP:{self.health:4.0f} "
            f"food={self.food_eaten} dist={self.distance_traveled:.1f}mm "
            f"[{status}]"
        )


class VitalsManager:
    """Manages vitals for all flies in the simulation."""

    def __init__(self, fly_names):
        self.vitals = {name: FlyVitals(name) for name in fly_names}

    def update_all(self, dt, fly_data: dict):
        """Update all fly vitals.

        fly_data: dict of fly_name -> {
            'pos': np.ndarray,
            'biome_temp': float,
            'biome_humidity': float,
            'ate_food': bool,
            'is_walking': bool,
        }
        """
        for name, data in fly_data.items():
            if name in self.vitals:
                self.vitals[name].update(dt, **data)

    def get(self, fly_name) -> FlyVitals:
        return self.vitals[fly_name]

    def any_alive(self) -> bool:
        return any(v.alive for v in self.vitals.values())

    def print_status(self):
        """Print full status bars for all flies."""
        for name, v in self.vitals.items():
            print(f"\n  {name} {'(ALIVE)' if v.alive else f'(DEAD: {v.cause_of_death})'}")
            print(v.get_status_bar())
