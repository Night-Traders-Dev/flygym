"""
biome.py — Multi-biome world for NeuroMechFly v2

Provides BiomeParams, preset biomes, BiomeWorld (tiled biome zones with
3D scatter objects: leaves, twigs, pebbles, grass, fruit, puddles).
"""

from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import mujoco as mj
import dm_control.mjcf as mjcf

from flygym.anatomy import LEG_LINKS
from flygym.compose import FlatGroundWorld
from flygym.compose.physics import ContactParams
from flygym.utils.math import Rotation3D


# ---------------------------------------------------------------------------
# Biome definition
# ---------------------------------------------------------------------------
@dataclass
class BiomeParams:
    name: str
    ground_rgb1: tuple = (0.3, 0.3, 0.3)
    ground_rgb2: tuple = (0.4, 0.4, 0.4)
    reflectance: float = 0.05
    specular: float = 0.05
    wind: tuple = (0.0, 0.0, 0.0)
    temperature: float = 22.0
    humidity: float = 0.5
    friction: tuple = (1.0, 0.005, 0.0001)
    food_density: float = 1.0
    elevation_amp: float = 0.0
    elevation_freq: float = 2.0


# ---------------------------------------------------------------------------
# Preset biomes
# ---------------------------------------------------------------------------
FOREST_FLOOR = BiomeParams(
    name="forest_floor",
    ground_rgb1=(0.28, 0.22, 0.14), ground_rgb2=(0.18, 0.14, 0.08),
    reflectance=0.02,
    wind=(0, 0, 0), temperature=20.0, humidity=0.75,
    friction=(1.0, 0.005, 0.0001), food_density=1.0,
)

MEADOW = BiomeParams(
    name="meadow",
    ground_rgb1=(0.40, 0.48, 0.22), ground_rgb2=(0.30, 0.38, 0.16),
    reflectance=0.08,
    wind=(0.0004, 0.0002, 0), temperature=26.0, humidity=0.40,
    friction=(0.8, 0.004, 0.0001), food_density=0.6,
)

WETLAND = BiomeParams(
    name="wetland",
    ground_rgb1=(0.14, 0.17, 0.11), ground_rgb2=(0.08, 0.11, 0.06),
    reflectance=0.30, specular=0.25,
    wind=(0, 0, 0), temperature=18.0, humidity=0.95,
    friction=(0.4, 0.002, 0.0001), food_density=0.8,
)

SANDY_ARID = BiomeParams(
    name="sandy_arid",
    ground_rgb1=(0.72, 0.62, 0.42), ground_rgb2=(0.58, 0.48, 0.30),
    reflectance=0.12,
    wind=(0.0008, 0, 0), temperature=34.0, humidity=0.12,
    friction=(1.4, 0.008, 0.0002), food_density=0.15,
)

FRUIT_GARDEN = BiomeParams(
    name="fruit_garden",
    ground_rgb1=(0.35, 0.26, 0.16), ground_rgb2=(0.24, 0.17, 0.10),
    reflectance=0.04,
    wind=(0.0002, 0.0001, 0), temperature=24.0, humidity=0.55,
    friction=(1.0, 0.005, 0.0001), food_density=2.5,
)

ALL_BIOMES = [FOREST_FLOOR, MEADOW, WETLAND, SANDY_ARID, FRUIT_GARDEN]


# ---------------------------------------------------------------------------
# Texture generator (1024x1024, multi-octave noise, no banding)
# ---------------------------------------------------------------------------
def _perlin_like(w, h, rng, octaves=5):
    """Multi-octave value noise (no sine banding)."""
    result = np.zeros((h, w), dtype=np.float64)
    amp = 1.0
    for o in range(octaves):
        freq = 2 ** (o + 1)
        # Low-res random grid, upscaled with smooth interpolation
        small = rng.uniform(-1, 1, (freq + 2, freq + 2))
        from scipy.ndimage import zoom
        big = zoom(small, (h / freq, w / freq), order=3)[:h, :w]
        result += big * amp
        amp *= 0.5
    # Normalize to [0, 1]
    result = (result - result.min()) / (result.max() - result.min() + 1e-8)
    return result


def generate_biome_texture(biome: BiomeParams, w=1024, h=1024, seed=None):
    """Generate a high-quality procedural ground texture."""
    rng = np.random.RandomState(seed or hash(biome.name) % 2**31)
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Multi-octave noise base (no sine banding)
    noise = _perlin_like(w, h, rng, octaves=6)
    grain = rng.normal(0, 1, (h, w)) * 0.03
    noise = np.clip(noise + grain, 0, 1)

    r1, r2 = np.array(biome.ground_rgb1) * 255, np.array(biome.ground_rgb2) * 255
    for c in range(3):
        img[:, :, c] = np.clip(r2[c] + noise * (r1[c] - r2[c]), 0, 255).astype(np.uint8)

    # Per-biome detail (large, visible features baked into texture)
    if biome.name == "forest_floor":
        # Large dark soil patches
        for _ in range(40):
            cx, cy, r = rng.randint(0, w), rng.randint(0, h), rng.randint(20, 60)
            yy, xx = np.ogrid[-cy:h - cy, -cx:w - cx]
            mask = xx**2 + yy**2 <= r**2
            img[mask] = (img[mask] * rng.uniform(0.55, 0.75)).astype(np.uint8)

    elif biome.name == "meadow":
        # Lighter green variation patches
        for _ in range(50):
            cx, cy = rng.randint(0, w), rng.randint(0, h)
            r = rng.randint(15, 50)
            yy, xx = np.ogrid[-cy:h - cy, -cx:w - cx]
            mask = xx**2 + yy**2 <= r**2
            bright = rng.uniform(1.05, 1.25)
            img[mask, 1] = np.clip(img[mask, 1] * bright, 0, 255).astype(np.uint8)

    elif biome.name == "wetland":
        # Large dark water pools
        for _ in range(25):
            cx, cy = rng.randint(0, w), rng.randint(0, h)
            rx, ry = rng.randint(30, 80), rng.randint(20, 60)
            yy, xx = np.ogrid[-cy:h - cy, -cx:w - cx]
            mask = (xx / max(rx, 1))**2 + (yy / max(ry, 1))**2 <= 1
            img[mask] = (img[mask] * 0.45).astype(np.uint8)
            # Bright rim (reflection)
            rim = ((xx / max(rx, 1))**2 + (yy / max(ry, 1))**2 > 0.7) & mask
            img[rim] = np.clip(img[rim] * 1.3 + 15, 0, 255).astype(np.uint8)

    elif biome.name == "sandy_arid":
        # Wind ripple patterns
        xs = np.arange(w) / w
        ys = np.arange(h) / h
        xx, yy = np.meshgrid(xs, ys)
        ripple = (np.sin(xx * 40 + yy * 5) * 0.5 + 0.5) * 15
        for c in range(3):
            img[:, :, c] = np.clip(img[:, :, c].astype(float) + ripple, 0, 255).astype(np.uint8)

    elif biome.name == "fruit_garden":
        # Rich dark soil patches
        for _ in range(30):
            cx, cy, r = rng.randint(0, w), rng.randint(0, h), rng.randint(15, 40)
            yy, xx = np.ogrid[-cy:h - cy, -cx:w - cx]
            mask = xx**2 + yy**2 <= r**2
            img[mask] = (img[mask] * rng.uniform(0.6, 0.8)).astype(np.uint8)

    return img


# ---------------------------------------------------------------------------
# 3D scatter object generators
# ---------------------------------------------------------------------------
def _add_scatter_objects(mjcf_root, biome, zone_x, zone_y, zone_size, rng, zone_id):
    """Add 3D geometry objects (leaves, twigs, pebbles, etc) to a biome zone."""
    half = zone_size / 2
    obj_count = 0

    if biome.name == "forest_floor":
        # Leaves: flat ellipsoid geoms
        leaf_colors = [
            (0.33, 0.24, 0.08, 0.9),  # dry brown
            (0.40, 0.32, 0.10, 0.9),  # ochre
            (0.22, 0.26, 0.12, 0.9),  # olive
            (0.28, 0.12, 0.08, 0.9),  # dark red
            (0.16, 0.20, 0.10, 0.9),  # dark moss
        ]
        for i in range(rng.randint(15, 30)):
            lx = zone_x + rng.uniform(-half * 0.9, half * 0.9)
            ly = zone_y + rng.uniform(-half * 0.9, half * 0.9)
            angle = rng.uniform(0, 180)
            color = leaf_colors[rng.randint(len(leaf_colors))]
            sx = rng.uniform(0.15, 0.5)
            sy = rng.uniform(0.3, 0.8)
            mjcf_root.worldbody.add(
                "geom", type="ellipsoid", name=f"leaf_{zone_id}_{i}",
                pos=(lx, ly, 0.01), euler=(0, 0, angle),
                size=(sx, sy, 0.008),
                rgba=color, contype=0, conaffinity=0,
            )
            obj_count += 1

        # Twigs: thin cylinders
        for i in range(rng.randint(5, 12)):
            tx = zone_x + rng.uniform(-half * 0.8, half * 0.8)
            ty = zone_y + rng.uniform(-half * 0.8, half * 0.8)
            angle = rng.uniform(0, 180)
            length = rng.uniform(0.5, 2.5)
            mjcf_root.worldbody.add(
                "geom", type="capsule", name=f"twig_{zone_id}_{i}",
                pos=(tx, ty, 0.02), euler=(0, 90, angle),
                size=(0.02, length / 2),
                rgba=(0.25, 0.18, 0.08, 0.9), contype=0, conaffinity=0,
            )
            obj_count += 1

        # Small pebbles
        for i in range(rng.randint(5, 15)):
            px = zone_x + rng.uniform(-half * 0.9, half * 0.9)
            py = zone_y + rng.uniform(-half * 0.9, half * 0.9)
            r = rng.uniform(0.05, 0.15)
            g = rng.uniform(0.3, 0.5)
            mjcf_root.worldbody.add(
                "geom", type="sphere", name=f"pebble_{zone_id}_{i}",
                pos=(px, py, r * 0.5), size=(r,),
                rgba=(g, g * 0.9, g * 0.7, 1), contype=0, conaffinity=0,
            )
            obj_count += 1

    elif biome.name == "meadow":
        # Grass blades: thin vertical capsules
        for i in range(rng.randint(30, 60)):
            gx = zone_x + rng.uniform(-half * 0.95, half * 0.95)
            gy = zone_y + rng.uniform(-half * 0.95, half * 0.95)
            gh = rng.uniform(0.3, 1.0)
            tilt = rng.uniform(-15, 15)
            green = rng.uniform(0.3, 0.55)
            mjcf_root.worldbody.add(
                "geom", type="capsule", name=f"grass_{zone_id}_{i}",
                pos=(gx, gy, gh / 2), euler=(tilt, 0, rng.uniform(0, 360)),
                size=(0.015, gh / 2),
                rgba=(green * 0.6, green, green * 0.3, 0.85),
                contype=0, conaffinity=0,
            )
            obj_count += 1

        # Seed heads: tiny spheres on stalks
        for i in range(rng.randint(8, 18)):
            sx = zone_x + rng.uniform(-half * 0.9, half * 0.9)
            sy = zone_y + rng.uniform(-half * 0.9, half * 0.9)
            mjcf_root.worldbody.add(
                "geom", type="sphere", name=f"seed_{zone_id}_{i}",
                pos=(sx, sy, rng.uniform(0.5, 1.2)),
                size=(rng.uniform(0.03, 0.07),),
                rgba=(0.7, 0.6, 0.3, 0.9), contype=0, conaffinity=0,
            )
            obj_count += 1

    elif biome.name == "wetland":
        # Puddles: flat dark discs
        for i in range(rng.randint(5, 12)):
            px = zone_x + rng.uniform(-half * 0.8, half * 0.8)
            py = zone_y + rng.uniform(-half * 0.8, half * 0.8)
            r = rng.uniform(0.5, 2.5)
            mjcf_root.worldbody.add(
                "geom", type="cylinder", name=f"puddle_{zone_id}_{i}",
                pos=(px, py, 0.002), size=(r, 0.002),
                rgba=(0.06, 0.08, 0.05, 0.7), contype=0, conaffinity=0,
            )
            obj_count += 1

        # Reeds: tall thin capsules
        for i in range(rng.randint(8, 20)):
            rx = zone_x + rng.uniform(-half * 0.9, half * 0.9)
            ry = zone_y + rng.uniform(-half * 0.9, half * 0.9)
            rh = rng.uniform(0.8, 2.0)
            mjcf_root.worldbody.add(
                "geom", type="capsule", name=f"reed_{zone_id}_{i}",
                pos=(rx, ry, rh / 2),
                euler=(rng.uniform(-8, 8), 0, rng.uniform(0, 360)),
                size=(0.02, rh / 2),
                rgba=(0.25, 0.35, 0.15, 0.85), contype=0, conaffinity=0,
            )
            obj_count += 1

    elif biome.name == "sandy_arid":
        # Pebbles and small rocks
        for i in range(rng.randint(10, 25)):
            px = zone_x + rng.uniform(-half * 0.9, half * 0.9)
            py = zone_y + rng.uniform(-half * 0.9, half * 0.9)
            r = rng.uniform(0.08, 0.3)
            g = rng.uniform(0.4, 0.65)
            mjcf_root.worldbody.add(
                "geom", type="ellipsoid", name=f"rock_{zone_id}_{i}",
                pos=(px, py, r * 0.4),
                euler=(rng.uniform(0, 30), 0, rng.uniform(0, 360)),
                size=(r, r * rng.uniform(0.6, 1.0), r * 0.5),
                rgba=(g, g * 0.88, g * 0.65, 1), contype=0, conaffinity=0,
            )
            obj_count += 1

        # Dried stick fragments
        for i in range(rng.randint(3, 8)):
            tx = zone_x + rng.uniform(-half * 0.8, half * 0.8)
            ty = zone_y + rng.uniform(-half * 0.8, half * 0.8)
            mjcf_root.worldbody.add(
                "geom", type="capsule", name=f"stick_{zone_id}_{i}",
                pos=(tx, ty, 0.03), euler=(0, 90, rng.uniform(0, 180)),
                size=(0.025, rng.uniform(0.3, 1.5)),
                rgba=(0.5, 0.4, 0.25, 0.9), contype=0, conaffinity=0,
            )
            obj_count += 1

    elif biome.name == "fruit_garden":
        # Fallen fruit: colorful spheres
        fruit_colors = [
            (0.8, 0.2, 0.1, 0.95),   # red apple
            (0.9, 0.75, 0.15, 0.95),  # yellow
            (0.6, 0.1, 0.5, 0.95),    # plum
            (0.4, 0.7, 0.2, 0.95),    # green
            (0.85, 0.45, 0.1, 0.95),  # orange
        ]
        for i in range(rng.randint(10, 25)):
            fx = zone_x + rng.uniform(-half * 0.9, half * 0.9)
            fy = zone_y + rng.uniform(-half * 0.9, half * 0.9)
            r = rng.uniform(0.15, 0.45)
            c = fruit_colors[rng.randint(len(fruit_colors))]
            mjcf_root.worldbody.add(
                "geom", type="sphere", name=f"fruit_{zone_id}_{i}",
                pos=(fx, fy, r), size=(r,),
                rgba=c, contype=0, conaffinity=0,
            )
            obj_count += 1

        # Leaves (similar to forest but greener)
        for i in range(rng.randint(10, 20)):
            lx = zone_x + rng.uniform(-half * 0.9, half * 0.9)
            ly = zone_y + rng.uniform(-half * 0.9, half * 0.9)
            green = rng.uniform(0.25, 0.45)
            mjcf_root.worldbody.add(
                "geom", type="ellipsoid", name=f"gleaf_{zone_id}_{i}",
                pos=(lx, ly, 0.01),
                euler=(0, 0, rng.uniform(0, 180)),
                size=(rng.uniform(0.15, 0.4), rng.uniform(0.25, 0.6), 0.008),
                rgba=(green * 0.7, green, green * 0.3, 0.85),
                contype=0, conaffinity=0,
            )
            obj_count += 1

    return obj_count


# ---------------------------------------------------------------------------
# BiomeWorld
# ---------------------------------------------------------------------------
class BiomeWorld(FlatGroundWorld):
    """Large world tiled with biome zones, each with unique visuals, physics,
    and 3D scatter objects (leaves, twigs, grass, puddles, fruit, etc)."""

    def __init__(
        self,
        biome_grid: list[list[BiomeParams]],
        zone_size: float = 20.0,
        n_food: int = 12,
        name: str = "biome_world",
    ):
        self.biome_grid = biome_grid
        self.nrows = len(biome_grid)
        self.ncols = len(biome_grid[0])
        self.zone_size = zone_size
        half = max(self.nrows, self.ncols) * zone_size / 2 + zone_size

        super().__init__(name=name, half_size=half)

        # Keep parent's infinite ground plane for physics collision (it works reliably)
        # Zone boxes are visual only (sit slightly above the collision plane)
        self.ground_geom.pos = (0, 0, -0.01)  # collision plane just below z=0
        # Restyle to neutral earth (visible in gaps between zones)
        dirt_tex = self.mjcf_root.asset.add(
            "texture", name="tex_dirt_fill", type="2d", builtin="flat",
            width=64, height=64, rgb1=(0.32, 0.28, 0.20), rgb2=(0.28, 0.24, 0.17),
        )
        dirt_mat = self.mjcf_root.asset.add(
            "material", name="mat_dirt_fill", texture=dirt_tex,
            texrepeat=(100, 100), reflectance=0.02,
        )
        self.ground_geom.material = dirt_mat

        self._zone_geoms = {}
        self._biome_textures = {}
        created_materials = {}
        scatter_rng = np.random.RandomState(777)
        total_objects = 0

        for row in range(self.nrows):
            for col in range(self.ncols):
                biome = biome_grid[row][col]
                x = (col - self.ncols / 2 + 0.5) * zone_size
                y = (row - self.nrows / 2 + 0.5) * zone_size

                if biome.name not in created_materials:
                    tex = self.mjcf_root.asset.add(
                        "texture", name=f"tex_{biome.name}", type="2d",
                        builtin="flat", width=1024, height=1024,
                        rgb1=biome.ground_rgb1, rgb2=biome.ground_rgb2,
                    )
                    mat = self.mjcf_root.asset.add(
                        "material", name=f"mat_{biome.name}", texture=tex,
                        texrepeat=(2, 2), reflectance=biome.reflectance,
                        specular=biome.specular,
                    )
                    created_materials[biome.name] = mat
                    self._biome_textures[biome.name] = generate_biome_texture(biome)

                geom = self.mjcf_root.worldbody.add(
                    "geom", type="box", name=f"zone_{row}_{col}",
                    pos=(x, y, -0.25),
                    size=(zone_size / 2, zone_size / 2, 0.25),
                    material=created_materials[biome.name],
                    friction=biome.friction,
                    contype=0, conaffinity=0,  # visual only — collision via parent ground plane
                )
                self._zone_geoms[(row, col)] = geom

                # Scatter 3D objects
                zone_id = f"{row}_{col}"
                n = _add_scatter_objects(
                    self.mjcf_root, biome, x, y, zone_size, scatter_rng, zone_id
                )
                total_objects += n

        print(f"  Scattered {total_objects} 3D objects across {self.nrows*self.ncols} zones")

        # Skybox
        self.mjcf_root.asset.add(
            "texture", name="biome_sky", type="skybox", builtin="gradient",
            rgb1=(0.55, 0.62, 0.75), rgb2=(0.30, 0.40, 0.28),
            width=512, height=512,
        )

        # Sun
        self.mjcf_root.worldbody.add(
            "light", name="sun", mode="fixed", directional=True,
            castshadow=True, pos=(0, 0, 100), dir=(0.3, 0.2, -1),
            ambient=(0.18, 0.16, 0.12), diffuse=(0.7, 0.6, 0.45),
            specular=(0.2, 0.18, 0.12),
        )

        # Food slots
        for i in range(n_food):
            body = self.mjcf_root.worldbody.add(
                "body", name=f"food_{i}", pos=(0, 0, -10), mocap=True)
            body.add("geom", type="sphere", size=(0.3,),
                     rgba=(0.85, 0.15, 0.1, 0.9), contype=0, conaffinity=0)

        # Border walls (prevent flies from falling off the edge)
        arena_half_x = self.ncols * zone_size / 2
        arena_half_y = self.nrows * zone_size / 2
        wall_h = 3.0   # mm tall (fly is ~0.5mm, this is a cliff)
        wall_t = 0.5    # mm thick
        wall_rgba = (0.35, 0.30, 0.20, 0.6)  # semi-transparent earth tone

        wall_specs = [
            ("wall_north", (0, arena_half_y + wall_t, wall_h / 2),
             (arena_half_x + wall_t, wall_t, wall_h / 2)),
            ("wall_south", (0, -arena_half_y - wall_t, wall_h / 2),
             (arena_half_x + wall_t, wall_t, wall_h / 2)),
            ("wall_east", (arena_half_x + wall_t, 0, wall_h / 2),
             (wall_t, arena_half_y + wall_t, wall_h / 2)),
            ("wall_west", (-arena_half_x - wall_t, 0, wall_h / 2),
             (wall_t, arena_half_y + wall_t, wall_h / 2)),
        ]
        self._wall_geoms = []
        for wname, wpos, wsize in wall_specs:
            wg = self.mjcf_root.worldbody.add(
                "geom", type="box", name=wname,
                pos=wpos, size=wsize,
                rgba=wall_rgba, contype=0, conaffinity=0,
            )
            self._wall_geoms.append(wg)

    # --- Multi-fly contact overrides ---

    def _set_ground_contact(self, fly, bodysegs, params):
        # Use parent's ground plane for collision (reliable, fast, single geom)
        # Zone boxes and scatter objects are visual only (contype=0)
        for bs in bodysegs:
            fly_geom = fly.mjcf_root.find("geom", bs.name)
            self.mjcf_root.contact.add(
                "pair", geom1=fly_geom, geom2=self.ground_geom,
                name=f"{fly.name}_{bs.name}-ground",
                friction=params.get_friction_tuple(),
                solref=params.get_solref_tuple(),
                solimp=params.get_solimp_tuple(),
                margin=params.margin,
            )
        # Wall contacts via explicit pairs
        for wg in self._wall_geoms:
            for bs in bodysegs:
                fly_geom = fly.mjcf_root.find("geom", bs.name)
                self.mjcf_root.contact.add(
                    "pair", geom1=fly_geom, geom2=wg,
                    name=f"{fly.name}_{bs.name}-{wg.name}",
                    friction=(1.0, 0.005, 0.0001, 0.001, 0.001),
                    solref=params.get_solref_tuple(),
                    solimp=params.get_solimp_tuple(),
                    margin=params.margin,
                )

    def _add_ground_contact_sensors(self, fly, bodysegs):
        if self.legpos_to_groundcontactsensors_by_fly is None:
            self.legpos_to_groundcontactsensors_by_fly = defaultdict(dict)
        ref_geom = self.ground_geom  # use parent's collision plane
        by_leg = defaultdict(list)
        for bs in bodysegs:
            if bs.is_leg(): by_leg[bs.pos].append(bs)
        for leg, segs in by_leg.items():
            segs.sort(key=lambda s: LEG_LINKS.index(s.link))
            body = fly.bodyseg_to_mjcfbody[segs[0]]
            sensor = self.mjcf_root.sensor.add(
                "contact", subtree1=body, geom2=ref_geom,
                num=1, reduce="netforce",
                data="found force torque pos normal tangent",
                name=f"{fly.name}_ground_contact_{leg}_leg",
            )
            self.legpos_to_groundcontactsensors_by_fly[fly.name][leg] = sensor

    def get_biome_at(self, x: float, y: float) -> BiomeParams:
        col = int((x / self.zone_size) + self.ncols / 2)
        row = int((y / self.zone_size) + self.nrows / 2)
        col = np.clip(col, 0, self.ncols - 1)
        row = np.clip(row, 0, self.nrows - 1)
        return self.biome_grid[row][col]

    def upload_textures(self, mj_model):
        for biome_name, tex_img in self._biome_textures.items():
            tex_name = f"tex_{biome_name}"
            for i in range(mj_model.ntex):
                name = mj.mj_id2name(mj_model, mj.mjtObj.mjOBJ_TEXTURE, i)
                if name == tex_name:
                    h, w = mj_model.tex_height[i], mj_model.tex_width[i]
                    adr = mj_model.tex_adr[i]
                    flat = tex_img[:h, :w, :].flatten()
                    mj_model.tex_data[adr:adr + len(flat)] = flat
                    break

    def apply_atmosphere(self, mj_model):
        mj_model.vis.rgba.fog[:] = [0.30, 0.35, 0.25, 1.0]
        mj_model.vis.map.fogstart = 25.0
        mj_model.vis.map.fogend = 100.0
