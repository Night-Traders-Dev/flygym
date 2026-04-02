"""
biome.py — Multi-biome world for NeuroMechFly v2

Provides BiomeParams (environment spec), preset biomes, and BiomeWorld
(a FlatGroundWorld subclass tiled with biome zones).
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
    # Ground appearance
    ground_rgb1: tuple = (0.3, 0.3, 0.3)
    ground_rgb2: tuple = (0.4, 0.4, 0.4)
    reflectance: float = 0.05
    specular: float = 0.05
    # Environment
    wind: tuple = (0.0, 0.0, 0.0)       # force vector (uN) applied to flies
    temperature: float = 22.0            # Celsius
    humidity: float = 0.5                # 0-1
    # Physics
    friction: tuple = (1.0, 0.005, 0.0001)  # sliding, torsional, rolling
    # Ecology
    food_density: float = 1.0            # relative spawn weight
    # Terrain
    elevation_amp: float = 0.0           # mm, heightfield amplitude
    elevation_freq: float = 2.0          # spatial frequency


# ---------------------------------------------------------------------------
# Preset biomes (Drosophila-appropriate)
# ---------------------------------------------------------------------------
FOREST_FLOOR = BiomeParams(
    name="forest_floor",
    ground_rgb1=(0.28, 0.22, 0.14),
    ground_rgb2=(0.18, 0.14, 0.08),
    reflectance=0.02,
    wind=(0.0, 0.0, 0.0),
    temperature=20.0,
    humidity=0.75,
    friction=(1.0, 0.005, 0.0001),
    food_density=1.0,
    elevation_amp=0.05,
    elevation_freq=3.0,
)

MEADOW = BiomeParams(
    name="meadow",
    ground_rgb1=(0.40, 0.48, 0.22),
    ground_rgb2=(0.30, 0.38, 0.16),
    reflectance=0.08,
    wind=(0.0004, 0.0002, 0.0),
    temperature=26.0,
    humidity=0.40,
    friction=(0.8, 0.004, 0.0001),
    food_density=0.6,
    elevation_amp=0.03,
    elevation_freq=2.0,
)

WETLAND = BiomeParams(
    name="wetland",
    ground_rgb1=(0.14, 0.17, 0.11),
    ground_rgb2=(0.08, 0.11, 0.06),
    reflectance=0.30,
    specular=0.25,
    wind=(0.0, 0.0, 0.0),
    temperature=18.0,
    humidity=0.95,
    friction=(0.4, 0.002, 0.0001),
    food_density=0.8,
    elevation_amp=0.01,
    elevation_freq=1.5,
)

SANDY_ARID = BiomeParams(
    name="sandy_arid",
    ground_rgb1=(0.72, 0.62, 0.42),
    ground_rgb2=(0.58, 0.48, 0.30),
    reflectance=0.12,
    wind=(0.0008, 0.0, 0.0),
    temperature=34.0,
    humidity=0.12,
    friction=(1.4, 0.008, 0.0002),
    food_density=0.15,
    elevation_amp=0.08,
    elevation_freq=4.0,
)

FRUIT_GARDEN = BiomeParams(
    name="fruit_garden",
    ground_rgb1=(0.35, 0.26, 0.16),
    ground_rgb2=(0.24, 0.17, 0.10),
    reflectance=0.04,
    wind=(0.0002, 0.0001, 0.0),
    temperature=24.0,
    humidity=0.55,
    friction=(1.0, 0.005, 0.0001),
    food_density=2.5,
    elevation_amp=0.04,
    elevation_freq=2.5,
)

ALL_BIOMES = [FOREST_FLOOR, MEADOW, WETLAND, SANDY_ARID, FRUIT_GARDEN]


# ---------------------------------------------------------------------------
# Procedural texture generators
# ---------------------------------------------------------------------------
def _base_noise(w, h, rng, rgb1, rgb2, grain=6.0):
    """Generate a noisy base ground texture."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    xs, ys = np.meshgrid(np.arange(w) / w, np.arange(h) / h)
    n = (np.sin(xs * 13.7 + ys * 9.3) * 0.3
         + np.sin(xs * 27.1 + ys * 19.7) * 0.15
         + np.sin(xs * 53.3 + ys * 41.1) * 0.08)
    bn = rng.normal(0, grain, (h, w))
    for c in range(3):
        lo = min(rgb1[c], rgb2[c]) * 255
        hi = max(rgb1[c], rgb2[c]) * 255
        mid = (lo + hi) / 2
        span = (hi - lo)
        img[:, :, c] = np.clip(mid + n * span * 2 + bn, lo * 0.6, hi * 1.3).astype(np.uint8)
    return img


def _scatter_patches(img, rng, colors, count=80, min_r=2, max_r=10, blend_range=(0.3, 0.7)):
    h, w = img.shape[:2]
    for _ in range(count):
        cx, cy = rng.randint(0, w), rng.randint(0, h)
        rx, ry = rng.randint(min_r, max_r), rng.randint(min_r + 2, max_r + 4)
        a = rng.uniform(0, np.pi)
        c = np.array(colors[rng.randint(len(colors))])
        yy, xx = np.ogrid[-cy:h - cy, -cx:w - cx]
        ca, sa = np.cos(a), np.sin(a)
        mask = ((ca * xx + sa * yy) / max(rx, 1)) ** 2 + ((-sa * xx + ca * yy) / max(ry, 1)) ** 2 <= 1
        bl = rng.uniform(*blend_range)
        img[mask] = (img[mask] * (1 - bl) + c * bl).astype(np.uint8)


def generate_biome_texture(biome: BiomeParams, w=256, h=256, seed=None):
    """Generate a procedural texture for the given biome."""
    rng = np.random.RandomState(seed or hash(biome.name) % 2**31)
    img = _base_noise(w, h, rng, biome.ground_rgb1, biome.ground_rgb2)

    if biome.name == "forest_floor":
        colors = [(85, 60, 20), (100, 80, 25), (55, 65, 30), (70, 30, 20), (40, 50, 25)]
        _scatter_patches(img, rng, colors, count=100)
        # twigs
        for _ in range(20):
            x0, y0 = rng.randint(0, w), rng.randint(0, h)
            a = rng.uniform(0, np.pi)
            for t in range(rng.randint(8, 30)):
                px, py = int(x0 + t * np.cos(a)), int(y0 + t * np.sin(a))
                if 0 <= px < w and 0 <= py < h:
                    img[py, px] = (img[py, px] * 0.4).astype(np.uint8)

    elif biome.name == "meadow":
        # grass blades and seed spots
        colors = [(70, 90, 30), (60, 80, 25), (80, 95, 35)]
        _scatter_patches(img, rng, colors, count=150, min_r=1, max_r=5)
        # tiny bright seeds
        for _ in range(200):
            x, y = rng.randint(0, w), rng.randint(0, h)
            img[y, x] = [rng.randint(160, 200), rng.randint(140, 180), rng.randint(60, 100)]

    elif biome.name == "wetland":
        # dark patches with bright reflective spots (water)
        for _ in range(60):
            cx, cy, r = rng.randint(0, w), rng.randint(0, h), rng.randint(5, 20)
            yy, xx = np.ogrid[-cy:h - cy, -cx:w - cx]
            mask = xx ** 2 + yy ** 2 <= r ** 2
            img[mask] = (img[mask] * 0.6).astype(np.uint8)
        # water glints
        for _ in range(100):
            x, y = rng.randint(0, w), rng.randint(0, h)
            img[y, x] = [rng.randint(80, 140), rng.randint(90, 150), rng.randint(100, 160)]

    elif biome.name == "sandy_arid":
        # fine grain noise + scattered pebbles
        grain = rng.normal(0, 12, (h, w))
        for c in range(3):
            img[:, :, c] = np.clip(img[:, :, c].astype(float) + grain, 0, 255).astype(np.uint8)
        for _ in range(60):
            x, y = rng.randint(0, w), rng.randint(0, h)
            b = rng.randint(100, 170)
            img[y, x] = [b, int(b * 0.85), int(b * 0.6)]

    elif biome.name == "fruit_garden":
        # rich soil with bright fruit spots
        colors = [(90, 60, 30), (70, 50, 25), (50, 40, 20)]
        _scatter_patches(img, rng, colors, count=80)
        # fallen fruit (bright spots)
        fruit_colors = [(200, 50, 30), (220, 180, 40), (180, 80, 160), (100, 180, 60)]
        for _ in range(25):
            x, y = rng.randint(0, w), rng.randint(0, h)
            r = rng.randint(2, 5)
            yy, xx = np.ogrid[-y:h - y, -x:w - x]
            mask = xx ** 2 + yy ** 2 <= r ** 2
            fc = fruit_colors[rng.randint(len(fruit_colors))]
            img[mask] = fc

    return img


# ---------------------------------------------------------------------------
# BiomeWorld
# ---------------------------------------------------------------------------
class BiomeWorld(FlatGroundWorld):
    """Large world tiled with biome zones, each with unique visuals and physics."""

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

        # Make parent's ground plane visual-only backdrop
        self.ground_geom.contype = 0
        self.ground_geom.conaffinity = 0

        # Create per-zone geoms and materials
        self._zone_geoms = {}
        self._biome_textures = {}  # name -> numpy array
        self._biome_tex_names = {}  # (row,col) -> texture name
        created_materials = {}  # biome.name -> material element

        for row in range(self.nrows):
            for col in range(self.ncols):
                biome = biome_grid[row][col]
                x = (col - self.ncols / 2 + 0.5) * zone_size
                y = (row - self.nrows / 2 + 0.5) * zone_size

                # Create texture/material once per biome type
                if biome.name not in created_materials:
                    tex = self.mjcf_root.asset.add(
                        "texture", name=f"tex_{biome.name}", type="2d",
                        builtin="flat", width=256, height=256,
                        rgb1=biome.ground_rgb1, rgb2=biome.ground_rgb2,
                    )
                    mat = self.mjcf_root.asset.add(
                        "material", name=f"mat_{biome.name}", texture=tex,
                        texrepeat=(4, 4), reflectance=biome.reflectance,
                        specular=biome.specular,
                    )
                    created_materials[biome.name] = mat
                    self._biome_textures[biome.name] = generate_biome_texture(biome)

                self._biome_tex_names[(row, col)] = f"tex_{biome.name}"

                # Zone ground geom (box with top surface at z=0)
                geom = self.mjcf_root.worldbody.add(
                    "geom", type="box", name=f"zone_{row}_{col}",
                    pos=(x, y, -0.25),
                    size=(zone_size / 2, zone_size / 2, 0.25),
                    material=created_materials[biome.name],
                    friction=biome.friction,
                    contype=0, conaffinity=0,  # contact via explicit pairs
                )
                self._zone_geoms[(row, col)] = geom

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

    # --- Multi-fly contact overrides ---

    def _set_ground_contact(self, fly, bodysegs, params):
        """Create contact pairs between fly body segments and ALL zone geoms."""
        for (row, col), geom in self._zone_geoms.items():
            biome = self.biome_grid[row][col]
            for bs in bodysegs:
                fly_geom = fly.mjcf_root.find("geom", bs.name)
                self.mjcf_root.contact.add(
                    "pair", geom1=fly_geom, geom2=geom,
                    name=f"{fly.name}_{bs.name}-z{row}_{col}",
                    friction=(*biome.friction, 0.001, 0.001),
                    solref=params.get_solref_tuple(),
                    solimp=params.get_solimp_tuple(),
                    margin=params.margin,
                )

    def _add_ground_contact_sensors(self, fly, bodysegs):
        """Add contact sensors per leg against the nearest zone geom (spawn zone)."""
        if self.legpos_to_groundcontactsensors_by_fly is None:
            self.legpos_to_groundcontactsensors_by_fly = defaultdict(dict)

        # Use the center zone as the sensor reference (sensors work via subtree)
        center_row, center_col = self.nrows // 2, self.ncols // 2
        ref_geom = self._zone_geoms[(center_row, center_col)]

        by_leg = defaultdict(list)
        for bs in bodysegs:
            if bs.is_leg():
                by_leg[bs.pos].append(bs)
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

    # --- Biome lookup ---

    def get_biome_at(self, x: float, y: float) -> BiomeParams:
        """Get the biome at world coordinates (x, y)."""
        col = int((x / self.zone_size) + self.ncols / 2)
        row = int((y / self.zone_size) + self.nrows / 2)
        col = np.clip(col, 0, self.ncols - 1)
        row = np.clip(row, 0, self.nrows - 1)
        return self.biome_grid[row][col]

    # --- Texture upload ---

    def upload_textures(self, mj_model):
        """Upload all biome textures into GPU after compilation."""
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
