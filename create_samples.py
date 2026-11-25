"""
Generates realistic benchmark sample images for AI Detection testing v2.
Produces images with more realistic properties:
  - Real: textured scene with authentic sensor noise profile
  - AI: complex scene with diffusion-like over-smoothing + fine detail disparity
  - Midjourney: portrait-style with synthetic color harmony + metadata
"""
import os
import numpy as np
from PIL import Image, PngImagePlugin
from scipy.ndimage import gaussian_filter

samples_dir = os.path.join(os.path.dirname(__file__), 'static', 'samples')
os.makedirs(samples_dir, exist_ok=True)
np.random.seed(42)

# =====================================================================
# 1. REAL CAMERA PHOTO  —  textured landscape with varied sensor noise
# =====================================================================
print("Generating sample_real.jpg (textured landscape + camera EXIF)...")
w, h = 640, 480
y, x = np.mgrid[:h, :w]

# Multi-layer natural scene: sky + clouds + grass + stones
sky_r = np.clip(100 + 60 * (1.0 - y / h) + np.random.normal(0, 1.5, (h, w)), 0, 255)
sky_g = np.clip(130 + 50 * (1.0 - y / h) + np.random.normal(0, 1.5, (h, w)), 0, 255)
sky_b = np.clip(185 + 40 * (1.0 - y / h) + np.random.normal(0, 1.5, (h, w)), 0, 255)

# Cloud turbulence (Perlin-like from summed sinusoids)
cloud = (30 * np.sin(x / 50.0) * np.cos(y / 40.0) +
         15 * np.sin(x / 25.0 + 1.0) * np.cos(y / 20.0 + 0.5) +
         8 * np.sin(x / 12.0 + 2.0) * np.cos(y / 10.0 + 1.5))
cloud_mask = (y < h * 0.4) & (cloud > 10)

# Ground with grass texture
hill_line = (h * 0.55 + 20 * np.sin(x[0] / 80.0) + 10 * np.cos(x[0] / 30.0)).astype(int)
ground_mask = y > hill_line[None, :]

# Grass texture: high-frequency random + chromatic variation
grass_r = 55 + 25 * np.sin(x / 3.0 + y / 4.0) + np.random.normal(0, 8, (h, w))
grass_g = 95 + 35 * np.sin(x / 4.0 + y / 3.0 + 0.5) + np.random.normal(0, 10, (h, w))
grass_b = 35 + 15 * np.sin(x / 5.0 + y / 5.0 + 1.0) + np.random.normal(0, 6, (h, w))

r = sky_r.copy(); g = sky_g.copy(); b = sky_b.copy()
r[cloud_mask] += 40; g[cloud_mask] += 35; b[cloud_mask] += 30
r[ground_mask] = grass_r[ground_mask]
g[ground_mask] = grass_g[ground_mask]
b[ground_mask] = grass_b[ground_mask]

# Add spatially varying sensor noise (brighter areas = more shot noise)
brightness = (r + g + b) / 3.0
noise_std = 2.0 + (brightness / 255.0) * 4.0  # Poisson-like
r += np.random.normal(0, 1, (h, w)) * noise_std
g += np.random.normal(0, 1, (h, w)) * noise_std
b += np.random.normal(0, 1, (h, w)) * noise_std

rgb = np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)
img_real = Image.fromarray(rgb)

# Camera EXIF
try:
    exif = img_real.getexif()
    exif[0x010f] = "Canon"
    exif[0x0110] = "Canon EOS 5D Mark IV"
    exif[0x0131] = "Firmware Version 1.3.3"
    exif[0x0132] = "2024:08:14 16:32:01"
    img_real.save(os.path.join(samples_dir, 'sample_real.jpg'), "JPEG", quality=88, exif=exif)
except Exception:
    img_real.save(os.path.join(samples_dir, 'sample_real.jpg'), "JPEG", quality=88)
print(" -> sample_real.jpg created.")

# =====================================================================
# 2. AI DIFFUSION SAMPLE  —  smooth zones + hyper-detail disparity + metadata
# =====================================================================
print("Generating sample_ai.png (SD-style with compression disparities)...")
w_ai, h_ai = 512, 512
y_ai, x_ai = np.mgrid[:h_ai, :w_ai]

# Background: very smooth gradient (over-smoothed like diffusion output)
bg_r = gaussian_filter(np.linspace(15, 55, w_ai)[None, :].repeat(h_ai, 0), sigma=8)
bg_g = gaussian_filter(np.linspace(10, 35, w_ai)[None, :].repeat(h_ai, 0), sigma=8)
bg_b = gaussian_filter(np.linspace(35, 85, w_ai)[None, :].repeat(h_ai, 0), sigma=8)

# Central glowing object (sphere with synthetic hyper-detail)
dist = np.hypot(x_ai - 256, y_ai - 256)
sphere_mask = dist < 150

glow = np.clip(1.0 - dist / 200.0, 0, 1) ** 2.0
sphere_r = 220 * glow + 30
sphere_g = 100 * glow + 15
sphere_b = 190 * glow + 50

# Add fine-detail "texture" on the sphere (hyper-detailed vs smooth bg = ELA anomaly)
fine_detail = (np.sin(x_ai * 2.5) * np.cos(y_ai * 2.5) * 12 +
               np.sin(x_ai * 5.0 + y_ai * 3.0) * 8 +
               np.sin(x_ai * 8.0) * np.cos(y_ai * 8.0) * 5)

r_ai = np.where(sphere_mask, sphere_r + fine_detail, bg_r)
g_ai = np.where(sphere_mask, sphere_g + fine_detail * 0.5, bg_g)
b_ai = np.where(sphere_mask, sphere_b + fine_detail * 0.7, bg_b)

# Checkerboard latent grid artifact
checker = ((x_ai // 2) % 2 == (y_ai // 2) % 2).astype(float) * 3.0
r_ai += checker; g_ai += checker; b_ai += checker

# NO noise (diffusion denoised output)
ai_rgb = np.clip(np.stack([r_ai, g_ai, b_ai], axis=-1), 0, 255).astype(np.uint8)
img_ai = Image.fromarray(ai_rgb)

# SD metadata
info = PngImagePlugin.PngInfo()
info.add_text("parameters",
    "neon synthwave glowing sphere, 8k, masterpiece\n"
    "Negative prompt: lowres, blurry\n"
    "Steps: 30, Sampler: DPM++ 2M Karras, CFG scale: 7.5, "
    "Seed: 19840212, Size: 512x512, Model: dreamshaper_8")
info.add_text("Software", "Stable Diffusion WebUI / ComfyUI v0.1.2")
img_ai.save(os.path.join(samples_dir, 'sample_ai.png'), "PNG", pnginfo=info)

# Also save a JPEG version so ELA can compare compression
img_ai.save(os.path.join(samples_dir, 'sample_ai.jpg'), "JPEG", quality=95)
print(" -> sample_ai.png created.")

# =====================================================================
# 3. MIDJOURNEY PORTRAIT  —  synthetic colour, over-smooth skin, metadata
# =====================================================================
print("Generating sample_midjourney.jpg (hyper-saturated portrait)...")
w_mj, h_mj = 512, 512
y_mj, x_mj = np.mgrid[:h_mj, :w_mj]

# Dramatic split lighting (warm left + cool right)
warm = np.clip(1.0 - np.hypot(x_mj - 180, y_mj - 200) / 260.0, 0, 1) ** 1.8
cool = np.clip(1.0 - np.hypot(x_mj - 340, y_mj - 280) / 280.0, 0, 1) ** 2.0

face_dist = np.hypot((x_mj - 256) * 1.15, y_mj - 250)
face_mask = face_dist < 140

r_mj = warm * 230 + 15
g_mj = warm * 140 + cool * 80 + 10
b_mj = cool * 220 + 25

# Unnaturally smooth face (zero noise, heavy gaussian smoothing)
r_mj[face_mask] = gaussian_filter(235 * warm + 45, sigma=5)[face_mask]
g_mj[face_mask] = gaussian_filter(165 * warm + 30, sigma=5)[face_mask]
b_mj[face_mask] = gaussian_filter(145 * warm + 25, sigma=5)[face_mask]

# Hair texture only on top (creates disparity with smooth face)
hair_mask = (y_mj < 180) & (np.abs(x_mj - 256) < 100) & ~face_mask
hair_texture = (np.sin(x_mj * 3.0 + y_mj * 1.5) * 25 +
                np.sin(x_mj * 7.0) * 15 +
                np.sin(x_mj * 12.0 + y_mj * 8.0) * 8)
r_mj[hair_mask] = 40 + hair_texture[hair_mask] * 0.5
g_mj[hair_mask] = 25 + hair_texture[hair_mask] * 0.3
b_mj[hair_mask] = 20 + hair_texture[hair_mask] * 0.2

# Detailed clothing / body texture below face (more disparity)
body_mask = (y_mj > 350) & (np.abs(x_mj - 256) < 150)
body_tex = (np.sin(x_mj * 4.0 + y_mj * 2.0) * 20 +
            np.sin(x_mj * 10.0) * np.cos(y_mj * 10.0) * 12)
r_mj[body_mask] = 60 + body_tex[body_mask] * 0.4
g_mj[body_mask] = 50 + body_tex[body_mask] * 0.3
b_mj[body_mask] = 70 + body_tex[body_mask] * 0.5

# Hyper-saturated background bokeh circles (AI colour clustering)
for cx_b, cy_b, rad, cr, cg, cb in [(80, 350, 40, 255, 100, 50),
                                       (420, 380, 35, 50, 200, 255),
                                       (450, 100, 30, 255, 50, 200)]:
    bokeh_d = np.hypot(x_mj - cx_b, y_mj - cy_b)
    bokeh = np.clip(1.0 - bokeh_d / rad, 0, 1) ** 1.5
    r_mj += bokeh * cr * 0.4
    g_mj += bokeh * cg * 0.4
    b_mj += bokeh * cb * 0.4

mj_rgb = np.stack([r_mj, g_mj, b_mj], axis=-1)
mj_rgb = gaussian_filter(mj_rgb, sigma=0.6)  # diffusion smoothing

# Latent frequency grid
grid_mj = ((x_mj % 4 == 0) & (y_mj % 4 == 0)).astype(float) * 8.0
mj_rgb = np.clip(mj_rgb + grid_mj[:, :, None], 0, 255).astype(np.uint8)

img_mj = Image.fromarray(mj_rgb)
try:
    exif = img_mj.getexif()
    exif[0x0131] = "Midjourney v6.0 / Upscale Latent"
    img_mj.save(os.path.join(samples_dir, 'sample_midjourney.jpg'),
                "JPEG", quality=92, exif=exif)
except Exception:
    img_mj.save(os.path.join(samples_dir, 'sample_midjourney.jpg'),
                "JPEG", quality=92)
print(" -> sample_midjourney.jpg created.")

print("\nAll v2 benchmark samples generated!")
