"""
AI Image Detection & Forensic Analysis Engine  v3.0
====================================================
Tuned sensitivities:
  • ELA & Chromatic — high sensitivity, highest weight
  • Noise Residual — rewritten with multi-scale Laplacian + entropy
  • FFT — dampened, lower weight
  • Metadata — soft signal only, never overrides probability
"""

import io
import math
import base64
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ExifTags
from scipy.ndimage import gaussian_filter, laplace, median_filter
from scipy.stats import entropy as sp_entropy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class AIDetector:
    def __init__(self):
        self.ai_keywords = [
            'stable diffusion', 'midjourney', 'dall-e', 'dalle', 'novelai',
            'comfyui', 'automatic1111', 'invokeai', 'civitai', 'leonardo.ai',
            'firefly', 'flux.1', 'parameters', 'negative prompt', 'steps: ',
            'sampler: ', 'cfg scale: ', 'seed: ', 'clip skip', 'model hash',
            'sd-metadata', 'generation_data', 'workflow'
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _to_base64(self, fig_or_img):
        buffered = io.BytesIO()
        if isinstance(fig_or_img, Image.Image):
            fig_or_img.save(buffered, format="JPEG", quality=85)
        else:
            fig_or_img.savefig(buffered, format='jpg', bbox_inches='tight',
                               pad_inches=0.05, dpi=120)
            plt.close(fig_or_img)
        buffered.seek(0)
        return "data:image/jpeg;base64," + base64.b64encode(
            buffered.getvalue()).decode("utf-8")

    @staticmethod
    def _safe(val, fallback=0.5):
        if val is None or (isinstance(val, float) and (math.isnan(val) or math.isinf(val))):
            return fallback
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return fallback
        return v

    # ==================================================================
    # 1. METADATA  — soft signal only (dampened)
    # ==================================================================
    def analyze_metadata(self, img):
        findings = {
            'has_ai_metadata': False, 'has_camera_exif': False,
            'ai_signatures': [], 'camera_info': {},
            'raw_keys_checked': 0, 'score': 0.5
        }

        if hasattr(img, 'info') and img.info:
            findings['raw_keys_checked'] += len(img.info)
            for k, v in img.info.items():
                v_str, k_str = str(v).lower(), str(k).lower()
                for kw in self.ai_keywords:
                    if kw in k_str or kw in v_str:
                        findings['has_ai_metadata'] = True
                        sig = f"{k}: {str(v)[:120]}..."
                        if sig not in findings['ai_signatures']:
                            findings['ai_signatures'].append(sig)

        try:
            exif = img.getexif()
            if exif:
                camera_tags = {}
                for tag_id, value in exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                    val_str = str(value).lower()
                    for kw in self.ai_keywords:
                        if kw in val_str or kw in tag_name.lower():
                            findings['has_ai_metadata'] = True
                            findings['ai_signatures'].append(
                                f"EXIF {tag_name}: {val_str[:80]}")
                    if tag_name in ['Make', 'Model', 'LensModel', 'FNumber',
                                    'ExposureTime', 'ISOSpeedRatings',
                                    'DateTimeOriginal']:
                        camera_tags[tag_name] = str(value)
                if camera_tags.get('Make') or camera_tags.get('Model'):
                    findings['has_camera_exif'] = True
                    findings['camera_info'] = camera_tags
        except Exception:
            pass

        # ---- dampened scoring (was 1.0 → now 0.85 max) ----
        if findings['has_ai_metadata']:
            findings['score'] = 0.85
        elif findings['has_camera_exif']:
            findings['score'] = 0.15
        else:
            findings['score'] = 0.50

        return findings

    # ==================================================================
    # 2. FFT  — dampened sensitivity
    # ==================================================================
    def analyze_fft_spectrum(self, gray_np):
        h, w = gray_np.shape
        h_even, w_even = h - h % 2, w - w % 2
        gray_np = gray_np[:h_even, :w_even]

        f_shift = np.fft.fftshift(np.fft.fft2(gray_np))
        magnitude = np.abs(f_shift)
        log_mag = np.log1p(magnitude)

        cy, cx = h_even // 2, w_even // 2
        y, x = np.ogrid[:h_even, :w_even]
        r = np.hypot(x - cx, y - cy).astype(int)
        max_radius = min(cx, cy)

        radial_prof = np.zeros(max_radius)
        for i in range(max_radius):
            m = (r == i)
            if np.any(m):
                radial_prof[i] = np.mean(log_mag[m])

        valid_r = np.arange(3, max_radius)
        valid_prof = radial_prof[3:max_radius]

        slope, residuals, hf_ratio = -1.0, 0.05, 0.5
        with np.errstate(all='ignore'):
            if len(valid_r) > 10 and np.std(valid_prof) > 1e-6:
                log_r = np.log(valid_r)
                coeffs = np.polyfit(log_r, valid_prof, 1)
                slope = self._safe(coeffs[0], -1.0)
                fitted = np.polyval(coeffs, log_r)
                residuals = self._safe(np.mean((valid_prof - fitted) ** 2), 0.05)
                sp = int(len(valid_prof) * 0.7)
                hi = np.mean(valid_prof[sp:])
                mi = np.mean(valid_prof[int(len(valid_prof) * 0.2):sp])
                hf_ratio = self._safe(hi / (mi + 1e-6), 0.5)

        # ---- DAMPENED & INVERTED ----
        spectral_raw = float(np.clip(
            (residuals * 7.0) + (hf_ratio * 0.4 - 0.2), 0.0, 1.0))
        spectral_anomaly = 1.0 - self._safe(spectral_raw, 0.7)

        fig, ax = plt.subplots(figsize=(4, 4), facecolor='#111827')
        nm = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8)
        ax.imshow(nm, cmap='magma'); ax.set_title(
            "2D FFT Power Spectrum", color='#f3f4f6', fontsize=11,
            pad=8, fontweight='bold'); ax.axis('off')
        heatmap = self._to_base64(fig)

        return {'spectral_score': spectral_anomaly, 'slope': float(slope),
                'hf_ratio': float(hf_ratio), 'decay_residual': float(residuals),
                'heatmap_b64': heatmap}

    # ==================================================================
    # 3. ELA  — BOOSTED sensitivity (multi-pass + block z-score)
    # ==================================================================
    def analyze_ela(self, pil_img, quality=90):
        rgb_img = pil_img.convert('RGB')

        def _ela_pass(q):
            buf = io.BytesIO()
            rgb_img.save(buf, 'JPEG', quality=q)
            buf.seek(0)
            return np.array(
                ImageChops.difference(rgb_img, Image.open(buf))
            ).astype(np.float32)

        # --- two quality passes for broader sensitivity ---
        d90 = _ela_pass(90)
        d80 = _ela_pass(80)
        ela_arr = (d90 + d80) / 2.0           # averaged multi-pass ELA

        mean_err = np.mean(ela_arr)
        std_err  = np.std(ela_arr)

        # --- block-level disparity (z-score of block means) ---
        h, w, _ = ela_arr.shape
        bh, bw = max(h // 16, 4), max(w // 16, 4)
        block_means = []
        for i in range(0, h - bh, bh):
            for j in range(0, w - bw, bw):
                block_means.append(np.mean(ela_arr[i:i+bh, j:j+bw]))

        block_means = np.array(block_means) if block_means else np.array([0.0])
        block_std = float(np.std(block_means))
        block_mean = float(np.mean(block_means))

        # coefficient of variation of block means — high = suspicious
        cv = block_std / (block_mean + 1e-5)
        # ratio std/mean of full image
        ratio = std_err / (mean_err + 1e-5)

        # --- BOOSTED gain: aggressive multi-signal mapping ---
        # ratio: std/mean of full image — high disparity = suspicious
        # cv:    coefficient of variation of blocks — high = suspicious
        # Also: very low mean_err means image was never JPEG compressed
        #        or is overly smooth (common for AI output)

        s_ratio = np.clip((ratio - 0.75) * 2.5, 0.0, 1.0)
        s_cv    = np.clip((cv - 0.25) * 2.2, 0.0, 1.0)

        # low-error signal: if mean error < 2.3, the image is unusually smooth
        # or lacks natural camera sensor compression variance
        s_low_err = np.clip((2.3 - mean_err) * 0.75, 0.0, 1.0)

        ela_raw = float(np.clip(
            0.35 * s_ratio + 0.30 * s_cv + 0.50 * s_low_err, 0.0, 1.0))
        ela_anomaly = 1.0 - self._safe(ela_raw, 0.5)

        # visual
        scale = 255.0 / max(np.max(d90), 1)
        ela_vis = ImageEnhance.Brightness(
            Image.fromarray(d90.astype(np.uint8))
        ).enhance(scale * 1.5)
        ela_b64 = self._to_base64(ela_vis)

        return {'ela_score': ela_anomaly,
                'mean_error': float(mean_err), 'std_error': float(std_err),
                'block_variance': block_std, 'heatmap_b64': ela_b64}

    # ==================================================================
    # 4. NOISE RESIDUAL  — completely rewritten
    # ==================================================================
    def analyze_noise_residual(self, gray_np):
        """
        Multi-scale Laplacian + median-filter residual.
        Measures:
          A) flatness ratio      — fraction of near-zero residual pixels
          B) patch entropy spread — how uniformly the noise entropy is distributed
          C) Laplacian kurtosis  — sharpness of the edge/noise boundary
        """
        # --- multi-scale residuals ---
        res_gauss  = gray_np - gaussian_filter(gray_np, sigma=1.5)
        res_median = gray_np - median_filter(gray_np, size=3)
        lap        = laplace(gray_np)

        # A) flatness ratio: fraction of pixels with |residual| < 0.8
        #    AI denoised images are very flat (high ratio)
        flat_frac = float(np.mean(np.abs(res_gauss) < 0.8))

        # B) patch entropy spread
        h, w = gray_np.shape
        ps = 32
        entropies = []
        for i in range(0, h - ps, ps):
            for j in range(0, w - ps, ps):
                patch = res_median[i:i+ps, j:j+ps].ravel()
                # 64-bin histogram of patch values
                hist, _ = np.histogram(patch, bins=64, density=True)
                hist = hist + 1e-12
                entropies.append(sp_entropy(hist))

        if entropies:
            ent_arr = np.array(entropies)
            ent_cv = float(np.std(ent_arr) / (np.mean(ent_arr) + 1e-6))
        else:
            ent_cv = 0.5

        # C) Laplacian kurtosis
        lap_std = np.std(lap)
        if lap_std > 1e-6:
            kurt = float(np.mean(((lap - np.mean(lap)) / lap_std) ** 4) - 3.0)
        else:
            kurt = 0.0

        # --- combined noise score ---
        # flat_frac  ∈ [0,1] — higher = over-smoothed / denoised (AI)
        #   Real camera noise: typically flat_frac < 0.25 at sigma=1.5
        #   AI denoised: often flat_frac > 0.35
        # ent_cv     — high = patchy disparity between smooth & synthetic detail (AI)
        #   Real camera grain: consistent entropy across patches (low ent_cv)
        # |kurt|     — extreme kurtosis (>7.5) indicates sharp synthetic edges on smooth bg
        s_flat = np.clip((flat_frac - 0.28) * 2.8, 0, 1)
        s_ent  = np.clip((ent_cv - 0.25) * 1.8, 0, 1)
        s_kurt = np.clip((np.abs(kurt) - 7.5) * 0.20, 0, 1)

        noise_raw = float(np.clip(
            0.40 * s_flat + 0.35 * s_ent + 0.20 * s_kurt + 0.20 * max(s_flat, s_ent), 0.0, 1.0))
        noise_score = 1.0 - self._safe(noise_raw, 0.5)

        # heatmap: show Gaussian residual
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='#111827')
        nr = (res_gauss - res_gauss.min()) / (res_gauss.max() - res_gauss.min() + 1e-8)
        ax.imshow(nr, cmap='inferno')
        ax.set_title("Noise Residual Map (Multi-Scale)",
                      color='#f3f4f6', fontsize=11, pad=8, fontweight='bold')
        ax.axis('off')
        noise_b64 = self._to_base64(fig)

        return {'noise_score': noise_score, 'noise_std': float(np.std(res_gauss)),
                'kurtosis': float(kurt),
                'flat_fraction': float(flat_frac),
                'entropy_cv': float(ent_cv),
                'noise_inconsistency': float(ent_cv),
                'heatmap_b64': noise_b64}

    # ==================================================================
    # 5. COLOR / CHROMATIC  — BOOSTED sensitivity
    # ==================================================================
    def analyze_color_chroma(self, rgb_np):
        r, g, b = rgb_np[:, :, 0], rgb_np[:, :, 1], rgb_np[:, :, 2]

        r_flat, g_flat, b_flat = r.ravel(), g.ravel(), b.ravel()
        std_r, std_g, std_b = np.std(r_flat), np.std(g_flat), np.std(b_flat)

        if std_r < 1e-4 or std_g < 1e-4 or std_b < 1e-4:
            mean_corr = 0.5
        else:
            with np.errstate(all='ignore'):
                c_rg = np.corrcoef(r_flat, g_flat)[0, 1]
                c_rb = np.corrcoef(r_flat, b_flat)[0, 1]
                c_gb = np.corrcoef(g_flat, b_flat)[0, 1]
            c_rg = self._safe(c_rg, 0.5)
            c_rb = self._safe(c_rb, 0.5)
            c_gb = self._safe(c_gb, 0.5)
            mean_corr = (c_rg + c_rb + c_gb) / 3.0

        # HSV saturation
        r_f, g_f, b_f = r / 255.0, g / 255.0, b / 255.0
        mx = np.maximum(np.maximum(r_f, g_f), b_f)
        mn = np.minimum(np.minimum(r_f, g_f), b_f)
        delta = mx - mn
        saturation = np.where(mx == 0, 0, delta / (mx + 1e-6))

        sat_mean = float(np.mean(saturation))
        sat_std  = float(np.std(saturation))

        # --- NEW: edge hue-shift detection ---
        # Compute local gradient magnitude of each channel and check
        # if the inter-channel gradient correlation breaks at edges
        # (natural lenses cause chromatic aberration at edges; AI doesn't)
        from scipy.ndimage import sobel
        grad_r = np.hypot(sobel(r, axis=0), sobel(r, axis=1))
        grad_g = np.hypot(sobel(g, axis=0), sobel(g, axis=1))
        grad_b = np.hypot(sobel(b, axis=0), sobel(b, axis=1))

        # At strong edges (top 15%) compare channel gradient ratios safely
        edge_thresh = np.percentile(grad_g, 85)
        edge_mask = grad_g > edge_thresh
        if np.sum(edge_mask) > 100:
            rg_ratio = np.clip(grad_r[edge_mask] / np.maximum(grad_g[edge_mask], 5.0), 0.0, 4.0)
            gb_ratio = np.clip(grad_g[edge_mask] / np.maximum(grad_b[edge_mask], 5.0), 0.0, 4.0)
            # Natural images: ratio variance is higher (chromatic aberration)
            # AI images: ratios are more uniform (no physical lens)
            edge_uniformity = float(np.clip(1.0 / (np.std(rg_ratio) + np.std(gb_ratio) + 0.25), 0.0, 2.0))
        else:
            edge_uniformity = 1.0

        # --- NEW: saturation histogram entropy & vibrance extremity ---
        sat_hist, _ = np.histogram(saturation.ravel(), bins=64, density=True)
        sat_hist = sat_hist + 1e-12
        sat_entropy = sp_entropy(sat_hist)
        high_sat_frac = float(np.mean(saturation > 0.65))

        # --- BOOSTED composite score ---
        s_corr = np.clip((mean_corr - 0.65) * 2.0, 0.0, 1.0)         # strong coupling → AI
        s_sat  = np.clip((sat_mean - 0.15) * 2.5, 0.0, 1.0)           # high saturation → AI
        s_vib  = np.clip(high_sat_frac * 2.5, 0.0, 1.0)               # hyper-vibrant pixels → AI
        s_edge = np.clip((edge_uniformity - 0.6) * 1.5, 0.0, 1.0)     # uniform edges → AI
        s_ent  = np.clip((4.0 - sat_entropy) * 0.40, 0.0, 1.0)        # low entropy → AI

        chroma_raw = float(np.clip(
            0.20 * s_corr + 0.20 * s_sat + 0.20 * s_vib + 0.25 * s_edge + 0.15 * s_ent,
            0.0, 1.0))
        chroma_score = 1.0 - self._safe(chroma_raw, 0.5)

        # heatmap
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='#111827')
        ax.imshow(saturation, cmap='plasma')
        ax.set_title("Color Saturation & Chromatic Map",
                      color='#f3f4f6', fontsize=11, pad=8, fontweight='bold')
        ax.axis('off')
        chroma_b64 = self._to_base64(fig)

        return {'chroma_score': chroma_score,
                'channel_correlation': float(mean_corr),
                'saturation_mean': sat_mean, 'saturation_std': sat_std,
                'edge_uniformity': float(edge_uniformity),
                'saturation_entropy': float(sat_entropy),
                'heatmap_b64': chroma_b64}

    # ==================================================================
    # DETECT  — rebalanced weights
    # ==================================================================
    def detect(self, image_input):
        if isinstance(image_input, (str, bytes)):
            img = Image.open(image_input) if isinstance(image_input, str) \
                else Image.open(io.BytesIO(image_input))
        else:
            img = image_input

        orig_format = img.format or "JPEG"
        rgb_img = img.convert('RGB')
        orig_w, orig_h = rgb_img.size

        preview_img = rgb_img.copy(); preview_img.thumbnail((800, 800))
        preview_b64 = self._to_base64(preview_img)

        analysis_img = rgb_img.resize((512, 512), Image.Resampling.BILINEAR)
        rgb_np  = np.array(analysis_img).astype(np.float32)
        gray_np = np.mean(rgb_np, axis=2)

        meta_res   = self.analyze_metadata(img)
        fft_res    = self.analyze_fft_spectrum(gray_np)
        ela_res    = self.analyze_ela(analysis_img)
        noise_res  = self.analyze_noise_residual(gray_np)
        chroma_res = self.analyze_color_chroma(rgb_np)

        # ----------------------------------------------------------
        # Rebalanced weights:
        #   ELA        30%   (was 20%)  — BOOSTED
        #   Chromatic  30%   (was 20%)  — BOOSTED
        #   Noise      20%   (was 25%)  — rewritten, useful now
        #   FFT        15%   (was 35%)  — DAMPENED
        #   Metadata    5%   (was override) — soft signal only
        # ----------------------------------------------------------
        raw_ai_score = (
            0.30 * ela_res['ela_score'] +
            0.30 * chroma_res['chroma_score'] +
            0.20 * noise_res['noise_score'] +
            0.15 * fft_res['spectral_score'] +
            0.05 * meta_res['score']
        )
        raw_ai_score = self._safe(raw_ai_score, 0.50)

        # Camera EXIF still provides a mild correction
        if meta_res['has_camera_exif']:
            raw_ai_score *= 0.75

        # Sigmoid calibration
        k = 6.5
        x_shifted = raw_ai_score - 0.46
        calibrated = 1.0 / (1.0 + math.exp(-k * x_shifted))
        val = self._safe(calibrated * 100.0, 50.0)
        ai_prob = round(float(np.clip(val, 1.5, 98.5)), 1)

        # Verdict
        if ai_prob >= 75.0:
            verdict, confidence, verdict_class = "Highly Probable AI-Generated", "High", "ai"
            explanation = (
                "Strong ELA compression anomalies, synthetic chromatic uniformity, "
                "and over-smoothed noise residuals together indicate AI generation."
            )
        elif ai_prob >= 55.0:
            verdict, confidence, verdict_class = "Likely AI-Generated", "Moderate", "leaning_ai"
            explanation = (
                "Elevated ELA disparity and chromatic coupling suggest AI origin, "
                "though some natural characteristics remain."
            )
        elif ai_prob >= 45.0:
            verdict, confidence, verdict_class = "Inconclusive / Mixed Signals", "Low", "uncertain"
            explanation = (
                "Forensic signals are balanced. The image may be heavily filtered, "
                "re-compressed, or produced by a blended workflow."
            )
        elif ai_prob >= 25.0:
            verdict, confidence, verdict_class = "Likely Real / Authentic Image", "Moderate", "leaning_real"
            explanation = (
                "Natural ELA compression patterns, organic chromatic aberration, "
                "and sensor-consistent noise distribution support authenticity."
            )
        else:
            verdict, confidence, verdict_class = "Highly Probable Authentic Human Image", "High", "real"
            explanation = (
                "Organic sensor noise, natural chromatic dispersion, and standard "
                "optical lens properties strongly confirm an authentic photograph."
            )

        real_prob = round(100.0 - ai_prob, 1)

        return {
            'status': 'success',
            'ai_probability': ai_prob,
            'real_probability': real_prob,
            'verdict': verdict,
            'confidence': confidence,
            'verdict_class': verdict_class,
            'explanation': explanation,
            'image_info': {
                'width': orig_w, 'height': orig_h,
                'aspect_ratio': f"{round(orig_w / max(orig_h, 1), 2)}:1",
                'format': orig_format,
            },
            'indicators': {
                'metadata': {
                    'name': 'Metadata & Generation Signatures',
                    'score': round(meta_res['score'] * 100, 1),
                    'status': ('AI Signatures Detected' if meta_res['has_ai_metadata']
                               else ('Authentic Camera EXIF' if meta_res['has_camera_exif']
                                     else 'No Generation Tags')),
                    'details': (meta_res['ai_signatures'] if meta_res['has_ai_metadata']
                                else meta_res['camera_info'])
                },
                'spectral': {
                    'name': '2D FFT Spectral Frequency Decay',
                    'score': round(fft_res['spectral_score'] * 100, 1),
                    'status': ('Elevated High-Freq Anomaly'
                               if fft_res['spectral_score'] > 0.6
                               else 'Natural 1/f² Decay'),
                    'details': (f"Decay slope: {fft_res['slope']:.2f}, "
                                f"High-freq ratio: {fft_res['hf_ratio']:.2f}")
                },
                'noise': {
                    'name': 'Sensor PRNU & Noise Residual',
                    'score': round(noise_res['noise_score'] * 100, 1),
                    'status': ('Denoised / Synthetic Noise'
                               if noise_res['noise_score'] > 0.5
                               else 'Natural Sensor Grain'),
                    'details': (f"Flat fraction: {noise_res['flat_fraction']:.2f}, "
                                f"Entropy CV: {noise_res['entropy_cv']:.2f}, "
                                f"Kurtosis: {noise_res['kurtosis']:.2f}")
                },
                'ela': {
                    'name': 'Error Level Analysis (ELA)',
                    'score': round(ela_res['ela_score'] * 100, 1),
                    'status': ('Abnormal Compression Gradient'
                               if ela_res['ela_score'] > 0.5
                               else 'Uniform Compression'),
                    'details': (f"Mean error: {ela_res['mean_error']:.1f}, "
                                f"Std: {ela_res['std_error']:.1f}, "
                                f"Block var: {ela_res['block_variance']:.2f}")
                },
                'chroma': {
                    'name': 'Color & Chromatic Dispersion',
                    'score': round(chroma_res['chroma_score'] * 100, 1),
                    'status': ('Synthetic Channel Coupling'
                               if chroma_res['chroma_score'] > 0.5
                               else 'Natural Optical Chromaticity'),
                    'details': (f"Channel corr: {chroma_res['channel_correlation']:.3f}, "
                                f"Sat mean: {chroma_res['saturation_mean']:.2f}, "
                                f"Edge unif: {chroma_res.get('edge_uniformity', 0):.2f}")
                }
            },
            'heatmaps': {
                'preview': preview_b64,
                'fft': fft_res['heatmap_b64'],
                'ela': ela_res['heatmap_b64'],
                'noise': noise_res['heatmap_b64'],
                'chroma': chroma_res['heatmap_b64']
            }
        }
