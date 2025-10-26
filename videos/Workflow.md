## Video Feature Extraction Workflow

This document describes our end-to-end workflow for extracting high-value visual signals from ad videos (no audio). It maps goals to concrete features, algorithms, and implementation steps. The notebook (`videos/main.py`) contains only brief step notes and executable code; refer here for full details.

### Constraints and Goals
- Fast and scalable: total run < 5 minutes on provided dataset; batch-friendly
- Robust across variable resolutions/aspect ratios/fps
- Clear, justifiable signals with low overlap; useful for recommendation models

---

## Step 0: Load and Normalize
- Normalize FPS for analysis (e.g., 10-15 fps) and resize short edge to ~576 px
- Cap frames per video (e.g., <= 400) to control runtime
- Keep timestamps for all frames

Outputs: `frames[t,img]`

---

## Step 1: Shot Detection and Keyframe Selection
Why: Provides structural backbone and representative frames for downstream analysis.

Method:
- Hybrid cut detection: color histogram distance (Bhattacharyya) + SSIM on grayscale
- Thresholds tuned to avoid over-segmentation; merge tiny shots if needed
- Keyframe per shot by composite score: `0.7 x sharpness (Laplacian var) + 0.3 x saliency`

Outputs:
- `shot_segments`, `keyframe_indices`, `keyframe_frames`
- Derived: number of shots, average shot duration, boundaries

---

## Step 2: First 3 Seconds Focus
Why: First 1-2 seconds are critical for retention.

Method:
- Uniformly sample 0-3s at 10 fps; rank frames by sharpness + saliency + text presence
- Extract 1.5-2.0s snippet around 0.5-2.5s

Outputs:
- `start_strength`, `t_first_action`, `t_first_text`, `t_first_face`, representative frames

---

## Step 3: CTA-Centric OCR Spans and Snippets
Why: CTAs drive engagement; timing and readability matter.

Method:
- Gate OCR with fast text detection (MSER/EAST/CRAFT)
- OCR detected boxes; search for CTA keywords (install, download, learn more, shop now, play, open, get, try) and offers (% off, free, limited, sale, prices)
- Merge spans over time; extract +/-0.75s clips for first and longest spans
- Measure text contrast (WCAG-like) within CTA boxes

Outputs:
- `cta_present`, `t_first_cta`, `cta_dwell`, `cta_longest`, `cta_contrast`, `cta_snippets`

---

## Step 4: Logo/Product Reveal Snippets
Why: Early product/brand presence correlates with recall and performance.

Method:
- Detector (YOLOv8n or similar) + simple tracking (ByteTrack/Norfair)
- Find first-appearance time and max-area time for product/logo; extract +/-1s clips
- If detector unavailable, approximate with salient large regions + template/logo proxy

Outputs:
- `t_first_product`, `t_first_logo`, `max_product_area_ratio`, `product_dwell`, `reveal_snippets`

---

## Step 5: High-Motion Burst Detection
Why: Motion spikes grab attention; distinguish camera vs object motion.

Method:
- Optical flow (Farneback/TVL1) on 5-10 fps sampled frames
- Rolling 1.0s window energy; pick top peaks; estimate global camera motion type (pan/tilt/zoom)

Outputs:
- `motion_intensity_mean`, `motion_intensity_peak`, `burst_timestamps`, `camera_motion_type`, burst clips

---

## Step 6: Text-Dense Frames
Why: Offers/prices/messages often correlate with conversions.

Method:
- Fast text detection across sampled frames; compute text coverage and box counts
- Rank top-3 frames by coverage; ensure contrast threshold; optional 1s snippet if persistent

Outputs:
- `text_coverage_stats`, `top_text_frames`, optional clip

---

## Step 7: End-Card Extraction
Why: Final impression; should be static, readable, and branded.

Method:
- Analyze last 3s; find most static, high-contrast frame
- Detect logo/CTA/text presence; compute readability and background simplicity

Outputs:
- `endcard_present`, `endcard_duration`, `endcard_frame`, `endcard_readability`

---

## Step 8: Thumbnail Candidate Ranking
Why: Good thumbnails improve engagement; used for previews.

Method:
- Composite score over keyframes + notable frames + end-card:
  - Sharpness, saliency, face presence/size, text readability, logo/product visibility,
  - Brightness range, absence of letterboxing, rule-of-thirds alignment
- Select top-3 candidates

Outputs:
- `thumbnail_candidates` (frame, timestamp, score, breakdown)

---

## Step 9: Montage/Fast-Cut Section
Why: Pacing and energy signal.

Method:
- Rolling cut density (cuts/s) over timeline; choose peak 1s window; ensure sharp frames

Outputs:
- `peak_cut_density`, `montage_window`, `montage_snippet`

---

## Step 10: Loop Readiness
Why: Loop-friendly ads suit platforms like TikTok/Reels.

Method:
- Compare first vs last 0.5-1s using SSIM, color hist distance (optional CLIP semantics)
- Mark loop-ready if above thresholds; propose candidate loop points

Outputs:
- `loop_readiness_score`, `ssim_first_last`, `candidate_loop_points`

---

## Step 11: Faces and Gaze
Why: Faces and eye contact strongly drive attention and trust.

Method:
- Lightweight face detector (e.g., Haar/DSFD-lite) over sampled frames
- Metrics: face_count, max_face_area_ratio, t_first_face, direct_gaze_ratio (proxy via frontalness), face_dwell

Outputs:
- `face_present`, `t_first_face`, `max_face_area_ratio`, `face_dwell`, `direct_gaze_ratio`

---

## Step 12: Color Tone and Palette
Why: Color contrast, saturation, and palette consistency affect readability and brand feel.

Method:
- HSV/Lab stats for saturation/contrast; warm-cool balance via hue bins
- Small KMeans palette; palette entropy and stability across time

Outputs:
- `saturation_mean`, `contrast_score`, `warm_cool_balance`, `palette_entropy`, `palette_stability`

---

## Step 13: Novelty and Surprise
Why: Abrupt visual changes can spur attention; too much can harm comprehension.

Method:
- Frame-to-frame deltas: color-hist distance + (1−SSIM); detect spikes vs rolling baseline
- Include shot-boundary novelty count

Outputs:
- `novelty_mean`, `novelty_peak`, `novelty_event_count`, `novelty_timestamps`

---

## Step 14: Gestalt / Simplicity (Clutter)
Why: Lower clutter and stronger figure-ground separation improve clarity.

Method:
- Edge density (Canny), connected-region count, saliency isolation ratio
- Composite clutter score (lower is better)

Outputs:
- `clutter_score`, `region_count`, `isolation_ratio`

---

## Step 15: Platform Norms and Safe Areas
Why: Native look and readable layouts increase watch-through.

Method:
- Aspect ratio class (9:16, 1:1, 16:9), letterbox ratio detection
- Center-weighted saliency; safe text ratio (avoid outer margins)

Outputs:
- `aspect_ratio_class`, `letterbox_ratio`, `center_saliency_ratio`, `safe_text_ratio`

---

## Step 16: Emotional Visuals (Proxy)
Why: Expressiveness and grading drive affect and recall.

Method:
- Close-up dwell (large faces over time), color temperature shift (blue<->orange), motion near faces

Outputs:
- `closeup_dwell`, `color_temperature_shift`, `expressive_motion_proxy`

---

## Step 17: Semantic Coherence
Why: Cohesive style and stable palette aid comprehension.

Method:
- Palette stability over time (hist variance), simplicity score, style-change flags

Outputs:
- `palette_stability`, `simplicity_score`, `style_change_events`

---

## Step 18: Aggregate, Score, and Rank Videos
Why: Produce a single composite for ranking and export artifacts.

Method:
- Consolidate per-step features into a single dict per video
- Weighted composite score (sharpness/saliency/opening/CTA/end-card/novelty/faces/color/gestalt/platform/loop)
- Save `video_features.csv`; print sorted rankings; cache reset per video; knobs via FAST_MODE

Outputs:
- `video_features.csv`, ranked list, per-video feature dictionaries

---

## Heuristics and Visual Principles

### Color Contrast and Saturation
Why: Contrast/readability; warm vs cool attention; brand palette consistency.

Signals: contrast ratios, saturation stats, warm/cool balance, brand palette similarity
Hooks: Steps 1, 3, 7, 8, 12

### Motion and Visual Salience
Why: Pre-attentive capture; central fixation.

Signals: motion onset/peaks, center-weighted saliency, camera vs object motion
Hooks: Steps 1, 5, 8, 15

### Faces and Eye Gaze
Why: Attention, trust, and attention guidance via gaze.

Signals: face presence/size/timing, direct gaze, gaze-to-product, emotion proxy
Hooks: Steps 2, 8, 11, 16

### Temporal Heuristics (Timing and Rhythm)
Why: First 1-2 seconds decide retention; pacing ramps then stabilize.

Signals: t_first_cut/action/text/logo, cut density slope
Hooks: Steps 1, 3-5, 9

### Text and Graphic Overlays
Why: Minimal, high-contrast text; early brand helps recall.

Signals: text coverage, CTA timing/dwell/contrast, safe-area compliance, brand early presence
Hooks: Steps 3, 6, 7, 15

### Novelty and Surprise
Why: Attention spikes from transitions/shifts; avoid fatigue.

Signals: transition novelty, embedding/color/style drift
Hooks: Steps 1, 5, 9, 13

### Gestalt and Visual Grouping
Why: Reduce cognitive load; highlight focus via isolation.

Signals: clutter score, region count, figure-ground contrast, isolation ratio
Hooks: Steps 1, 8, 14

### Consistency with Platform Norms
Why: Native-looking content improves watch-through.

Signals: UGC cues, aspect ratio/orientation, watermarks/stickers
Hooks: Loader, 4, 15

### Emotional Visuals
Why: Expressiveness and color grading drive affect.

Signals: emotion proxy, close-up dwell, color temperature shift
Hooks: 11, 12, 16

### Semantic Coherence and Visual Metaphors
Why: Simplicity and metaphors aid comprehension.

Signals: scene simplicity, palette stability, metaphor icons
Hooks: Steps 1, 4, 17


