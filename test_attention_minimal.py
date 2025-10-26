# --- TPP-Gaze sequential demo (minimal) --------------------------------------
import os
from pathlib import Path

def run_tppgaze_demo():
    import torch
    from tppgaze.tppgaze import TPPGaze
    try:
        from tppgaze.utils import visualize_scanpaths
    except Exception:
        visualize_scanpaths = None

    repo_root = Path(__file__).resolve().parent
    # Prefer env overrides; else use your cloned repo defaults
    cfg_path = Path(os.getenv("TPPGAZE_CFG", repo_root / "tppgaze" / "data" / "config.yaml"))
    ckpt_path = Path(os.getenv("TPPGAZE_CKPT", repo_root / "tppgaze" / "data" / "model_transformer.pth"))

    # Use your standardized test image
    img_path = repo_root / "ads" / "standardized_images" / "i0006_standardized.png"

    # Device & run params (can be overridden via env)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_simulations = int(os.getenv("TPPGAZE_N_SIM", "5"))
    sample_duration = float(os.getenv("TPPGAZE_DURATION_SEC", "2.0"))

    print(f"\n[TPPGaze demo] cfg={cfg_path}  ckpt={ckpt_path}  img={img_path}  "
          f"device={device}  n_sim={n_simulations}  secs={sample_duration}")

    # Init & load
    model = TPPGaze(str(cfg_path), str(ckpt_path), device)
    model.load_model()

    # Generate scanpaths: list of arrays (n_fix, 3) with [x, y, duration]
    scanpaths = model.generate_predictions(str(img_path), sample_duration, n_simulations)

    # Light textual summary
    total_fix = sum(sp.shape[0] for sp in scanpaths if sp is not None)
    total_sec = sum(float(sp[:, 2].sum()) for sp in scanpaths if sp is not None)
    print(f"[TPPGaze demo] got {len(scanpaths)} scanpaths, {total_fix} fixations, total_dur≈{total_sec:.2f}")

    # Show first few fixations for sanity
    if scanpaths and scanpaths[0] is not None and len(scanpaths[0]) > 0:
        p0 = scanpaths[0]
        print("[TPPGaze demo] first path head (x, y, dur):")
        for row in p0[:min(5, len(p0))]:
            print("   ", tuple(round(float(v), 2) for v in row))

    # Visualization (if utils available)
    out_dir = repo_root / "viz_out"
    out_dir.mkdir(exist_ok=True)
    if visualize_scanpaths is not None:
        try:
            fig = visualize_scanpaths(str(img_path), scanpaths)  # matches repo utils signature
            # Try saving if a Matplotlib figure was returned
            try:
                import matplotlib.pyplot as plt
                out_png = out_dir / "i0006_scanpaths.png"
                plt.savefig(out_png, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"[TPPGaze demo] saved visualization → {out_png}")
            except Exception:
                pass
        except Exception as e:
            print(f"[TPPGaze demo] visualize_scanpaths failed: {e}")
    else:
        print("[TPPGaze demo] tppgaze.utils.visualize_scanpaths not available")

# Run the demo after the attention-map section
run_tppgaze_demo()
