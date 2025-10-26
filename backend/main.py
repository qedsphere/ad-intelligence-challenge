from __future__ import annotations

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import List
import tempfile
import shutil
import zipfile
import time
import json

# use your existing extractor
from image.extraction import Runner

app = FastAPI(title="Adjacent Feature Extractor")

# allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

IMAGES_DIR = Path("ads/images")
OUT_DIR = Path("extracted_features")


@app.get("/")
def root():
    return {"message": "Adjacent Feature Extractor API is running."}


@app.post("/extract")
async def extract(files: List[UploadFile] = File(...)):
    """Accept image uploads, clear old results, run extraction, and return a zip."""
    start = time.time()

    # Save uploads to a temp dir
    tmp_in = Path(tempfile.mkdtemp(prefix="adj_imgs_"))
    for f in files:
        dest = tmp_in / f.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)

    # Clear previous outputs before run
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run feature extraction
    runner = Runner(
        images_dir=str(tmp_in),
        out_dir=str(OUT_DIR),
        target_size=(1024, 1024),
        std_workers=4,
        ext_workers=0,
        force=True,
        deadline_seconds=300,
        per_image_timeout=120,
    )
    summaries = runner.run()

    # Zip all outputs
    zip_tmp = Path(tempfile.mkdtemp(prefix="adj_zip_"))
    zip_path = zip_tmp / "extracted_features.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for f in OUT_DIR.rglob("*"):
            zipf.write(f, f.relative_to(OUT_DIR))

    elapsed = time.time() - start
    print(f"[API] Completed extraction of {len(summaries)} images in {elapsed:.1f}s")

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename="extracted_features.zip",
    )


@app.get("/summaries")
def get_summaries():
    """Return readable summaries from the current extracted_features folder."""
    if not OUT_DIR.exists():
        return {"summaries": []}

    summaries = []
    for folder in sorted([p for p in OUT_DIR.iterdir() if p.is_dir()]):
        image_id = folder.name
        txt_file = folder / f"{image_id}_summary.txt"
        meta_json = folder / f"{image_id}_summary.json"

        display_name = image_id
        if meta_json.exists():
            try:
                with meta_json.open("r") as f:
                    meta = json.load(f)
                display_name = meta.get("metadata", {}).get("image_id", image_id)
            except Exception:
                pass

        if txt_file.exists():
            with txt_file.open("r") as f:
                summaries.append({
                    "file": display_name,
                    "summary": f.read().strip(),
                })

    return {"summaries": summaries}


@app.post("/restart")
def restart_extractor():
    """Clears all uploaded images and extracted features for a fresh start."""
    cleared = []
    for folder in [IMAGES_DIR, OUT_DIR]:
        if folder.exists():
            shutil.rmtree(folder)
            cleared.append(str(folder))
        folder.mkdir(parents=True, exist_ok=True)

    return JSONResponse({
        "status": "reset",
        "cleared_folders": cleared,
        "message": "Extractor has been reset. All previous files and summaries are cleared.",
    })
