from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import tempfile, shutil, zipfile, time
from image.extraction import Runner

app = FastAPI(title="Ad Feature Extraction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Ad Intelligence Feature Extractor API"}

@app.post("/extract")
async def extract(files: list[UploadFile] = File(...)):
    start = time.time()
    tmp_in = tempfile.mkdtemp()
    tmp_out = tempfile.mkdtemp()

    # save uploaded images
    for f in files:
        dest = Path(tmp_in) / f.filename
        with open(dest, "wb") as out:
            shutil.copyfileobj(f.file, out)

    # run extraction
    runner = Runner(
        images_dir=tmp_in,
        out_dir=tmp_out,
        target_size=(1024, 1024),
        std_workers=4,
        ext_workers=0,  # auto
        force=True,
        deadline_seconds=300,
        per_image_timeout=120,
    )
    summaries = runner.run()

    # zip results
    zip_path = Path(tmp_out) / "extracted_features.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for f in Path(tmp_out).rglob("*"):
            zipf.write(f, f.relative_to(tmp_out))

    elapsed = time.time() - start
    print(f"[API] Completed extraction of {len(summaries)} images in {elapsed:.1f}s")

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename="extracted_features.zip",
    )
