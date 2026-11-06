"""
Public Hugging Face Space for flat2svg - custom HTML page + small FastAPI backend.

The vectorizer source code is NOT stored in this Space. At startup the app downloads it
from a private Hugging Face repo (CORE_REPO) using the HF_TOKEN secret, so visitors only
see this thin web layer.

Flow:
  1. The page uploads a PNG/JPG to /api/check; the flatness detector says whether it fits flat2svg.
  2. Flat images are converted at once (/api/convert); others get a warning and a
     "Convert anyway" button.
  3. The page shows source and SVG side by side with synchronized mouse zoom and pan.
  4. The user rates the result (/api/rate).
  5. Every event is logged anonymously to data/usage/<date>.jsonl (time, session id, verdict,
     size, rating) - no images.
  6. With the user's consent, the source, the SVG, the detector report and the rating are
     saved to a private dataset (LOG_REPO) for quality analysis.

Space secrets / variables:
  HF_TOKEN   - token with read access to CORE_REPO and write access to LOG_REPO
  CORE_REPO  - private repo with flat2svg.py, vecmath.py, denoise_flat.py, flatness_detector.py
  LOG_REPO   - private dataset repo for uploads and ratings
"""

import json
import os
import sys
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

os.environ["GRADIO_SSR_MODE"] = "False"   # SSR would start its own server on port 7860 and block uvicorn

import gradio as gr
import spaces
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from huggingface_hub import CommitScheduler, snapshot_download

HF_TOKEN = os.environ.get("HF_TOKEN")
CORE_REPO = os.environ.get("CORE_REPO", "Ellenanet/flat2svg-core")
LOG_REPO = os.environ.get("LOG_REPO", "Ellenanet/flat2svg-feedback")
MAX_MB = 10
MAX_SIDE = 4000
KEEP_RUNS = 40                      # uploads and results kept in memory / temp folder
HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Load the private vectorizer code
# ---------------------------------------------------------------------------
core_dir = snapshot_download(CORE_REPO, token=HF_TOKEN, repo_type="model")
sys.path.insert(0, core_dir)
import flat2svg as f2s  # noqa: E402
import flatness_detector as fd  # noqa: E402

OPTIONS = {**f2s.DEFAULTS, "jobs": 1}


@spaces.GPU(duration=1)
def _zerogpu_placeholder():
    """ZeroGPU hardware refuses to start without a @spaces.GPU function.
    flat2svg runs on CPU only, so this function exists just to satisfy that check and is never called."""
    return None


# ---------------------------------------------------------------------------
# Logging to a private dataset (pushed in the background every few minutes)
# ---------------------------------------------------------------------------
LOG_DIR = Path("feedback_data")
LOG_DIR.mkdir(exist_ok=True)
scheduler = None
if HF_TOKEN:
    scheduler = CommitScheduler(repo_id=LOG_REPO, repo_type="dataset", folder_path=LOG_DIR,
                                path_in_repo="data", every=5, token=HF_TOKEN, private=True)


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _lock():
    return scheduler.lock if scheduler else _NoLock()


def save_record(run_id, src_path, svg_text, report, extra):
    """Write the source, the SVG and a JSON record into data/<date>/<run_id>/."""
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    folder = LOG_DIR / day / run_id
    with _lock():
        folder.mkdir(parents=True, exist_ok=True)
        if src_path:
            dst = folder / ("source" + Path(src_path).suffix.lower())
            if not dst.exists():
                dst.write_bytes(Path(src_path).read_bytes())
        if svg_text:
            (folder / "result.svg").write_text(svg_text, encoding="utf-8")
        meta_path = folder / "record.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta.update({"run_id": run_id, "time": datetime.now(timezone.utc).isoformat(), "report": report, **extra})
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def log_usage(event, session, **fields):
    """Anonymous usage line for every event, saved even without consent (no image, no SVG)."""
    now = datetime.now(timezone.utc)
    line = {"time": now.isoformat(), "event": event, "session": session, **fields}
    with _lock():
        folder = LOG_DIR / "usage"
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / f"{now.strftime('%Y-%m-%d')}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")


# ---------------------------------------------------------------------------
# In-memory run store (one CPU worker: jobs are serialized by a lock)
# ---------------------------------------------------------------------------
RUNS = OrderedDict()                # run_id -> dict(path, name, report, svg, info, forced)
RUNS_LOCK = threading.Lock()
WORK_LOCK = threading.Lock()
UPLOAD_DIR = Path(tempfile.mkdtemp(prefix="f2s_uploads_"))


def get_run(run_id):
    with RUNS_LOCK:
        run = RUNS.get(run_id)
    if run is None:
        raise HTTPException(404, "This result has expired. Please upload the image again.")
    return run


def put_run(run_id, run):
    with RUNS_LOCK:
        RUNS[run_id] = run
        while len(RUNS) > KEEP_RUNS:
            _, old = RUNS.popitem(last=False)
            Path(old["path"]).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
router = APIRouter()

VERDICT_TEXT = {
    "flat": "Flat illustration, suitable for flat2svg.",
    "shaded": "This image has gradients or many shades. flat2svg works best with even fills; "
              "the result may look posterized.",
    "line_art": "This image has outlines or hatching. flat2svg works best with flat fills; "
                "a tool for detailed illustrations is coming later.",
    "line_art_dark": "Detailed illustration on a dark background - not supported yet.",
}


@router.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@router.post("/api/check")
def api_check(file: UploadFile = File(...), consent: str = Form("0"), session: str = Form("")):
    data = file.file.read(MAX_MB * 1024 * 1024 + 1)
    if len(data) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"File is larger than {MAX_MB} MB.")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg"):
        raise HTTPException(415, "Please upload a PNG or JPG image.")
    run_id = uuid.uuid4().hex[:12]
    path = UPLOAD_DIR / f"{run_id}{suffix}"
    path.write_bytes(data)
    try:
        with WORK_LOCK:
            rep = fd.analyze(path)
    except Exception:
        path.unlink(missing_ok=True)
        raise HTTPException(400, "This file could not be read as an image.")
    if max(rep.width, rep.height) > MAX_SIDE:
        path.unlink(missing_ok=True)
        raise HTTPException(413, f"Image is larger than {MAX_SIDE} px on the long side.")
    report = asdict(rep)
    put_run(run_id, {"path": str(path), "name": Path(file.filename or "image").stem, "report": report,
                     "svg": None, "info": ""})
    log_usage("upload", session, run_id=run_id, verdict=rep.verdict, width=rep.width, height=rep.height,
              consent=consent == "1")
    if consent == "1":
        save_record(run_id, str(path), None, report, {"converted": False, "session": session})
    return {"run_id": run_id, "flat": rep.verdict == "flat", "verdict": rep.verdict,
            "message": VERDICT_TEXT.get(rep.verdict, rep.verdict), "reason": rep.reason,
            "width": rep.width, "height": rep.height}


@router.post("/api/convert")
def api_convert(run_id: str = Form(...), forced: str = Form("0"), consent: str = Form("0"),
                session: str = Form("")):
    run = get_run(run_id)
    t0 = time.time()
    with WORK_LOCK:
        rgb, alpha = f2s.load_image(Path(run["path"]))
        with tempfile.TemporaryDirectory() as work:
            res = f2s.adaptive_convert(rgb, alpha, OPTIONS, Path(work))
    svg = res["svg"]
    info = f"{len(res['used'])} colors · {len(svg.encode('utf-8')) / 1024:.0f} KB · {time.time() - t0:.1f} s"
    run["svg"], run["info"] = svg, info
    log_usage("force" if forced == "1" else "convert", session, run_id=run_id,
              verdict=run["report"].get("verdict"), info=info, consent=consent == "1")
    if consent == "1":
        save_record(run_id, run["path"], svg, run["report"],
                    {"converted": True, "forced": forced == "1", "info": info, "session": session})
    return {"run_id": run_id, "info": info, "width": run["report"]["width"], "height": run["report"]["height"]}


@router.get("/api/src/{run_id}")
def api_src(run_id: str):
    run = get_run(run_id)
    return FileResponse(run["path"], headers={"Cache-Control": "no-store"})


@router.get("/api/svg/{run_id}")
def api_svg(run_id: str, download: int = 0):
    run = get_run(run_id)
    if not run["svg"]:
        raise HTTPException(404, "Not converted yet.")
    headers = {"Cache-Control": "no-store"}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{run["name"]}.svg"'
    return Response(run["svg"], media_type="image/svg+xml", headers=headers)


@router.post("/api/rate")
def api_rate(run_id: str = Form(...), rating: str = Form(...), comment: str = Form(""),
             consent: str = Form("0"), session: str = Form("")):
    if rating not in ("Excellent", "Acceptable", "Error"):
        raise HTTPException(400, "Unknown rating.")
    run = get_run(run_id)
    log_usage("rate", session, run_id=run_id, rating=rating, verdict=run["report"].get("verdict"),
              consent=consent == "1")
    if consent == "1":
        save_record(run_id, None, None, run["report"],
                    {"rating": rating, "comment": comment[:1000], "session": session})
        return {"message": f"Thank you! Your rating: {rating}."}
    return {"message": "Thank you! (Ratings are only saved when the checkbox is on.)"}


# ---------------------------------------------------------------------------
# ZeroGPU only accepts a Space that is started through Gradio's own launch(), so a tiny
# Gradio app is launched normally and our routes are put in front of Gradio's routes:
# visitors only ever see the custom page at "/".
# ---------------------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("flat2svg")

demo.launch(server_name="0.0.0.0", server_port=7860, prevent_thread_lock=True)
gradio_app = demo.app
_before = list(gradio_app.router.routes)
gradio_app.include_router(router)
_mine = gradio_app.router.routes[len(_before):]
gradio_app.router.routes[:] = _mine + _before      # our "/" and "/api/..." win over Gradio's
demo.block_thread()
