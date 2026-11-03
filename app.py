"""
Public Hugging Face Space for flat2svg.

The vectorizer source code is NOT stored in this Space. At startup the app downloads it
from a private Hugging Face repo (CORE_REPO) using the HF_TOKEN secret, so visitors only
see this thin interface.

Flow:
  1. The user uploads a PNG/JPG.
  2. The flatness detector checks whether the image fits flat2svg (even fills, sharp edges).
  3. Flat images are converted at once; other images get an explanation and a
     "Convert anyway" button.
  4. The user rates the result (Excellent / Acceptable / Error).
  5. With the user's consent, the source, the SVG, the detector report and the rating
     are saved to a private dataset (LOG_REPO) for quality analysis.

Space secrets / variables:
  HF_TOKEN   - token with read access to CORE_REPO and write access to LOG_REPO
  CORE_REPO  - private repo with flat2svg.py, vecmath.py, denoise_flat.py, flatness_detector.py
  LOG_REPO   - private dataset repo for uploads and ratings
"""

import json
import os
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
from huggingface_hub import CommitScheduler, snapshot_download

HF_TOKEN = os.environ.get("HF_TOKEN")
CORE_REPO = os.environ.get("CORE_REPO", "ElenaKu/flat2svg-core")
LOG_REPO = os.environ.get("LOG_REPO", "ElenaKu/flat2svg-feedback")
MAX_MB = 10
MAX_SIDE = 4000

# ---------------------------------------------------------------------------
# Load the private vectorizer code
# ---------------------------------------------------------------------------
core_dir = snapshot_download(CORE_REPO, token=HF_TOKEN, repo_type="model")
sys.path.insert(0, core_dir)
import flat2svg as f2s  # noqa: E402
import flatness_detector as fd  # noqa: E402

OPTIONS = {**f2s.DEFAULTS, "jobs": 1}

# ---------------------------------------------------------------------------
# Logging to a private dataset (pushed in the background every few minutes)
# ---------------------------------------------------------------------------
LOG_DIR = Path("feedback_data")
LOG_DIR.mkdir(exist_ok=True)
scheduler = None
if HF_TOKEN:
    scheduler = CommitScheduler(repo_id=LOG_REPO, repo_type="dataset", folder_path=LOG_DIR,
                                path_in_repo="data", every=5, token=HF_TOKEN, private=True)


def save_record(run_id, src_path, svg_text, report, extra):
    """Write the source, the SVG and a JSON record into data/<date>/<run_id>/."""
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    folder = LOG_DIR / day / run_id
    lock = scheduler.lock if scheduler else _NoLock()
    with lock:
        folder.mkdir(parents=True, exist_ok=True)
        if src_path and not (folder / ("source" + Path(src_path).suffix.lower())).exists():
            (folder / ("source" + Path(src_path).suffix.lower())).write_bytes(Path(src_path).read_bytes())
        if svg_text:
            (folder / "result.svg").write_text(svg_text, encoding="utf-8")
        meta_path = folder / "record.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta.update({"run_id": run_id, "time": datetime.now(timezone.utc).isoformat(), "report": report, **extra})
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


class _NoLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
def check_upload(path):
    if path is None:
        raise gr.Error("Please upload an image.")
    if Path(path).stat().st_size > MAX_MB * 1024 * 1024:
        raise gr.Error(f"File is larger than {MAX_MB} MB.")


def detect(path):
    rep = fd.analyze(Path(path))
    if max(rep.width, rep.height) > MAX_SIDE:
        raise gr.Error(f"Image is larger than {MAX_SIDE} px on the long side.")
    return rep


def convert(path):
    rgb, alpha = f2s.load_image(Path(path))
    with tempfile.TemporaryDirectory() as work:
        t0 = time.time()
        res = f2s.adaptive_convert(rgb, alpha, OPTIONS, Path(work))
    out = Path(tempfile.mkdtemp()) / (Path(path).stem + ".svg")
    out.write_text(res["svg"], encoding="utf-8")
    info = f"Colors: {len(res['used'])}, time: {time.time() - t0:.1f} s, size: {out.stat().st_size / 1024:.0f} KB"
    return res["svg"], str(out), info


VERDICT_TEXT = {
    "flat": "Flat illustration - suitable for flat2svg.",
    "shaded": "This image has gradients or many shades. flat2svg works best with even fills; "
              "the result may look posterized.",
    "line_art": "This image has outlines or hatching. flat2svg works best with flat fills; "
                "a tool for detailed illustrations is coming later.",
    "line_art_dark": "Detailed illustration on a dark background - not supported yet.",
}


def on_upload(path, consent):
    """Run the detector; convert at once only if the image is flat."""
    check_upload(path)
    run_id = uuid.uuid4().hex[:12]
    rep = detect(path)
    report = rep.__dict__ if hasattr(rep, "__dict__") else dict(rep)
    msg = f"**{VERDICT_TEXT.get(rep.verdict, rep.verdict)}**\n\n<small>{rep.reason}</small>"
    if rep.verdict != "flat":
        if consent:
            save_record(run_id, path, None, report, {"converted": False})
        return (msg, None, None, "", gr.update(visible=True), gr.update(visible=False), run_id, report)
    svg, svg_file, info = convert(path)
    if consent:
        save_record(run_id, path, svg, report, {"converted": True, "forced": False, "info": info})
    return (msg, svg_html(svg), svg_file, info, gr.update(visible=False), gr.update(visible=True), run_id, report)


def on_force(path, consent, run_id, report):
    svg, svg_file, info = convert(path)
    if consent:
        save_record(run_id, path, svg, report, {"converted": True, "forced": True, "info": info})
    return svg_html(svg), svg_file, info, gr.update(visible=False), gr.update(visible=True)


def on_rate(rating, comment, consent, run_id, report):
    if not run_id:
        return "Nothing to rate yet."
    if consent:
        save_record(run_id, None, None, report, {"rating": rating, "comment": comment})
        return f"Thank you! Your rating: {rating}."
    return "Thank you! (Ratings are only saved when the checkbox above is on.)"


def svg_html(svg):
    return f'<div style="background:#fff;border:1px solid #ddd;padding:8px;max-height:70vh;overflow:auto">{svg}</div>'


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
with gr.Blocks(title="flat2svg - flat PNG to SVG") as demo:
    gr.Markdown("# flat2svg\nFree vectorizer for **flat illustrations**: even color fills, sharp edges, "
                "no gradients. Upload a PNG or JPG and download the SVG.")
    consent = gr.Checkbox(value=True, label="Allow saving my image, the SVG and my rating to help improve "
                                            "the tool (stored privately, never shared or sold)")
    run_id = gr.State("")
    report = gr.State({})
    with gr.Row():
        with gr.Column():
            src = gr.Image(type="filepath", label=f"Source image (up to {MAX_MB} MB, {MAX_SIDE} px)",
                           sources=["upload"])
            go = gr.Button("Check and vectorize", variant="primary")
            verdict = gr.Markdown()
            force = gr.Button("Convert anyway", visible=False)
        with gr.Column():
            preview = gr.HTML(label="SVG preview")
            svg_file = gr.File(label="Download SVG")
            info = gr.Markdown()
    with gr.Group(visible=False) as rate_box:
        gr.Markdown("### How good is the result?")
        comment = gr.Textbox(label="Comment (optional)", placeholder="What went wrong or right?")
        with gr.Row():
            b_ok = gr.Button("Excellent")
            b_mid = gr.Button("Acceptable")
            b_err = gr.Button("Error")
        thanks = gr.Markdown()

    go.click(on_upload, [src, consent], [verdict, preview, svg_file, info, force, rate_box, run_id, report])
    force.click(on_force, [src, consent, run_id, report], [preview, svg_file, info, force, rate_box])
    for btn, label in ((b_ok, "Excellent"), (b_mid, "Acceptable"), (b_err, "Error")):
        btn.click(lambda c, cs, r, rp, lab=label: on_rate(lab, c, cs, r, rp),
                  [comment, consent, run_id, report], thanks)

demo.queue(default_concurrency_limit=1, max_size=20)

if __name__ == "__main__":
    demo.launch()
