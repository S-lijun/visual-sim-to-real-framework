"""Validate sim and real RGB episode tensors.

Reads only ``obses_15fps/episode_*.pth``. Each file is one trajectory stored as
``uint8`` ``(T, H, W, 3)``. Robot states, actions, and lidar are ignored.

Run with the cvhw environment (it has PyTorch; the base env does not):

    C:\\Users\\lijun\\anaconda3\\envs\\cvhw\\python.exe validate_datasets.py
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "validation_reports"
DECLARED_FPS = 15.0
EXPECTED_HWC = (480, 640, 3)
DOMAINS = {
    "real": DATA / "real_obses_15fps" / "obses_15fps",
    "sim": DATA / "sim_obses_15fps" / "obses_15fps",
}
BLANK_STD = 1.5
STILL_MAD = 1.0
EP_RE = re.compile(r"episode_(\d+)\.pth$")


@dataclass
class EpisodeReport:
    domain: str
    episode: int
    path: str
    ok: bool
    error: str = ""
    frames: int = 0
    height: int = 0
    width: int = 0
    channels: int = 0
    dtype: str = ""
    pix_min: int | None = None
    pix_max: int | None = None
    mean_rgb: list[float] = field(default_factory=list)
    n_blank: int = 0
    n_still_pairs: int = 0
    n_pairs: int = 0
    mad_p50: float | None = None
    chromatic_rgb: list[float] = field(default_factory=list)


def episode_id(path: Path) -> int:
    match = EP_RE.search(path.name)
    if match is None:
        raise ValueError(f"Unrecognized episode file name: {path.name}")
    return int(match.group(1))


def list_episodes(folder: Path) -> list[Path]:
    files = [p for p in folder.glob("episode_*.pth") if p.is_file()]
    return sorted(files, key=episode_id)


def _quantile(values: torch.Tensor, q: float) -> float | None:
    if values.numel() == 0:
        return None
    return float(torch.quantile(values, q))


def inspect_episode(path: Path, domain: str, save_preview: Path | None) -> EpisodeReport:
    report = EpisodeReport(domain=domain, episode=episode_id(path), path=str(path), ok=False)
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - report the unreadable file and continue
        report.error = f"unreadable: {exc}"
        return report

    if not torch.is_tensor(tensor):
        report.error = f"expected a tensor, got {type(tensor).__name__}"
        return report

    report.dtype = str(tensor.dtype).replace("torch.", "")
    report.frames = int(tensor.shape[0]) if tensor.ndim >= 1 else 0
    if tensor.ndim != 4:
        report.error = f"expected 4 dims (T,H,W,C), got shape {tuple(tensor.shape)}"
        return report
    report.height, report.width, report.channels = (int(v) for v in tensor.shape[1:])
    if tuple(tensor.shape[1:]) != EXPECTED_HWC:
        report.error = f"expected HWC {EXPECTED_HWC}, got {tuple(tensor.shape[1:])}"
        return report
    if tensor.dtype != torch.uint8:
        report.error = f"expected uint8, got {tensor.dtype}"
        return report
    if tensor.shape[0] < 1:
        report.error = "episode has no frames"
        return report

    means = []
    stds = []
    pix_min = 255
    pix_max = 0
    chroma_sum = torch.zeros(3, dtype=torch.float64)
    chroma_count = 0
    for start in range(0, tensor.shape[0], 16):
        chunk = tensor[start : start + 16].to(torch.float32)
        means.append(chunk.mean(dim=(1, 2)))
        stds.append(chunk.std(dim=(1, 2), unbiased=False))
        pix_min = min(pix_min, int(chunk.min()))
        pix_max = max(pix_max, int(chunk.max()))
        flat = chunk.reshape(-1, 3)
        chroma = flat.max(dim=-1).values - flat.min(dim=-1).values
        mask = chroma > 40
        if bool(mask.any()):
            chroma_sum += flat[mask].sum(dim=0).to(torch.float64)
            chroma_count += int(mask.sum())

    mean_rgb = torch.cat(means, dim=0)
    std_rgb = torch.cat(stds, dim=0)
    frame_std = std_rgb.mean(dim=-1)
    report.n_blank = int((frame_std < BLANK_STD).sum())
    report.pix_min = pix_min
    report.pix_max = pix_max
    report.mean_rgb = [round(float(v), 4) for v in mean_rgb.mean(dim=0)]
    if chroma_count:
        report.chromatic_rgb = [round(float(v), 2) for v in (chroma_sum / chroma_count)]

    small = tensor[:, ::8, ::8].to(torch.float32).mean(dim=-1)
    if small.shape[0] > 1:
        mad = (small[1:] - small[:-1]).abs().mean(dim=(1, 2))
        report.n_pairs = int(mad.numel())
        report.n_still_pairs = int((mad < STILL_MAD).sum())
        report.mad_p50 = round(float(mad.median()), 4)

    if save_preview is not None:
        Image.fromarray(tensor[0].numpy(), mode="RGB").save(save_preview)

    if pix_min < 0 or pix_max > 255:
        report.error = f"pixel range {pix_min}..{pix_max} outside 0..255"
        return report

    report.ok = True
    return report


def summarize(domain: str, rows: list[EpisodeReport], n_files: int) -> dict:
    ok_rows = [row for row in rows if row.ok]
    frames = [row.frames for row in ok_rows]
    total = int(sum(frames))
    still = int(sum(row.n_still_pairs for row in ok_rows))
    pairs = int(sum(row.n_pairs for row in ok_rows))
    blanks = int(sum(row.n_blank for row in ok_rows))
    failures = [
        {"episode": row.episode, "file": Path(row.path).name, "error": row.error}
        for row in rows
        if not row.ok
    ]
    mean_rgb = None
    if total:
        acc = torch.zeros(3, dtype=torch.float64)
        for row in ok_rows:
            acc += torch.tensor(row.mean_rgb, dtype=torch.float64) * row.frames
        mean_rgb = [round(float(v), 4) for v in (acc / total)]
    ids = [row.episode for row in rows]
    expected = list(range(max(ids) + 1)) if ids else []
    missing = [i for i in expected if i not in set(ids)]
    return {
        "domain": domain,
        "directory": str(DOMAINS[domain]),
        "n_files": n_files,
        "n_episodes_read": len(rows),
        "n_episodes_ok": len(ok_rows),
        "episode_id_min": min(ids) if ids else None,
        "episode_id_max": max(ids) if ids else None,
        "missing_episode_ids": missing,
        "n_failures": len(failures),
        "failures": failures,
        "total_frames": total,
        "frames_min": min(frames) if frames else None,
        "frames_mean": round(total / len(ok_rows), 2) if ok_rows else None,
        "frames_max": max(frames) if frames else None,
        "resolution_hwc": list(EXPECTED_HWC),
        "dtype": "uint8",
        "layout": "(T, H, W, 3)",
        "declared_fps": DECLARED_FPS,
        "duration_sec": round(total / DECLARED_FPS, 2) if total else 0,
        "mean_rgb": mean_rgb,
        "n_blank_frames": blanks,
        "blank_std_threshold": BLANK_STD,
        "n_consecutive_pairs": pairs,
        "n_still_pairs_mad_lt_1": still,
        "still_pair_fraction": round(still / pairs, 6) if pairs else None,
        "per_frame_timestamps_stored": False,
        "notes": (
            "RGB tensors have no per-frame timestamps. "
            "Frame order inside each episode is the tensor index. "
            "Effective rate is the declared 15 fps shared by both folders."
        ),
    }


def write_csv(path: Path, rows: list[EpisodeReport]) -> None:
    fields = [
        "domain",
        "episode",
        "ok",
        "frames",
        "height",
        "width",
        "channels",
        "dtype",
        "pix_min",
        "pix_max",
        "mean_r",
        "mean_g",
        "mean_b",
        "n_blank",
        "n_still_pairs",
        "n_pairs",
        "mad_p50",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            rgb = row.mean_rgb + [None, None, None]
            writer.writerow(
                {
                    "domain": row.domain,
                    "episode": row.episode,
                    "ok": row.ok,
                    "frames": row.frames,
                    "height": row.height,
                    "width": row.width,
                    "channels": row.channels,
                    "dtype": row.dtype,
                    "pix_min": row.pix_min,
                    "pix_max": row.pix_max,
                    "mean_r": rgb[0],
                    "mean_g": rgb[1],
                    "mean_b": rgb[2],
                    "n_blank": row.n_blank,
                    "n_still_pairs": row.n_still_pairs,
                    "n_pairs": row.n_pairs,
                    "mad_p50": row.mad_p50,
                    "error": row.error,
                }
            )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows: list[EpisodeReport] = []
    summaries = []
    for domain, folder in DOMAINS.items():
        files = list_episodes(folder)
        print(f"[{domain}] {len(files)} episodes in {folder}")
        rows = []
        for index, path in enumerate(files):
            preview = OUT / f"preview_{domain}_ep0.png" if episode_id(path) == 0 else None
            row = inspect_episode(path, domain, preview)
            rows.append(row)
            if not row.ok:
                print(f"  FAIL episode {row.episode}: {row.error}")
            elif index % 20 == 0:
                print(f"  episode {row.episode}: {row.frames} frames, still {row.n_still_pairs}/{row.n_pairs}")
        all_rows.extend(rows)
        summaries.append(summarize(domain, rows, len(files)))

    passed = all(item["n_failures"] == 0 and item["n_episodes_ok"] == 157 and not item["missing_episode_ids"] for item in summaries)
    payload = {
        "passed_format_checks": passed,
        "ignored": ["states.pth", "actions.pth", "costs.pth", "traj/", "lidar"],
        "domains": summaries,
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(OUT / "episodes.csv", all_rows)
    print("passed_format_checks:", passed)
    print("wrote", OUT / "summary.json")


if __name__ == "__main__":
    main()
