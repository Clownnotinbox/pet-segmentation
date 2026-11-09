import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
import torch

from run import load_book


def zoom_mask(ns, net, cfg, pic, full):
    w, h = pic.size
    tw, th = max(1, round(w * 0.68)), max(1, round(h * 0.68))
    corners = [(x, y) for y in (0, h - th) for x in (0, w - tw)]
    items = [ns["pack"](pic.crop((x, y, x + tw, y + th)), None, cfg) for x, y in corners]
    x = torch.stack([item[0] for item in items]).to(ns["device"])
    probs = ns["probs"](net, x)
    zoom = np.zeros_like(full)
    size = torch.tensor([th, tw])
    for (left, top), item, p in zip(corners, items, probs):
        part = ns["restore"](p, item[2], size)
        wx = np.ones(tw, dtype=np.float32)
        wy = np.ones(th, dtype=np.float32)
        nx, ny = max(1, tw // 8), max(1, th // 8)
        if left:
            wx[:nx] = np.linspace(0.4, 1, nx)
        if left + tw < w:
            wx[-nx:] = np.linspace(1, 0.4, nx)
        if top:
            wy[:ny] = np.linspace(0.4, 1, ny)
        if top + th < h:
            wy[-ny:] = np.linspace(1, 0.4, ny)
        area = zoom[top:top+th, left:left+tw]
        np.maximum(area, part * wy[:, None] * wx, out=area)
    return np.maximum(full, 0.35 * full + 0.65 * zoom)


def masks(ns, job, part, tile=False):
    net, pack = ns["load"](job)
    rows = {}
    for x, _, boxes, sizes, names in ns["loader"](part, pack["cfg"]):
        batch = ns["probs"](net, x.to(ns["device"], non_blocking=True), pack["flip"])
        for p, box, size, name in zip(batch, boxes, sizes, names):
            full = ns["restore"](p, box, size)
            if tile and np.mean(full >= 0.5) < 0.18:
                if part == "stress":
                    pic, _ = ns["stress_pair"](name)
                elif part == "test":
                    pic = ns["read_pic"](ns["data"] / "test" / f"{name}.png")
                else:
                    pic, _ = ns["pair"](name)
                full = zoom_mask(ns, net, pack["cfg"], pic, full)
            rows[name] = full
    print(job, part, len(rows), flush=True)
    return rows


def labels(ns, part):
    if part == "stress":
        return {name: ns["target"](np.array(ns["stress_pair"](name)[1])) for name in ns["stress_ids"]}
    return {
        name: ns["target"](np.array(ns["read_mask"](name)))
        for name in ns["split"][part]
    }


def average(pics, jobs, name):
    out = pics[jobs[0]][name].copy()
    for job in jobs[1:]:
        out += pics[job][name]
    return out / len(jobs)


def weight_hash(job):
    with (Path("runs") / job / "best.pt").open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def select(ns, jobs):
    pics = {job: masks(ns, job, "val") for job in jobs}
    y = labels(ns, "val")
    rows = []
    thresholds = [0.5, 0.45, 0.55, 0.4, 0.6, 0.35, 0.65]
    for n in range(1, len(jobs) + 1):
        for group in itertools.combinations(jobs, n):
            scores = np.zeros(len(thresholds))
            for name, true in y.items():
                p = average(pics, group, name)
                for i, t in enumerate(thresholds):
                    scores[i] += ns["iou"](p >= t, true)
            for t, s in zip(thresholds, scores / len(y)):
                rows.append((group, t, float(s)))
    group, t, val = max(rows, key=lambda row: row[-1])
    plan = dict(jobs=list(group), threshold=t, val_iou=val, split=ns["split_id"],
                weights={job: weight_hash(job) for job in group})
    with Path("mix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["models", "threshold", "eval_iou"])
        writer.writerows(("+".join(g), t, score) for g, t, score in rows)
    Path("mix.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(plan, flush=True)


def robust(ns, jobs):
    y = {part: labels(ns, part) for part in ("val", "stress")}
    small = set(sorted(y["val"], key=lambda name: y["val"][name].mean())[:148])
    rows = []
    groups = [g for n in (1, 2, 3, len(jobs)) for g in itertools.combinations(jobs, n)]
    groups = list(dict.fromkeys(groups))
    levels = [0.5, 0.45, 0.55, 0.4, 0.6]
    for tile in (False, True):
        pics = {part: {job: masks(ns, job, part, tile) for job in jobs} for part in y}
        for group in groups:
            scores = {}
            small_scores = np.zeros(len(levels))
            for part, targets in y.items():
                values = np.zeros(len(levels))
                for name, true in targets.items():
                    p = average(pics[part], group, name)
                    for i, t in enumerate(levels):
                        value = ns["iou"](p >= t, true)
                        values[i] += value
                        if part == "val" and name in small:
                            small_scores[i] += value
                scores[part] = values / len(targets)
            for t, val, low, hard in zip(levels, scores["val"], small_scores / len(small), scores["stress"]):
                score = 0.45 * val + 0.35 * low + 0.20 * hard
                rows.append((group, t, float(val), float(low), float(hard), tile, float(score)))
        del pics
    group, t, val, low, hard, tile, score = max(rows, key=lambda row: row[-1])
    plan = dict(jobs=list(group), threshold=t, val_iou=val, small_iou=low, stress_iou=hard,
                tile=tile, score=score, split=ns["split_id"],
                weights={job: weight_hash(job) for job in group})
    with Path("mix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["models", "threshold", "eval_iou", "small_iou", "stress_iou", "tile", "score"])
        writer.writerows(("+".join(g), t, v, low, h, tile, s) for g, t, v, low, h, tile, s in rows)
    Path("mix.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(plan, flush=True)


def final(ns, part, out=None, plan_path="mix.json"):
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    if plan["split"] != ns["split_id"]:
        raise ValueError("Не совпало разбиение")
    jobs = plan["jobs"]
    if any(weight_hash(job) != plan["weights"][job] for job in jobs):
        raise ValueError("Веса отличаются от выбранных на eval")
    t = plan["threshold"]
    pics = {job: masks(ns, job, part, plan.get("tile", False)) for job in jobs}
    if part == "hold":
        y = labels(ns, part)
        values = [ns["iou"](average(pics, jobs, name) >= t, true) for name, true in y.items()]
        result = dict(plan, hold_iou=float(np.mean(values)), count=len(values))
        Path(out or "hold.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(result, flush=True)
    else:
        ids = ns["split"]["test"]
        if any(set(pics[job]) != {str(i) for i in ids} for job in jobs):
            raise ValueError("Не совпали ID теста")
        with Path(out or "submission.csv").open("x", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ImageID", "RLE"])
            for i in ids:
                p = average(pics, jobs, str(i))
                writer.writerow([i, ns["rle_encode"](p >= t)])
        print("Строк в CSV:", len(ids), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=["val", "robust", "hold", "test"])
    parser.add_argument("jobs", nargs="*")
    parser.add_argument("--out")
    parser.add_argument("--plan", default="mix.json")
    args = parser.parse_args()
    ns = load_book(workers=0 if os.name == "nt" else 5)
    ns["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.part in ("val", "robust"):
        if not args.jobs:
            parser.error("Укажи модели для сравнения")
        if args.part == "robust":
            robust(ns, args.jobs)
        else:
            select(ns, args.jobs)
    else:
        final(ns, args.part, args.out, args.plan)
