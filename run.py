import argparse
import json
from pathlib import Path


def load_book(job="res320", stage="idle", workers=5):
    book = json.loads(Path("pet.ipynb").read_text(encoding="utf-8"))
    ns = {"__name__": "__main__"}
    for cell in book["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if src.startswith(("ids = ", "test_cfg = ", "fig, ax = ", "export = ")):
            continue
        if src.startswith("import csv"):
            src = src.replace('job = "res320"', f"job = {job!r}", 1)
            src = src.replace("train = False", f"train = {stage == 'train'}", 1)
            src = src.replace("check_hold = False", f"check_hold = {stage in ('hold', 'both')}", 1)
            src = src.replace("predict = False", f"predict = {stage in ('csv', 'both')}", 1)
            src = src.replace("workers = 0", f"workers = {workers}", 1)
            src += "\ntorch.backends.cudnn.benchmark = True\n"
        exec(compile(src, "pet.ipynb", "exec"), ns)
    return ns


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("job")
    parser.add_argument("stage", choices=["train", "csv", "hold", "both"])
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    load_book(args.job, args.stage, args.workers)
