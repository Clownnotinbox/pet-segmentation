import argparse
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
from PIL import Image


ANN = "train/data/annotations/annotations/"
IMG = "train/data/images/images/"
TEST = "test/test/images/"


def image_key(pic):
    data = str(pic.size).encode() + pic.tobytes()
    return hashlib.sha256(data).hexdigest()


def split_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["breed"]].append(row["id"])
    rng = random.Random(42)
    split = {"train": [], "val": [], "hold": []}
    for breed in sorted(groups):
        ids = sorted(groups[breed])
        rng.shuffle(ids)
        n = max(1, round(len(ids) * 0.10))
        split["val"].extend(ids[:n])
        split["hold"].extend(ids[n:2*n])
        split["train"].extend(ids[2*n:])
    return split


def prepare(source, out):
    out = Path(out)
    split_path = out / "split.json"
    if split_path.exists():
        raise FileExistsError(f"{split_path} уже существует")
    for folder in ("img", "mask", "test"):
        (out / folder).mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = {}
    flags = defaultdict(list)
    mask_values = Counter()
    img_modes = Counter()
    test_sizes = Counter()
    with ZipFile(source) as z:
        names = set(z.namelist())
        labels = {}
        for part in ("trainval", "test"):
            for line in z.read(f"{ANN}{part}.txt").decode().splitlines():
                if not line or line.startswith("#"):
                    continue
                name, breed, species, _ = line.split()
                if name in labels:
                    raise ValueError(f"Повтор в списках Oxford: {name}")
                labels[name] = (int(breed), int(species), part)
        jpg = [n for n in names if n.startswith(IMG) and n.endswith(".jpg")]
        png = [n for n in names if n.startswith(ANN + "trimaps/") and n.endswith(".png")]
        mac = [n for n in png if PurePosixPath(n).name.startswith("._")]
        test_names = sorted(
            (n for n in names if n.startswith(TEST) and n.endswith(".png")),
            key=lambda n: int(PurePosixPath(n).stem),
        )
        keys = set()
        test_ids = []
        for path in test_names:
            name = PurePosixPath(path).stem
            raw = z.read(path)
            with Image.open(io.BytesIO(raw)) as pic:
                keys.add(image_key(pic.convert("RGB")))
                w, h = pic.size
                test_sizes[f"{w}x{h}"] += 1
            test_ids.append(int(name))
            (out / "test" / f"{name}.png").write_bytes(raw)
        if test_ids != list(range(len(test_ids))) or not test_ids:
            raise ValueError("Неполный набор ID теста Kaggle")

        groups = defaultdict(list)
        for i, (name, (breed, species, part)) in enumerate(labels.items(), 1):
            if Path(name).name != name or name in (".", ".."):
                raise ValueError(f"Некорректное имя: {name}")
            img_path = f"{IMG}{name}.jpg"
            mask_path = f"{ANN}trimaps/{name}.png"
            if img_path not in names or mask_path not in names:
                skipped[name] = "missing"
                continue
            try:
                raw = z.read(img_path)
                with Image.open(io.BytesIO(raw)) as pic:
                    mode = pic.mode
                    rgb = pic.convert("RGB")
                with Image.open(io.BytesIO(z.read(mask_path))) as pic:
                    mask = np.array(pic)
            except (OSError, ValueError) as err:
                skipped[name] = f"decode: {err}"
                continue
            if mask.shape != (rgb.height, rgb.width):
                skipped[name] = "shape"
                continue
            values = set(np.unique(mask).tolist())
            mask_values.update(values)
            img_modes[mode] += 1
            if not values <= {1, 2, 3}:
                skipped[name] = f"values: {sorted(values)}"
                continue
            if 1 not in values:
                skipped[name] = "no_pet"
                continue
            if 2 not in values:
                flags["no_background"].append(name)
            key = image_key(rgb)
            if key in keys:
                skipped[name] = "kaggle_duplicate"
                continue
            groups[key].append({
                "id": name, "breed": breed, "species": species, "part": part,
                "mask": hashlib.sha256(mask.tobytes()).hexdigest(),
            })
            if i % 2000 == 0:
                print(f"Проверено пар: {i}/{len(labels)}", flush=True)

        for group in groups.values():
            if len({row["mask"] for row in group}) > 1:
                for row in group:
                    skipped[row["id"]] = "conflicting_duplicate"
                continue
            row = group[0]
            for other in group[1:]:
                skipped[other["id"]] = "duplicate"
            name = row["id"]
            (out / "img" / f"{name}.jpg").write_bytes(z.read(f"{IMG}{name}.jpg"))
            (out / "mask" / f"{name}.png").write_bytes(z.read(f"{ANN}trimaps/{name}.png"))
            rows.append(row)

    split = split_rows(rows)
    split["test"] = test_ids
    audit = {
        "oxford": len(labels),
        "official": dict(Counter(v[2] for v in labels.values())),
        "jpg": len(jpg),
        "unlisted_jpg": len(jpg) - sum(f"{IMG}{n}.jpg" in names for n in labels),
        "mac_files": len(mac),
        "breeds": len({row["breed"] for row in rows}),
        "sizes": {k: len(v) for k, v in split.items()},
        "test_shapes": dict(test_sizes),
        "image_modes": dict(img_modes),
        "mask_values": dict(mask_values),
        "skipped": skipped,
        "flags": dict(flags),
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    split_path.write_text(json.dumps(split, ensure_ascii=False), encoding="utf-8")
    print(audit["sizes"], "пропущено:", len(skipped), flush=True)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()
    prepare(args.source, args.out)
