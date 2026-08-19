#!/usr/bin/env python3
import os
import re
import json
from collections import defaultdict
from PIL import Image

from tqdm import tqdm
import pandas as pd

LABEL_PATH = "" # Set as the GazeCapture label path
label_file = pd.read_csv(LABEL_PATH, sep=" ")

IMAGE_DIR = "" # Set as the GazeCapture image folder path
OUT_JSON = "gazecapture_test.json"

MAKE_PER_FRAME_ANN = True

PLACEHOLDER_GAZE = [0.0, 0.0, 0.0]  # or None

# Filename pattern: 00010_00000.jpg
PATTERN = re.compile(r"^(\d+)[_-](\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)

def get_image_size(path):
    with Image.open(path) as im:
        return im.size[1], im.size[0]

def main():
    groups = defaultdict(list)

    for fname in os.listdir(IMAGE_DIR):
        m = PATTERN.match(fname)
        if not m:
            continue
        vid_str, frame_str, _ext = m.group(1), m.group(2), m.group(3)
        vid = int(vid_str)
        frame = int(frame_str)
        groups[vid].append((frame, fname))

    if not groups:
        raise SystemExit(f"No matching images found in '{IMAGE_DIR}' with pattern like 00010_00000.jpg")

    videos = []
    annotations = []

    next_video_id = 1
    next_ann_id = 1
    label_file["filename"] = label_file["Face"].str.rsplit("/", n=1).str[-1]

    for vid_key in tqdm(sorted(groups.keys())):
        frames = sorted(groups[vid_key], key=lambda x: x[0])  # sort by frame index
        first_path = os.path.join(IMAGE_DIR, frames[0][1])
        height, width = get_image_size(first_path)

        file_names = [f"{IMAGE_DIR}/{fname}" for (_frame_idx, fname) in frames]

        video_entry = {
            "height": height,
            "width": width,
            "length": len(file_names),
            "file_names": file_names,
            "id": next_video_id,
            "source_video_key": vid_key,
        }
        videos.append(video_entry)

        if MAKE_PER_FRAME_ANN:
            for i, (_frame_idx, fname) in enumerate(frames):
                gaze = label_file.loc[label_file["filename"] == fname, "3DGaze"].item().split(",")
                ann = {
                    "id": next_ann_id,
                    "video_id": next_video_id,
                    "frame_index": i,
                    "height": height,
                    "width": width,
                    "length": 1,
                    "category_id": 1,
                    "gaze": gaze,
                }
                annotations.append(ann)
                next_ann_id += 1
        else:
            ann = {
                "id": next_ann_id,
                "video_id": next_video_id,
                "height": height,
                "width": width,
                "length": len(file_names),
                "category_id": 1,
                "gaze": PLACEHOLDER_GAZE,
            }
            annotations.append(ann)
            next_ann_id += 1

        next_video_id += 1

    out = {
        "info": {
            "description": "generated_from_Image_folder",
            "version": "1",
        },
        "licenses": "only for research",
        "videos": videos,
        "annotations": annotations,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {OUT_JSON}")
    print(f"Videos: {len(videos)}")
    print(f"Annotations: {len(annotations)}")

if __name__ == "__main__":
    main()