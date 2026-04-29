#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageSequence, ImageDraw


def list_gifs_in_folder(folder: Path) -> List[Path]:
    gifs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".gif"]

    def key(p: Path):
        try:
            return (0, int(p.stem))
        except ValueError:
            return (1, p.stem.lower())

    return sorted(gifs, key=key)


def load_middle_third_frames(gif_path: Path) -> Tuple[List[Image.Image], List[int]]:
    img = Image.open(gif_path)
    frames: List[Image.Image] = []
    durations: List[int] = []

    for frame in ImageSequence.Iterator(img):
        rgba = frame.convert("RGBA")
        w, h = rgba.size
        x0 = w // 3
        x1 = (2 * w) // 3
        rgba = rgba.crop((x0, 0, x1, h))

        dur = frame.info.get("duration", img.info.get("duration", 80))
        if not isinstance(dur, int) or dur <= 0:
            dur = 80

        frames.append(rgba)
        durations.append(dur)

    if not frames:
        raise ValueError(f"No frames found in {gif_path}")

    return frames, durations


def resize_to_height(img: Image.Image, target_h: int) -> Image.Image:
    w, h = img.size
    if h == target_h:
        return img
    new_w = max(1, round(w * target_h / h))
    return img.resize((new_w, target_h), Image.Resampling.LANCZOS)


def expand_by_duration(
    frames: List[Image.Image],
    durations: List[int],
    target_ms: int,
) -> List[Image.Image]:
    target_ms = max(20, int(target_ms))
    out: List[Image.Image] = []
    for frame, dur in zip(frames, durations):
        repeats = max(1, int(round(dur / target_ms)))
        out.extend([frame] * repeats)
    return out


def prepare_task_gif(gif_path: Path, task_height: int, target_ms: int) -> List[Image.Image]:
    frames, durations = load_middle_third_frames(gif_path)
    frames = [resize_to_height(f, task_height) for f in frames]
    frames = expand_by_duration(frames, durations, target_ms)
    return frames


def load_suite(folder: Path, task_height: int, target_ms: int) -> List[List[Image.Image]]:
    gif_paths = list_gifs_in_folder(folder)
    if not gif_paths:
        raise ValueError(f"No GIFs found in {folder}")
    return [prepare_task_gif(p, task_height, target_ms) for p in gif_paths]


def row_colors() -> List[Tuple[int, int, int]]:
    return [
        (230, 57, 70),
        (29, 53, 87),
        (69, 123, 157),
        (42, 157, 143),
        (244, 162, 97),
        (142, 68, 173),
        (231, 111, 81),
        (38, 70, 83),
    ]


def compose_frames(
    suites: List[List[List[Image.Image]]],
    task_height: int,
    separator_h: int,
    left_pad: int,
    right_pad: int,
    top_pad: int,
    bottom_pad: int,
) -> List[Image.Image]:
    gap = 0  # no padding whatsoever inside each suite row

    row_heights: List[int] = []
    row_widths: List[int] = []
    max_frames = 1

    for suite in suites:
        row_heights.append(task_height)
        row_width = 0
        for task in suite:
            if task:
                row_width += task[0].width
                max_frames = max(max_frames, len(task))
        row_width += gap * max(0, len(suite) - 1)
        row_widths.append(row_width)

    canvas_w = left_pad + max(row_widths) + right_pad
    canvas_h = top_pad + sum(row_heights) + separator_h * (len(suites) - 1) + bottom_pad

    row_y_positions: List[int] = []
    y = top_pad
    for idx in range(len(suites)):
        row_y_positions.append(y)
        y += row_heights[idx]
        if idx < len(suites) - 1:
            y += separator_h

    colors = row_colors()
    frames_out: List[Image.Image] = []

    for t in range(max_frames):
        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

        for suite_idx, suite in enumerate(suites):
            y0 = row_y_positions[suite_idx]
            x = left_pad

            for task in suite:
                frame = task[t % len(task)]
                canvas.alpha_composite(frame, (x, y0))
                x += frame.width  # directly adjacent, no gap

            if suite_idx < len(suites) - 1 and separator_h > 0:
                color = colors[suite_idx % len(colors)]
                sep_y = y0 + task_height
                draw = ImageDraw.Draw(canvas)
                draw.rectangle(
                    [left_pad, sep_y, canvas_w - right_pad - 1, sep_y + separator_h - 1],
                    fill=color + (255,),
                )

        frames_out.append(canvas)

    return frames_out


def save_gif(frames: List[Image.Image], output_path: Path, frame_ms: int) -> None:
    palette_frames = []

    for i, frame in enumerate(frames):
        # Convert RGBA → P with adaptive palette
        p = frame.convert("P", palette=Image.ADAPTIVE, colors=255)

        # Create transparency mask
        alpha = frame.getchannel("A")
        mask = Image.eval(alpha, lambda a: 255 if a <= 128 else 0)

        # Use last palette index as transparent
        p.paste(255, mask)

        p.info["transparency"] = 255
        p.info["disposal"] = 2

        palette_frames.append(p)

    palette_frames[0].save(
        output_path,
        save_all=True,
        append_images=palette_frames[1:],
        duration=frame_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def save_webp(frames: List[Image.Image], output_path: Path, frame_ms: int) -> None:
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_ms,
        loop=0,
        lossless=True,
        method=6,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("folders", nargs=6, type=Path)
    parser.add_argument("--task-height", type=int, default=180)
    parser.add_argument("--separator-height", type=int, default=8)
    parser.add_argument("--left-pad", type=int, default=0)
    parser.add_argument("--right-pad", type=int, default=0)
    parser.add_argument("--top-pad", type=int, default=0)
    parser.add_argument("--bottom-pad", type=int, default=0)
    parser.add_argument("--frame-ms", type=int, default=80)
    args = parser.parse_args()

    suites = [load_suite(folder, args.task_height, args.frame_ms) for folder in args.folders]

    frames = compose_frames(
        suites=suites,
        task_height=args.task_height,
        separator_h=args.separator_height,
        left_pad=args.left_pad,
        right_pad=args.right_pad,
        top_pad=args.top_pad,
        bottom_pad=args.bottom_pad,
    )

    suffix = args.output.suffix.lower()
    if suffix == ".webp":
        save_webp(frames, args.output, args.frame_ms)
    else:
        save_gif(frames, args.output, args.frame_ms)


if __name__ == "__main__":
    main()