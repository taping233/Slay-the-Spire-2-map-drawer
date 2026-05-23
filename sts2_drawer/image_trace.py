from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageOps

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover - optional fallback for minimal installs.
    cv2 = None
    np = None


@dataclass(frozen=True)
class Stroke:
    """A single drag stroke in canvas coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    def to_dict(self) -> dict:
        return asdict(self)


def parse_canvas_size(value: str) -> Tuple[int, int]:
    cleaned = value.lower().replace(" ", "")
    if "x" not in cleaned:
        raise ValueError("Canvas size must look like WIDTHxHEIGHT, for example 640x360.")
    width_text, height_text = cleaned.split("x", 1)
    width = int(width_text)
    height = int(height_text)
    if width <= 0 or height <= 0:
        raise ValueError("Canvas width and height must be positive.")
    return width, height


def load_mask(
    image_path: Path,
    canvas_size: Tuple[int, int],
    threshold: int,
    invert: bool,
) -> Image.Image:
    """Return a 1-bit mask where black pixels are the parts to draw."""

    canvas = load_luminance_canvas(image_path, canvas_size, invert)
    return canvas.point(lambda pixel: 0 if pixel < threshold else 255, mode="1")


def load_luminance_canvas(
    image_path: Path,
    canvas_size: Tuple[int, int],
    invert: bool,
) -> Image.Image:
    """Return a centered grayscale canvas while preserving source brightness."""

    image = Image.open(image_path).convert("L")
    image = ImageOps.contain(image, canvas_size, method=Image.Resampling.LANCZOS)

    canvas = Image.new("L", canvas_size, 255)
    offset = ((canvas_size[0] - image.width) // 2, (canvas_size[1] - image.height) // 2)
    canvas.paste(image, offset)
    canvas = ImageOps.autocontrast(canvas)

    if invert:
        canvas = ImageOps.invert(canvas)

    return canvas


def analyze_image_parameters(
    image_path: Path,
    canvas_size: Tuple[int, int],
    invert: bool,
) -> dict:
    luminance = load_luminance_canvas(image_path, canvas_size, invert)
    width, height = luminance.size

    if np is not None:
        pixels = np.array(luminance, dtype=np.uint8)
        mean_luminance = float(pixels.mean())
        contrast_std = float(pixels.std())
        dark_ratio = float((pixels < 170).mean())
        mid_dark_ratio = float((pixels < 135).mean())
        if cv2 is not None:
            edges = adaptive_canny_edges(pixels, 170)
            edge_density = float((edges > 0).mean())
        else:
            edge_density = estimate_edge_density_pil(luminance)
    else:
        histogram = luminance.histogram()
        total = max(1, width * height)
        mean_luminance = sum(value * count for value, count in enumerate(histogram)) / total
        contrast_std = estimate_histogram_std(histogram, mean_luminance, total)
        dark_ratio = sum(histogram[:170]) / total
        mid_dark_ratio = sum(histogram[:135]) / total
        edge_density = estimate_edge_density_pil(luminance)

    complexity = min(1.0, edge_density * 4.6 + contrast_std / 128.0 + dark_ratio * 0.65)
    suggested_mode = "handdrawn" if complexity >= 0.34 else "lines"
    if dark_ratio >= 0.34 and edge_density < 0.08:
        suggested_mode = "scanline"

    threshold = clamp(round(160 + (128 - mean_luminance) * 0.46 + complexity * 28 - dark_ratio * 16), 110, 225)
    step = clamp(round(6 - complexity * 10), 2, 8)
    min_run = clamp(round(5 - complexity * 5 - edge_density * 8), 2, 8)
    line_count = clamp(round(260 + complexity * 2350 + dark_ratio * 1000 + mid_dark_ratio * 520), 160, 5000)
    move_duration = 0.006 if line_count <= 700 else 0.008 if line_count <= 1500 else 0.01
    jump_duration = 0.0
    pause_duration = 0.0 if edge_density < 0.15 else 0.003

    return {
        "canvas_size": canvas_size,
        "mean_luminance": mean_luminance,
        "contrast_std": contrast_std,
        "dark_ratio": dark_ratio,
        "mid_dark_ratio": mid_dark_ratio,
        "edge_density": edge_density,
        "complexity": complexity,
        "suggested_mode": suggested_mode,
        "threshold": threshold,
        "step": step,
        "min_run": min_run,
        "line_count": line_count,
        "move_duration": move_duration,
        "jump_duration": jump_duration,
        "pause_duration": pause_duration,
    }


def estimate_histogram_std(histogram: List[int], mean_luminance: float, total: int) -> float:
    variance = sum(((value - mean_luminance) ** 2) * count for value, count in enumerate(histogram)) / max(total, 1)
    return math.sqrt(max(0.0, variance))


def estimate_edge_density_pil(luminance: Image.Image) -> float:
    edges = ImageOps.autocontrast(luminance.filter(ImageFilter.FIND_EDGES))
    histogram = edges.histogram()
    total = max(1, edges.width * edges.height)
    return sum(histogram[48:]) / total


def trace_scanline_strokes(mask: Image.Image, step: int, min_run: int) -> List[Stroke]:
    """Convert a mask to horizontal drag strokes.

    This intentionally favors predictability over artistic vectorization. A map-screen
    brush usually tolerates slow scanlines better than complex contour chasing.
    """

    if step <= 0:
        raise ValueError("Step must be at least 1.")
    if min_run <= 0:
        raise ValueError("Minimum run must be at least 1.")

    width, height = mask.size
    pixels = mask.load()
    strokes: List[Stroke] = []

    for y in range(0, height, step):
        runs: List[Tuple[int, int]] = []
        start = None

        for x in range(width):
            should_draw = pixels[x, y] == 0
            if should_draw and start is None:
                start = x
            elif not should_draw and start is not None:
                end = x - 1
                if end - start + 1 >= min_run:
                    runs.append((start, end))
                start = None

        if start is not None:
            end = width - 1
            if end - start + 1 >= min_run:
                runs.append((start, end))

        if (y // step) % 2 == 1:
            runs.reverse()
            strokes.extend(Stroke(end, y, start, y) for start, end in runs)
        else:
            strokes.extend(Stroke(start, y, end, y) for start, end in runs)

    return strokes


def trace_outline_strokes(mask: Image.Image, step: int, min_run: int) -> List[Stroke]:
    """Trace mask boundaries as short horizontal and vertical strokes."""

    edges = build_boundary_mask(mask)
    outline_step = max(1, step // 2)
    strokes = trace_scanline_strokes(edges, outline_step, min_run)
    strokes.extend(trace_vertical_strokes(edges, outline_step, min_run))
    return strokes


def build_boundary_mask(mask: Image.Image) -> Image.Image:
    width, height = mask.size
    pixels = mask.load()
    edges = Image.new("1", mask.size, 255)
    edge_pixels = edges.load()

    for y in range(height):
        for x in range(width):
            if pixels[x, y] != 0:
                continue
            touches_background = False
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if nx < 0 or ny < 0 or nx >= width or ny >= height or pixels[nx, ny] != 0:
                    touches_background = True
                    break
            if touches_background:
                edge_pixels[x, y] = 0
    return edges


def trace_vertical_strokes(mask: Image.Image, step: int, min_run: int) -> List[Stroke]:
    if step <= 0:
        raise ValueError("Step must be at least 1.")
    if min_run <= 0:
        raise ValueError("Minimum run must be at least 1.")

    width, height = mask.size
    pixels = mask.load()
    strokes: List[Stroke] = []

    for x in range(0, width, step):
        runs: List[Tuple[int, int]] = []
        start: Optional[int] = None

        for y in range(height):
            should_draw = pixels[x, y] == 0
            if should_draw and start is None:
                start = y
            elif not should_draw and start is not None:
                end = y - 1
                if end - start + 1 >= min_run:
                    runs.append((start, end))
                start = None

        if start is not None:
            end = height - 1
            if end - start + 1 >= min_run:
                runs.append((start, end))

        if (x // step) % 2 == 1:
            runs.reverse()
            strokes.extend(Stroke(x, end, x, start) for start, end in runs)
        else:
            strokes.extend(Stroke(x, start, x, end) for start, end in runs)

    return strokes


def trace_diagonal_hatch_strokes(
    mask: Image.Image,
    step: int,
    min_run: int,
    direction: int = 1,
) -> List[Stroke]:
    """Create diagonal hatching inside the mask."""

    if step <= 0:
        raise ValueError("Step must be at least 1.")
    if min_run <= 0:
        raise ValueError("Minimum run must be at least 1.")
    if direction not in (1, -1):
        raise ValueError("Direction must be 1 or -1.")

    width, height = mask.size
    pixels = mask.load()
    strokes: List[Stroke] = []
    hatch_step = max(step * 3, min_run)

    offsets = range(-height, width + height, hatch_step)
    for line_index, offset in enumerate(offsets):
        run_start: Optional[Tuple[int, int]] = None
        last_point: Optional[Tuple[int, int]] = None

        for y in range(height):
            x = offset + y if direction == 1 else offset + (height - 1 - y)
            if x < 0 or x >= width:
                if run_start is not None and last_point is not None:
                    append_diagonal_run(strokes, run_start, last_point, min_run, line_index)
                run_start = None
                last_point = None
                continue

            should_draw = pixels[x, y] == 0
            if should_draw:
                if run_start is None:
                    run_start = (x, y)
                last_point = (x, y)
            elif run_start is not None and last_point is not None:
                append_diagonal_run(strokes, run_start, last_point, min_run, line_index)
                run_start = None
                last_point = None

        if run_start is not None and last_point is not None:
            append_diagonal_run(strokes, run_start, last_point, min_run, line_index)

    return strokes


def append_diagonal_run(
    strokes: List[Stroke],
    start: Tuple[int, int],
    end: Tuple[int, int],
    min_run: int,
    line_index: int,
) -> None:
    length = max(abs(end[0] - start[0]), abs(end[1] - start[1])) + 1
    if length < min_run:
        return
    if line_index % 2 == 1:
        start, end = end, start
    strokes.append(Stroke(start[0], start[1], end[0], end[1]))


def trace_pixel_strokes(
    luminance: Image.Image,
    pixel_size: int,
    threshold: int,
    min_run: int,
    density: float,
) -> List[Stroke]:
    """Redraw an image as deterministic dot-matrix marks based on local brightness."""

    if pixel_size <= 0:
        raise ValueError("Pixel size must be at least 1.")
    if min_run <= 0:
        raise ValueError("Minimum run must be at least 1.")
    if density <= 0:
        raise ValueError("Density must be greater than 0.")

    width, height = luminance.size
    strokes: List[Stroke] = []
    cell = max(2, pixel_size)
    mark_length = max(1, min_run, round(cell * 0.45))
    pixels = luminance.load()

    for row_index, y in enumerate(range(0, height, cell)):
        columns = list(range(0, width, cell))
        if row_index % 2 == 1:
            columns.reverse()

        for x in columns:
            average = average_cell_luminance(pixels, x, y, cell, width, height)
            if average >= threshold:
                continue

            darkness = (threshold - average) / max(threshold, 1)
            marks = min(4, max(1, math.ceil(darkness * 3.0 * density)))
            if darkness < 0.08 and density <= 1:
                continue

            for mark_index in range(marks):
                seed = (x * 73856093) ^ (y * 19349663) ^ (mark_index * 83492791)
                rng = random.Random(seed)
                center_x = x + cell * (0.25 + rng.random() * 0.5)
                center_y = y + cell * (0.25 + rng.random() * 0.5)
                angle = rng.choice((0.0, math.pi / 4, -math.pi / 4, math.pi / 2))
                length = mark_length * (0.65 + rng.random() * 0.55)

                dx = math.cos(angle) * length / 2
                dy = math.sin(angle) * length / 2
                x1 = clamp(round(center_x - dx), x, min(x + cell - 1, width - 1))
                y1 = clamp(round(center_y - dy), y, min(y + cell - 1, height - 1))
                x2 = clamp(round(center_x + dx), x, min(x + cell - 1, width - 1))
                y2 = clamp(round(center_y + dy), y, min(y + cell - 1, height - 1))
                strokes.append(Stroke(x1, y1, x2, y2))

    return strokes


def trace_smart_pixel_strokes(
    luminance: Image.Image,
    pixel_size: int,
    threshold: int,
    min_run: int,
    density: float,
) -> List[Stroke]:
    """Redraw image brightness as ordered cell marks with edge reinforcement."""

    if pixel_size <= 0:
        raise ValueError("Pixel size must be at least 1.")
    if min_run <= 0:
        raise ValueError("Minimum run must be at least 1.")
    if density <= 0:
        raise ValueError("Density must be greater than 0.")

    width, height = luminance.size
    pixels = luminance.load()
    cell = max(3, pixel_size)
    strokes: List[Stroke] = []
    bayer = (
        (0, 8, 2, 10),
        (12, 4, 14, 6),
        (3, 11, 1, 9),
        (15, 7, 13, 5),
    )

    for row_index, y in enumerate(range(0, height, cell)):
        columns = list(range(0, width, cell))
        if row_index % 2 == 1:
            columns.reverse()

        for col_index, x in enumerate(columns):
            average = average_cell_luminance(pixels, x, y, cell, width, height)
            contrast = cell_contrast(pixels, x, y, cell, width, height)
            edge_strength = min(1.0, contrast / 90.0)

            effective = average - edge_strength * 34
            darkness = (threshold - effective) / max(threshold, 1)
            darkness = max(0.0, min(1.0, darkness))

            ordered_limit = (bayer[row_index % 4][col_index % 4] + 0.5) / 16.0
            if darkness <= 0 and edge_strength < 0.35:
                continue
            if darkness < ordered_limit * 0.55 and edge_strength < 0.45:
                continue

            mark_count = max(1, round((darkness * 3.2 + edge_strength * 1.1) * density))
            mark_count = min(mark_count, 5)
            strokes.extend(cell_marks(x, y, cell, width, height, mark_count, min_run, row_index + col_index))

            if edge_strength > 0.42:
                strokes.append(edge_mark(pixels, x, y, cell, width, height, min_run))

    return strokes


def trace_line_reduction_strokes(
    luminance: Image.Image,
    spacing: int,
    threshold: int,
    min_run: int,
    line_count: int,
) -> List[Stroke]:
    """Trace the image's major visible lines, with outline reinforcement."""

    if spacing <= 0:
        raise ValueError("Line spacing must be at least 1.")
    if min_run <= 0:
        raise ValueError("Minimum run must be at least 1.")
    if line_count <= 0:
        raise ValueError("Line count must be at least 1.")

    detail_mask, edge_strength = build_major_line_mask(luminance, threshold, spacing)
    main_mask = extract_main_object_mask(luminance, threshold)
    boundary = build_boundary_mask(main_mask)
    detail_step = max(1, spacing)
    outline_step = max(1, spacing)
    strokes = []

    strokes.extend(trace_scanline_strokes(detail_mask, detail_step, min_run))
    strokes.extend(trace_vertical_strokes(detail_mask, detail_step, min_run))
    strokes.extend(trace_diagonal_hatch_strokes(detail_mask, max(1, detail_step * 2), min_run, direction=1))
    strokes.extend(trace_diagonal_hatch_strokes(detail_mask, max(1, detail_step * 2), min_run, direction=-1))

    strokes.extend(trace_scanline_strokes(boundary, outline_step, min_run))
    strokes.extend(trace_vertical_strokes(boundary, outline_step, min_run))
    strokes.extend(trace_boundary_marks(boundary, outline_step, min_run))
    strokes = unique_strokes(strokes)
    strokes = discard_frame_like_strokes(strokes, luminance.size)
    if len(strokes) > line_count * 3:
        strokes = keep_strongest_line_strokes(strokes, luminance, edge_strength, threshold, line_count * 3)
    strokes = merge_nearby_line_strokes(strokes, spacing)
    strokes = discard_short_line_strokes(strokes, spacing, min_run)
    strokes = unique_strokes(strokes)
    strokes = discard_frame_like_strokes(strokes, luminance.size)
    if len(strokes) > line_count:
        strokes = keep_strongest_line_strokes(strokes, luminance, edge_strength, threshold, line_count)
        strokes = merge_nearby_line_strokes(strokes, spacing)
        strokes = discard_short_line_strokes(strokes, spacing, min_run)
        strokes = unique_strokes(strokes)
        strokes = discard_frame_like_strokes(strokes, luminance.size)
    return sort_strokes_for_drawing(strokes)


def trace_handdrawn_strokes(
    luminance: Image.Image,
    spacing: int,
    threshold: int,
    min_run: int,
    line_count: int,
) -> List[Stroke]:
    """Trace curved image structure, then render it as human-like sketch strokes."""

    if spacing <= 0:
        raise ValueError("Line spacing must be at least 1.")
    if min_run <= 0:
        raise ValueError("Minimum run must be at least 1.")
    if line_count <= 0:
        raise ValueError("Line count must be at least 1.")

    seed = image_trace_seed(luminance, threshold, spacing)
    structure_ratio = 0.95 if line_count >= 1000 else 0.78
    structure_budget = max(1, round(line_count * structure_ratio))
    ink_mask = luminance.point(lambda pixel: 0 if pixel < threshold else 255, mode="1")
    ink_mask = erase_frame_pixels(remove_border_dark_background(luminance, ink_mask, threshold))
    ink_coverage = black_pixel_ratio(ink_mask)
    paths = trace_opencv_handdrawn_paths(luminance, threshold, spacing)
    if not paths:
        skeleton = build_handdrawn_skeleton_mask(luminance, threshold, spacing)
        paths = trace_skeleton_paths(skeleton, max(2, min_run))
    path_groups = handdrawn_stroke_groups_from_paths(
        paths,
        luminance.size,
        spacing,
        min_run,
        seed,
        structure_budget,
    )
    structure = flatten_ordered_stroke_groups(path_groups)

    if not structure:
        base_strokes = trace_line_reduction_strokes(
            luminance,
            spacing,
            threshold,
            min_run,
            min(line_count, max(8, line_count // 2)),
        )
        structure = humanize_line_strokes(
            base_strokes,
            luminance.size,
            spacing,
            min_run,
            seed,
            structure_budget,
        )

    detail_ratio = 0.05 if line_count >= 1000 else 0.1
    detail_budget = max(0, min(line_count - len(structure), round(line_count * detail_ratio)))
    if detail_budget > 0:
        detail_strokes = trace_internal_line_detail_strokes(
            luminance,
            max(1, spacing),
            threshold,
            min_run,
            detail_budget,
        )
        detail_strokes = discard_redundant_detail_strokes(detail_strokes, structure, spacing)
        structure.extend(
            humanize_line_strokes(
                detail_strokes,
                luminance.size,
                spacing,
                min_run,
                seed ^ 0xA511E9B3,
                detail_budget,
            )
        )

    shading_budget = 0
    shading = trace_freehand_shading_strokes(
        luminance,
        spacing,
        threshold,
        min_run,
        shading_budget,
        seed ^ 0x9E3779B9,
    )
    return order_handdrawn_strokes(structure, shading)[:line_count]


def build_handdrawn_skeleton_mask(luminance: Image.Image, threshold: int, spacing: int) -> Image.Image:
    opencv_mask = build_opencv_handdrawn_mask(luminance, threshold, spacing)
    if opencv_mask is not None:
        return thin_binary_mask(opencv_mask)

    raw_ink = luminance.point(lambda pixel: 0 if pixel < threshold else 255, mode="1")
    raw_ink = erase_frame_pixels(remove_border_dark_background(luminance, raw_ink, threshold))
    detail_mask, _edge_strength = build_major_line_mask(luminance, threshold, spacing)
    ink_coverage = black_pixel_ratio(raw_ink)
    combined = raw_ink if ink_coverage < 0.18 else combine_black_masks(raw_ink, detail_mask)

    # Lightly connect nearly-touching ink before thinning, then skeletonize its centerline.
    connected = combined.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    return thin_binary_mask(connected)


def black_pixel_ratio(mask: Image.Image) -> float:
    width, height = mask.size
    pixels = mask.load()
    black = 0
    for y in range(height):
        for x in range(width):
            if pixels[x, y] == 0:
                black += 1
    return black / max(1, width * height)


def erase_frame_pixels(mask: Image.Image) -> Image.Image:
    width, height = mask.size
    output = mask.copy()
    pixels = output.load()
    border = min(8, max(1, min(width, height) // 20))

    for y in range(border):
        if row_black_count(pixels, y, width) >= width * 0.55:
            clear_row(pixels, y, width)
    for y in range(max(0, height - border), height):
        if row_black_count(pixels, y, width) >= width * 0.55:
            clear_row(pixels, y, width)
    for x in range(border):
        if column_black_count(pixels, x, height) >= height * 0.55:
            clear_column(pixels, x, height)
    for x in range(max(0, width - border), width):
        if column_black_count(pixels, x, height) >= height * 0.55:
            clear_column(pixels, x, height)

    return output


def remove_border_dark_background(luminance: Image.Image, ink_mask: Image.Image, threshold: int) -> Image.Image:
    width, height = luminance.size
    luminance_pixels = luminance.load()
    dark_cutoff = max(8, min(58, threshold // 3))
    starts = []

    for x in range(width):
        if luminance_pixels[x, 0] <= dark_cutoff:
            starts.append((x, 0))
        if luminance_pixels[x, height - 1] <= dark_cutoff:
            starts.append((x, height - 1))
    for y in range(height):
        if luminance_pixels[0, y] <= dark_cutoff:
            starts.append((0, y))
        if luminance_pixels[width - 1, y] <= dark_cutoff:
            starts.append((width - 1, y))

    if not starts:
        return ink_mask

    visited: Set[Tuple[int, int]] = set()
    stack = starts[:]
    while stack:
        x, y = stack.pop()
        if (x, y) in visited or luminance_pixels[x, y] > dark_cutoff:
            continue
        visited.add((x, y))
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                if luminance_pixels[nx, ny] <= dark_cutoff:
                    stack.append((nx, ny))

    if len(visited) < width * height * 0.12:
        return ink_mask

    output = ink_mask.copy()
    output_pixels = output.load()
    for x, y in visited:
        output_pixels[x, y] = 255
    return output


def remove_large_border_connected_ink(mask: Image.Image) -> Image.Image:
    width, height = mask.size
    pixels = mask.load()
    starts = []
    for x in range(width):
        if pixels[x, 0] == 0:
            starts.append((x, 0))
        if pixels[x, height - 1] == 0:
            starts.append((x, height - 1))
    for y in range(height):
        if pixels[0, y] == 0:
            starts.append((0, y))
        if pixels[width - 1, y] == 0:
            starts.append((width - 1, y))

    if not starts:
        return mask

    visited: Set[Tuple[int, int]] = set()
    stack = starts[:]
    while stack:
        x, y = stack.pop()
        if (x, y) in visited or pixels[x, y] != 0:
            continue
        visited.add((x, y))
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited and pixels[nx, ny] == 0:
                stack.append((nx, ny))

    if len(visited) < width * height * 0.12:
        return mask

    output = mask.copy()
    output_pixels = output.load()
    for x, y in visited:
        output_pixels[x, y] = 255
    return output


def row_black_count(pixels, y: int, width: int) -> int:
    return sum(1 for x in range(width) if pixels[x, y] == 0)


def column_black_count(pixels, x: int, height: int) -> int:
    return sum(1 for y in range(height) if pixels[x, y] == 0)


def clear_row(pixels, y: int, width: int) -> None:
    for x in range(width):
        pixels[x, y] = 255


def clear_column(pixels, x: int, height: int) -> None:
    for y in range(height):
        pixels[x, y] = 255


def combine_black_masks(*masks: Image.Image) -> Image.Image:
    if not masks:
        raise ValueError("At least one mask is required.")

    width, height = masks[0].size
    output = Image.new("1", (width, height), 255)
    output_pixels = output.load()
    loaded = [mask.load() for mask in masks]

    for y in range(height):
        for x in range(width):
            if any(pixels[x, y] == 0 for pixels in loaded):
                output_pixels[x, y] = 0
    return output


def build_opencv_handdrawn_mask(luminance: Image.Image, threshold: int, spacing: int) -> Optional[Image.Image]:
    if cv2 is None or np is None:
        return None

    gray = np.array(luminance, dtype=np.uint8)
    denoised = cv2.bilateralFilter(gray, 5, 35, 35)
    edges = adaptive_canny_edges(denoised, threshold)
    raw_ink = luminance.point(lambda pixel: 0 if pixel < threshold else 255, mode="1")
    raw_ink = erase_frame_pixels(remove_border_dark_background(luminance, raw_ink, threshold))
    dark = mask_to_binary_array(raw_ink)

    kernel_size = max(2, min(5, spacing + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dark_open = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=1)
    dark_connected = cv2.morphologyEx(dark_open, cv2.MORPH_CLOSE, kernel, iterations=1)
    edge_weight = cv2.bitwise_and(cv2.dilate(edges, kernel, iterations=1), cv2.dilate(dark, kernel, iterations=1))
    floating_edges = remove_tiny_components_array(edges, max(4, spacing * spacing))
    if float(np.count_nonzero(dark)) / max(1, dark.size) < 0.035:
        edge_weight = cv2.bitwise_or(edge_weight, floating_edges)
    combined = cv2.bitwise_or(dark, edge_weight)
    combined = cv2.bitwise_or(combined, dark_connected)
    combined = remove_tiny_components_array(combined, max(2, spacing * spacing // 2))

    if float(np.count_nonzero(combined)) / max(1, combined.size) < 0.002:
        return None
    return erase_frame_pixels(binary_array_to_mask(combined))


def trace_opencv_handdrawn_paths(
    luminance: Image.Image,
    threshold: int,
    spacing: int,
) -> List[List[Tuple[int, int]]]:
    if cv2 is None or np is None:
        return []

    line_map = build_opencv_handdrawn_line_array(luminance, threshold, spacing)
    if line_map is None:
        return []

    contours, _hierarchy = cv2.findContours(line_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    paths: List[List[Tuple[int, int]]] = []
    min_length = max(10.0, spacing * 3.0)

    for contour in contours:
        if len(contour) < 2:
            continue

        length = float(cv2.arcLength(contour, closed=False))
        if length < min_length:
            continue

        epsilon = max(0.45, min(2.0, spacing * 0.22))
        approximated = cv2.approxPolyDP(contour, epsilon, closed=False)
        points = [(int(point[0][0]), int(point[0][1])) for point in approximated]
        points = dedupe_adjacent_points(points)
        if len(points) < 2:
            continue

        if length > spacing * 16:
            paths.extend(split_path_at_corners(points, spacing))
        else:
            paths.append(points)

    return sorted(paths, key=path_length, reverse=True)


def build_opencv_handdrawn_line_array(luminance: Image.Image, threshold: int, spacing: int):
    if cv2 is None or np is None:
        return None

    gray = np.array(luminance, dtype=np.uint8)
    denoised = cv2.bilateralFilter(gray, 5, 35, 35)
    equalized = cv2.createCLAHE(clipLimit=1.7, tileGridSize=(8, 8)).apply(denoised)
    raw_ink = luminance.point(lambda pixel: 0 if pixel < threshold else 255, mode="1")
    raw_ink = erase_frame_pixels(remove_border_dark_background(luminance, raw_ink, threshold))
    dark = mask_to_binary_array(raw_ink)

    edges = adaptive_canny_edges(equalized, threshold)
    kernel_size = max(2, min(4, spacing))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dark_gradient = cv2.morphologyEx(dark, cv2.MORPH_GRADIENT, kernel)
    local_radius = max(5, min(11, spacing * 2 + 1))
    if local_radius % 2 == 0:
        local_radius += 1
    local_kernel = np.ones((local_radius, local_radius), dtype=np.uint8)
    local_high = cv2.dilate(denoised, local_kernel)
    local_low = cv2.erode(denoised, local_kernel)
    local_contrast = cv2.subtract(local_high, local_low)
    contrast_edges = cv2.bitwise_and(
        edges,
        cv2.threshold(local_contrast, max(12, threshold // 6), 255, cv2.THRESH_BINARY)[1],
    )
    near_ink_edges = cv2.bitwise_and(cv2.dilate(edges, kernel, iterations=1), cv2.dilate(dark, kernel, iterations=2))
    line_map = cv2.bitwise_or(dark_gradient, contrast_edges)
    line_map = cv2.bitwise_or(line_map, near_ink_edges)

    if float(np.count_nonzero(dark)) / max(1, dark.size) < 0.035:
        line_map = cv2.bitwise_or(line_map, remove_tiny_components_array(edges, max(4, spacing * spacing)))

    line_map = cv2.morphologyEx(line_map, cv2.MORPH_CLOSE, kernel, iterations=1)
    line_map = remove_tiny_components_array(line_map, max(3, spacing * spacing // 3))
    line_map = mask_to_binary_array(erase_frame_pixels(binary_array_to_mask(line_map)))
    return line_map


def dedupe_adjacent_points(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    result: List[Tuple[int, int]] = []
    for point in points:
        if not result or point != result[-1]:
            result.append(point)
    return result


def split_path_at_corners(points: List[Tuple[int, int]], spacing: int) -> List[List[Tuple[int, int]]]:
    if len(points) < 4:
        return [points]

    min_piece_length = max(8.0, spacing * 3.0)
    pieces: List[List[Tuple[int, int]]] = []
    current = [points[0]]
    current_length = 0.0

    for index in range(1, len(points) - 1):
        previous = points[index - 1]
        point = points[index]
        following = points[index + 1]
        current.append(point)
        current_length += math.hypot(point[0] - previous[0], point[1] - previous[1])

        if current_length < min_piece_length:
            continue
        if corner_angle(previous, point, following) < 118:
            pieces.append(current)
            current = [point]
            current_length = 0.0

    current.append(points[-1])
    if path_length(current) >= min_piece_length:
        pieces.append(current)
    elif pieces:
        pieces[-1].extend(current[1:])
    else:
        pieces.append(points)

    return pieces


def corner_angle(
    previous: Tuple[int, int],
    point: Tuple[int, int],
    following: Tuple[int, int],
) -> float:
    ax = previous[0] - point[0]
    ay = previous[1] - point[1]
    bx = following[0] - point[0]
    by = following[1] - point[1]
    denominator = math.hypot(ax, ay) * math.hypot(bx, by)
    if denominator <= 0:
        return 180.0
    cosine = max(-1.0, min(1.0, (ax * bx + ay * by) / denominator))
    return math.degrees(math.acos(cosine))


def adaptive_canny_edges(gray, threshold: int):
    median = float(np.median(gray))
    lower = int(max(12, min(threshold * 0.45, median * 0.68)))
    upper = int(max(lower + 24, min(245, median * 1.35 + threshold * 0.28)))
    return cv2.Canny(gray, lower, upper, L2gradient=True)


def remove_large_border_component_array(mask):
    height, width = mask.shape
    flood = mask.copy()
    fill_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    border_points = []

    for x in range(width):
        if flood[0, x]:
            border_points.append((x, 0))
        if flood[height - 1, x]:
            border_points.append((x, height - 1))
    for y in range(height):
        if flood[y, 0]:
            border_points.append((0, y))
        if flood[y, width - 1]:
            border_points.append((width - 1, y))

    if not border_points:
        return mask

    removed = np.zeros_like(mask)
    for x, y in border_points:
        if flood[y, x] == 0:
            continue
        component = np.zeros_like(mask)
        cv2.floodFill(flood, fill_mask, (x, y), 0)
        component[(mask > 0) & (flood == 0) & (removed == 0)] = 255
        removed = cv2.bitwise_or(removed, component)

    removed_ratio = float(np.count_nonzero(removed)) / max(1, mask.size)
    if removed_ratio < 0.12:
        return mask

    output = mask.copy()
    output[removed > 0] = 0
    return output


def binary_array_to_mask(array) -> Image.Image:
    mask = Image.fromarray(np.where(array > 0, 0, 255).astype(np.uint8), mode="L")
    return mask.convert("1")


def mask_to_binary_array(mask: Image.Image):
    return np.where(np.array(mask.convert("L"), dtype=np.uint8) < 128, 255, 0).astype(np.uint8)


def thin_binary_mask(mask: Image.Image) -> Image.Image:
    width, height = mask.size
    source = mask.load()
    foreground = [[source[x, y] == 0 for x in range(width)] for y in range(height)]

    changed = True
    iterations = 0
    max_iterations = max(width, height)
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        for step in (0, 1):
            removals: List[Tuple[int, int]] = []
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    if not foreground[y][x] or should_keep_skeleton_pixel(foreground, x, y, step):
                        continue
                    removals.append((x, y))

            if removals:
                changed = True
                for x, y in removals:
                    foreground[y][x] = False

    output = Image.new("1", mask.size, 255)
    output_pixels = output.load()
    for y in range(height):
        for x in range(width):
            if foreground[y][x]:
                output_pixels[x, y] = 0
    return output


def should_keep_skeleton_pixel(foreground: List[List[bool]], x: int, y: int, step: int) -> bool:
    neighbors = zhang_suen_neighbors(foreground, x, y)
    neighbor_count = sum(1 for value in neighbors if value)
    if neighbor_count < 2 or neighbor_count > 6:
        return True
    if black_white_transition_count(neighbors) != 1:
        return True

    p2, _p3, p4, _p5, p6, _p7, p8, _p9 = neighbors
    if step == 0:
        return (p2 and p4 and p6) or (p4 and p6 and p8)
    return (p2 and p4 and p8) or (p2 and p6 and p8)


def zhang_suen_neighbors(foreground: List[List[bool]], x: int, y: int) -> Tuple[bool, ...]:
    return (
        foreground[y - 1][x],
        foreground[y - 1][x + 1],
        foreground[y][x + 1],
        foreground[y + 1][x + 1],
        foreground[y + 1][x],
        foreground[y + 1][x - 1],
        foreground[y][x - 1],
        foreground[y - 1][x - 1],
    )


def black_white_transition_count(values: Tuple[bool, ...]) -> int:
    total = 0
    wrapped = values + (values[0],)
    for index in range(len(values)):
        if not wrapped[index] and wrapped[index + 1]:
            total += 1
    return total


def trace_skeleton_paths(mask: Image.Image, min_points: int) -> List[List[Tuple[int, int]]]:
    pixels = skeleton_pixel_set(mask)
    if not pixels:
        return []

    neighbor_map = {point: skeleton_neighbors(point, pixels) for point in pixels}
    visited_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    paths: List[List[Tuple[int, int]]] = []
    starts = sorted((point for point, neighbors in neighbor_map.items() if len(neighbors) != 2), key=point_sort_key)

    for start in starts:
        for neighbor in sorted(neighbor_map[start], key=point_sort_key):
            edge = ordered_edge(start, neighbor)
            if edge in visited_edges:
                continue
            path = walk_skeleton_path(start, neighbor, neighbor_map, visited_edges)
            if len(path) >= min_points:
                paths.append(path)

    for start in sorted(pixels, key=point_sort_key):
        for neighbor in sorted(neighbor_map[start], key=point_sort_key):
            edge = ordered_edge(start, neighbor)
            if edge in visited_edges:
                continue
            path = walk_skeleton_path(start, neighbor, neighbor_map, visited_edges, allow_cycle=True)
            if len(path) >= min_points:
                paths.append(path)

    return sorted(paths, key=path_length, reverse=True)


def skeleton_pixel_set(mask: Image.Image) -> Set[Tuple[int, int]]:
    width, height = mask.size
    pixels = mask.load()
    return {(x, y) for y in range(height) for x in range(width) if pixels[x, y] == 0}


def skeleton_neighbors(point: Tuple[int, int], pixels: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    x, y = point
    neighbors = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            candidate = (x + dx, y + dy)
            if candidate in pixels:
                neighbors.append(candidate)
    return neighbors


def walk_skeleton_path(
    start: Tuple[int, int],
    first: Tuple[int, int],
    neighbor_map: Dict[Tuple[int, int], List[Tuple[int, int]]],
    visited_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]],
    allow_cycle: bool = False,
) -> List[Tuple[int, int]]:
    path = [start]
    previous = start
    current = first

    while True:
        visited_edges.add(ordered_edge(previous, current))
        path.append(current)
        neighbors = [point for point in neighbor_map[current] if point != previous]

        if allow_cycle and current == start:
            break
        if not allow_cycle and len(neighbor_map[current]) != 2:
            break
        if not neighbors:
            break

        next_point = choose_next_skeleton_point(previous, current, neighbors, visited_edges)
        if next_point is None:
            break
        if allow_cycle and next_point == start and len(path) > 3:
            previous, current = current, next_point
            continue

        previous, current = current, next_point

    return path


def choose_next_skeleton_point(
    previous: Tuple[int, int],
    current: Tuple[int, int],
    candidates: List[Tuple[int, int]],
    visited_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]],
) -> Optional[Tuple[int, int]]:
    unvisited = [point for point in candidates if ordered_edge(current, point) not in visited_edges]
    if not unvisited:
        return None

    incoming_x = current[0] - previous[0]
    incoming_y = current[1] - previous[1]
    return max(
        unvisited,
        key=lambda point: (point[0] - current[0]) * incoming_x + (point[1] - current[1]) * incoming_y,
    )


def ordered_edge(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def point_sort_key(point: Tuple[int, int]) -> Tuple[int, int]:
    return (point[1], point[0])


def handdrawn_stroke_groups_from_paths(
    paths: List[List[Tuple[int, int]]],
    canvas_size: Tuple[int, int],
    spacing: int,
    min_run: int,
    seed: int,
    max_count: int,
) -> List[List[Stroke]]:
    if max_count <= 0:
        return []

    useful_paths = [
        path
        for path in paths
        if path_length(path) >= max(min_run, spacing * 1.5) and not is_frame_like_path(path, canvas_size)
    ]
    if not useful_paths:
        return []

    total_length = sum(path_length(path) for path in useful_paths)
    target_step = max(1.8, total_length / max(max_count, 1), min(5.5, spacing * 0.85))
    groups: List[List[Stroke]] = []
    used = 0

    for path_index, path in enumerate(useful_paths):
        if used >= max_count:
            break

        rng = random.Random(seed ^ (path_index * 1000003) ^ int(path_length(path) * 97))
        simplified = simplify_path(path, max(0.65, min(1.45, spacing * 0.2)))
        curve_points = catmull_rom_sample(simplified, target_step)
        curve_points = gently_wobble_points(curve_points, spacing, rng)
        group = strokes_from_points(curve_points, canvas_size, min_run)
        if not group:
            continue

        remaining = max_count - used
        if len(group) > remaining:
            group = decimate_strokes(group, remaining)
        groups.append(group)
        used += len(group)

    return order_stroke_groups_by_nearest(groups, (0, 0))


def flatten_ordered_stroke_groups(groups: List[List[Stroke]]) -> List[Stroke]:
    return [stroke for group in groups for stroke in group]


def order_stroke_groups_by_nearest(groups: List[List[Stroke]], start: Tuple[int, int]) -> List[List[Stroke]]:
    items = [group[:] for group in groups if group]
    ordered: List[List[Stroke]] = []
    current = start
    cell_size = 64
    spatial = SpatialIndex(cell_size)

    for index, group in enumerate(items):
        spatial.add(index, group_endpoint_points(group))

    while spatial.active_count:
        best_index, best_reverse = nearest_group_index(items, spatial, current)
        group = items[best_index]
        spatial.remove(best_index, group_endpoint_points(group))
        if best_reverse:
            group = [Stroke(stroke.x2, stroke.y2, stroke.x1, stroke.y1) for stroke in reversed(group)]
        ordered.append(group)
        current = (group[-1].x2, group[-1].y2)

    return ordered


def simplify_path(points: List[Tuple[int, int]], epsilon: float) -> List[Tuple[float, float]]:
    if len(points) <= 2:
        return [(float(x), float(y)) for x, y in points]

    first = points[0]
    last = points[-1]
    best_distance = -1.0
    best_index = 0
    for index in range(1, len(points) - 1):
        distance = point_to_segment_distance(points[index], first, last)
        if distance > best_distance:
            best_distance = distance
            best_index = index

    if best_distance <= epsilon:
        return [(float(first[0]), float(first[1])), (float(last[0]), float(last[1]))]

    left = simplify_path(points[: best_index + 1], epsilon)
    right = simplify_path(points[best_index:], epsilon)
    return left[:-1] + right


def point_to_segment_distance(
    point: Tuple[int, int],
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    denominator = dx * dx + dy * dy
    if denominator == 0:
        return math.hypot(px - sx, py - sy)

    amount = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / denominator))
    closest_x = sx + dx * amount
    closest_y = sy + dy * amount
    return math.hypot(px - closest_x, py - closest_y)


def catmull_rom_sample(points: List[Tuple[float, float]], target_step: float) -> List[Tuple[float, float]]:
    if len(points) <= 2:
        return points

    result: List[Tuple[float, float]] = [points[0]]
    for index in range(len(points) - 1):
        p0 = points[max(0, index - 1)]
        p1 = points[index]
        p2 = points[index + 1]
        p3 = points[min(len(points) - 1, index + 2)]
        distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        steps = max(1, round(distance / max(target_step, 0.1)))
        for step in range(1, steps + 1):
            t = step / steps
            result.append(catmull_rom_point(p0, p1, p2, p3, t))
    return result


def catmull_rom_point(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    t: float,
) -> Tuple[float, float]:
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * (
        (2 * p1[0])
        + (-p0[0] + p2[0]) * t
        + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
        + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
    )
    y = 0.5 * (
        (2 * p1[1])
        + (-p0[1] + p2[1]) * t
        + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
        + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
    )
    return (x, y)


def gently_wobble_points(
    points: List[Tuple[float, float]],
    spacing: int,
    rng: random.Random,
) -> List[Tuple[float, float]]:
    if len(points) <= 2:
        return points

    amplitude = min(0.85, max(0.15, spacing * 0.12))
    phase = rng.random() * math.tau
    frequency = rng.uniform(0.8, 1.8)
    result = [points[0]]

    for index in range(1, len(points) - 1):
        prev_point = points[index - 1]
        point = points[index]
        next_point = points[index + 1]
        dx = next_point[0] - prev_point[0]
        dy = next_point[1] - prev_point[1]
        distance = max(0.001, math.hypot(dx, dy))
        normal_x = -dy / distance
        normal_y = dx / distance
        t = index / max(1, len(points) - 1)
        wobble = math.sin(phase + t * math.tau * frequency) * amplitude
        wobble += rng.uniform(-amplitude * 0.2, amplitude * 0.2)
        result.append((point[0] + normal_x * wobble, point[1] + normal_y * wobble))

    result.append(points[-1])
    return result


def strokes_from_points(
    points: List[Tuple[float, float]],
    canvas_size: Tuple[int, int],
    min_run: int,
) -> List[Stroke]:
    result: List[Stroke] = []
    for index in range(len(points) - 1):
        append_if_long_enough(
            result,
            points[index][0],
            points[index][1],
            points[index + 1][0],
            points[index + 1][1],
            canvas_size,
            min_run,
        )
    return result


def decimate_strokes(strokes: List[Stroke], limit: int) -> List[Stroke]:
    if limit <= 0:
        return []
    if len(strokes) <= limit:
        return strokes

    result = []
    for index in range(limit):
        source_index = round(index * (len(strokes) - 1) / max(1, limit - 1))
        result.append(strokes[source_index])
    return result


def offset_curve_group(
    group: List[Stroke],
    canvas_size: Tuple[int, int],
    spacing: int,
    rng: random.Random,
    limit: int,
) -> List[Stroke]:
    if limit <= 0:
        return []

    offset_x = rng.uniform(-0.75, 0.75) * min(1.8, max(0.6, spacing * 0.28))
    offset_y = rng.uniform(-0.75, 0.75) * min(1.8, max(0.6, spacing * 0.28))
    start = rng.randint(0, max(0, len(group) // 5))
    end = rng.randint(max(start + 1, len(group) * 3 // 5), len(group))
    result = []
    for stroke in group[start:end]:
        if len(result) >= limit:
            break
        shifted = clamped_stroke(
            stroke.x1 + offset_x,
            stroke.y1 + offset_y,
            stroke.x2 + offset_x,
            stroke.y2 + offset_y,
            canvas_size,
        )
        result.append(shifted)
    return result


def path_length(points: List[Tuple[int, int]]) -> float:
    total = 0.0
    for index in range(len(points) - 1):
        total += math.hypot(points[index + 1][0] - points[index][0], points[index + 1][1] - points[index][1])
    return total


def is_frame_like_path(path: List[Tuple[int, int]], canvas_size: Tuple[int, int]) -> bool:
    if len(path) < 2:
        return False

    width, height = canvas_size
    xs = [point[0] for point in path]
    ys = [point[1] for point in path]
    left = min(xs)
    right = max(xs)
    top = min(ys)
    bottom = max(ys)
    path_width = right - left + 1
    path_height = bottom - top + 1
    near_left = left <= 2
    near_right = right >= width - 3
    near_top = top <= 2
    near_bottom = bottom >= height - 3

    if path_width >= width * 0.82 and path_height <= max(4, height * 0.05) and (near_top or near_bottom):
        return True
    if path_height >= height * 0.82 and path_width <= max(4, width * 0.05) and (near_left or near_right):
        return True
    if path_width >= width * 0.82 and path_height <= max(4, height * 0.025):
        return True
    if path_height >= height * 0.82 and path_width <= max(4, width * 0.025):
        return True
    if path_width >= width * 0.92 and (near_top or near_bottom):
        return True
    if path_height >= height * 0.92 and (near_left or near_right):
        return True
    return False


def image_trace_seed(luminance: Image.Image, threshold: int, spacing: int) -> int:
    width, height = luminance.size
    pixels = luminance.load()
    seed = (width * 1009) ^ (height * 9176) ^ (threshold * 131) ^ (spacing * 31337)
    x_stride = max(1, width // 16)
    y_stride = max(1, height // 16)

    for y in range(0, height, y_stride):
        for x in range(0, width, x_stride):
            seed = ((seed << 5) - seed + pixels[x, y] + x * 17 + y * 31) & 0xFFFFFFFF
    return seed


def humanize_line_strokes(
    strokes: List[Stroke],
    canvas_size: Tuple[int, int],
    spacing: int,
    min_run: int,
    seed: int,
    max_count: int,
) -> List[Stroke]:
    if max_count <= 0:
        return []

    result: List[Stroke] = []
    for index, stroke in enumerate(strokes):
        if len(result) >= max_count:
            break

        rng = random.Random(stroke_seed(seed, stroke, index))
        append_limited(
            result,
            handdrawn_segments_for_stroke(stroke, canvas_size, spacing, min_run, rng),
            max_count,
        )

        length = stroke_length(stroke)
        if len(result) >= max_count or length < max(spacing * 8, min_run * 5):
            continue
        if rng.random() > 0.34:
            continue

        repeated = partial_offset_stroke(stroke, canvas_size, rng, spacing)
        append_limited(
            result,
            handdrawn_segments_for_stroke(repeated, canvas_size, spacing, min_run, rng),
            max_count,
        )

    return result


def discard_redundant_detail_strokes(
    details: List[Stroke],
    structure: List[Stroke],
    spacing: int,
) -> List[Stroke]:
    if not details or not structure:
        return details

    occupied = set()
    cell = max(4, spacing * 2)
    for stroke in structure:
        occupied.add((stroke.x1 // cell, stroke.y1 // cell))
        occupied.add((stroke.x2 // cell, stroke.y2 // cell))
        occupied.add(((stroke.x1 + stroke.x2) // 2 // cell, (stroke.y1 + stroke.y2) // 2 // cell))

    result: List[Stroke] = []
    for stroke in details:
        key = ((stroke.x1 + stroke.x2) // 2 // cell, (stroke.y1 + stroke.y2) // 2 // cell)
        if key in occupied and stroke_length(stroke) < spacing * 6:
            continue
        result.append(stroke)
    return result


def trace_internal_line_detail_strokes(
    luminance: Image.Image,
    spacing: int,
    threshold: int,
    min_run: int,
    limit: int,
) -> List[Stroke]:
    if limit <= 0:
        return []

    detail_mask, edge_strength = build_major_line_mask(luminance, threshold, spacing)
    strokes: List[Stroke] = []
    strokes.extend(trace_scanline_strokes(detail_mask, max(1, spacing), min_run))
    strokes.extend(trace_vertical_strokes(detail_mask, max(1, spacing), min_run))
    strokes = unique_strokes(strokes)
    strokes = discard_frame_like_strokes(strokes, luminance.size)
    strokes = discard_short_line_strokes(strokes, max(1, spacing // 2), min_run)
    if len(strokes) > limit:
        strokes = keep_strongest_line_strokes(strokes, luminance, edge_strength, threshold, limit)
    return sort_strokes_for_drawing(strokes)


def trace_freehand_shading_strokes(
    luminance: Image.Image,
    spacing: int,
    threshold: int,
    min_run: int,
    max_count: int,
    seed: int,
) -> List[Stroke]:
    if max_count <= 0:
        return []

    width, height = luminance.size
    pixels = luminance.load()
    cell = max(8, spacing * 4)
    candidates: List[Tuple[float, Stroke]] = []

    for row_index, y in enumerate(range(0, height, cell)):
        columns = list(range(0, width, cell))
        if row_index % 2 == 1:
            columns.reverse()

        for column_index, x in enumerate(columns):
            average = average_cell_luminance(pixels, x, y, cell, width, height)
            darkness = pixel_darkness(round(average), threshold)
            if darkness < 0.08:
                continue

            cell_seed = seed ^ (x * 73856093) ^ (y * 19349663) ^ (column_index * 83492791)
            rng = random.Random(cell_seed)
            if darkness < 0.22 and rng.random() > darkness * 2.6:
                continue

            contrast = cell_contrast(pixels, x, y, cell, width, height)
            mark_count = 1
            if darkness > 0.46:
                mark_count += 1
            if darkness > 0.72 or contrast > 92:
                mark_count += 1

            for mark_index in range(mark_count):
                angle = math.radians(-30 if (row_index + column_index) % 2 == 0 else 24)
                angle += rng.uniform(-0.24, 0.24)
                if darkness > 0.68 and mark_index == mark_count - 1:
                    angle += math.pi / 2

                right = min(x + cell - 1, width - 1)
                bottom = min(y + cell - 1, height - 1)
                cell_width = max(0, right - x)
                cell_height = max(0, bottom - y)
                center_x = rng.uniform(x + cell_width * 0.22, x + cell_width * 0.88)
                center_y = rng.uniform(y + cell_height * 0.22, y + cell_height * 0.88)
                length = cell * (0.42 + darkness * 0.58) * rng.uniform(0.72, 1.16)
                stroke = freehand_mark(center_x, center_y, angle, length, luminance.size, spacing, min_run, rng)
                if stroke_length(stroke) < min_run:
                    continue
                score = darkness * 1.6 + min(1.0, contrast / 120.0) * 0.35 + rng.random() * 0.04
                candidates.append((score, stroke))

    strongest = sorted(candidates, key=lambda item: item[0], reverse=True)[:max_count]
    return [stroke for _score, stroke in strongest]


def handdrawn_segments_for_stroke(
    stroke: Stroke,
    canvas_size: Tuple[int, int],
    spacing: int,
    min_run: int,
    rng: random.Random,
) -> List[Stroke]:
    length = stroke_length(stroke)
    if length < min_run:
        return []

    dx = stroke.x2 - stroke.x1
    dy = stroke.y2 - stroke.y1
    distance = max(1.0, math.hypot(dx, dy))
    normal_x = -dy / distance
    normal_y = dx / distance
    tangent_x = dx / distance
    tangent_y = dy / distance
    segment_count = max(1, min(5, round(length / max(spacing * 5, min_run * 4, 10))))
    amplitude = min(3.0, max(0.45, spacing * 0.32))
    phase = rng.random() * math.tau
    frequency = rng.uniform(0.75, 1.55)
    points: List[Tuple[float, float]] = []

    for point_index in range(segment_count + 1):
        t = point_index / max(segment_count, 1)
        edge_fade = min(1.0, t * 3.2, (1.0 - t) * 3.2)
        wobble = math.sin(phase + t * math.pi * frequency) * amplitude * edge_fade
        wobble += rng.uniform(-amplitude * 0.35, amplitude * 0.35) * edge_fade
        tangent = rng.uniform(-amplitude * 0.22, amplitude * 0.22) * edge_fade
        x = stroke.x1 + dx * t + normal_x * wobble + tangent_x * tangent
        y = stroke.y1 + dy * t + normal_y * wobble + tangent_y * tangent
        points.append((x, y))

    result: List[Stroke] = []
    for index in range(len(points) - 1):
        start = points[index]
        end = points[index + 1]
        gap = 0.0
        if length > spacing * 7:
            gap = rng.uniform(0.0, 0.08)
        x1, y1 = lerp_point(start, end, gap)
        x2, y2 = lerp_point(start, end, 1.0 - gap * 0.55)
        append_if_long_enough(result, x1, y1, x2, y2, canvas_size, min_run)
    return result


def freehand_mark(
    center_x: float,
    center_y: float,
    angle: float,
    length: float,
    canvas_size: Tuple[int, int],
    spacing: int,
    min_run: int,
    rng: random.Random,
) -> Stroke:
    half_length = max(min_run, length) / 2
    bend = rng.uniform(-0.8, 0.8) * min(2.2, max(0.5, spacing * 0.28))
    dx = math.cos(angle) * half_length
    dy = math.sin(angle) * half_length
    normal_x = -math.sin(angle)
    normal_y = math.cos(angle)
    x1 = center_x - dx + normal_x * bend
    y1 = center_y - dy + normal_y * bend
    x2 = center_x + dx - normal_x * bend * 0.45
    y2 = center_y + dy - normal_y * bend * 0.45
    return clamped_stroke(x1, y1, x2, y2, canvas_size)


def partial_offset_stroke(
    stroke: Stroke,
    canvas_size: Tuple[int, int],
    rng: random.Random,
    spacing: int,
) -> Stroke:
    dx = stroke.x2 - stroke.x1
    dy = stroke.y2 - stroke.y1
    distance = max(1.0, math.hypot(dx, dy))
    normal_x = -dy / distance
    normal_y = dx / distance
    offset = rng.choice((-1, 1)) * rng.uniform(0.55, min(2.4, spacing * 0.45 + 0.7))
    start_t = rng.uniform(0.02, 0.18)
    end_t = rng.uniform(0.74, 0.98)
    x1 = stroke.x1 + dx * start_t + normal_x * offset
    y1 = stroke.y1 + dy * start_t + normal_y * offset
    x2 = stroke.x1 + dx * end_t + normal_x * offset * rng.uniform(0.65, 1.2)
    y2 = stroke.y1 + dy * end_t + normal_y * offset * rng.uniform(0.65, 1.2)
    return clamped_stroke(x1, y1, x2, y2, canvas_size)


def order_handdrawn_strokes(structure: List[Stroke], shading: List[Stroke]) -> List[Stroke]:
    ordered_structure = order_strokes_by_nearest(structure, (0, 0))
    start = (ordered_structure[-1].x2, ordered_structure[-1].y2) if ordered_structure else (0, 0)
    ordered_shading = order_strokes_by_nearest(shading, start)
    return ordered_structure + ordered_shading


def order_strokes_by_nearest(strokes: List[Stroke], start: Tuple[int, int]) -> List[Stroke]:
    items = strokes[:]
    ordered: List[Stroke] = []
    current = start
    cell_size = 48
    spatial = SpatialIndex(cell_size)

    for index, stroke in enumerate(items):
        spatial.add(index, ((stroke.x1, stroke.y1), (stroke.x2, stroke.y2)))

    while spatial.active_count:
        best_index, best_reverse = nearest_stroke_index(items, spatial, current)
        stroke = items[best_index]
        spatial.remove(best_index, ((stroke.x1, stroke.y1), (stroke.x2, stroke.y2)))
        if best_reverse:
            stroke = Stroke(stroke.x2, stroke.y2, stroke.x1, stroke.y1)
        ordered.append(stroke)
        current = (stroke.x2, stroke.y2)

    return ordered


class SpatialIndex:
    def __init__(self, cell_size: int) -> None:
        self.cell_size = max(1, cell_size)
        self.cells: Dict[Tuple[int, int], Set[int]] = {}
        self.active: Set[int] = set()

    @property
    def active_count(self) -> int:
        return len(self.active)

    def add(self, index: int, points: Tuple[Tuple[int, int], ...]) -> None:
        self.active.add(index)
        for point in points:
            self.cells.setdefault(self.cell_key(point), set()).add(index)

    def remove(self, index: int, points: Tuple[Tuple[int, int], ...]) -> None:
        self.active.discard(index)
        for point in points:
            key = self.cell_key(point)
            bucket = self.cells.get(key)
            if not bucket:
                continue
            bucket.discard(index)
            if not bucket:
                self.cells.pop(key, None)

    def nearby_candidates(self, point: Tuple[int, int], max_radius: int = 4) -> Set[int]:
        cx, cy = self.cell_key(point)
        candidates: Set[int] = set()
        for radius in range(max_radius + 1):
            found = False
            for nx in range(cx - radius, cx + radius + 1):
                for ny in range(cy - radius, cy + radius + 1):
                    bucket = self.cells.get((nx, ny))
                    if not bucket:
                        continue
                    candidates.update(index for index in bucket if index in self.active)
                    found = True
            if found and candidates:
                return candidates
        return set(self.active)

    def cell_key(self, point: Tuple[int, int]) -> Tuple[int, int]:
        return (point[0] // self.cell_size, point[1] // self.cell_size)


def group_endpoint_points(group: List[Stroke]) -> Tuple[Tuple[int, int], ...]:
    return ((group[0].x1, group[0].y1), (group[-1].x2, group[-1].y2))


def nearest_group_index(
    groups: List[List[Stroke]],
    spatial: SpatialIndex,
    current: Tuple[int, int],
) -> Tuple[int, bool]:
    best_index = -1
    best_reverse = False
    best_distance = float("inf")
    for index in spatial.nearby_candidates(current):
        group = groups[index]
        start_distance = squared_distance(current, (group[0].x1, group[0].y1))
        end_distance = squared_distance(current, (group[-1].x2, group[-1].y2))
        if start_distance < best_distance:
            best_index = index
            best_reverse = False
            best_distance = start_distance
        if end_distance < best_distance:
            best_index = index
            best_reverse = True
            best_distance = end_distance
    return best_index, best_reverse


def nearest_stroke_index(
    strokes: List[Stroke],
    spatial: SpatialIndex,
    current: Tuple[int, int],
) -> Tuple[int, bool]:
    best_index = -1
    best_reverse = False
    best_distance = float("inf")
    for index in spatial.nearby_candidates(current):
        stroke = strokes[index]
        start_distance = squared_distance(current, (stroke.x1, stroke.y1))
        end_distance = squared_distance(current, (stroke.x2, stroke.y2))
        if start_distance < best_distance:
            best_index = index
            best_reverse = False
            best_distance = start_distance
        if end_distance < best_distance:
            best_index = index
            best_reverse = True
            best_distance = end_distance
    return best_index, best_reverse


def stroke_seed(seed: int, stroke: Stroke, index: int) -> int:
    return (
        seed
        ^ (stroke.x1 * 73856093)
        ^ (stroke.y1 * 19349663)
        ^ (stroke.x2 * 83492791)
        ^ (stroke.y2 * 2654435761)
        ^ (index * 97531)
    ) & 0xFFFFFFFF


def append_limited(target: List[Stroke], source: List[Stroke], max_count: int) -> None:
    for stroke in source:
        if len(target) >= max_count:
            return
        target.append(stroke)


def append_if_long_enough(
    target: List[Stroke],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    canvas_size: Tuple[int, int],
    min_run: int,
) -> None:
    stroke = clamped_stroke(x1, y1, x2, y2, canvas_size)
    if stroke_length(stroke) >= min_run:
        target.append(stroke)


def clamped_stroke(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    canvas_size: Tuple[int, int],
) -> Stroke:
    width, height = canvas_size
    return Stroke(
        clamp(round(x1), 0, width - 1),
        clamp(round(y1), 0, height - 1),
        clamp(round(x2), 0, width - 1),
        clamp(round(y2), 0, height - 1),
    )


def lerp_point(start: Tuple[float, float], end: Tuple[float, float], amount: float) -> Tuple[float, float]:
    return (
        start[0] + (end[0] - start[0]) * amount,
        start[1] + (end[1] - start[1]) * amount,
    )


def squared_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def build_major_line_mask(luminance: Image.Image, threshold: int, spacing: int) -> Tuple[Image.Image, Image.Image]:
    """Return likely visible line pixels and their edge-strength image.

    The old lines mode selected the largest dark component, which threw away
    eyes, mouths, folds, and other internal marks. This mask keeps high-contrast
    edges and dark pixels that behave like boundaries anywhere in the image.
    """

    opencv_result = build_opencv_major_line_mask(luminance, threshold, spacing)
    if opencv_result is not None:
        return opencv_result

    width, height = luminance.size
    smoothed = luminance.filter(ImageFilter.MedianFilter(3))
    edge_strength = ImageOps.autocontrast(smoothed.filter(ImageFilter.FIND_EDGES))
    edge_cutoff = adaptive_edge_cutoff(edge_strength)
    contrast_cutoff = max(18, min(55, threshold // 4))
    radius = max(1, min(3, spacing // 2))
    filter_size = radius * 2 + 1
    local_high = smoothed.filter(ImageFilter.MaxFilter(filter_size))
    local_low = smoothed.filter(ImageFilter.MinFilter(filter_size))

    pixels = smoothed.load()
    edge_pixels = edge_strength.load()
    high_pixels = local_high.load()
    low_pixels = local_low.load()
    line_mask = Image.new("1", luminance.size, 255)
    line_pixels = line_mask.load()

    for y in range(height):
        for x in range(width):
            local_contrast = high_pixels[x, y] - low_pixels[x, y]
            strong_edge = edge_pixels[x, y] >= edge_cutoff and local_contrast >= contrast_cutoff
            dark_boundary = pixels[x, y] < threshold and local_contrast >= contrast_cutoff
            if strong_edge or dark_boundary:
                line_pixels[x, y] = 0

    return line_mask.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3)), edge_strength


def build_opencv_major_line_mask(
    luminance: Image.Image,
    threshold: int,
    spacing: int,
) -> Optional[Tuple[Image.Image, Image.Image]]:
    if cv2 is None or np is None:
        return None

    gray = np.array(luminance, dtype=np.uint8)
    denoised = cv2.bilateralFilter(gray, 5, 42, 42)
    equalized = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(denoised)
    edges = adaptive_canny_edges(equalized, threshold)

    kernel_size = max(2, min(5, spacing + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    local_radius = max(3, min(9, spacing * 2 + 1))
    if local_radius % 2 == 0:
        local_radius += 1
    local_high = cv2.dilate(denoised, np.ones((local_radius, local_radius), dtype=np.uint8))
    local_low = cv2.erode(denoised, np.ones((local_radius, local_radius), dtype=np.uint8))
    local_contrast = cv2.subtract(local_high, local_low)
    dark = cv2.threshold(denoised, threshold, 255, cv2.THRESH_BINARY_INV)[1]
    contrast_cutoff = max(16, min(52, threshold // 4))
    textured_dark = cv2.bitwise_and(dark, cv2.threshold(local_contrast, contrast_cutoff, 255, cv2.THRESH_BINARY)[1])

    combined = cv2.bitwise_or(edges, textured_dark)
    combined = remove_tiny_components_array(combined, max(3, spacing * spacing))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)

    edge_strength = Image.fromarray(cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), mode="L")
    return binary_array_to_mask(combined), edge_strength


def remove_tiny_components_array(mask, min_area: int):
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    output = np.zeros_like(mask)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            output[labels == label] = 255
    return output


def adaptive_edge_cutoff(edge_strength: Image.Image) -> int:
    cutoff = image_histogram_percentile(edge_strength, 0.72, ignore_zero=True)
    return max(18, min(96, cutoff))


def image_histogram_percentile(image: Image.Image, percentile: float, ignore_zero: bool = False) -> int:
    histogram = image.histogram()
    start = 1 if ignore_zero else 0
    total = sum(histogram[start:])
    if total <= 0:
        return 255

    target = total * percentile
    running = 0
    for value in range(start, 256):
        running += histogram[value]
        if running >= target:
            return value
    return 255


def unique_strokes(strokes: List[Stroke]) -> List[Stroke]:
    seen = set()
    result: List[Stroke] = []
    for stroke in strokes:
        key = (stroke.x1, stroke.y1, stroke.x2, stroke.y2)
        reverse_key = (stroke.x2, stroke.y2, stroke.x1, stroke.y1)
        if key in seen or reverse_key in seen:
            continue
        seen.add(key)
        result.append(stroke)
    return result


def merge_nearby_line_strokes(strokes: List[Stroke], spacing: int) -> List[Stroke]:
    merge_gap = max(1, spacing * 2)
    horizontal = {}
    vertical = {}
    diagonal_down = {}
    diagonal_up = {}
    passthrough: List[Stroke] = []

    for stroke in strokes:
        dx = stroke.x2 - stroke.x1
        dy = stroke.y2 - stroke.y1
        if dy == 0:
            add_interval(horizontal, stroke.y1, stroke.x1, stroke.x2)
        elif dx == 0:
            add_interval(vertical, stroke.x1, stroke.y1, stroke.y2)
        elif abs(dx) == abs(dy) and dx * dy > 0:
            add_interval(diagonal_down, stroke.y1 - stroke.x1, stroke.x1, stroke.x2)
        elif abs(dx) == abs(dy) and dx * dy < 0:
            add_interval(diagonal_up, stroke.y1 + stroke.x1, stroke.x1, stroke.x2)
        else:
            passthrough.append(stroke)

    merged: List[Stroke] = []
    for y, intervals in horizontal.items():
        merged.extend(Stroke(start, y, end, y) for start, end in merge_intervals(intervals, merge_gap))
    for x, intervals in vertical.items():
        merged.extend(Stroke(x, start, x, end) for start, end in merge_intervals(intervals, merge_gap))
    for offset, intervals in diagonal_down.items():
        for start, end in merge_intervals(intervals, merge_gap):
            merged.append(Stroke(start, start + offset, end, end + offset))
    for offset, intervals in diagonal_up.items():
        for start, end in merge_intervals(intervals, merge_gap):
            merged.append(Stroke(start, offset - start, end, offset - end))

    return passthrough + merged


def add_interval(groups, key: int, start: int, end: int) -> None:
    left, right = sorted((start, end))
    groups.setdefault(key, []).append((left, right))


def merge_intervals(intervals: List[Tuple[int, int]], merge_gap: int) -> List[Tuple[int, int]]:
    if not intervals:
        return []

    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + merge_gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def discard_short_line_strokes(strokes: List[Stroke], spacing: int, min_run: int) -> List[Stroke]:
    minimum_length = max(min_run, spacing * 4)
    return [stroke for stroke in strokes if stroke_length(stroke) >= minimum_length]


def discard_frame_like_strokes(strokes: List[Stroke], canvas_size: Tuple[int, int]) -> List[Stroke]:
    width, height = canvas_size
    vertical_coverage = {}
    horizontal_coverage = {}
    for stroke in strokes:
        if stroke.x1 == stroke.x2:
            vertical_coverage[stroke.x1] = vertical_coverage.get(stroke.x1, 0) + stroke_length(stroke)
        if stroke.y1 == stroke.y2:
            horizontal_coverage[stroke.y1] = horizontal_coverage.get(stroke.y1, 0) + stroke_length(stroke)

    frame_columns = {x for x, coverage in vertical_coverage.items() if coverage >= height * 0.85}
    frame_rows = {y for y, coverage in horizontal_coverage.items() if coverage >= width * 0.85}

    return [
        stroke
        for stroke in strokes
        if not (
            is_frame_like_stroke(stroke, canvas_size)
            or (stroke.x1 == stroke.x2 and stroke.x1 in frame_columns)
            or (stroke.y1 == stroke.y2 and stroke.y1 in frame_rows)
        )
    ]


def is_frame_like_stroke(stroke: Stroke, canvas_size: Tuple[int, int]) -> bool:
    width, height = canvas_size
    length = stroke_length(stroke)
    is_vertical = stroke.x1 == stroke.x2
    is_horizontal = stroke.y1 == stroke.y2
    near_horizontal_edge = min(stroke.y1, height - 1 - stroke.y1) <= 1
    near_vertical_edge = min(stroke.x1, width - 1 - stroke.x1) <= 1
    if is_vertical and length >= height * 0.72:
        return True
    if is_horizontal and length >= width * 0.72:
        return True
    if is_vertical and length >= height * 0.62:
        return True
    if is_horizontal and length >= width * 0.62:
        return True
    if is_horizontal and near_horizontal_edge and length >= min(width, height) * 0.45:
        return True
    if is_vertical and near_vertical_edge and length >= min(width, height) * 0.45:
        return True
    return False


def trace_boundary_marks(boundary: Image.Image, spacing: int, min_run: int) -> List[Stroke]:
    width, height = boundary.size
    pixels = boundary.load()
    stride = max(1, spacing)
    length = max(2, min_run, spacing * 2)
    strokes: List[Stroke] = []

    for y in range(0, height, stride):
        for x in range(0, width, stride):
            if pixels[x, y] != 0:
                continue
            vertical_neighbors = count_boundary_neighbors(pixels, x, y, width, height, vertical=True)
            horizontal_neighbors = count_boundary_neighbors(pixels, x, y, width, height, vertical=False)
            if vertical_neighbors > horizontal_neighbors:
                strokes.append(
                    Stroke(
                        x,
                        clamp(y - length // 2, 0, height - 1),
                        x,
                        clamp(y + length // 2, 0, height - 1),
                    )
                )
            else:
                strokes.append(
                    Stroke(
                        clamp(x - length // 2, 0, width - 1),
                        y,
                        clamp(x + length // 2, 0, width - 1),
                        y,
                    )
                )

    return strokes


def count_boundary_neighbors(pixels, x: int, y: int, width: int, height: int, vertical: bool) -> int:
    count = 0
    offsets = ((0, -1), (0, 1)) if vertical else ((-1, 0), (1, 0))
    for dx, dy in offsets:
        nx = x + dx
        ny = y + dy
        if 0 <= nx < width and 0 <= ny < height and pixels[nx, ny] == 0:
            count += 1
    return count


def extract_main_object_mask(luminance: Image.Image, threshold: int) -> Image.Image:
    """Return a binary mask containing only the largest dark connected component."""

    width, height = luminance.size
    raw = luminance.point(lambda pixel: 255 if pixel < threshold else 0, mode="L")
    # Connect close strokes and small gaps before component selection.
    connected = raw.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(3))
    pixels = connected.load()
    foreground = [[pixels[x, y] > 0 for x in range(width)] for y in range(height)]
    seen = [[False for _x in range(width)] for _y in range(height)]
    best_component: List[Tuple[int, int]] = []
    best_score = 0.0

    for y in range(height):
        for x in range(width):
            if seen[y][x] or not foreground[y][x]:
                continue
            component = collect_component(foreground, seen, x, y, width, height)
            score = component_score(component)
            if score > best_score:
                best_component = component
                best_score = score

    if not best_component:
        return Image.new("1", luminance.size, 255)

    mask = Image.new("1", luminance.size, 255)
    mask_pixels = mask.load()
    for x, y in best_component:
        mask_pixels[x, y] = 0
    return mask


def component_score(component: List[Tuple[int, int]]) -> float:
    if not component:
        return 0.0
    xs = [point[0] for point in component]
    ys = [point[1] for point in component]
    width = max(xs) - min(xs) + 1
    height = max(ys) - min(ys) + 1
    return len(component) + math.sqrt(width * height)


def collect_component(
    foreground: List[List[bool]],
    seen: List[List[bool]],
    start_x: int,
    start_y: int,
    width: int,
    height: int,
) -> List[Tuple[int, int]]:
    stack = [(start_x, start_y)]
    seen[start_y][start_x] = True
    component: List[Tuple[int, int]] = []

    while stack:
        x, y = stack.pop()
        component.append((x, y))
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if seen[ny][nx] or not foreground[ny][nx]:
                continue
            seen[ny][nx] = True
            stack.append((nx, ny))

    return component


def keep_longest_strokes(strokes: List[Stroke], limit: int) -> List[Stroke]:
    ranked = sorted(strokes, key=stroke_length, reverse=True)
    return ranked[:limit]


def keep_strongest_line_strokes(
    strokes: List[Stroke],
    luminance: Image.Image,
    edge_strength: Image.Image,
    threshold: int,
    limit: int,
) -> List[Stroke]:
    ranked = sorted(
        strokes,
        key=lambda stroke: (
            stroke_importance(stroke, luminance, edge_strength, threshold),
            stroke_length(stroke),
        ),
        reverse=True,
    )
    return ranked[:limit]


def stroke_importance(
    stroke: Stroke,
    luminance: Image.Image,
    edge_strength: Image.Image,
    threshold: int,
) -> float:
    length = stroke_length(stroke)
    if length <= 0:
        return 0.0

    luminance_pixels = luminance.load()
    edge_pixels = edge_strength.load()
    width, height = luminance.size
    steps = max(abs(stroke.x2 - stroke.x1), abs(stroke.y2 - stroke.y1))
    total = 0.0
    samples = 0

    for index in range(steps + 1):
        amount = index / max(steps, 1)
        x = round(stroke.x1 + (stroke.x2 - stroke.x1) * amount)
        y = round(stroke.y1 + (stroke.y2 - stroke.y1) * amount)
        if x < 0 or y < 0 or x >= width or y >= height:
            continue
        darkness = pixel_darkness(luminance_pixels[x, y], threshold)
        edge = edge_pixels[x, y] / 255.0
        total += darkness * 0.65 + edge * 0.85
        samples += 1

    if samples == 0:
        return 0.0
    return (total / samples) * math.sqrt(length)


def stroke_length(stroke: Stroke) -> int:
    return max(abs(stroke.x2 - stroke.x1), abs(stroke.y2 - stroke.y1)) + 1


def collect_horizontal_line_candidates(
    candidates: List[Tuple[float, Stroke]],
    pixels,
    y: int,
    width: int,
    threshold: int,
    min_run: int,
) -> None:
    start: Optional[int] = None
    darkness_total = 0.0

    for x in range(width):
        darkness = pixel_darkness(pixels[x, y], threshold)
        if darkness > 0:
            if start is None:
                start = x
                darkness_total = 0.0
            darkness_total += darkness
        elif start is not None:
            append_scored_line(candidates, Stroke(start, y, x - 1, y), darkness_total, min_run)
            start = None
            darkness_total = 0.0

    if start is not None:
        append_scored_line(candidates, Stroke(start, y, width - 1, y), darkness_total, min_run)


def collect_diagonal_line_candidates(
    candidates: List[Tuple[float, Stroke]],
    pixels,
    offset: int,
    width: int,
    height: int,
    threshold: int,
    min_run: int,
    direction: int,
) -> None:
    start: Optional[Tuple[int, int]] = None
    last: Optional[Tuple[int, int]] = None
    darkness_total = 0.0

    for y in range(height):
        x = offset + y if direction == 1 else offset - y
        if x < 0 or x >= width:
            if start is not None and last is not None:
                append_scored_line(candidates, Stroke(start[0], start[1], last[0], last[1]), darkness_total, min_run)
            start = None
            last = None
            darkness_total = 0.0
            continue

        darkness = pixel_darkness(pixels[x, y], threshold)
        if darkness > 0:
            if start is None:
                start = (x, y)
                darkness_total = 0.0
            last = (x, y)
            darkness_total += darkness
        elif start is not None and last is not None:
            append_scored_line(candidates, Stroke(start[0], start[1], last[0], last[1]), darkness_total, min_run)
            start = None
            last = None
            darkness_total = 0.0

    if start is not None and last is not None:
        append_scored_line(candidates, Stroke(start[0], start[1], last[0], last[1]), darkness_total, min_run)


def append_scored_line(
    candidates: List[Tuple[float, Stroke]],
    stroke: Stroke,
    darkness_total: float,
    min_run: int,
) -> None:
    length = max(abs(stroke.x2 - stroke.x1), abs(stroke.y2 - stroke.y1)) + 1
    if length < min_run:
        return
    average_darkness = darkness_total / max(length, 1)
    score = average_darkness * math.sqrt(length)
    if score <= 0:
        return
    candidates.append((score, stroke))


def pixel_darkness(pixel: int, threshold: int) -> float:
    return max(0.0, (threshold - pixel) / max(threshold, 1))


def sort_strokes_for_drawing(strokes: List[Stroke]) -> List[Stroke]:
    def key(stroke: Stroke) -> Tuple[int, int]:
        return ((stroke.y1 + stroke.y2) // 2, (stroke.x1 + stroke.x2) // 2)

    ordered = sorted(strokes, key=key)
    result: List[Stroke] = []
    for row_index, stroke in enumerate(ordered):
        if row_index % 2 == 1:
            result.append(Stroke(stroke.x2, stroke.y2, stroke.x1, stroke.y1))
        else:
            result.append(stroke)
    return result


def cell_marks(
    x: int,
    y: int,
    cell: int,
    width: int,
    height: int,
    mark_count: int,
    min_run: int,
    seed: int,
) -> List[Stroke]:
    left = x
    top = y
    right = min(x + cell - 1, width - 1)
    bottom = min(y + cell - 1, height - 1)
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    short = max(1, min_run, round(cell * 0.35))
    long = max(short, round(cell * 0.72))
    jitter = (seed % 3) - 1

    patterns = [
        Stroke(cx - short // 2, cy + jitter, cx + short // 2, cy + jitter),
        Stroke(cx + jitter, cy - short // 2, cx + jitter, cy + short // 2),
        Stroke(cx - long // 2, cy - long // 2, cx + long // 2, cy + long // 2),
        Stroke(cx - long // 2, cy + long // 2, cx + long // 2, cy - long // 2),
        Stroke(left + 1, cy, right - 1, cy),
    ]

    result: List[Stroke] = []
    order = [0, 2, 1, 3, 4] if seed % 2 == 0 else [1, 3, 0, 2, 4]
    for index in order[:mark_count]:
        stroke = patterns[index]
        result.append(
            Stroke(
                clamp(stroke.x1, left, right),
                clamp(stroke.y1, top, bottom),
                clamp(stroke.x2, left, right),
                clamp(stroke.y2, top, bottom),
            )
        )
    return result


def edge_mark(pixels, x: int, y: int, cell: int, width: int, height: int, min_run: int) -> Stroke:
    left = x
    top = y
    right = min(x + cell - 1, width - 1)
    bottom = min(y + cell - 1, height - 1)
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    length = max(min_run, round(cell * 0.8))

    left_avg = average_region(pixels, left, top, cx, bottom)
    right_avg = average_region(pixels, cx, top, right, bottom)
    top_avg = average_region(pixels, left, top, right, cy)
    bottom_avg = average_region(pixels, left, cy, right, bottom)
    gx = right_avg - left_avg
    gy = bottom_avg - top_avg

    if abs(gx) > abs(gy):
        return Stroke(cx, clamp(cy - length // 2, top, bottom), cx, clamp(cy + length // 2, top, bottom))
    return Stroke(clamp(cx - length // 2, left, right), cy, clamp(cx + length // 2, left, right), cy)


def cell_contrast(pixels, x: int, y: int, cell: int, width: int, height: int) -> float:
    values = []
    stride = max(1, cell // 3)
    for py in range(y, min(y + cell, height), stride):
        for px in range(x, min(x + cell, width), stride):
            values.append(pixels[px, py])
    if not values:
        return 0.0
    return float(max(values) - min(values))


def average_cell_luminance(pixels, x: int, y: int, cell: int, width: int, height: int) -> float:
    total = 0
    count = 0
    for py in range(y, min(y + cell, height)):
        for px in range(x, min(x + cell, width)):
            total += pixels[px, py]
            count += 1
    return total / max(count, 1)


def average_region(pixels, left: int, top: int, right: int, bottom: int) -> float:
    total = 0
    count = 0
    for py in range(top, max(top + 1, bottom + 1)):
        for px in range(left, max(left + 1, right + 1)):
            total += pixels[px, py]
            count += 1
    return total / max(count, 1)


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(value, upper))


def trace_strokes(mask: Image.Image, step: int, min_run: int, mode: str) -> List[Stroke]:
    if mode == "outline":
        return trace_outline_strokes(mask, step, min_run)
    if mode == "hatch":
        return trace_diagonal_hatch_strokes(mask, step, min_run, direction=1)
    if mode == "sketch":
        outline = trace_outline_strokes(mask, step, min_run)
        hatch = trace_diagonal_hatch_strokes(mask, step, min_run, direction=1)
        second_hatch = trace_diagonal_hatch_strokes(mask, step * 2, min_run, direction=-1)
        return outline + hatch + second_hatch
    return trace_scanline_strokes(mask, step, min_run)


def render_preview(
    strokes: Iterable[Stroke],
    canvas_size: Tuple[int, int],
    output_path: Path,
    line_width: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview = build_preview_image(strokes, canvas_size, line_width)
    preview.save(output_path)


def build_preview_image(
    strokes: Iterable[Stroke],
    canvas_size: Tuple[int, int],
    line_width: int,
) -> Image.Image:
    preview = Image.new("RGB", canvas_size, "white")
    draw = ImageDraw.Draw(preview)

    for stroke in strokes:
        draw.line((stroke.x1, stroke.y1, stroke.x2, stroke.y2), fill="black", width=line_width)

    return preview
