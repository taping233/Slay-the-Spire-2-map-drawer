from __future__ import annotations

import argparse
import json
from pathlib import Path

from .image_trace import (
    parse_canvas_size,
    render_preview,
    load_luminance_canvas,
    load_mask,
    trace_handdrawn_strokes,
    trace_line_reduction_strokes,
    trace_strokes,
)
from .mouse_draw import calibrate, draw_strokes, load_draw_area


def build_strokes(args: argparse.Namespace):
    canvas_size = parse_canvas_size(args.canvas)
    if args.mode in ("lines", "handdrawn"):
        luminance = load_luminance_canvas(Path(args.image), canvas_size, args.invert)
        if args.mode == "handdrawn":
            strokes = trace_handdrawn_strokes(
                luminance,
                args.step,
                args.threshold,
                args.min_run,
                args.line_count,
            )
        else:
            strokes = trace_line_reduction_strokes(
                luminance,
                args.step,
                args.threshold,
                args.min_run,
                args.line_count,
            )
        return canvas_size, strokes

    mask = load_mask(Path(args.image), canvas_size, args.threshold, args.invert)
    strokes = trace_strokes(mask, args.step, args.min_run, args.mode)
    return canvas_size, strokes


def preview_line_width(args: argparse.Namespace) -> int:
    if args.mode in ("lines", "handdrawn"):
        return max(1, min(args.step // 3, 4))
    return max(args.step, 1)


def add_trace_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("image", help="Path to the image to draw.")
    parser.add_argument("--canvas", default="640x360", help="Virtual drawing canvas, for example 640x360.")
    parser.add_argument("--threshold", type=int, default=170, help="0-255 cutoff; lower means fewer dark pixels.")
    parser.add_argument("--step", type=int, default=4, help="Vertical pixel spacing between scanlines.")
    parser.add_argument("--min-run", type=int, default=3, help="Ignore shorter horizontal runs.")
    parser.add_argument(
        "--mode",
        default="lines",
        choices=("scanline", "lines", "handdrawn"),
        help="Stroke style: lines, handdrawn, or scanline.",
    )
    parser.add_argument("--line-count", type=int, default=500, help="Maximum number of strokes for lines/handdrawn mode.")
    parser.add_argument("--invert", action="store_true", help="Draw bright parts instead of dark parts.")


def preview_command(args: argparse.Namespace) -> None:
    canvas_size, strokes = build_strokes(args)
    render_preview(strokes, canvas_size, Path(args.output), preview_line_width(args))
    print(f"Preview saved to {args.output}")
    print(f"Generated {len(strokes)} strokes")


def plan_command(args: argparse.Namespace) -> None:
    canvas_size, strokes = build_strokes(args)
    payload = {
        "canvas": {"width": canvas_size[0], "height": canvas_size[1]},
        "stroke_count": len(strokes),
        "strokes": [stroke.to_dict() for stroke in strokes],
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Stroke plan saved to {args.output}")


def calibrate_command(args: argparse.Namespace) -> None:
    calibrate(Path(args.config))
    print(f"Calibration saved to {args.config}")


def draw_command(args: argparse.Namespace) -> None:
    canvas_size, strokes = build_strokes(args)
    print(f"Generated {len(strokes)} strokes")

    if args.preview:
        render_preview(strokes, canvas_size, Path(args.preview), preview_line_width(args))
        print(f"Preview saved to {args.preview}")

    if args.dry_run:
        return

    area = load_draw_area(Path(args.config))
    draw_strokes(
        strokes=strokes,
        area=area,
        canvas_size=canvas_size,
        countdown=args.countdown,
        move_duration=args.move_duration,
        stroke_pause=args.stroke_pause,
        button=args.button,
        jump_duration=args.jump_duration,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sts2-drawer",
        description="Convert an image into slow mouse strokes for drawing in Slay the Spire 2 map screens.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview_parser = subparsers.add_parser("preview", help="Create a preview PNG from an image.")
    add_trace_options(preview_parser)
    preview_parser.add_argument("--output", default="preview.png", help="Preview image output path.")
    preview_parser.set_defaults(func=preview_command)

    plan_parser = subparsers.add_parser("plan", help="Export the generated stroke plan as JSON.")
    add_trace_options(plan_parser)
    plan_parser.add_argument("--output", default="stroke_plan.json", help="Stroke plan output path.")
    plan_parser.set_defaults(func=plan_command)

    calibrate_parser = subparsers.add_parser("calibrate", help="Record the drawable screen area.")
    calibrate_parser.add_argument("--config", default="config.json", help="Config file to write.")
    calibrate_parser.set_defaults(func=calibrate_command)

    gui_parser = subparsers.add_parser("gui", help="Open the visual overlay UI.")
    gui_parser.set_defaults(func=gui_command)

    draw_parser = subparsers.add_parser("draw", help="Draw an image using mouse drags.")
    add_trace_options(draw_parser)
    draw_parser.add_argument("--config", default="config.json", help="Config file created by calibrate.")
    draw_parser.add_argument("--preview", help="Optionally save a preview before drawing.")
    draw_parser.add_argument("--dry-run", action="store_true", help="Generate strokes and preview without moving the mouse.")
    draw_parser.add_argument("--countdown", type=int, default=5, help="Seconds before drawing starts.")
    draw_parser.add_argument("--move-duration", type=float, default=0.01, help="Seconds for each drawing drag.")
    draw_parser.add_argument("--jump-duration", type=float, default=0.0, help="Seconds for moving to the next stroke start.")
    draw_parser.add_argument("--stroke-pause", type=float, default=0.0, help="Pause between strokes.")
    draw_parser.add_argument("--button", default="left", choices=("left", "right", "middle"), help="Mouse button to drag.")
    draw_parser.set_defaults(func=draw_command)

    args = parser.parse_args()
    args.func(args)


def gui_command(args: argparse.Namespace) -> None:
    from .gui import run_gui

    run_gui()


if __name__ == "__main__":
    main()
