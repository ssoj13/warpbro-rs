"""Compare --world-bench raw HDR artifacts against independent higher-SPP references.

Usage: python tools/convergence.py BASELINE CANDIDATE --output REPORT.json
Requires NumPy. Times include the per-batch presentation work recorded by the harness.
MSE*time is an efficiency estimate, not a directly observed stopping-time speedup.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(directory):
    cases = {}
    for path in sorted(Path(directory).glob("*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("schema") != 1 or "raw" not in item:
            continue
        assert item["world_render"] and not item["denoise_enabled"]
        assert item["exposure_multiplier"] == 1.0 and item["saturation"] == 1.0
        image = np.fromfile(path.parent / item["raw"]["file"], dtype="<f4")
        assert image.size == item["width"] * item["height"] * 3
        assert np.isfinite(image).all()
        cases.setdefault(item["case"], []).append((item, image.astype(np.float64)))
    return cases


def summarize(entries, reference, count):
    trials = [(meta, image) for meta, image in entries if meta["samples"] == count]
    assert len(trials) >= 3, "At least three independent low-SPP seeds required"
    assert len({meta["seed"] for meta, _ in trials}) == len(trials)
    images = np.stack([image for _, image in trials])
    times = np.array([meta["elapsed_seconds"] for meta, _ in trials])
    errors = np.mean((images - reference) ** 2, axis=1)
    return {
        "samples": count, "seeds": [meta["seed"] for meta, _ in trials],
        "elapsed_seconds": times.tolist(),
        "median_seconds": float(np.median(times)),
        "linear_rgb_mse": errors.tolist(),
        "mean_mse": float(errors.mean()),
        "mse_standard_error": float(errors.std(ddof=1) / np.sqrt(len(errors))),
        "independent_seed_variance": float(images.var(axis=0, ddof=1).mean()),
        "mean_mse_times_seconds": float(np.mean(errors * times)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    baseline, candidate = load(args.baseline), load(args.candidate)
    results = {}
    for name in sorted(baseline.keys() & candidate.keys()):
        before, after = baseline[name], candidate[name]
        maximum = max(meta["samples"] for meta, _ in after)
        references = [(meta, image) for meta, image in after if meta["samples"] == maximum]
        assert maximum >= 512
        ref_seeds = {meta["seed"] for meta, _ in references}
        count = min(meta["samples"] for meta, _ in after)
        assert count == min(meta["samples"] for meta, _ in before)
        assert not ref_seeds & {meta["seed"] for meta, _ in after if meta["samples"] == count}
        reference_images = np.stack([image for _, image in references])
        reference = reference_images.mean(axis=0)
        reference_noise = (float(reference_images.var(axis=0, ddof=1).mean()) / len(references)
                           if len(references) > 1 else None)
        # Settings are frozen; only implementation/revision differs.
        a, b = before[0][0], after[0][0]
        for key in ("width", "height", "batch", "frame", "evaluated_scene",
                    "evaluated_objects", "evaluated_lights"):
            assert a[key] == b[key], f"Changed authored inputs: {name} / {key}"
        old = summarize(before, reference, count)
        new = summarize(after, reference, count)
        results[name] = {
            "reference_spp_per_seed": maximum,
            "reference_seeds": sorted(ref_seeds),
            "reference_mean_rgb": reference.reshape(-1, 3).mean(axis=0).tolist(),
            "reference_noise_mse_estimate": reference_noise,
            "valid_convergence_comparison": reference_noise is not None and reference_noise < new["mean_mse"] * 0.2,
            "reference_warning": (None if reference_noise is not None and reference_noise < new["mean_mse"] * 0.2
                                  else "Independent references are too noisy to establish image-error convergence; inspect rare outliers before claiming an improvement."),
            "before": old, "after": new,
            "throughput_ratio": old["median_seconds"] / new["median_seconds"],
            "mse_ratio": old["mean_mse"] / new["mean_mse"],
            "estimated_error_time_efficiency_ratio":
                old["mean_mse_times_seconds"] / new["mean_mse_times_seconds"],
            "seed_variance_ratio":
                old["independent_seed_variance"] / new["independent_seed_variance"],
        }
    assert results
    report = {
        "metric": "raw scene-linear Rec.709 RGB, no denoise or display transform",
        "limits": "Finite noisy reference; at least three independent low-SPP seeds; MSE*time predicts asymptotic efficiency, not measured time to a stopping threshold. Timing includes presentation and background GPU contention.",
        "cases": results,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
