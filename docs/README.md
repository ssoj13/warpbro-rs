# WarpBro documentation

The guides below describe the current implementation; open work is in [PLAN.md](../PLAN.md).

- [Colour](color.md): BT.709 / BT.2020 / ACES gamuts, sRGB / PQ / HLG curves, and the ACEScg render pipeline.
- [Workspace, materials, and cached playback](workspace.md): node selection, assignment, gesture Undo, panel shortcuts, and final/draft preview caches.
- [Transparent and absorbing glass](glass.md): transmission, green glass/water, depth controls, legacy scenes, and geometry limits.
- [CUDA startup investigation](cuda-startup.md): measured context/module timings, system-load caveats, reproduction, and pending work.
- [Changelog](../CHANGELOG.md): published behavior and the current update.

## Validation limits

The current release executable passed 211 tests with 8 ignored GPU/visual probes (62.07 s). The active CUDA glass regression covers signed/unsigned geometry, legacy/World dispatch, and depth absorption. The ignored glass visual probe was also run separately and produced six inspected clear/green sphere and Mandelbulb images.

Automated tests and headless render checks do not establish native UI latency or visual alignment at every DPI. Native scrolling, cached playback, and interaction under GPU load still need inspection. Disk preview caching, a unified Curve Editor, and a runtime compiled-module cache are not completed features. The bootstrap now warms the CUDA driver's existing JIT cache after building instead.
