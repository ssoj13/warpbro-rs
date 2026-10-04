# WarpBro documentation

The guides below describe the current implementation. Dated plans preserve the decisions and verification history of earlier revisions.

- [Workspace, materials, and cached playback](workspace.md): node selection, assignment, gesture Undo, panel shortcuts, and final/draft preview caches.
- [Transparent and absorbing glass](glass.md): transmission, green glass/water, depth controls, legacy scenes, and geometry limits.
- [CUDA startup investigation](cuda-startup.md): measured context/module timings, system-load caveats, reproduction, and pending work.
- [Changelog](../CHANGELOG.md): published behavior and the current update.
- [Render throughput and convergence](../plans/plan7.md): production-world measurements and Nsight evidence.
- [Export and video quality](../plans/plan6.md): Vulkan HEVC, software mitigation, and graceful cancellation.
- [Reusable workspace roadmap](../plans/8-reusable-workspace.md): completed primitives and remaining panel extraction.

## Validation limits

The current release executable passed 211 tests with 8 ignored GPU/visual probes (62.07 s). The active CUDA glass regression covers signed/unsigned geometry, legacy/World dispatch, and depth absorption. The ignored glass visual probe was also run separately and produced six inspected clear/green sphere and Mandelbulb images.

Automated tests and headless render checks do not establish native UI latency or visual alignment at every DPI. Native scrolling, cached playback, and interaction under GPU load still need inspection. Disk preview caching, a unified Curve Editor, and a runtime compiled-module cache are not completed features. The bootstrap now warms the CUDA driver's existing JIT cache after building instead.
