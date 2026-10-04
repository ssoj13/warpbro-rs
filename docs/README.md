# WarpBro documentation

The guides below describe the current implementation. Dated plans preserve the decisions and verification history of earlier revisions.

- [Workspace, materials, and cached playback](workspace.md): node selection, assignment, gesture Undo, panel shortcuts, and final/draft preview caches.
- [CUDA startup investigation](cuda-startup.md): measured context/module timings, system-load caveats, reproduction, and pending work.
- [Changelog](../CHANGELOG.md): published behavior and the current update.
- [Render throughput and convergence](../plans/plan7.md): production-world measurements and Nsight evidence.
- [Export and video quality](../plans/plan6.md): Vulkan HEVC, software mitigation, and graceful cancellation.
- [Reusable workspace roadmap](../plans/8-reusable-workspace.md): completed primitives and remaining panel extraction.

## Validation limits

The final repeat of the current release test executable passed 199 tests with 7 ignored GPU tests (72.71 s). The first run hit the export-cancellation test's 90-second deadline under load; its isolated repeat passed in 1.39 s. Those skipped GPU tests were not rerun in this documentation update.

Automated tests and headless render checks do not establish native UI latency or visual alignment at every DPI. Native scrolling, cached playback, and interaction under GPU load still need inspection. Disk preview caching, a unified Curve Editor, true refractive glass, and a compiled CUDA-module cache are not completed features.
