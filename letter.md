Interesting weekend!
OpenFX port suddenly turned out into a separate fractal engine.
Raw fractal engines are not very interesting, so I quickly added some harness around:

- Nvidia has released cuda-oxide recently and it works kind of just on Linux. This gentlemen patched it for Windows: https://github.com/ansidium/cuda-oxide-windows. So I decided to give it a try this weekend and suddenly it worked! I had to do couple of fixes for the latest Stable and made it work on my Windows, so technically fractals are Rust CUDA kernels.
- A Path Tracer. Small custom pure Rust "render-rs" engine I'm using in my tools.
- Standard Surface material (a Rust port of https://github.com/Autodesk/standard-surface from MaterialX: https://github.com/AcademySoftwareFoundation/MaterialX)
- The app itself is node-based, it's easier to add/remove things this way. So World, Environment, Fractal, Camera, Light, materials - are all nodes. And timeline layers at the same time (timeline layers = nodes displayed as a frame range, that's it).
- A set of standard viewers: Outliner, Timeline, Material Library, Color settings, Attribute Editor, Video Encoder.
- ACES2 color system (Rust version)
- HDR display capabilities
- Encoding EXR files with exr-rs (a full Rust port of https://github.com/academysoftwarefoundation/openexr)
- Video encoding via Rust Ffmpeg-rs.
- OIDN: a Rust port of https://github.com/RenderKit/oidn, Intel Open Image Denoise library.

I'm not sure if I went too far here, but now I have a full stack of "Lego blocks" allowing to build quite a complex Rust graphics apps in a day or two.

Funny enough I cannot even make screenshots of this app in PQ mode (HDR), none of standard tools work, and it looks crazy at 500 nit.


Next up: a Rust port of OpenFX (yep, it's done).