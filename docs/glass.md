# Render transparent and absorbing glass

Select a Material node in Materials or the Outliner, then edit it in the shared Attribute Editor. Assign its UUID through the object's **Material** field or **Apply to object**. Transmission, tint, roughness, and depth use the same animation and Undo system as other node attributes.

## Choose a glass preset

The Material Library includes **GlassClear**, **GlassAmber**, **GlassFrosted**, **GlassBottleGreen**, and **GlassWaterGreen**, alongside the legacy **Glass** preset. Bottle green uses IOR 1.52 and absorption depth 0.5; green water uses IOR 1.333 and depth 2.0. These are curated RGB looks, not measured optical data.

Glass presets now select Standard Surface and map library opacity to `transmission = 1 - opacity`. They do not approximate transparency by reducing diffuse and adding a coat.

## Set transmission and absorption

| Attribute                    | Meaning                                                                                                                          |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Transmission                 | Fraction of dielectric base energy assigned to refraction; 0 is opaque, 1 is fully transmissive apart from interface reflection. |
| Specular IOR                 | Refractive index; the BSDF uses its inverse at the exit interface.                                                               |
| Specular roughness           | Reflection and refraction roughness.                                                                                             |
| Transmission extra roughness | Additional roughness for refraction.                                                                                             |
| Transmission color           | Interface tint at depth 0; transmittance at the reference depth otherwise.                                                       |
| Transmission depth           | Reference distance in world units for absorption; 0 disables volume absorption.                                                  |

For positive depth, each segment inside the material multiplies radiance by:

```text
transmittance[channel] = transmission_color[channel] ^ (distance / transmission_depth)
```

With color `[0.12, 0.82, 0.25]` and depth 0.5, a 0.5-unit path retains those fractions of RGB light; a 1-unit path retains their squares. Short paths stay light, while thick areas become darker and greener. The interface tint is white in this mode, so color is not applied again at each crossing. Internal reflections add their traveled distance to absorption.

A material with positive Transmission uses Full Standard Surface kernels even if its stored model is Fast. Opaque Fast materials keep the existing fast path. Changing transmission or depth invalidates renderer preparation, accumulated samples, preview frames, and the material thumbnail through the existing render identities.

## Bounces

Raise the quality profile's bounce limit when rays need several interfaces or internal reflections. A very low interactive bounce cap (the Moving profile's default is two) can terminate glass paths before they reach the environment.

## Understand the current geometry scope

The path tracer tracks one occupied dielectric object at a time. Disjoint objects can be seen through glass after its exit. Nested or overlapping dielectric media and objects contained inside a medium need a medium stack and are not implemented. Automatic camera-medium initialization for cameras starting inside a solid is also pending.

Signed fields support interior stepping. Exterior-only fields use the entrance pixel tolerance to define a resolved solid, scan its bounded interior, and refine the first detected exit. Cavities smaller than the configured probe spacing may be missed; this is not an exact signed-distance conversion. An unresolved exit terminates the path instead of leaking environment light through it. Direct OFX shading remains an environment approximation; the new transport is in the path-traced kernels.

The implementation reuses render-rs Standard Surface sampling for Fresnel reflection, Snell refraction, and total internal reflection. Squarebob's enter/exit and ray-offset handling informed the integration; its mesh intersection code cannot replace a fractal interior search.

## Tests

The glass tests cover an unsigned Mandelbulb and a signed zero-iteration KIFS cube in both the single-object and World dispatch, an absorbing cube against the expected RGB transmittance, and the CPU interior and Beer-Lambert math:

```powershell
cargo oxide test -- --release -- glass
```

The optional `cuda_glass_visual_probe` test writes checker-environment EXR and comparison PNGs under `target/glass-probe/`. Run that test explicitly with `--ignored --nocapture`. These diagnostic images are not bundled scene templates.
