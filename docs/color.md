# Colour in WarpBro

How colour is represented, rendered and shown in WarpBro, and what the standards behind it mean.
Code references point at the single place each step lives.

## 1. Three independent properties of an RGB space

An "RGB colour space" is three separate choices. Most confusion comes from mixing them up.

| Property | Question it answers | Examples |
|---|---|---|
| **Primaries** | Which exact red, green and blue the three numbers mean (a triangle on the CIE xy chart): the **gamut** | BT.709, P3, BT.2020, ACES AP1, AP0 |
| **White point** | Which colour R = G = B is | D65 (~6500 K daylight), D60 (ACES), DCI (cinema) |
| **Transfer function** | How light is encoded into signal values | linear, sRGB, BT.1886 (gamma 2.4), PQ, HLG, ACEScct (log) |

- **Linear** values are proportional to light: physics (adding, multiplying, path tracing) happens here.
- **Encoded** values are a perceptual / display signal: storage, monitors and video files use them.
- **OETF** (opto-electronic) encodes light into a signal, **EOTF** (electro-optical) is what a display
  does with the signal, **OOTF** is the end-to-end look between scene and display light.
- **Scene-referred** light has no ceiling (a sun is 100 000x a wall). **Display-referred** light is what
  a given display emits, in nits (cd/m²). Turning one into the other is **tone mapping**: the view
  transform.

## 2. The standards and how they relate

```
ITU-R BT.709 (HDTV, 1990)        primaries R/G/B + white D65 + camera OETF + YCbCr matrix
 ├── IEC sRGB (1996)             SAME primaries and white, own transfer curve (~2.2): monitors, web
 ├── ITU-R BT.1886 (2011)        the EOTF of a BT.709 display: gamma 2.4
 └── "Linear Rec.709"            BT.709 primaries + D65 + linear light  (= linear sRGB)

ITU-R BT.2020 (UHDTV, 2012)      much wider primaries, white D65, an SDR OETF like BT.709's
 └── ITU-R BT.2100 (HDR, 2016)   BT.2020 primaries + one of two HDR transfer functions:
      ├── PQ  = SMPTE ST 2084        absolute: signal value = luminance in nits (0..10 000)
      │     └── HDR10 = PQ + BT.2020 + 10-bit + static metadata (mastering display, MaxCLL/MaxFALL)
      └── HLG = ARIB STD-B67         relative: scene light, SDR-compatible, no metadata (broadcast)

SMPTE / Academy ACES (2014-)     a production colour system, white ~D60
 ├── AP0 primaries -> ACES2065-1     linear AP0: archive / interchange ("aces_interchange" role in OCIO)
 ├── AP1 primaries -> ACEScg         linear AP1: RENDERING and compositing   <- WarpBro works here
 │                 -> ACEScct        log AP1: grading
 └── Output transforms (ACES 2.0)    scene-referred ACES -> any display (SDR BT.709, HDR PQ/HLG BT.2020...)

OpenColorIO (OCIO)               the library + config files that implement all of the above as named spaces
```

So **Rec.709 and BT.709 are the same standard** ("Rec." = Recommendation, "BT" = broadcasting
television). **BT.2020** is a set of primaries; the HDR curves **PQ** and **HLG** come from **BT.2100**,
which uses BT.2020 primaries.

## 3. Gamuts: primaries on the CIE 1931 xy chart

| Space | Red (x, y) | Green (x, y) | Blue (x, y) | White |
|---|---|---|---|---|
| BT.709 / sRGB | 0.640, 0.330 | 0.300, 0.600 | 0.150, 0.060 | D65 (0.3127, 0.3290) |
| Display P3 | 0.680, 0.320 | 0.265, 0.690 | 0.150, 0.060 | D65 |
| BT.2020 / BT.2100 | 0.708, 0.292 | 0.170, 0.797 | 0.131, 0.046 | D65 |
| ACES AP1 (ACEScg) | 0.713, 0.293 | 0.165, 0.830 | 0.128, 0.044 | ACES (0.32168, 0.33767) |
| ACES AP0 (ACES2065-1) | 0.7347, 0.2653 | 0.0, 1.0 | 0.0001, -0.0770 | ACES |

To scale, x to the right (0..0.76), y up (0..0.86). AP0 is left out: its green and blue lie outside the
chart; it encloses the whole spectral locus.

```
      ********  ##
    ***      ***#####
   **           #===###
  **            #=  ===###
  *             #=     ==###
  *             #        ===###
 **             #        ~~~==####
 *             ##        ~   ~~==###
 *             #=       ~~      ~==####
 **            #=       ~   -----  ~==###
  *            #=      ~~   -   ----  ~=####
  *            #=      ~   --      ---- ~==###
  *            #=     ~~  --          ---- ~==###
  *            #      ~   -              ---- ~=###
  **           #      ~  --                 ---- ~=###
   *           #     ~  --                     ---- =####
   **         ##     ~  -                          ----=###
    *         #=    ~~ --                             ----####
    *         #=    ~ --       w                         ----####
    **        #=   ~~--       W                          ----- =###
     *        #    ~ -                               -----~~~~~  ####
     **       #   ~~--                          ------~~~~ #######  ***
      **      #   ~--                       -----~~~~=######  *******
       *      #  ~~-                   -----~~~=######   ******
       **    ##  ~--              ------~~######    ******
        **   #= ~--           -----~######    ******
         **  #= ~-       -----#######    ******
          ** #= --   ---#######    *******
           **# ---#######     ******
            *######      ******
              ***  ******
                ****
```

`*` spectral locus (every visible colour lies inside) · `-` BT.709 · `~` P3 · `=` BT.2020 · `#` AP1 ·
`W` D65 white · `w` ACES white

BT.709 is the small inner triangle: many real saturated colours (deep greens and cyans, saturated
yellows) cannot be written in it without negative numbers. BT.2020 and AP1 almost coincide; AP1 reaches
a little further into green. Every BT.709 colour is inside AP1, so converting authored BT.709 colours to
AP1 loses nothing.

## 4. Transfer functions: sRGB, PQ and HLG

Signal (0..1, what is stored or sent) against the display luminance it produces, log scale.

```
 1.0 |                                    sssssssssssssssssssssssPP
     |                                              hh         PP
     |                                   s        hh        PPP
     |                                  s        h        PP
     |                                 s       hh       PP
0.75 |                                       hh      PPP
     |                                s     h      PP
     |                               s    hh    PPP
     |                              s    h    PP
     |                             s    h  PPP
 0.5 |                            s   hh PP
     |                           s   hPPP
     |                         ss  PPP
     |                        s PPP
     |                      sPPPhh
0.25 |                   PPPPhh
     |               PPPPshhh
     |          PPPPPsshhh
     |   PPPPPPPsssshh
     |PPPhssssss
 0.0 |ssss
     +-------------------------------------------------------------
     0.1          1          10          100        1000        10000  nits
```

`s` sRGB on an SDR display (white = 100 nits; everything above clips) · `P` PQ (absolute, up to
10 000 nits) · `h` HLG on a 1000-nit display (system gamma 1.2)

| | sRGB / BT.1886 (SDR) | PQ, SMPTE ST 2084 | HLG, ARIB STD-B67 |
|---|---|---|---|
| Meaning of the signal | relative: 1.0 = whatever the display white is | **absolute**: a code value is a fixed luminance | **relative to scene light**; the display maps it to its own peak |
| Range | ~0.1..100 nits (SDR) | 0..10 000 nits | up to the display peak (typically 1000 nits) |
| Shape | power ~2.2 / 2.4 | Barten-model curve: equal visible steps from 0.001 to 10 000 nits | lower half square root (SDR-like), upper half logarithmic |
| Bits | 8 are enough | 10 / 12 needed | 10 |
| Metadata | none | mastering display (`mDCV`, ST 2086), MaxCLL / MaxFALL (`cLLI`) | none |
| Used for | monitors, web, Rec.709 video | HDR10, Dolby Vision, streaming, HDR stills | live HDR broadcast; plays acceptably on SDR TVs |

Reference values:

| Display light | sRGB (SDR, 100-nit white) | PQ code |
|---|---|---|
| 1 nit | 0.10 | 0.150 |
| 18 nits (mid grey 0.18) | 0.461 | 0.348 |
| 100 nits (SDR white) | 1.000 | **0.508** |
| 203 nits (BT.2408 HDR reference white) | clips | 0.581 |
| 1000 nits | clips | **0.752** |
| 10 000 nits | clips | 1.000 |

HLG reference white (203 nits on a 1000-nit display) sits at about **75 %** of the signal; the upper
quarter holds the highlights. PQ spends half of its code values below 100 nits, because the eye is more
sensitive to steps in the dark.

## 5. WarpBro's pipeline

```
 AUTHORING (user-facing, BT.709)                          FILES IN
 material / palette / sky / sun / light colours           environment EXR: its `chromaticities`
 linear BT.709, D65  (= linear sRGB)                      (BT.709 when untagged), .hdr = BT.709
        │ color::to_working (Bradford D65 -> D60,                 │ color::working_from(primaries)
        │ vfx-ocio matrix; scene::put_rgb, palette::build_lut)    │ environment::Map::load
        ▼                                                         ▼
 ┌──────────────────────────── RENDER: linear ACEScg (AP1, ACES white) ────────────────────────────┐
 │ CUDA path tracer (gpu.rs): BSDF, NEE + MIS, environment importance sampling                       │
 │ luminance everywhere = AP1 Y row 0.2722 / 0.6741 / 0.0537 (SS_LUMA = color::LUMA)                 │
 │ accumulate -> OIDN (pt-denoise-oidn, AP1 autoexposure) -> exposure, saturation (AP1 luminance)    │
 └────────────────┬────────────────────────────────────────────────────────┬───────────────────────┘
                  │ scene-linear ACEScg                                     │
                  ▼                                                         ▼
       EXR sequence (export.rs)                          DISPLAY (color::ColorPipeline::apply)
       float RGB, AP1 `chromaticities`                   ├─ OCIO on: input = the config's linear AP1
       (scene-referred master for compositing)           │  space (found by its transform to
                                                         │  aces_interchange) -> ACES 2.0 view:
                                                         │  tone map + gamut compress -> display light
                                                         └─ OCIO off / Reinhard: color::to_709
                                                            -> Reinhard -> sRGB OETF
                                                                    │ display light: linear BT.709;
                                                                    │ HDR view: absolute, 1.0 = 100 nits
                                                                    │ SDR view: relative, 1.0 = SDR white
                  ┌──────────────────────────┬──────────────────────┴─────┬────────────────────────┐
                  ▼                          ▼                            ▼                        ▼
          SDR monitor / PNG 8-bit    HDR monitor / PNG HDR10        PNG HLG                  SDR video
          sRGB curve, BT.709         BT.709 -> BT.2020 matrix,      BT.2020, HLG curve       BT.1886 code
          (sRGB chunk)               x nits per 1.0 (HDR view 100,  for the view's peak     L^(1/2.4), BT.709
                                     SDR view: monitor white or     (relative light: the
                                     BT.2408 203), PQ (cICP 9/16,   BT.2100 1000-nit
                                     mDCV = measured peak, cLLI)    reference; cICP 9/18)
          Display EXR: linear BT.709 display light, `whiteLuminance` 100
```

Rules this keeps:

- **One conversion point in, one out.** Authored colours enter AP1 only at upload; raw tracer RGB meets
  an sRGB curve only in the OCIO-off branch. Everything after OCIO is already display light, so nothing
  downstream converts primaries again.
- **Authoring stays BT.709**, so existing scenes, presets and pickers keep their meaning. The one library
  value authored in ACEScg (the emissive preset) is stored as its BT.709 equivalent.
- **Files say what they are:** scene-linear EXR is tagged AP1, display EXR BT.709, an HDR PNG carries
  `cICP` (and is named `.pq.png` / `.hlg.png`), an SDR PNG the `sRGB` chunk.
- **Window screenshots** (`src/window_shot.rs`) start from the composited egui-display canvas
  (extended sRGB, 1.0 = SDR reference white), captured as scRGB at 80-nit white, which is exactly the
  canvas decoded to linear BT.709. Reference white = the monitor's SDR white on an HDR output, else
  BT.2408's 203 nits (`Monitor::sdr_white_nits`, shared with viewport snapshots). EXR: that linear light,
  BT.709 `chromaticities`, `whiteLuminance` = the reference white. PQ PNG: BT.2020 nits through ST 2084
  (`cICP` 9/16/0/1), so 1.0 lands on PQ 0.5807 at 203 nits.

## 6. Before and after the switch to ACEScg

| | Before | Now |
|---|---|---|
| Render space | linear BT.709 | linear ACEScg (AP1) |
| OCIO input | "Linear Rec.709 (sRGB)", a free choice | the config's linear AP1 space (auto or picked, verified) |
| Luminance weights | BT.709 (0.2126 / 0.7152 / 0.0722), except the BSDF, which already used AP1 | AP1 everywhere |
| Environment EXR | read as-is | converted from its `chromaticities` |
| EXR sequence tag | BT.709 | AP1 |
| OIDN | BT.709 luminance | AP1 luminance |

What changes in the picture:

- **Neutral scenes:** the same within noise (white stays white, both weight sets sum to 1).
- **Saturated colours under GI:** each bounce multiplies colours per channel. In the narrow BT.709
  triangle such products fall outside the gamut and lose hue; in AP1 they stay inside, so coloured
  inter-reflections are cleaner and closer to a spectral result. The final squeeze to the display gamut
  is done once, by the ACES 2.0 output transform, which compresses rather than clips.
- **Per-channel products** (palette x tint, Beer-Lambert glass) give slightly different results: the
  product is taken in a different basis.
- **Noise pattern** (not its expectation): light and lobe selection probabilities moved; saturation != 1
  pivots on AP1 luminance.
