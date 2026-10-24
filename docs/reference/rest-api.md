# Local REST API reference

WarpBro exposes the running desktop app at `http://127.0.0.1:9877/api`. Use it to inspect the document, edit nodes, navigate the viewport, control playback, and start exports.

Updated: 2026-10-24. The command definitions live in [the REST decoder](../../src/app_control.rs), [shared typed events](../../src/events.rs), [the dispatcher](../../src/app_events.rs), [action handlers](../../src/app_actions.rs), [the HTTP transport](../../crates/app-control/src/lib.rs), and [WorldCommand](../../src/world.rs).

## Contents

- [Connection and configuration](#connection-and-configuration)
- [Requests and replies](#requests-and-replies)
- [Queries](#queries)
- [Viewport navigation](#viewport-navigation)
- [Scene and document commands](#scene-and-document-commands)
- [World commands and attributes](#world-commands-and-attributes)
- [Playback, materials, settings, and exports](#playback-materials-settings-and-exports)
- [Screenshots and diagnostics](#screenshots-and-diagnostics)
- [Errors](#errors)

## Connection and configuration

The listener starts with the desktop UI. It binds only to IPv4 loopback, `127.0.0.1`, and has no authentication. Other local processes can control the app. Requests with an `Origin` header are rejected with HTTP 403, so use a local CLI or script rather than browser-page JavaScript.

| Environment variable | Behavior                                                                                               |
| -------------------- | ------------------------------------------------------------------------------------------------------ |
| `WARPBRO_API`        | Set to exactly `0` before launching to disable the listener.                                           |
| `WARPBRO_API_PORT`   | Port number; default `9877`. Set `0` to let the OS choose a free port. The log reports the chosen URL. |

For a different port, launch the existing executable from PowerShell:

```powershell
$env:WARPBRO_API_PORT = '9878'
Start-Process .\target\release\WarpBro.exe
```

A port conflict or invalid port is reported in the application status and log. Check the process identity when more than one app instance is running:

```powershell
$api = 'http://127.0.0.1:9877/api'
Invoke-RestMethod "$api/health"
Invoke-RestMethod "$api/status" | ConvertTo-Json -Depth 20
```

Health returns `pid` and `protocol` (currently `1`) without waiting for the UI. Status waits for the app to handle the query and also identifies the application as `WarpBro`.

## Requests and replies

GET routes read state. POST routes execute commands with JSON payloads. Slashes map to dots in command names: `POST /api/viewport/orbit` executes `viewport.orbit`.

HTTP decodes requests into the same typed `app_control::Event<UiCommand>` that widgets and hotkeys publish to App's Playa EventBus. The central dispatcher applies App, World, Take, and Query actions on the app thread. HTTP parser channels carry sockets, and reply channels carry acknowledgements; there is no separate REST application-command queue. `GET /api/commands` lists the exposed HTTP commands; UI-only transient intents do not automatically have HTTP endpoints.

Use this helper for the examples below:

```powershell
$api = 'http://127.0.0.1:9877/api'
function Send-WarpBro {
    param([string]$Command, [object]$Payload = @{})
    $json = ConvertTo-Json -InputObject $Payload -Depth 100 -Compress
    $body = '{"event_type":"' + $Command + '","payload":' + $json + '}'
    Invoke-RestMethod -Method Post -Uri "$api/event" -ContentType 'application/json' -Body $body
}
Send-WarpBro 'viewport.orbit' @{ yaw_degrees = 15; pitch_degrees = 5 }
```

The event envelope contains `event_type` and optional `payload`; unknown envelope fields fail. The equivalent direct route is:

```powershell
Invoke-RestMethod -Method Post -Uri "$api/viewport/orbit" -ContentType 'application/json' -Body '{"yaw_degrees":15,"pitch_degrees":5}'
```

Direct POST routes accept an empty body as `{}`. Use `{}` for commands with no payload. Request bodies are limited to 1 MiB. Send document JSON as an object, rather than a JSON string containing another document.

A normal successful command returns HTTP 200 after the shared dispatcher handles it. Camera commands return `success`, the resulting `camera`, and document `revision`. Other result fields depend on the command. Read queries return their objects or arrays directly, without a result wrapper.

An acknowledgement does not mean a new image has finished rendering or a requested file operation has completed. `scene.open` and `scene.save` dispatch background I/O; inspect application status for its result. Picking and export explicitly return `{"queued":true}` because their GPU/file work continues asynchronously. `app.exit` acknowledges the close request before the process exits. A command waiting for the app times out after 10 seconds; a request that has expired before dispatch is skipped. If a timeout overlaps execution, inspect current state before retrying a mutation.

## Queries

| GET route under `/api` | Reply                                                                                                               |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `/health`              | Transport identity: process ID and protocol version.                                                                |
| `/commands`            | Supported command/query names, `event_bus` identity, and WorldCommand format.                                       |
| `/status`              | Scene name, document UUID, revision, selection, camera, timeline, render, thumbnail, diagnostic, and status fields. |
| `/scene`               | Persistent World document, suitable for strict document loading.                                                    |
| `/nodes`               | Node array: `id`, `kind`, `name`, `parent`, `visible`, `locked`, `solo`, `start`, `end`.                            |
| `/attributes/<uuid>`   | Node attributes evaluated at the playhead: `path`, `label`, `value`, `keyable`, `frames`, `choices`, `range`.       |
| `/gallery`             | Gallery entries: zero-based `index`, `name`, `formula`, and thumbnail-ready flag `thumbnail`.                       |
| `/materials`           | Material-library entries: zero-based `index` and preset name `preset`.                                              |
| `/takes`               | Recorded-take list from the current document.                                                                       |
| `/camera`              | Camera evaluated from the document at the current playhead, including during cached preview.                        |
| `/settings`            | Current application preferences, including controls, Auto Key, and export settings.                                 |
| `/render`              | The render portion of status.                                                                                       |
| `/diagnostics`         | Active pipeline compilation, recent pipeline timings, and thumbnail-cache counters.                                 |
| `/screenshot.png`      | Current viewport image as `image/png`; fails if no frame exists or its colour output failed.                        |

For the running binary's command list:

```powershell
Invoke-RestMethod "$api/commands" | ConvertTo-Json -Depth 10
```

## Viewport navigation

Each camera navigation call stops timeline/cache playback, evaluates the document at the current playhead, applies one camera change, stops active flight/inertia, authors it through the active Camera node, and closes its Undo step. Auto Key follows the current application setting. These calls do not simulate held mouse buttons or a timed flight.

| POST command             | Required payload                                                                  | Effect                                                                                                                                                        |
| ------------------------ | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `viewport.orbit`         | `yaw_degrees`, `pitch_degrees`: numbers                                           | Orbit deltas in degrees using the shared Houdini orbit rig; pitch is limited by the rig to ±89°.                                                              |
| `viewport.pan`           | `dx_px`, `dy_px`: numbers                                                         | Pan deltas in viewport pixels; scale follows vertical FOV, viewport height, distance, and camera roll.                                                        |
| `viewport.zoom`          | `factor`: positive finite number                                                  | Apply a multiplicative orbit-distance change.                                                                                                                 |
| `viewport.fly`           | `translation`: three numbers; `rotation_degrees`: three numbers                   | Translate in the camera's local axes, then add yaw, pitch, and roll deltas. Translation is right, up, forward in scene units.                                 |
| `viewport.camera`        | Object containing camera fields to replace                                        | Patch the evaluated camera; omitted fields retain their current values.                                                                                       |
| `viewport.frame`         | `{}`                                                                              | Frame the scene bounds.                                                                                                                                       |
| `viewport.home`          | `{}`                                                                              | Restore the camera of the loaded origin scene.                                                                                                                |
| `viewport.pick`          | `ndc`: two numbers in `[-1,1]`; optional `add`, `focus`: booleans (default false) | Stop timeline/cache playback, evaluate the current playhead, and queue a GPU pick; returns `queued` and `pick_id`. Selection changes when the result arrives. |
| `viewport.raw`           | `enabled`: boolean                                                                | Toggle raw viewport display.                                                                                                                                  |
| `viewport.pause`         | `enabled`: boolean                                                                | Author the viewport pause flag.                                                                                                                               |
| `viewport.flight.toggle` | `{}`                                                                              | Toggle saved free-flight/horizon-lock mode.                                                                                                                   |
| `viewport.slot.store`    | `index`: integer                                                                  | Store the current camera in a zero-based CamClip slot (0–4).                                                                                                  |
| `viewport.slot.restore`  | `index`: integer                                                                  | Restore a nonempty CamClip slot (0–4).                                                                                                                        |
| `viewport.tool`          | `tool`: string                                                                    | Choose `select`, `move`, `rotate`, or `scale`.                                                                                                                |

Navigation examples:

```powershell
Send-WarpBro 'viewport.orbit' @{ yaw_degrees = 20; pitch_degrees = 5 }
Send-WarpBro 'viewport.pan' @{ dx_px = 80; dy_px = -30 }
Send-WarpBro 'viewport.zoom' @{ factor = 0.8 }
Send-WarpBro 'viewport.fly' @{
    translation = @(0, 0, 0.25)
    rotation_degrees = @(10, 0, 0)
}
Send-WarpBro 'viewport.frame'
Send-WarpBro 'viewport.home'
```

Camera patch fields:

| Field                                          | Type and units                                                                                     |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `target`                                       | Three numbers, world position in scene units.                                                      |
| `yaw_degrees`, `pitch_degrees`, `roll_degrees` | Finite numbers, degrees.                                                                           |
| `free_flight`                                  | Boolean, saved flight mode.                                                                        |
| `distance`                                     | Positive finite number, in the camera's framing radii.                                             |
| `fov_y_degrees`                                | Vertical FOV, strictly between 0 and 180 degrees.                                                  |
| `f_number`                                     | Nonnegative physical f-number; `0` disables depth of field.                                        |
| `sensor_height`                                | Positive finite vertical film gate in scene units; `0.024` means 24 mm under the metre convention. |
| `focus_distance`                               | Distance in framing radii; `0` focuses on the target.                                              |

Unknown camera fields fail. Lens dimensions must remain finite.

```powershell
Send-WarpBro 'viewport.camera' @{
    fov_y_degrees = 45
    f_number = 0
    roll_degrees = 0
}
Invoke-RestMethod "$api/camera"
Send-WarpBro 'history.undo'
Send-WarpBro 'history.redo'
Send-WarpBro 'viewport.pick' @{ ndc = @(0, 0) }
```

## Scene and document commands

| POST command                   | Payload and behavior                                                             |
| ------------------------------ | -------------------------------------------------------------------------------- |
| `scene.new`                    | `{}`; load a new default scene.                                                  |
| `scene.load`                   | Full World document object; validate and load it through the scene-loading path. |
| `scene.document`               | Alias of `scene.load`; full World document object.                               |
| `scene.open`                   | `path`: string; submit a scene-file open request to background I/O.              |
| `scene.save`                   | `path`: string; submit saving the current document to background I/O.            |
| `gallery.load`                 | `{"index":19}`; load the gallery entry at a zero-based index.                    |
| `world.command`                | One externally tagged WorldCommand object; see below.                            |
| `selection.set`                | `{"ids":[...uuid strings...]}`; validate each node and replace selection.        |
| `history.undo`, `history.redo` | `{}`; use document history.                                                      |
| `history.reset`                | `{}`; restore the loaded origin document as one undoable step.                   |

Gallery entries can change; query their names before choosing one:

```powershell
$gallery = Invoke-RestMethod "$api/gallery"
$gallery | Format-Table index, name, formula, thumbnail
Send-WarpBro 'gallery.load' @{ index = 19 }
$document = Invoke-RestMethod "$api/scene"
Send-WarpBro 'scene.load' $document
```

`scene.load` and `scene.document` use the same strict schema validation as opening a scene file. They accept the document object, rather than an evaluated Scene snapshot or a filename. Use `scene.open` or `scene.save` for filesystem paths.

## World commands and attributes

`world.command` accepts every variant of the [WorldCommand enum](../../src/world.rs), including `Batch`. Serde uses an externally tagged object: the variant name is the sole outer key, with its exact capitalization. For example:

```json
{ "Create": { "kind": "Group", "name": "REST group", "parent": null } }
```

UUIDs are JSON strings. Optional references use a UUID or `null`. Attribute values are ordinary JSON values, matching the type returned by the attribute query. Attribute paths are document property addresses, not labels from the editor.

This example finds a Fractal node, queries its Translate attribute, and moves it 0.25 scene units on X at the current playhead:

```powershell
$nodes = Invoke-RestMethod "$api/nodes"
$node = $nodes | Where-Object kind -eq 'Fractal' | Select-Object -First 1
if ($null -eq $node) { throw 'The current scene has no Fractal node.' }
$attributes = Invoke-RestMethod "$api/attributes/$($node.id)"
$position = $attributes | Where-Object path -eq '/transform/position'
if ($null -eq $position) { throw 'The node has no Translate attribute.' }
$frame = (Invoke-RestMethod "$api/status").timeline.frame
$next = @($position.value)
$next[0] = [double]$next[0] + 0.25
Send-WarpBro 'world.command' @{
    SetAttribute = @{
        id = $node.id
        path = $position.path
        value = $next
        frame = $frame
    }
}
```

The enum payloads below use field names from the Rust definition. Supply the stated object inside the variant key, or the direct value for variants marked "direct". `frame`, `frames`, `start`, `end`, and `delta` use timeline frames.

| Variant                                                        | Payload                                                             |
| -------------------------------------------------------------- | ------------------------------------------------------------------- |
| `Create`                                                       | `kind`, `name`, `parent`                                            |
| `Delete`                                                       | Direct UUID string                                                  |
| `Duplicate`                                                    | Direct array of UUID strings                                        |
| `Rename`                                                       | `id`, `name`                                                        |
| `Reparent`                                                     | `id`, `parent`                                                      |
| `SetAttribute`                                                 | `id`, `path`, `value`, `frame`                                      |
| `Key`, `RemoveKey`                                             | `id`, `path`, `frame`                                               |
| `MoveKeys`                                                     | `id`, `path`, `frames`, `delta`                                     |
| `Interpolation`                                                | `id`, `path`, `frames`, `kind` (curve tangent enum)                 |
| `CurveEffect`                                                  | `targets` (arrays of `[uuid,path]`), `effect` (curve effect object) |
| `SetMetadata`                                                  | `id`, `path`, `value`                                               |
| `Reorder`                                                      | `ids`                                                               |
| `SetTimeRange`                                                 | `first`, `last`, `fps`                                              |
| `SetMark`                                                      | `slot` (0–9), `frame` (integer or null)                             |
| `AssignMaterial`                                               | `id`, `material` (UUID or null)                                     |
| `CreateMaterial`                                               | `material` (Material object), `name`                                |
| `CreateMaterialFor`                                            | `material` (Material object), `name`, `targets`                     |
| `ApplyMaterial`                                                | `id`, `material` (Material object), `name`, `frame`                 |
| `CreateRenderProfile`                                          | `name`, `role`, `source`, `target`                                  |
| `InstantiateRenderTemplate`                                    | `id`, `name`, `target`                                              |
| `SetCatalogRole`                                               | `id`, `role`                                                        |
| `SetOutputRender`, `SetOutputSettings`                         | Direct UUID string                                                  |
| `CreateOutputSettings`                                         | `name`, `role`, `source`, `assign`                                  |
| `InstantiateOutputTemplate`                                    | `id`, `name`, `assign`                                              |
| `SetActiveCamera`, `SetActiveEnvironment`, `ReloadEnvironment` | Direct UUID string                                                  |
| `Batch`                                                        | Direct array of WorldCommand objects                                |
| `SetSpan`                                                      | `id`, `start`, `end`                                                |
| `SetAnimation`                                                 | `id`, `path`, `enabled`, `frame`                                    |

Node kinds are `Fractal`, `Camera`, `DirectionalLight`, `Environment`, `Group`, `Material`, `RenderSettings`, `ViewportSettings`, `OutputSettings`, and `Mask`. For profile routing and settings-node attributes, see [render profiles](../render-profiles.md). Query current attributes to discover choices and numeric ranges.

Group related edits into one validated Undo command:

```powershell
Send-WarpBro 'world.command' @{
    Batch = @(
        @{ Rename = @{ id = $node.id; name = 'REST fractal' } }
        @{ SetAttribute = @{
            id = $node.id; path = $position.path; value = $next; frame = $frame
        } }
    )
}
Send-WarpBro 'selection.set' @{ ids = @($node.id) }
```

## Playback, materials, settings, and exports

| POST command      | Payload and behavior                                                                                                 |
| ----------------- | -------------------------------------------------------------------------------------------------------------------- |
| `timeline.seek`   | `{"frame":0}`; unsigned integer playhead frame; cancels cache-preview playback.                                      |
| `timeline.play`   | `{}`; start timeline playback.                                                                                       |
| `timeline.pause`  | `{}`; pause playback and cancel cache preview.                                                                       |
| `timeline.loop`   | `enabled`: boolean; set timeline looping.                                                                            |
| `material.create` | `{"index":0}`; create/select a standalone material from a library preset.                                            |
| `material.apply`  | `{"index":0}`; apply a library preset to selected eligible objects.                                                  |
| `settings.set`    | Recursive preferences patch; omitted fields retain their current values, unknown fields fail.                        |
| `colour.set`      | Full serialized colour-selection object (`ocio::Sel`); author the scene selection through the shared action handler. |
| `colour.preset`   | `index`: integer; restore a saved colour preset.                                                                     |
| `colour.reload`   | `{}`; reload the selected OCIO configuration.                                                                        |
| `export.start`    | `{}`; freeze the current World scene and queue export with current export settings.                                  |
| `export.cancel`   | `{}`; request cancellation.                                                                                          |
| `app.ui`          | `{"enabled":false}`; hide/show panels.                                                                               |
| `app.exit`        | `{}`; request window close.                                                                                          |

```powershell
Send-WarpBro 'timeline.pause'
Send-WarpBro 'timeline.seek' @{ frame = 24 }
Send-WarpBro 'settings.set' @{ auto_key = $false; gui_fps = 60 }
$materials = Invoke-RestMethod "$api/materials"
$materials | Format-Table index, preset
Send-WarpBro 'material.create' @{ index = 0 }
```

Read `GET /api/settings` before patching nested settings. The `export` object contains the current export recipe; patch its actual field names, then call `export.start`. Encoding, rendering, and writing happen asynchronously; a queued reply does not confirm an output file. The status string and Render / Encode panel report job progress and errors. See [export behavior](../workspace.md#render-final-images-and-animation) and [render profile semantics](../render-profiles.md).

## Screenshots and diagnostics

Save the existing viewport frame:

```powershell
Invoke-WebRequest -Uri "$api/screenshot.png" -OutFile '.\warpbro-viewport.png'
Invoke-RestMethod "$api/render" | ConvertTo-Json -Depth 10
Invoke-RestMethod "$api/diagnostics" | ConvertTo-Json -Depth 10
```

This route returns an 8-bit RGBA SDR viewport PNG at the current rendered frame dimensions. It does not trigger a new render or capture panels, menus, or the whole window. A previous frame may still be displayed while a changed scene renders.

Status includes:

- `render.requested_generation` and `render.frame_generation`: compare them to detect whether the current frame belongs to the latest viewport request.
- `render.samples`, `target_spp`, `last_ms`, `last_spp`, and `denoise_ms`: current frame progress and recorded worker timings; unavailable frame fields are null.
- `render.colour_error`, `denoise_error`, and `worker_stopped`: render failure details.
- `thumbnails`: ready/total counts, current pending entry, and failed count.
- `diagnostics.compiling`: active WGPU pipeline entry and elapsed seconds, or null.
- `diagnostics.pipelines`: recent completed pipeline timings.
- `diagnostics.thumbnail_cache`: hit, miss, and write counters.
- `status`: application status text.

A healthy HTTP listener alone does not establish render progress. Inspect the generation, sample, compilation, and error fields together. The diagnostics route reads CPU-side state and does not wait for GPU completion.

## Errors

Errors return JSON such as:

```json
{ "success": false, "error": "Zoom factor must be finite and positive" }
```

| HTTP status | Meaning                                                                                                                                                                     |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 200         | Query or command handled; inspect the body for asynchronous `queued` results.                                                                                               |
| 400         | Invalid JSON/payload, unknown command/field, validation failure, or application command error.                                                                              |
| 403         | The request contains an `Origin` header; browser-origin requests are refused.                                                                                               |
| 404         | Unsupported HTTP method or route outside `/api/`. Unknown commands under `/api/` return 400.                                                                                |
| 503         | HTTP parser queue full/stopped, application shutting down, or EventBus admission refused. A full bus leaves older accepted actions intact and does not publish this action. |
| 504         | The app did not acknowledge the command within 10 seconds.                                                                                                                  |

Commands run on the app thread through Playa's shared EventBus. Reliable `try_emit` refuses overflow without evicting an accepted action. Prefer sequential mutations so later requests observe earlier results. A disconnected listener is a connection error, rather than a JSON response. Check the startup log for disabled API, an invalid port, or an occupied port.
