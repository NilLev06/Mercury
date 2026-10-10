# Live map gallery Implementation Plan

> **For agentic workers:** Use subagent-driven-development for the independent frontend and inline backend; requesting-code-review for integration. User approved the design and local live format. Existing dirty checkout contains required uncommitted model/runtime code: work in place using isolated new gallery files rather than omitting it in a new checkout. Do not commit unrelated work.

**Goal:** Let the user browse231 maps and observe the unchanged favourite driving real MuJoCo episodes in the browser.
**Architecture:** Existing lab HTTP server with gallery routes; bounded process manager; standard neural episode worker. Vanilla HTML/CSS/JS drawing actual positions and camera, no browser physics.
**Tech Stack:** Python stdlib HTTP/subprocess, existing PyTorch/MuJoCo, canvas.

## Global Constraints

Follow docs/superpowers/specs/2026-10-10-live-map-gallery-design.md, especially its exact API. No model/training writes; no sealed104 maps; default favourite immutablee7155c9a…; old lab flows unchanged.

### Task1: Backend catalogue, manager and worker
Files: fly_brain_rc/dashboard/gallery.py, scripts/run_gallery_episode.py, tests/test_gallery.py.
- [x] Write tests first for231 unique cases, sealed exclusion, invalid map/speed rejection and lifecycle isolation.
- [x] Implement catalogue using validation_cases/mixed_cases; known results from actual independent report plus legacy report; missing historical results remain null.
- [x] Implement locked GalleryManager snapshots, one subprocess, cancellation and stdout frame/result handling, capped trail. Fixed argv and server-owned map ID. Add latest-frame session history only for noncancelled completed runs.
- [x] Worker loads immutable favourite, checks hash, reconstructs server-owned map, calls collect_episode betaNone with actual camera/telemetry/state hook and wall pacing. Prints frame/result JSON; closes renderer and checks checkpoint unchanged. SIGINT cancels cleanly.

### Task2: Browser catalogue and live observation
Files: fly_brain_rc/dashboard/static/gallery.html, gallery.css, gallery.js.
- [x] Delegate frontend with exact API/spec. Responsive map catalogue/filter/search/layout previews; honest known result badges; preview selected map before start.
- [x] Live world-coordinate canvas, path and car heading; actual camera, range indicators, action/time/contact metrics and recurrent activity. Accessible controls, visible errors and reconnect handling; no false autoplay.
- [x] Start/stop/repeat speed controls, retain final frame and distinguish cancelled/contact arrival/clean arrival/timeout.

### Task3: Integration and verification
Files: fly_brain_rc/dashboard/lab.py, static/lab.html, scripts/run_map_gallery.py, Fly Map Gallery.command, README.md.
- [x] Add same-origin protected routes, static allowlist, manager cleanup, optional open_path to run_lab; launcher dedicated8766 /maps; existing lab link.
- [x] HTTP tests for route validation and lifecycle; full test suite and independent review.
- [x] Launch actual app, inspect with browser, execute live map, stop/restart and complete known old map. Compare outcome to verified favourite baseline, hash all protected models before/after.
- [x] Update docs and hand user working URL + double-click launcher. No automatic browser activity before app is actually available.
