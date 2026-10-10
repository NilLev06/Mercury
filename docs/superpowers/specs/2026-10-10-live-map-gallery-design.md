# Live map gallery

User approved the proposed catalogue + live simulation design and explicitly selected local browser simulation. Implement in the existing project; preserve all model artifacts. No new training or scientific ablations in this UI task.

231 deterministic maps: old7, previously opened complex32 (950001..4), newly opened independent192 (1050001..16). Earlier sealed1040001..8 excluded. Catalogue includes failures; no cherry-picked success-only demo. Each card contains a true layout thumbnail, tier/group and known result when a verified report exists. User can search/filter maps, select, start, stop, repeat and choose0.5x/1x/2x/4x wall playback speed. Default favourite immutable main-23of32 only.

Browser shows live top-down 2D observation of actual MuJoCo poses and path, goal, obstacles, actual camera preview, six ranges, throttle/steering, elapsed simulation time and contacts. Optional255-cell recurrent activity view represents actual rate states, never spikes. Stop/timeout/arrival are distinct; success with contacts is distinct from clean success. Display known historical results separately from current runs. No invented perception, no JavaScript driving controller or physics. World coordinates are for viewer only.

Reuse LabServer with new /maps page and /api/gallery routes. GalleryManager owns a separate MuJoCo worker process, uses a fixed checkpoint and server-owned map ID only. Worker calls the standard collect_episode(beta=None); pacing in the telemetry emitter only changes wall time. Cancel file checked every physics step, then graceful process escalation if necessary. Port8766 dedicated launcher opens /maps; existing lab on8765 gains a /maps link but its existing modes remain available. Local same-origin mutations, bounded JSON, no user-selected filesystem paths or executable commands.

API:
- GET /api/gallery/maps -> {maps:[{id,name,title,group,tier,obstacles:[{position:[x,y],colour:[r,g,b]}],goal:[x,y],pose:[x,y,yaw],known_result:object|null}],model:{label,sha256,neurons:255,edges:4088},arena:{half:3,obstacle_radius:.28,wheelbase:.18,track:.36}}.
- GET /api/gallery/state -> {status:idle|starting|running|finished|stopped|error,run_id,map_id,speed,frame:null|{step,time,pose:[x,y,yaw],speed,contacts,target_visible,goal_distance,arrival_elapsed,action:[throttle,steering],ranges:[6],preview:dataJPEG,activity:[255]},trail:[[x,y]],result:null|report,error:null|string,history:{map_id:report}}.
- POST /api/gallery/start {map_id:string,speed:0.5|1|2|4}; returns state,409 while active.
- POST /api/gallery/stop {}; returns state, retains last frame/result.

One worker per gallery instance. Bounded trail and logs. Latest session run results in memory; no weights, optimizer or demonstration writes. Fresh workers reset recurrent state each run. Start failures visible in UI; never pretend the simulator is running. On server shutdown close both old job manager and gallery worker.

Verification: catalogue uniqueness/count/reserved exclusion; input validation and local origin; serial start/stop/restart and worker late-message isolation; normal GUI episode equals existing known report; checkpoint hashes unchanged; full tests; browser inspect at desktop/mobile widths, selecting map/filtering/live camera/path/stop/restart/completion.
