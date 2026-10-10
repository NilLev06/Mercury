# Experimental camera sector inputs

Goal: test whether discarding all but the largest blue obstacle limits complex navigation.
This is a hypothesis, not a diagnosed sole cause. Teacher-only warm experiments and live-core experiments remain separate.

Add an opt-in v2 observation schema: keep the original nineteen inputs unchanged and append twelve left-to-right blue pixel coverage values from the camera. Use exactly the existing blue/teal HSV range and morphology. No scene coordinates, obstacle ids, route, target coordinates, compass or action choices enter perception. Each sector reports normalized blue pixel coverage, including all visible obstacle components.

Upgrade a copied v1 encoder with zero-initialized extra columns. Before training, identical histories must yield identical neural outputs and states even with arbitrary extra inputs. Save the schema in checkpoints; preserve legacy default/load behavior and reject unknown schemas. Connectome topology stays unchanged.

Collect new successful teacher and assisted student histories using the new camera features on training maps. Source anchors use v2 observations from autonomous old seven-map baseline rollouts. Train existing neural parameters with full histories. Autonomous development uses old seven maps and mixed selection maps. Never deploy a candidate with old regressions. Independent final maps are reserved until robust development performance; a success is autonomous arrival, not teacher-assisted arrival.
