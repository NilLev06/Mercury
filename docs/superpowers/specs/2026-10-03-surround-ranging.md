# Surround ranging for bypass experiment

Approved by user: move front diagonals toward wheel edges at +/-45deg,
add +/-90deg side rays and rear ray. These are ideal MuJoCo rangefinders,
not ultrasound cone simulation. Six total. Sensors use the existing car
envelope, never shrink hitboxes. Origins: nose x=wheelbase+.04, y=+/-(track/2+.022);
sides at body center with outer wheel y; rear at x=-.04.
Add range_side_left, range_side_right, range_rear to SensorFrame, default1.2.
Negative range means miss; zero means zero clearance, not clear.

Bypass retains task bypass-v1 but sensor_schema=2. Append one compact context
category to its nine existing state fields: which sides are near (<.25m),
and whether rear is near (<.35m). Eight contexts, not27 raw range combinations.
Legacy nine-field values are approximate read-only warm-start fallbacks;
new context rows learn independently. Retain training_runs and before-run
backups. Frozen evaluation does not rewrite model. Main navigation Q schema
and rewards unchanged. No scripted turn/automatic bypass added. Bumpers remain
the only action restriction. Do not promise training success from sensor tests.

Dashboard shows actual six distances and schematic rays; viewer enables
rangefinder visualization. Reading source determines observed geometry, not
true obstacle coordinates. Test side cylinder gap, side/rear wall gaps,
diagonal direction and envelope placement, warm-start isolation and zero/miss.
Short verification only; no full retraining or arbitrary penalty tuning.
