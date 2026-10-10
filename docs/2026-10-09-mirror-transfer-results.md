# Physical mirror transfer experiment

Independent frozen-history probe of gentle-corrected mastery candidate7:
known four complex routes mean action MSE .000934; other twenty gentle
training routes .02275, about24x. Initial steering sign agrees on all24,
but held-out steering direction sometimes reverses roughly ten decisions too
early. These errors precede autonomous distribution shift. Approximate feature
mirroring also raises MSE, but unequal sector widths preclude treating this as
an exact reflection experiment. No proof of absent memory or unavoidable
observation aliasing was found.

Collection:45 original gentle teacher histories, physically reflected obstacles,
goal, initial y/yaw and steering commands; traction retained. Actual camera and
range features are recollected. All45 reflected episodes reached the stable-stop
criterion with zero contacts. Playback commands are teaching data only. Source
checkpoint and live policy remain unchanged.

Controlled A/B from the same candidate7:

- control:45 originals (21 old canonical/perturbed starts,24 complex);
- augmented: same45 plus45 physical mirrors;
- old teacher anchors exactly identical in both;
- full-history fitting, all neural weights trainable, topology fixed;
- opening weight1, anchor2, Adam1e-4, six1000-update stages;
- matching stage seeds51000..51005, independently fresh optimizers;
- old7 and same repeated development12 checked autonomously after each stage.

Both arms broaden complex training from four to24 original geometries. Their
comparison isolates augmentation at equal update count; neither is a clean
single-factor comparison with earlier four-route curriculum. Independent final
maps remain unopened until reliability gate. No inference mirror rule or
planner is introduced, and no automatic live promotion is performed.

## Completed A/B

| Arm | Updates | Old/7 | Old contacts | Complex/6 | Complex contacts |
|---|---:|---:|---:|---:|---:|
| control | 1000 | 7 | 1 | 1 | 232 |
| control | 2000 | 7 | 0 | 1 | 209 |
| control | 3000 | 7 | 0 | 0 | 578 |
| control | 4000 | 7 | 0 | 1 | 574 |
| control | 5000 | 7 | 0 | 2 | 412 |
| control | 6000 | 6 | 0 | 1 | 543 |
| augmented | 1000 | 4 | 99 | 3 | 203 |
| augmented | 2000 | 6 | 210 | 1 | 404 |
| augmented | 3000 | 7 | 0 | 1 | 513 |
| augmented | 4000 | 6 | 10 | 3 | 284 |
| augmented | 5000 | 6 | 9 | 2 | 136 |
| augmented | 6000 | 6 | 47 | 1 | 79 |

Neither arm met the reliability gate. No robust advantage from mirroring was established. Source/replay hashes remained unchanged in both arms. Independent review verified exact command reflection, identical anchors, A/B metadata and successful physical recollection.

A frozen-history probe at3000 updates found original complex MSE .00616(control) vs .00785(augmented), and reflected complex MSE .01239 vs .00842. This supports improved fit to reflected histories at that checkpoint, not autonomous transfer or a final scientific conclusion. Source MSE was .01911 original and .01893 reflected.
