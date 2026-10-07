# Synthetic heading stabilization and anatomical gate

User approved fixing the diagnosed weaknesses and continuing the staged path.
No deployed vehicle, checkpoint, sensor or teacher changes.

1. Regression tests: sixty-second holds under ±5% feedback gain; long maximum
   yaw turns; amplitude recovery after a radial disturbance; recurrence lesion.
2. Add explicit synthetic radial feedback to restore first-harmonic amplitude.
   Compensate the two-population lag using the characteristic equation, not
   a simulator heading or a stored scalar angle.
3. Repeat unchanged stress cases and full suite. Bias/noise phase diffusion
   remains a measured limitation requiring genuine sensory anchoring.
4. Extend anatomical evidence checks only after the synthetic regression gate.
   Never claim that synthetic gains or sectors constitute FlyWire physiology.

Design: target harmonic amplitude 0.8, radial stiffness 0.2; at stationary unity
gain the original sinusoidal profile remains an equilibrium. A lag-corrected
real gain 1−tauE*tauP*yaw² and complementary loop gate target the imaginary
eigenvalue i*yaw. Effective weights and the radial controller remain synthetic.
Acceptance: supported ±2 rad/s over 60 s has <5° error; stationary gain 0.95–1.05
holds a decodable heading with <2° drift and contrast >0.1; lesions still erase
memory. Do not force phase after disturbances or invent a spontaneous direction.
