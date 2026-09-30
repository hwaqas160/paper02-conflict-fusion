# Sim-to-real check — findings (2026-09-21)

Run: `python -m src.sim2real --sequence boreas-2021-01-26-11-22 boreas-2020-12-18-13-44 boreas-2020-12-01-13-26
boreas-2020-11-26-13-58 --n 60`. Output: `results/sim2real/sim2real.md`, contact sheet
`results/sim2real/real_sequences_contact.jpg` (4 evenly spaced camera frames per sequence).

Boreas publishes weather *tags* (Fig. 6 of the IJRR paper: 5 "snowing" sequences) but not which sequences carry them,
and pyboreas has no metadata. Weather below is **labelled by eye from the contact sheet**, not from Boreas metadata.
`boreas-2021-01-26-10-59` could not be used: it ships only raw IMU/DMI, no post-processed sensor poses.

| real sequence | visual label | nearest simulated (overall) | per modality (L / C / R) | agrees? |
|---|---|---|---|---|
| 2021-01-26-11-22 | heavy active snowfall, whiteout haze, snow-covered road | fog2 | rain2 / fog2 / rain2 | partly |
| 2020-12-01-13-26 | overcast, water droplets on the lens, wet road | rain2 | snow1 / rain2 / rain2 | yes |
| 2020-11-26-13-58 | overcast, dry | clear0 | clear0 / snow1 / rain2 | yes |
| 2020-12-18-13-44 | low sun with glare, dry, patchy old snow | rain2 | clear0 / rain2 / rain2 | LiDAR yes, camera no |

## What this says about the simulator

1. **Real heavy snowfall looks like fog to the camera.** Its camera signature (low contrast, high dark-channel haze, bright
   snow-covered ground) is nearest to simulated *fog2* (70 m visibility), not to any simulated snow severity. Our snow
   camera model adds flakes and lens occlusion but too little atmospheric extinction. Paper wording: simulated snow
   ≠ real heavy snowfall for the camera; simulated fog is the closer proxy for visibility loss in heavy snow.
2. **Real snowfall's LiDAR signature is weaker than simulated snow1–3** (nearest rain2), consistent with our flake-return
   model being too aggressive or with the Alpha Prime's own clutter filtering. Recommendation before the final run: none
   (the cache is frozen and pre-registered); report as a limitation and calibrate snow LiDAR against 11-22 in future work.
3. **Radar is uninformative for matching:** every real sequence maps to rain2 on radar features, including the dry clear
   one. Real-vs-labelled-day differences in the radar (firmware/range-resolution change noted in Boreas Table 2, and a
   different scene mix) dominate weather. Do not use radar frame features as a sim-to-real argument.
4. The two clean controls work: dry overcast → clear0; wet lens → rain2 (the lens-droplet model).

**Consequence for claims:** Boreas "snow1–3" results are about *our simulated snow model*, whose camera effect is milder
than real heavy snowfall. The real-snow claim rests on CADC (Exp 5). State both in the paper's Limitations.
