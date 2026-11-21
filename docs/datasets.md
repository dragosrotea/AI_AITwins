# Datasets: vehicle data (LEAD-T07)

Everything below was checked against the downloaded files on 2026-11-21, not taken from memory.
CV datasets are documented by the CV owner in `training/DATASETS.md`.

## UAH-DriveSet v1

Smartphone recordings (DriveSafe app, phone on the windshield) by 6 drivers (D1 to D6), each
driving normal, drowsy and aggressive on a motorway (~25 km, 120 km/h limit) and a secondary
road (~16 km, 90 km/h), near Alcalá de Henares, Madrid. 40 trips, about 500 minutes in all.

- Site: http://www.robesafe.uah.es/personal/eduardo.romera/uah-driveset/ (plain `http`, no https).
  The download is a form: accept the license, enter an email, and `download.php` returns
  `UAH-DRIVESET-v1.zip` (3.5 GB, store-compressed, mostly `.mp4` videos).
- **License:** non-commercial use only; cite the dataset; **do not redistribute the data or
  modified versions**. So it never goes into git (`data/` is ignored) and never into the deck as
  raw files. Dashboards show it labelled "replay" (DB-02).
- **Cite:** E. Romera, L.M. Bergasa, R. Arroyo, "Need Data for Driving Behavior Analysis?
  Presenting the Public UAH-DriveSet", IEEE ITSC 2016, pp. 387-392.

### Setup on a laptop

1. Download the zip into `data/` (see above).
2. Extract without the videos (53 MB instead of 3.5 GB):
   `cd data && unzip -q UAH-DRIVESET-v1.zip -x "*.mp4" "*~"`
3. Result: `data/UAH-DRIVESET-v1/D<n>/<YYYYMMDDhhmmss>-<km>km-D<n>-<BEHAVIOUR>-<ROAD>/`

Each trip folder holds 9 whitespace-separated text files with **no header row**, plus the video.
Column 0 of every time-series file is **seconds since route start** (float). The README only
documents the two SEMANTIC files; column names for the rest come from the bundled reader
(`uah_driveset_reader/driveset_reader.py`, `setColumnInfo`), and the axis meanings and event
codes below were then checked against the data.

### The files we need

**`RAW_ACCELEROMETERS.txt`: acceleration, 10 Hz.** Median step 0.100 s (5th-95th percentile
0.090-0.110 s, a few duplicate timestamps), so it needs light resampling onto an even 10 Hz grid.

| Col | Meaning (reader) | Unit | Checked |
| --- | --- | --- | --- |
| 0 | time since start | s | |
| 1 | activation flag (1 when speed > 50 km/h) | bool | |
| 2, 3, 4 | raw acceleration X, Y, Z | g | means ~0, so gravity is already removed |
| 5, 6, 7 | **Kalman-filtered** acceleration X, Y, Z | g | use these |
| 8, 9, 10 | roll, pitch, yaw | reader says degrees | values look like radians (roll ≈ −1.55); not needed |

Axes, worked out from the data (the README does not say):
- **Z (col 7) = longitudinal, positive = speeding up, negative = braking.** Braking events
  (type 1) peak at negative Z; acceleration events (type 3) at positive Z. Z also follows the
  change in GPS speed on every trip (positive correlation on all 40).
- **GPS lag: about 0.75 s.** Over the 25 braking peaks below −0.3 g, the biggest GPS speed drop
  comes a median 0.75 s after the accelerometer peak (middle half 0 to 1.6 s; the GPS is only
  1 Hz). An earlier smoothed cross-correlation suggested 3 s, but that put the speed drop
  *before* the braking, so it was wrong. `core.vehicle.gps_lag_s` uses 0.75.
- **Y (col 6) = lateral, positive when turning right.** Turning events (type 2) show large |Y|
  and almost no Z, and Y follows speed × rate of course change (course in degrees clockwise, so
  positive = right turn; correlation +0.62 on the D1 normal motorway trip).
- **X (col 5) = vertical** (road bumps); small, not used.

So `vehicle.motion` maps to: `accel_long_g` = col 7, `accel_lat_g` = col 6,
`accel_mag_g` = √(long² + lat²).

**`RAW_GPS.txt`: speed and position, 1 Hz** (median step 1.000 s). Shift by the ~0.75 s lag, then interpolate to 10 Hz.

| Col | Meaning | Unit |
| --- | --- | --- |
| 0 | time since start | s |
| 1 | **speed** | km/h |
| 2, 3 | **latitude, longitude** | degrees (WGS84) |
| 4 | altitude | m |
| 5, 6 | vertical, horizontal accuracy | m |
| 7 | course (heading) | degrees; −1 before the first fix |
| 8 | course change | degrees |
| 9-11 | internal DriveSafe values | |

The first row can be before a GPS fix (speed 0, course −1); drop rows with course −1.

**`EVENTS_INERTIAL.txt`: DriveSafe's own harsh-manoeuvre events.** It may be empty
(D4 NORMAL2 SECONDARY has none).

| Col | Meaning |
| --- | --- |
| 0 | time since start (s) |
| 1 | type: **1 braking, 2 turning, 3 acceleration** |
| 2 | level: **1 low, 2 medium, 3 high** |
| 3, 4 | latitude, longitude |
| 5 | wall-clock timestamp `YYYYMMDDhhmmss` |

Type and level are confirmed against the low/medium/high counts in `SEMANTIC_FINAL.txt`:
358 of 360 counts match across all 40 trips (the 2 misses are both in D1 DROWSY SECONDARY).

### Other files (not needed for the replay)

| File | Rate (average; irregular) | Contents |
| --- | --- | --- |
| `PROC_OPENSTREETMAP_DATA.txt` | ~0.4 Hz | max speed (km/h), reliability flag, road type (`motorway`, `motorway_link`, ...), lanes, current lane, lat, lon, query delay (s), speed (km/h) |
| `PROC_LANE_DETECTION.txt` | ~24 Hz | position from lane centre (m), phi, road width (m), estimator state; −9 = not available |
| `PROC_VEHICLE_DETECTION.txt` | ~5 Hz | distance to the car ahead (m), time to impact (s), number of cars, GPS speed; −1 = none |
| `EVENTS_LIST_LANE_CHANGES.txt` | on event | time, direction (±1), lat, lon, duration, threshold |
| `SEMANTIC_ONLINE.txt` | 1 Hz | DriveSafe scores; columns listed in the bundle README |
| `SEMANTIC_FINAL.txt` | one value per line | trip summary (avg/max speed, event counts, scores); line 11 is a value with a unit (`25 km`) |

### What this means for the replay (T08) and thresholds

- **Harsh braking above 0.4 g (RR-04, DB-02) is rare in this data.** Across all 40 trips the
  strongest Kalman braking is −0.14 g median for normal trips, −0.33 g for drowsy and −0.25 g for
  aggressive; only 5 trips go below −0.4 g at all. The aggressive trips show their style through
  many more events (D6 aggressive motorway has 80), not bigger peaks. So the live demo relies on
  the injector (T09: −0.6 g) for a visible harsh brake, and RR-04 may need calibrating (for example
  0.3 g, or DriveSafe's own EVENTS_INERTIAL levels) on Saturday.
- **No crash-like spikes:** the largest filtered value on any axis is 0.62 g, so CR-01 (4 g) is
  only ever tested by injection, as planned.

Proposed replay trips (all motorway, so the monotony rule ZO-01 can fire):

| Trip | Folder | Length | Max speed | Strongest braking | Events |
| --- | --- | --- | --- | --- | --- |
| normal | `D1/20151111123124-25km-D1-NORMAL-MOTORWAY` | 14.6 min | 131 km/h | −0.17 g | 6 |
| drowsy | `D2/20151120135152-25km-D2-DROWSY-MOTORWAY` | 15.9 min | 122 km/h | −0.44 g | 14 |
| aggressive | `D5/20151209153137-25km-D5-AGGRESSIVE-MOTORWAY` | 13.4 min | 147 km/h | −0.46 g | 43 |

The drowsy and aggressive picks are the two motorway trips with a real braking peak beyond
0.4 g, so the harsh-event marker also appears on genuine data.
