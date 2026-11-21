# CodeLap

## DriveMate CV process

Start fake-mode CV output without waiting for a camera or broker implementation:

```bash
python3 -m backend.cv.run --fake --sequence all
```

Choose a single fake sequence with `--sequence normal`, `eyes_closing`, `looking_down`, `angry`, or `staring`.

Download the MediaPipe Face Landmarker model:

```bash
bash data/download_datasets.sh mediapipe
```

Live camera capture requires OpenCV and the shared broker client; see `backend/cv/camera.py` and `backend/cv/run.py`. The CV process publishes `driver.state` every 100 ms and is owned by the CV module.

Run the CV unit tests:

```bash
python3 -m unittest discover -s tests -v
```
