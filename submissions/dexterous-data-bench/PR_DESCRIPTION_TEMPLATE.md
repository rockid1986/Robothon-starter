Registration UUID: 36a76b5b-9ef3-44dd-9033-c677884bfb2d

## Project

Dexterous Data Bench is a self-contained MuJoCo dexterous manipulation and data
collection benchmark. A five-fingertip hand presses a four-key sequence,
executes a two-finger chord, sweeps a rotary dial, and records synchronized
controls, joint states, touch sensors, button depths, dial angle, contacts, and
camera-rendered demo frames.

## AI tools used

Codex

## How to run

```bash
python -m pip install -r requirements.txt
python submissions/dexterous-data-bench/run_demo.py
python submissions/dexterous-data-bench/collect_data.py --episodes 5
```

## Demo artifact

`submissions/dexterous-data-bench/demo.mp4`

## Notes

The project is self-contained and does not require external mesh downloads or a
GPU. The UUID above matches `registration.json`.
