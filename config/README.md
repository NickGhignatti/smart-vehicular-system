# Tuned configuration

JSON files of config overrides (`"section.field": value`), produced by the tools and committed so
the whole team uses the same calibrated values:

- `calibration.json` — from `python -m svs calibrate` on CARLA approach recordings (sensor bias/noise)
- `best.json` — from `python -m svs tune` on the tuning recordings (filter / warning parameters)

Use them with `python -m svs --config-file config/calibration.json --config-file config/best.json <command>`
(later files win; `--set section.field=value` wins over both). Keys starting with `_` are notes.
Defaults live in `svs/config.py`; every value that is meant to be tuned has a `TUNE:` comment there.
