# Overview

Welcome to Peruze! A place for traffic analytics and anomaly detection for packet captures.

Here we read a capture, roll the packets up into flows, score those flows with several different detectors, and write the results out as charts.

## Peruze is a project for perusing network data.

Packet captures are great for keeping fresh on data skills including feature engineering, unsupervised and supervised models, and building charts to visual key data metrics.

## Layout

```
src/peruze/
  capture/      reading packets from a pcap, a live interface, or the generator
  features/     packets to flows, flows to model ready feature tables
  analytics/    descriptive statistics and time series rollups
  detect/       the detectors, all behind one interface
  evaluation/   scoring detectors and comparing them against each other
  viz/          Plotly figures, one function per chart
data/raw/       drop .pcap files here, contents are gitignored
reports/        generated charts land in reports/figures/
docs/sections/  notes on why each part works the way it does
tests/          mirrors src/peruze
```

## Setup

Python 3.13

```
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Tests

```
pytest
```
