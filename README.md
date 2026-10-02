# flowsight

Traffic analytics and anomaly detection for packet captures.

flowsight reads a capture, rolls the packets up into flows, scores those flows
with several different detectors, and writes the results out as charts. Captures
can come from a pcap file, from a live interface, or from the built in synthetic
generator when you want data that is the same on every run.

## Why it exists

Packet captures are a good excuse to practice a full data workflow end to end:
awkward input, feature engineering, unsupervised and supervised models, picking a
threshold that is defensible, and charts someone could actually act on. The repo
is laid out so each of those concerns sits on its own and can be read without the
others.

## Layout

```
src/flowsight/
  capture/      reading packets from a pcap, a live interface, or the generator
  features/     packets to flows, flows to model ready feature tables
  analytics/    descriptive statistics and time series rollups
  detect/       the detectors, all behind one interface
  evaluation/   scoring detectors and comparing them against each other
  viz/          Plotly figures, one function per chart
data/raw/       drop .pcap files here, contents are gitignored
reports/        generated charts land in reports/figures/
docs/sections/  notes on why each part works the way it does
tests/          mirrors src/flowsight
```

## Setup

Python 3.13. The version is pinned only because TensorFlow has no 3.14 wheel yet.

```
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Tests

```
pytest
```
