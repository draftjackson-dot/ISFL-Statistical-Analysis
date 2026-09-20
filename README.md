# ISFL EPA - Play-by-Play Analyzer & EPA/WPA Calculator

Analyzes play-by-play data from the [ISFL](https://index.sim-football.com/) (International Simulation Football League) to extract player/team stats and calculate Expected Points Added (EPA) as well as Win Probability Added (WPA).

## Overview

The ISFL uses the DDSPF simulation engine across two eras:

- **Seasons 27–59** (DDSPF 2022 engine): PBP via LZString-compressed JSON files
- **Seasons 1–26** (DDSPF 2016 engine): PBP via individual HTML game pages

This project:

1. **Scrapes & caches** PBP and boxscore data from the index (both formats)
2. **Parses** natural-language play descriptions into structured data
3. **Aggregates** player and team statistics
4. **Calculates EPA** (Expected Points Added) per play using a model trained on historical sim data

## Project Status

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Data Extraction | Done | Scraper, decompression, caching for both engine eras |
| 2. Play Parsing | Done | Regex-based parser — ~100% parse rate, cross-validated against boxscores |
| 3. Stats Aggregation | Done | Player/team stats, player registry, PostgreSQL + Parquet storage, FastAPI |
| 4. EPA Model | Done | Expected points model, EPA/play, per-player/team aggregation, API |
