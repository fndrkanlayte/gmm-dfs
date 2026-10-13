# SUMO scenarios

```
raw_data/manhattan_28x7/                  196 intersections, 16 phases   (600 s episodes, demand scale 2.0)
raw_data/hangzhou_4x4_gudang_18041610_1h/  16 intersections, 8 phases    (3600 s, raw demand)
raw_data/atlanta_1x5/                       5 intersections              (1200 s, raw demand)
```

Each directory holds `<name>.net.xml` + `<name>.rou.xml`. The scenarios are
redistributed from the LibSignal benchmarks
(github.com/DaRL-LibSignal/LibSignal) for reproducibility. To add a scenario,
drop a new `<name>.net.xml` / `.rou.xml` pair here and pass
`--scenario <name>`; the SF reward, zone machinery and baselines are
scenario-agnostic.
