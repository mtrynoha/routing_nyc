# routing_nyc — мережева модель часу проїзду на даних Нью-Йорка

Опис методу й результатів: звіт «Мережева модель часу проїзду на даних Нью-Йорка» (розділ 2 зведеного звіту етапу 1).
Модель IGP і TD-алгоритм беруться з `..\routing_td` (`td_core.py`, `td_batch.py`).

## Дані
- Завантаження: `python nyc_download.py` (TLC yellow taxi, NYC DOT Traffic Speeds, EZ Pass Readers, OSM; 01.04–31.05.2024).
  Опції: `--only tlc dot ez osm`, `--hvfhv` (Uber/Lyft). EZ Pass качається 1–3 год; повторний запуск продовжує з місця зупинки.
  Скрипт пише в папку, де лежить сам (`tlc\`, `dot\`, `ez\`, `osm\`).
- Граф OSM окремо (з повторними спробами): `python nyc_osm.py` → `osm\manhattan_drive.graphml`, `manhattan_edges.csv`, `manhattan_nodes.csv`.
- Решта скриптів очікує ці дані в підпапці `raw\` робочої папки: `raw\tlc`, `raw\dot`, `raw\ez`, `raw\osm`.
- Прогнози DGCRN: `dgcrn_nyc_pred.npy` — з `..\Traffic-Benchmark_Fork` (навчання на `nyc_series.npz`, `adj_mx_nyc.pkl`).

Залежності: numpy, pandas, pyarrow, scipy, shapely, pyproj, osmnx, requests, matplotlib.

## Порядок запуску
```
python nyc_quality.py                  # покриття датчиків, придатність поїздок -> ez_seg_cov / dot_seg_cov.parquet
python nyc_match.py                    # прив'язка датчиків до ребер OSM -> osm_edges_matched.csv, nyc_observed_edges.png (рис. 8)
python nyc_map.py                      # карта датчиків -> nyc_sensors_map.png (необов'язково)
python nyc_sensor_series.py            # 15-хв ряди швидкостей датчиків -> sensor_speed_15min.parquet
python nyc_edge_times.py               # ядро Мідтауна (nyc_match_core.py), швидкості ребер -> nyc_core_graph.parquet
python nyc_zones_trips.py              # зони таксі ядра і поїздки -> core_trips.parquet, zone_nodes.json
python nyc_calibrate.py                # калібрування alpha і c на квітні -> calibration_grid.csv, calibration_best.json
ren calibration_grid.csv calibration_grid_1.csv
python nyc_calibrate2.py               # розширення сітки (alpha_main 1.3–2.0); найкраще лишається 0.7 / 1.2, c = 2.06 хв
python nyc_td_run.py 0 2               # матриці static / snapshot / TD (дві половини, можна паралельно)
python nyc_td_run.py 1 2
python nyc_eval.py                     # перевірка на травні -> nyc_eval.json, nyc_eval.png (табл. 7, рис. 9)
python nyc_eval_hybrid.py              # гібриди td_pair і HA x TD -> nyc_eval_hybrid.json (табл. 7)
python nyc_turns_calibrate.py          # затримки на поворотах (модель у nyc_turns.py) -> turns_grid.csv, turns_result.json
python nyc_build_dgcrn.py              # набір для DGCRN -> nyc_series.npz, adj_mx_nyc.pkl, nyc_sensors.csv
#   навчання DGCRN у ..\Traffic-Benchmark_Fork -> dgcrn_nyc_pred.npy
python nyc_forecast_routing.py 0 2     # матриці persistence / DGCRN / оракул на тесті (19–31.05)
python nyc_forecast_routing.py 1 2
python nyc_forecast_eval.py            # порівняння з поїздками -> nyc_forecast_eval.json (табл. 9)
python nyc_tours.py 0 2 300            # 300 турів доставки, дві половини (поля політик — nyc_fields.py)
python nyc_tours.py 1 2 300
python nyc_tours_eval.py               # ETA, програш порядку, бутстреп-ДІ -> nyc_tours_eval.json, nyc_tours_eval.png (табл. 10, рис. 10)
```
Номери таблиць і рисунків — за зведеним звітом етапу 1.

Проміжні файли `zone_mats_shard*.npz`, `fc_mats_shard*.npz` і сирі дані в `raw\` не включено.
