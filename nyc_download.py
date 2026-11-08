"""Завантаження даних Нью-Йорка для часо-залежної матриці T_ij (запускати на Windows).

Що завантажується (за замовчуванням 01.04.2024 - 31.05.2024):
  1. TLC: поїздки yellow taxi (+ за бажанням Uber/Lyft, HVFHV) і довідник/межі 263 зон таксі
     -> реальні часи поїздок між зонами (ground truth для T_ij).
  2. NYC DOT Traffic Speeds (i4gi-tjb9): ~123 ланки швидкісних доріг/магістралей, крок ~5 хв.
  3. EZ Pass Readers (erdf-2akx): ~351 ділянка міських вулиць, виміряний медіанний час проїзду.
  4. OpenStreetMap: граф вулиць Мангеттена для автомобілів (osmnx).

Запуск:   python nyc_download.py            (усе)
          python nyc_download.py --hvfhv     (додатково Uber/Lyft, ~0.5 ГБ на місяць)
          python nyc_download.py --only tlc dot ez osm
Повторний запуск продовжує з місця зупинки (готові файли пропускаються).
Потрібно: pip install requests pandas pyarrow osmnx
Час: TLC і OSM — хвилини; DOT — ~10-20 хв; EZ Pass — 1-3 год (API віддає ~17 тис. рядків за годину даних).
"""
import argparse, io, json, sys, time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent
START, END = datetime(2024, 4, 1), datetime(2024, 6, 1)          # [START, END)
SOCRATA = 'https://data.cityofnewyork.us/resource/{id}.csv'
TLC = 'https://d37ci6vzurychx.cloudfront.net'
S = requests.Session(); S.headers['User-Agent'] = 'thesis-research-download/1.0'


def get(url, params=None, tries=6, timeout=300):
    for k in range(tries):
        try:
            r = S.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r
            print(f'   HTTP {r.status_code}, повтор {k + 1}/{tries}', flush=True)
        except requests.RequestException as e:
            print(f'   {type(e).__name__}, повтор {k + 1}/{tries}', flush=True)
        time.sleep(10 * (k + 1))
    raise RuntimeError(f'не вдалося завантажити {url}')


def save(df, path):
    try:
        df.to_parquet(path.with_suffix('.parquet'), index=False)
    except Exception:
        df.to_csv(path.with_suffix('.csv.gz'), index=False)


def done(path):
    return path.with_suffix('.parquet').exists() or path.with_suffix('.csv.gz').exists()


def months():
    m = START.replace(day=1)
    while m < END:
        yield m.strftime('%Y-%m')
        m = (m + timedelta(days=32)).replace(day=1)


def tlc(hvfhv):
    d = OUT / 'tlc'; d.mkdir(exist_ok=True)
    files = [('misc/taxi_zone_lookup.csv', 'taxi_zone_lookup.csv'), ('misc/taxi_zones.zip', 'taxi_zones.zip')]
    kinds = ['yellow'] + (['fhvhv'] if hvfhv else [])
    files += [(f'trip-data/{k}_tripdata_{m}.parquet', f'{k}_tripdata_{m}.parquet') for k in kinds for m in months()]
    for src, name in files:
        p = d / name
        if p.exists():
            continue
        print(f'[TLC] {name}', flush=True)
        r = get(f'{TLC}/{src}', timeout=900)
        p.write_bytes(r.content)


def socrata_range(ds_id, select, ts_field, step_hours, tag):
    d = OUT / tag; d.mkdir(exist_ok=True)
    t = START
    while t < END:
        day_path = d / f'{tag}_{t:%Y-%m-%d}'
        if done(day_path):
            t += timedelta(days=1); continue
        parts = []
        for h in range(0, 24, step_hours):
            a = t + timedelta(hours=h); b = a + timedelta(hours=step_hours)
            off = 0
            while True:
                q = {'$select': select, '$where': f"{ts_field} >= '{a:%Y-%m-%dT%H:%M:%S}' AND {ts_field} < '{b:%Y-%m-%dT%H:%M:%S}'",
                     '$order': f'{ts_field}, :id', '$limit': 50000, '$offset': off}
                df = pd.read_csv(io.StringIO(get(SOCRATA.format(id=ds_id), q).text))
                parts.append(df); off += len(df)
                if len(df) < 50000:
                    break
        day = pd.concat(parts, ignore_index=True)
        save(day, day_path)
        print(f'[{tag}] {t:%Y-%m-%d}: {len(day):,} рядків', flush=True)
        t += timedelta(days=1)


def geometry(ds_id, cols, key, ts_field, tag):
    p = OUT / tag / f'{tag}_segments'
    if done(p):
        return
    q = {'$select': cols, '$where': f"{ts_field} >= '{START:%Y-%m-%dT00:00:00}' AND {ts_field} < '{START + timedelta(days=1):%Y-%m-%dT00:00:00}'",
         '$limit': 500000}
    df = pd.read_csv(io.StringIO(get(SOCRATA.format(id=ds_id), q).text)).drop_duplicates(key)
    save(df, p); print(f'[{tag}] геометрія: {len(df)} ділянок', flush=True)


def dot():
    (OUT / 'dot').mkdir(exist_ok=True)
    geometry('i4gi-tjb9', 'link_id,link_name,borough,owner,link_points,encoded_poly_line', 'link_id', 'data_as_of', 'dot')
    socrata_range('i4gi-tjb9', 'link_id,speed,travel_time,status,data_as_of', 'data_as_of', 24, 'dot')


def ez():
    (OUT / 'ez').mkdir(exist_ok=True)
    geometry('erdf-2akx', 'sid,link_name,borough,polyline,link_length_ft', 'sid', 'median_calculation_timestamp', 'ez')
    socrata_range('erdf-2akx', 'sid,median_calculation_timestamp,median_tt_sec,median_speed_fps,n_samples,aggregation_period_sec',
                  'median_calculation_timestamp', 6, 'ez')


def osm():
    p = OUT / 'osm' / 'manhattan_drive.graphml'
    if p.exists():
        return
    p.parent.mkdir(exist_ok=True)
    try:
        import osmnx as ox
    except ImportError:
        print('[OSM] встановіть osmnx: pip install osmnx — і запустіть скрипт ще раз'); return
    print('[OSM] граф вулиць Мангеттена...', flush=True)
    G = ox.graph_from_place('Manhattan, New York County, New York, USA', network_type='drive', simplify=True)
    ox.save_graphml(G, p)
    print(f'[OSM] вузлів {G.number_of_nodes():,}, ребер {G.number_of_edges():,}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', default=['tlc', 'dot', 'osm', 'ez'])
    ap.add_argument('--hvfhv', action='store_true')
    a = ap.parse_args()
    steps = {'tlc': lambda: tlc(a.hvfhv), 'dot': dot, 'osm': osm, 'ez': ez}
    for s in a.only:
        print(f'=== {s} ===', flush=True)
        try:
            steps[s]()
        except Exception as e:
            print(f'[{s}] ПОМИЛКА: {e}. Запустіть скрипт ще раз — він продовжить.', flush=True)
    json.dump({'start': str(START.date()), 'end': str(END.date()), 'finished': datetime.now().isoformat(timespec='seconds')},
              open(OUT / 'download_log.json', 'w'))
    print('готово', flush=True)
