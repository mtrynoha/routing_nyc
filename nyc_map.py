import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
def decode(s):
    pts, i, lat, lng = [], 0, 0, 0
    while i < len(s):
        for k in (0, 1):
            shift = res = 0
            while True:
                b = ord(s[i]) - 63; i += 1; res |= (b & 0x1f) << shift; shift += 5
                if b < 0x20: break
            d = ~(res >> 1) if res & 1 else res >> 1
            if k == 0: lat += d
            else: lng += d
        pts.append((lat / 1e5, lng / 1e5))
    return np.array(pts)
ez = pd.read_parquet('ez_seg_cov.parquet'); dot = pd.read_parquet('dot_seg_cov.parquet')
fig, ax = plt.subplots(1, 2, figsize=(12, 7.5))
for a, (bb, title) in zip(ax, [((-74.27, -73.68, 40.49, 40.92), 'Усі райони'), ((-74.03, -73.90, 40.70, 40.82), 'Мангеттен')]):
    for _, r in ez.iterrows():
        p = decode(r.polyline) if isinstance(r.polyline, str) else None
        if p is not None and len(p) > 1:
            a.plot(p[:, 1], p[:, 0], color=plt.cm.viridis(r['cov']), lw=2.2, solid_capstyle='round')
    for _, r in dot.iterrows():
        p = np.array([[float(v) for v in xy.split(',')] for xy in str(r.link_points).split() if xy.count(',') == 1 and all(c.strip('-.').replace('.','',1).isdigit() for c in xy.split(','))]) if isinstance(r.link_points, str) else None
        if p is not None and len(p) > 1:
            p = p[(p[:, 0] > 40.45) & (p[:, 0] < 40.95) & (p[:, 1] > -74.3) & (p[:, 1] < -73.65)]
            if len(p) > 1:
                jump = np.r_[0, np.hypot(np.diff(p[:, 0]) * 111, np.diff(p[:, 1]) * 84)]
                for q in np.split(p, np.nonzero(jump > 1.5)[0]):
                    if len(q) > 1: a.plot(q[:, 1], q[:, 0], color='#c05621', lw=1.2, alpha=.8)
    a.set_xlim(bb[0], bb[1]); a.set_ylim(bb[2], bb[3]); a.set_aspect(1 / np.cos(np.radians(40.75))); a.set_title(title)
    a.set_xticks([]); a.set_yticks([])
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1)); sm.set_array([])
fig.colorbar(sm, ax=ax, shrink=.6, label='EZ Pass: частка 15-хв слотів з вимірами (кві–тра 2024)')
ax[0].plot([], [], color='#c05621', label='NYC DOT Traffic Speeds (123 ланки)'); ax[0].plot([], [], color=plt.cm.viridis(.8), lw=2.2, label='EZ Pass Readers (351 ділянка)'); ax[0].legend(loc='upper left', fontsize=8, frameon=False)
plt.savefig('nyc_sensors_map.png', dpi=140, bbox_inches='tight'); print('ok')
