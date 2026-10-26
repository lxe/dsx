#!/usr/bin/env python3
"""Build --rank JSON from completed DSX handoff.rs DSX_LOG_ROUTES logs.

This complements make_rank.py (quantizer calibration statistics). It does not
change the runtime placement algorithm. Standard library only.
"""
import argparse
import json
from pathlib import Path
import re


def build_rank(paths, layers=40, experts=384, picks=6, single_token_only=False):
    if min(layers, experts, picks) <= 0 or picks > experts:
        raise ValueError('invalid model dimensions')
    counts = {(l, e): 0 for l in range(layers) for e in range(experts)}
    seen = set()
    lines = used = skipped = 0
    for path in paths:
        with Path(path).open(encoding='utf-8') as f:
            for number, raw in enumerate(f, 1):
                lines += 1
                s = raw.rstrip('\n')
                if not re.fullmatch(r'[0-9]+(?: [0-9]+)+', s):
                    raise ValueError(f'{path}:{number}: expected handoff format: layer expert ...')
                layer, *ids = map(int, s.split(' '))
                if layer >= layers or len(ids) % picks or any(e >= experts for e in ids):
                    raise ValueError(f'{path}:{number}: invalid layer, expert or pick count')
                if any(len(set(ids[i:i+picks])) != picks for i in range(0, len(ids), picks)):
                    raise ValueError(f'{path}:{number}: duplicate expert within a token')
                if single_token_only and len(ids) != picks:
                    skipped += 1
                    continue
                seen.add(layer)
                used += 1
                for expert in ids:
                    counts[layer, expert] += 1
    if not used:
        raise ValueError('no usable routes')
    if seen != set(range(layers)):
        raise ValueError(f'calibration missing layers: {sorted(set(range(layers)) - seen)}')
    # Historical workstation calibration used expert-first tie breaking.
    rank = [[l, e, n] for (l, e), n in sorted(counts.items(), key=lambda item: (-item[1], item[0][1], item[0][0]))]
    stats = dict(lines=lines, used_lines=used, skipped_multi_token_lines=skipped,
                 total_picks=sum(counts.values()), unseen_experts=sum(n == 0 for n in counts.values()),
                 per_layer_picks=[sum(counts[l, e] for e in range(experts)) for l in range(layers)],
                 single_token_only=single_token_only,
                 note='Single-token filtering is a heuristic: one-token prefill is not distinguished. Counts do not predict byte-based placement or tokens/s.')
    return rank, stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calib', nargs='+', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--stats', type=Path)
    p.add_argument('--layers', type=int, default=40)
    p.add_argument('--experts', type=int, default=384)
    p.add_argument('--picks', type=int, default=6)
    p.add_argument('--single-token-only', action='store_true')
    a = p.parse_args()
    outputs = [a.out] + ([a.stats] if a.stats else [])
    if len({x.resolve() for x in outputs}) != len(outputs) or any(x.resolve() in {s.resolve() for s in a.calib} for x in outputs):
        p.error('output paths must differ from inputs and each other')
    if any(x.exists() for x in outputs):
        p.error('output already exists; choose new output paths')
    try:
        rank, stats = build_rank(a.calib, a.layers, a.experts, a.picks, a.single_token_only)
    except (ValueError, OSError) as e:
        p.error(str(e))
    with a.out.open('x', encoding='utf-8') as f:
        json.dump(rank, f)
        f.write('\n')
    if a.stats:
        with a.stats.open('x', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
            f.write('\n')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
