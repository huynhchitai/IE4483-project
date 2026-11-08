"""Create the shared exact-text-grouped, class-stratified 80/20 split.

Run: python analysis/split_data.py (standard library only).
Output order follows the source data; manifest indices are zero-based.
"""
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / 'data'
SEED = 42
VALID_FRACTION = 0.2


def build_split(rows):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        assert isinstance(row['reviews'], str) and row['reviews'].strip()
        assert type(row['sentiments']) is int and row['sentiments'] in (0, 1)
        groups[row['reviews']].append(i)
    assert all(len({rows[i]['sentiments'] for i in ids}) == 1 for ids in groups.values()), 'Conflicting labels in exact-text group'
    valid = []
    for label in (0, 1):
        class_groups = sorted(
            (ids for ids in groups.values() if rows[ids[0]]['sentiments'] == label),
            key=lambda ids: ids[0])
        random.Random(SEED + label).shuffle(class_groups)
        totals = [0]
        for ids in class_groups:
            totals.append(totals[-1] + len(ids))
        target = VALID_FRACTION * totals[-1]
        k = min(range(len(totals)), key=lambda j: abs(totals[j] - target))
        valid.extend(i for ids in class_groups[:k] for i in ids)
    valid = sorted(valid)
    valid_set = set(valid)
    train = [i for i in range(len(rows)) if i not in valid_set]
    assert len(valid_set) == len(valid)
    assert set(train).isdisjoint(valid_set)
    assert set(train) | valid_set == set(range(len(rows)))
    assert not ({rows[i]['reviews'] for i in train} & {rows[i]['reviews'] for i in valid})
    assert all({rows[i]['sentiments'] for i in ids} == {0, 1} for ids in (train, valid))
    return train, valid


def main():
    source = (DATA / 'train.json').read_bytes()
    rows = json.loads(source)
    train, valid = build_split(rows)
    assert (train, valid) == build_split(rows), 'Split is not reproducible'
    manifest = {
        'source': 'train.json', 'source_sha256': hashlib.sha256(source).hexdigest(),
        'seed': SEED, 'validation_fraction_target': VALID_FRACTION,
        'grouping': 'exact review string, preserving case and whitespace',
        'algorithm': 'Per class: sort groups by first source index; shuffle with Python random.Random(seed + label); choose prefix closest to target row count; ties prefer shorter prefix.',
        'index_base': 0, 'output_order': 'ascending source row index',
        'train_indices': train, 'validation_indices': valid,
        'counts': {}, 'output_sha256': {},
    }
    for name, ids in [('train_splitted', train), ('valid_splitted', valid)]:
        records = [rows[i] for i in ids]
        content = (json.dumps(records, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
        path = DATA / f'{name}.json'
        path.write_bytes(content)
        assert json.loads(path.read_bytes()) == records
        manifest['output_sha256'][path.name] = hashlib.sha256(content).hexdigest()
        manifest['counts'][name] = {'rows': len(records), 'labels': dict(sorted(Counter(r['sentiments'] for r in records).items()))}
    (DATA / 'split_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['counts'], indent=2))
    print('Verified: reproducible, full row coverage, disjoint indices, no exact-text overlap.')


if __name__ == '__main__':
    main()
