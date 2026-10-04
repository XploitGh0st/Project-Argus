# argus_digraph

Merges `argus_cfg` and `argus_dfg` artifacts into a unified Dependency Digraph.

## Quickstart

```bash
python -m argus_digraph cfg.json dfg.json --out ./out/
```

This generates `./out/digraph.json`.

## Testing

```bash
pytest modules/dependency_digraph/tests
```
