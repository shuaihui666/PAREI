# PAREI

Source code for PAREI

## Data

The dataset originates from [SRaSLR](https://ieeexplore.ieee.org/document/9590240).

- **PAREI-1 / PAREI-2 / PAREI-3 / PAREI-4**: Use the **validation set**  for evaluation.
- **PAREICodeEval**: Uses the **validation set**  for evaluation.
- **PAREICodeTest**: Uses the **test set**  for evaluation.

## Environment Setup

```bash
pip install -r requirements.txt
```

## Running

The first run will download the SimCSE checkpoint from Hugging Face (~400 MB), which requires an internet connection. Subsequent runs load from the cache.

| Script | Dataset | Output Directory | Description |
|--------|---------|-----------------|-------------|
| `python PAREI-1.py` | Validation set | `./PAREI-1/` | PAREI variant 1 |
| `python PAREI-2.py` | Validation set | `./PAREI-2/` | PAREI variant 2 |
| `python PAREI-3.py` | Validation set | `./PAREI-3/` | PAREI variant 3 |
| `python PAREI-4.py` | Validation set | `./PAREI-4/` | PAREI variant 4 |
| `python PAREICodeEval.py` | Validation set | `./EvalResult/` | Full PAREI evaluation |
| `python PAREICodeTest.py` | Test set | `./TestResult/` | Full PAREI test |


## Output

After execution, the terminal prints an evaluation table:

```
+-------+-----------+--------+--------+
| top-N | precision | recall | MAP    |
+-------+-----------+--------+--------+
```

results are saved under the corresponding output directory:
- `./PAREI-1/` — results for PAREI-1 (validation set)
- `./PAREI-2/` — results for PAREI-2 (validation set)
- `./PAREI-3/` — results for PAREI-3 (validation set)
- `./PAREI-4/` — results for PAREI-4 (validation set)
- `./EvalResult/` — results for PAREICodeEval (validation set)
- `./TestResult/` — results for PAREICodeTest (test set)
