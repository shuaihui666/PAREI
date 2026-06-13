# PAREI

Source code for PAREI

## Data

The dataset originates from [SRaSLR](https://ieeexplore.ieee.org/document/9590240).


## Environment Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
python PAREICode.py
```


The first run will download the SimCSE checkpoint from Hugging Face (~400 MB), which requires an internet connection. Subsequent runs load from the cache.

## Output

After execution, the terminal prints an evaluation table:

```
+-------+-----------+--------+--------+
| top-N | precision | recall | MAP    |
+-------+-----------+--------+--------+
```


Per-query similarity results are saved under `./result/demo*.txt`.
