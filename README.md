## Example usage

Install companion conda `environment.yml`:

```bash
conda env create -f environment.yml
conda activate asm
```

---

Fetch proteins for a list of GenBank Accessions `fetch_proteins.py`:

```bash
python3 fetch_proteins.py -i example_data/accession_list.txt -o example_data/proteomes
```

---

Predict structures using ESM3 for a given list of protein sequences `esm_fold.py`

```bash
python3 esm_fold.py -i example_data/protein_list.tsv -o example_data/structures -t your_token
```
