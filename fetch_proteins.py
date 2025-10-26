#!/usr/bin/env python3
"""
Fetch phage protein sequences from NCBI given a list of GenBank accessions.

Example usage:
    python fetch_phage_proteins.py \
        --input ../data/accession_list.csv \
        --outdir ../data/proteins \
        --email your_email@example.com
"""

from Bio import Entrez, SeqIO
import pandas as pd
from pathlib import Path
import argparse
import time
import sys


def fetch_proteins(accession, outdir, retries=3, delay=2):
    """Fetch protein sequences for one phage accession and save to FASTA."""
    print(f"Fetching phage data with id: {accession}")
    for attempt in range(1, retries + 1):
        try:
            handle = Entrez.efetch(
                db="nucleotide",
                id=accession,
                rettype="fasta_cds_aa",
                retmode="text",
            )
            records = list(SeqIO.parse(handle, "fasta"))
            handle.close()

            if not records:
                print(f"No protein records found for {accession}")
                return

            outpath = Path(outdir) / f"{accession}.fasta"
            with open(outpath, "w") as outfile:
                SeqIO.write(records, outfile, "fasta")
            print(f"Saved {len(records)} proteins to {outpath}")
            return
        except Exception as e:
            print(f"Attempt {attempt}/{retries} failed for {accession}: {e}")
            time.sleep(delay)

    print(f"Failed to fetch {accession} after {retries} attempts.")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch phage protein FASTA files from NCBI."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="CSV file with a column 'Accession'",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        required=True,
        help="Output directory for protein FASTA files",
    )
    parser.add_argument(
        "-e",
        "--email",
        required=True,
        help="Your email address (required by Entrez)",
    )
    args = parser.parse_args()

    Entrez.email = args.email

    # Read accessions
    with open(args.input) as f:
        accessions = [line.strip() for line in f if line.strip()]

    if not accessions:
        sys.exit("Error: No accessions found in input file.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for acc in accessions:
        fetch_proteins(acc, outdir)


if __name__ == "__main__":
    main()
