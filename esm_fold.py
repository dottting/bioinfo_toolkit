#!/usr/bin/env python3
"""Generates ESM3 protein folds from an input file and saves them as PDB files."""

import argparse, logging, tqdm, tarfile
import pandas as pd
from pathlib import Path
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig


def setup_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def get_esm_folds(df, client, output_dir):
    """Generates protein folds from sequences in a DataFrame and saves them as PDB files."""

    for _, row in tqdm.tqdm(
        df.iterrows(), total=len(df), desc="Predicting protein structures"
    ):
        seq = row["sequence"][:1500]  # limit sequence length
        protein = ESMProtein(sequence=seq)
        config = GenerationConfig(
            track="structure",
            schedule="cosine",
            num_steps=int(len(seq) / 16),
            temperature=0.7,
        )
        try:
            folded_protein = client.generate(protein, config)
            if not isinstance(folded_protein, ESMProtein):
                raise TypeError(
                    f"Expected ESMProtein, got {type(folded_protein)} for ID {row['protein_id']}"
                )
            pdb_file = output_dir / f"{row['protein_id']}.pdb"
            folded_protein.to_pdb(pdb_file)
            logging.info(
                f"Saved folded structure for {row['protein_id']} to {pdb_file}"
            )
        except Exception as e:
            logging.info(f"Skipped {row['protein_id']} with exception {e}")


def compress_pdbs(output_dir, archive_name="all_pdbs.tar.gz"):
    """Compress all PDB files in output_dir into a single gzip tar file."""
    archive_path = output_dir / archive_name
    with tarfile.open(archive_path, "w:gz") as tar:
        for pdb_file in output_dir.glob("*.pdb"):
            tar.add(pdb_file, arcname=pdb_file.name)
    logging.info(f"Compressed all PDBs into {archive_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Predict protein folds for protein sequences with ESM3."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to input TSV with 'protein_id' and 'sequence' columns.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=False,
        default=Path("./data/protein_structures"),
        help="Directory to save output PDB files.",
    )
    parser.add_argument(
        "-t",
        "--token",
        type=str,
        required=True,
        help="Hugging Face token for authentication.",
    )
    parser.add_argument(
        "-c",
        "--compress",
        action="store_true",
        help="Compress all generated PDB files into a single gzip tarball.",
    )

    args = parser.parse_args()
    setup_logger()

    # Login and load model
    login(token=args.token)
    logging.info("Logged into Hugging Face Hub.")
    client = ESM3.from_pretrained("esm3-open").to("cuda")
    logging.info("Loaded ESM3 model.")

    # Read input
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    df = pd.read_csv(args.input, sep="\t")

    if not {"protein_id", "sequence"}.issubset(df.columns):
        raise ValueError(
            "Input DataFrame must contain 'protein_id' and 'sequence' columns."
        )

    # Create output dir if it doesnt exist
    args.output.mkdir(parents=True, exist_ok=True)

    # Process sequences
    get_esm_folds(df, client, args.output)

    # Optional compression
    if args.compress:
        logging.info("Compressing files...")
        compress_pdbs(args.output)


if __name__ == "__main__":
    main()
