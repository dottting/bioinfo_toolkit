#!/usr/bin/env python3
"""Generates ESM3 protein folds from an input file and saves them as PDB files."""

import argparse, logging, tqdm, tarfile, torch, io
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


def get_esm_folds(df, client, output_dir, max_len, tar=None):
    """Generates protein folds from sequences in a DataFrame.

    If tar is provided, writes PDBs directly into the tar archive.
    Otherwise, saves them as individual files.
    """
    for _, row in tqdm.tqdm(
        df.iterrows(), total=len(df), desc="Predicting protein structures"
    ):
        seq = row["sequence"]
        protein = ESMProtein(sequence=seq)
        config = GenerationConfig(
            track="structure",
            schedule="cosine",
            num_steps=int(len(seq) / 16),
            temperature=0.7,
        )
        try:
            if len(seq) > max_len:
                raise Exception("Protein longer than max length")
            folded_protein = client.generate(protein, config)
            if not isinstance(folded_protein, ESMProtein):
                raise TypeError(
                    f"Expected ESMProtein, got {type(folded_protein)} for ID {row['protein_id']}"
                )

            pdb_filename = f"{row['protein_id']}.pdb"

            if tar is not None:
                # Write directly into tar
                pdb_buffer = io.StringIO()
                folded_protein.to_pdb(pdb_buffer)
                pdb_bytes = pdb_buffer.getvalue().encode("utf-8")
                info = tarfile.TarInfo(name=pdb_filename)
                info.size = len(pdb_bytes)
                tar.addfile(info, io.BytesIO(pdb_bytes))
                pdb_buffer.close()
                logging.info(f"Added {pdb_filename} to tar archive.")
            else:
                # Write to disk normally
                pdb_file = output_dir / pdb_filename
                folded_protein.to_pdb(pdb_file)
                logging.info(
                    f"Saved folded structure for {row['protein_id']} to {pdb_file}"
                )

        except Exception as e:
            logging.info(f"Skipped {row['protein_id']} with exception: {e}")


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
        help="Directory to save output PDB files or archive.",
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
        help="If set, directly write all generated PDBs into a single gzip tar instead of saving separately.",
    )
    parser.add_argument(
        "--len_cutoff",
        default=1500,
        type=int,
        help="Maximum protein length to fold.",
    )

    args = parser.parse_args()
    setup_logger()

    # Check CUDA
    if not torch.cuda.is_available():
        raise Exception("CUDA is not available.")

    # Read input
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    df = pd.read_csv(args.input, sep="\t")

    if not {"protein_id", "sequence"}.issubset(df.columns):
        raise ValueError("Input TSV must contain 'protein_id' and 'sequence' columns.")

    # Create output dir if not compressing
    args.output.mkdir(parents=True, exist_ok=True)

    # Login and load model
    login(token=args.token)
    logging.info("Logged into Hugging Face Hub.")
    client = ESM3.from_pretrained("esm3-open").to("cuda")
    logging.info("Loaded ESM3 model.")

    if args.compress:
        archive_path = args.output / "all_pdbs.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            get_esm_folds(df, client, args.output, args.len_cutoff, tar=tar)
        logging.info(f"All PDBs written into archive {archive_path}")
    else:
        get_esm_folds(df, client, args.output, args.len_cutoff)


if __name__ == "__main__":
    main()
