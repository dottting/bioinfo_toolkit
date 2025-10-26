#!/usr/bin/env python3
"""
Generates ESM3 protein embeddings from an input file and stores the dataframe as a partquet file.

This script requires a huggingface token to use.
Input is a tsv with the columns "protein_id" and "sequence".
"""

import argparse, logging, tqdm
import polars as pl
from pathlib import Path
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig


def setup_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def get_esm_embedding(sequence, client):
    """Generates ESM3 embeddings for a given protein sequence."""
    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)

    result = client.forward_and_sample(
        protein_tensor,
        SamplingConfig(return_per_residue_embeddings=True, return_mean_embedding=True),
    )
    return result


def process_sequences(df, client, outputdir):
    """
    Generate ESM3 embeddings for sequences (limited to length 1500) in the input DataFrame.
    Saves embeddings and ids to a parquet file.
    """
    ids = []
    embeddings = []

    for row in tqdm.tqdm(
        df.iter_rows(named=True), total=df.height, desc="Processing protein sequences"
    ):
        seq = row["sequence"]
        protein_id = row["protein_id"][:1500]

        try:
            embedding = get_esm_embedding(seq, client)
            ids.append(protein_id)
            embeddings.append(embedding.mean_embedding.cpu().numpy())
        except Exception as e:
            logging.error(f"Failed to processing sequence {protein_id}: {e}")

    result_df = pl.DataFrame({"protein_id": ids, "embedding": embeddings})
    outpath = outputdir / "embeddings.parquet"
    result_df.write_parquet(outpath, compression="zstd")

    logging.info(f"Saved {len(ids)} embeddings to {outputdir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ESM3 embeddings for protein sequences."
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
        default=Path("./data/embeddings"),
        help="Output directory for embedding files. ['./data/embeddings']",
    )
    parser.add_argument(
        "-t",
        "--token",
        type=str,
        required=True,
        help="Hugging Face token for authentication.",
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
    df = pl.read_csv(args.input, separator="\t")

    if not {"protein_id", "sequence"}.issubset(df.columns):
        raise ValueError(
            "Input DataFrame must contain 'protein_id' and 'sequence' columns."
        )

    # Create output dir if it doesnt exist
    args.output.mkdir(parents=True, exist_ok=True)

    # Process sequences
    process_sequences(df, client, args.output)


if __name__ == "__main__":
    main()
