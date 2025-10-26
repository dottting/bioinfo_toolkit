"""
parse_fasta.py

Tool to correctly parse NCBI FASTA records with the following pattern
>lcl|ORG_ID_FEATURE_ID [locus_tag=x] [db_xref=GeneID:x] [protein=protein func] [protein_id=x.x] [location=X..Y] [gbkey=X]

Takes a FASTA file.
Returns a Biopython SeqRecord object.
"""

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation
import re


def parse_fasta(infile: str):
    """
    Parses a FASTA file and yields enriched SeqRecord objects.

    Args:
        infile (str): Path to the FASTA file.

    Yields:
        SeqRecord: Annotated Biopython SeqRecord object.
    """
    for record in SeqIO.parse(infile, "fasta"):
        name, desc, id_, features, dbxrefs = parse_description(record.description)

        yield SeqRecord(
            seq=Seq(str(record.seq)),
            id=id_,
            name=name,
            description=desc,
            features=features,
            dbxrefs=dbxrefs,
        )


def parse_description(description) -> list:
    dbxrefs = []
    features = []
    name = desc = id = "None"
    for item in description.split("["):
        item = item.strip(" ]")

        # NCBI identifiers
        if len(item.split("|")) != 1 and not item.startswith("protein"):
            dbxrefs.append(item.split("|")[0])

        if item.startswith("locus_tag="):
            name = item.split("=")[1]

        if item.startswith("protein="):
            desc = item.split("=")[1]

        if item.startswith("protein_id="):
            id = item.split("=")[1]

        if item.startswith("location="):
            it = item.split("=")[1]

            match = re.search(r"(\d+)\.\.(\d+)", it)

            strand = -1 if it.startswith("complement") else 1
            start, end = int(match.group(1)), int(match.group(2))

            floc = FeatureLocation(start, end, strand)
            feature = SeqFeature(floc)

        if item.startswith("gbkey"):
            feature.type = item.split("=")[1]

            features.append(feature)

        if item.startswith("db_xref"):
            dbxrefs.append(item.split("=")[1])

    return [name, desc, id, features, dbxrefs]
