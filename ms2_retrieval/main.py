import json
import re
import subprocess
from pathlib import Path

import spacy
import typer
from pyserini.search.faiss import FaissSearcher, TctColBertQueryEncoder

from ms2_retrieval import msg
from typing import Dict, List

app = typer.Typer()

_SPACY_MODEL = "en_core_web_sm"
_QUESTION_WORDS = ["Is", "Does", "Can", "Are"]
_ENCODER = "castorini/tct_colbert-v2-hnp-msmarco"


def _sanitize_text(text: str, lowercase: bool = False) -> str:
    """Cleans text by removing whitespace, newlines and tabs and (optionally) lowercasing."""
    sanitized_text = " ".join(text.strip().split())
    sanitized_text = sanitized_text.lower() if lowercase else sanitized_text
    return sanitized_text


@app.command()
def create_examples(input_dir: Path, output_dir: Path) -> None:
    """Create the preprocessed example files.

    input_dir: Path to a local copy of the MS2 dataset.
    output_dir: Path to the directory where the example files should be saved.
    """
    msg.divider("Create Examples")

    with msg.loading("Loading spaCy model..."):
        nlp = spacy.load("en_core_web_sm")
    msg.good(f"Loaded spaCy model {_SPACY_MODEL}.")

    # Queries for select reviews that is expected to surface the included studies of that review.
    examples = []
    # All included studies in the reviews of the examples.
    to_index = []

    with msg.loading("Creating example files (this could take several minutes)..."):
        for split_fp in Path(input_dir).glob("[training|validation|testing]*.jsonl"):
            with open(split_fp, "r") as f:
                for line in f:
                    parsed = json.loads(line)
                    doc = nlp(parsed["title"], disable="ner")

                    # Construct the example
                    example = {
                        "pmid": parsed["pmid"],
                        "included_studies": [],
                    }

                    # Add the neccessary information about the included studies
                    for study in parsed["included_studies"]:
                        example["included_studies"].append({"pmid": study["pmid"]})
                        # These are sometimes None, so handle that here.
                        title = study["references"][0]["title"] or ""
                        abstract = study["references"][0]["abstract"] or ""
                        # Add the included study to the index
                        to_index.append(
                            {
                                "id": study["pmid"],
                                # Need to use some basic cleaning on this text or pyserini will fail.
                                "contents": _sanitize_text(title) + " " + _sanitize_text(abstract),
                            }
                        )

                    # This is a bit of a hack, but we can create queries by retaining the first sentence
                    # of titles that end with a question mark and start with any of several question words
                    query = next(doc.sents).text
                    # spaCy has trouble with the proceeding ":"'s which are common.
                    query = query.rstrip(":")
                    if (
                        any(re.match(rf"{word}\b", parsed["title"]) for word in _QUESTION_WORDS)
                        # The minimum length check prevents us from creating really sparse queries,
                        # like "Does formulation matter?".
                        and len(query.split()) >= 4
                        # A sanity check that this is actually a question
                        and query.endswith("?")
                    ):
                        example["query"] = query
                        examples.append(example)

        msg.good(f"Created {len(examples)} examples with an index size of {len(to_index)}.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples_jsonl = "\n".join([json.dumps(example) for example in examples])
    to_index_jsonl = "\n".join([json.dumps(item) for item in to_index])

    (output_dir / "examples.jsonl").write_text(examples_jsonl)
    (output_dir / "to_index.jsonl").write_text(to_index_jsonl)
    msg.good(f"Preprocessed data saved to {output_dir.resolve()}.")


@app.command()
def create_index(input_fp: Path, output_dir: Path, device: str = "cpu") -> None:
    """Create the dense vector index.

    input_fp: Path to the file containing the preprocessed examples.
    output_dir: Path to the directory where the index should be saved.
    device: The device to use for embedding. Should be "cpu" or "cuda:0, cuda:1...".
    """
    msg.divider("Create Index")

    script_fp = Path(__file__).parent.parent.resolve() / "scripts" / "encode.sh"
    output_dir = Path(output_dir) / "index"
    subprocess.check_call(f"bash {script_fp} {input_fp} {output_dir} {device}", shell=True)
    msg.good(f"Local FAISS index saved to {output_dir}.")


@app.command()
def search_and_score(input_fp: Path, index_fp: Path) -> None:
    """Query the dense retriever with each example and score the retrieved results.

    input_fp: Path to the file containing the preprocessed examples.
    index_fp: Path to the directory containing the index.
    """
    msg.divider("Search and Score")

    with msg.loading("Loading examples..."):
        examples = [json.loads(line) for line in Path(input_fp).read_text().strip().splitlines()]
    msg.good(f"Loaded examples at {input_fp}.")
    with msg.loading("Loading encoder..."):
        encoder = TctColBertQueryEncoder(_ENCODER)
    msg.good(f"Loaded encoder {_ENCODER}.")
    with msg.loading("Loading index..."):
        searcher = FaissSearcher(index_fp, encoder)
    msg.good(f"Loaded index at {index_fp}.")

    metrics: Dict[str, List[float]] = {
        "mean-r-precision": [],
        "recall@10": [],
        "recall@50": [],
        "recall@100": [],
        "recall@1000": [],
    }
    with typer.progressbar(examples, label="Scoring examples") as pbar:
        for example in pbar:
            relevant_docids = [study["pmid"] for study in example["included_studies"]]
            # Compute R-Precision and Recall@K
            r = len(relevant_docids)
            for k in (r, 10, 50, 100, 1000):
                hits = searcher.search(example["query"], k=k)
                score = sum(True if hit.docid in relevant_docids else False for hit in hits) / r
                if k == r:
                    metrics["mean-r-precision"].append(score)
                else:
                    metrics[f"recall@{k}"].append(score)

    # typer.secho(f"* Mean R-Precision: {round(sum(mean_r_prec) / len(mean_r_prec), 4)}", bold=True)
    for metric, scores in metrics.items():
        typer.secho(f"* {metric}: {round(sum(scores) / len(scores), 4)}", bold=True)


if __name__ == "__main__":
    app()
