"""Generate bge-m3 dense embeddings for exported RAG JSONL documents."""

import argparse
import json
from pathlib import Path

import numpy as np

from ragflow_style_pipeline.embedding_text import embedding_text
from ragflow_style_pipeline.local_search import load_documents


def parse_args(argv=None):
    """Parse bge-m3 embedding CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate bge-m3 dense embeddings.")
    parser.add_argument("--input", required=True, help="Input redacted document JSONL.")
    parser.add_argument("--vectors", required=True, help="Output .npy vector path.")
    parser.add_argument("--meta", required=True, help="Output metadata JSONL sidecar path.")
    parser.add_argument("--model", default="BAAI/bge-m3", help="Embedding model name or local path.")
    parser.add_argument("--device", default="cuda", help="Embedding device, such as cuda or cpu.")
    parser.add_argument("--batch-size", type=int, default=8, help="Encoding batch size.")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum token length for bge-m3.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum documents to encode.")
    return parser.parse_args(argv)


def normalize_dense_output(output):
    """Return dense vectors from a FlagEmbedding output as float32 NumPy array."""
    return np.asarray(output["dense_vecs"], dtype=np.float32)


def _load_encoder(model_name, device):
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(model_name, use_fp16=device == "cuda", device=device)


def _encode_batch(model, texts, batch_size, max_length):
    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    return normalize_dense_output(output)


def _sidecar_document(document):
    display_text = str(document.get("display_text") or document.get("text") or "")
    return {
        "doc_id": document["doc_id"],
        "case_content_clean": document.get("case_content_clean", ""),
        "case_goal_clean": document.get("case_goal_clean", ""),
        "embedding_text": document.get("embedding_text", ""),
        "display_text": display_text,
        "text": display_text,
        "metadata": document.get("metadata", {}),
        "derived": document.get("derived", {}),
    }


def write_embedding_outputs(documents, vectors, vector_path, meta_path):
    """Write vectors and matching document sidecar."""
    vector_path = Path(vector_path)
    meta_path = Path(meta_path)
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(vector_path, vectors)
    with meta_path.open("w", encoding="utf-8") as output_file:
        for document in documents:
            output_file.write(json.dumps(_sidecar_document(document), ensure_ascii=False) + "\n")


def main(argv=None):
    """Generate and save dense vectors for exported documents."""
    args = parse_args(argv)
    documents = load_documents(args.input, limit=args.limit)
    texts = [embedding_text(document) for document in documents]
    model = _load_encoder(args.model, args.device)
    vectors = _encode_batch(model, texts, args.batch_size, args.max_length)
    write_embedding_outputs(documents, vectors, args.vectors, args.meta)

    print(
        json.dumps(
            {
                "documents": len(documents),
                "model": args.model,
                "device": args.device,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "vectors": args.vectors,
                "meta": args.meta,
                "shape": list(vectors.shape),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
