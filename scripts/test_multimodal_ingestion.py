import os
import sys

from src.rag.ingestion_multimodal import ingest_paths, search_multimodal


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_multimodal_ingestion.py <sample-folder>")
        return 1

    sample_dir = os.path.abspath(sys.argv[1])
    if not os.path.isdir(sample_dir):
        print(f"Sample folder not found: {sample_dir}")
        return 1

    files = [
        os.path.join(sample_dir, name)
        for name in os.listdir(sample_dir)
        if os.path.isfile(os.path.join(sample_dir, name))
    ]
    successes, total_rows = ingest_paths(files)
    print(f"Ingested files={successes}, rows={total_rows}")

    queries = [
        "screenshot of the error message",
        "what does the document say about refunds",
        "diagram or screenshot",
    ]
    for query in queries:
        print(f"\nQuery: {query}")
        for index, row in enumerate(search_multimodal(query, top_k=5), start=1):
            print(
                f"{index}. modality={row.get('modality')} source={row.get('source_path')} "
                f"page={row.get('page')} chunk={row.get('chunk_index')} image={row.get('image_index')}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
