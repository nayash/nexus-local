import os

from src.core.config import Config
from src.embeddings import multimodal_onnx
from src.rag import ingestion_multimodal
from src.rag.query_filters import CompiledFilterPlan, compile_multimodal_filter, compile_multimodal_filter_plan


class FakeEmbedder:
    def embed_text(self, text: str):
        return [0.1, 0.2, 0.3]

    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embed_image(self, _image):
        return [0.4, 0.5, 0.6]


class FakeNomicEmbedder:
    def embed_query(self, text: str):
        _ = text
        return [0.7, 0.8, 0.9]

    def embed_documents(self, texts):
        return [[0.7, 0.8, 0.9] for _ in texts]


class TestMultimodalSetup:
    def test_config_has_multimodal_defaults(self):
        assert Config.MULTIMODAL_EMBEDDINGS_ENABLED is True
        assert Config.MULTIMODAL_MODEL_DIR.endswith(os.path.join("models", "clip_onnx"))
        assert Config.EMBEDDING_DEVICE == "cuda"
        assert Config.ORT_PROVIDER == "CUDAExecutionProvider"

    def test_embedder_gracefully_disables_when_model_dir_missing(self, monkeypatch):
        monkeypatch.setattr(Config, "MULTIMODAL_EMBEDDINGS_ENABLED", True)
        monkeypatch.setattr(Config, "MULTIMODAL_MODEL_DIR", "/definitely/missing/model_dir")
        monkeypatch.setattr(multimodal_onnx, "_EMBEDDER_SINGLETON", None)
        monkeypatch.setattr(multimodal_onnx, "_EMBEDDER_ERROR", None)

        embedder = multimodal_onnx.get_multimodal_embedder(force_refresh=True)

        assert embedder is None
        assert multimodal_onnx._EMBEDDER_ERROR is not None

    def test_missing_asset_description_calls_out_onnx_files(self, tmp_path):
        model_dir = tmp_path / "clip_onnx"
        model_dir.mkdir()
        for filename in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
        ):
            (model_dir / filename).write_text("{}", encoding="utf-8")

        message = multimodal_onnx._describe_missing_assets(str(model_dir))

        assert "text ONNX model" in message
        assert "vision ONNX model" in message
        assert "nexus-local setup --download-onnx" in message

    def test_text_file_ingests_multimodal_rows_with_mocked_embedder(self, tmp_path, monkeypatch):
        sample = tmp_path / "sample.txt"
        sample.write_text(
            "Refund policy: customers may request a refund within thirty days. " * 20,
            encoding="utf-8",
        )

        captured_upserts = []
        captured_deletes = []

        monkeypatch.setattr(ingestion_multimodal, "get_multimodal_embedder", lambda: FakeEmbedder())
        monkeypatch.setattr(ingestion_multimodal, "_NOMIC_EMBEDDER", FakeNomicEmbedder())
        monkeypatch.setattr(ingestion_multimodal, "_load_registry_row", lambda _path, workspace_id=None: None)
        monkeypatch.setattr(ingestion_multimodal, "_update_registry", lambda *args, **kwargs: None)

        def fake_upsert_rows(table_name, rows, delete_filter=None):
            captured_upserts.append(
                {
                    "table_name": table_name,
                    "rows": rows,
                    "delete_filter": delete_filter,
                }
            )
            return rows

        def fake_delete_rows(table_name, delete_filter=None):
            captured_deletes.append((table_name, delete_filter))

        monkeypatch.setattr(ingestion_multimodal, "upsert_rows", fake_upsert_rows)
        monkeypatch.setattr(ingestion_multimodal, "delete_rows", fake_delete_rows)

        success, row_count, doc_hash = ingestion_multimodal.ingest_file_multimodal(str(sample))

        assert success is True
        assert row_count > 0
        assert doc_hash
        assert len(captured_deletes) == 3
        upsert_tables = {item["table_name"] for item in captured_upserts}
        assert Config.MULTIMODAL_PARENT_TABLE in upsert_tables
        assert Config.MULTIMODAL_TEXT_CHILD_TABLE in upsert_tables
        assert Config.MULTIMODAL_CLIP_CHILD_TABLE in upsert_tables

        parent_upsert = next(item for item in captured_upserts if item["table_name"] == Config.MULTIMODAL_PARENT_TABLE)
        nomic_upsert = next(item for item in captured_upserts if item["table_name"] == Config.MULTIMODAL_TEXT_CHILD_TABLE)
        clip_upsert = next(item for item in captured_upserts if item["table_name"] == Config.MULTIMODAL_CLIP_CHILD_TABLE)

        assert parent_upsert["rows"]
        assert nomic_upsert["rows"]
        assert clip_upsert["rows"]

        first_parent = parent_upsert["rows"][0]
        assert first_parent["modality"] == "text"
        assert first_parent["source_type"] == "txt"
        assert first_parent["source_path"] == str(sample.resolve())
        assert first_parent["text"]
        assert first_parent["doc_hash"] == doc_hash
        assert first_parent["source_mtime_date"]
        assert first_parent["source_ctime_date"]
        assert first_parent["source_size_bytes"] > 0

        first_nomic_child = nomic_upsert["rows"][0]
        assert first_nomic_child["embedding_family"] == "nomic"
        assert first_nomic_child["vector"] == [0.7, 0.8, 0.9]
        assert first_nomic_child["parent_id"] == first_parent["parent_id"]
        assert first_nomic_child["source_ctime_date"] == first_parent["source_ctime_date"]

        first_clip_child = clip_upsert["rows"][0]
        assert first_clip_child["embedding_family"] == "clip"
        assert first_clip_child["vector"] == [0.1, 0.2, 0.3]
        assert first_clip_child["parent_id"] == first_parent["parent_id"]

    def test_search_local_image_results_render_as_placeholders(self, monkeypatch):
        fake_rows = [
            {
                "modality": "image",
                "source_type": "pdf",
                "source_path": "/tmp/example.pdf",
                "page": 2,
                "image_index": 0,
                "mime": "image/png",
                "width": 800,
                "height": 600,
                "extra": '{"cached_path": "/tmp/cached.png"}',
            }
        ]

        monkeypatch.setattr("src.tools.local.search_multimodal", lambda *args, **kwargs: fake_rows)

        from src.tools.local import _search_multimodal_results

        results = _search_multimodal_results("screenshot of the error message", top_k=5)

        assert len(results) == 1
        result = results[0]
        assert result.title == "Local Image (pdf): example.pdf"
        assert result.url == "/tmp/example.pdf"
        assert "[IMAGE]" in (result.content or "")
        assert "page=2" in (result.content or "")

    def test_search_multimodal_merges_nomic_and_clip_results(self, monkeypatch):
        parent_id_text = "parent-text"
        parent_id_image = "parent-image"

        monkeypatch.setattr(
            ingestion_multimodal,
            "compile_multimodal_filter_plan",
            lambda query, file_filter=None, workspace_id=None: CompiledFilterPlan(
                text_query=query,
                strict_sql_filter=None,
                relaxed_sql_filter=None,
                should_try_relaxed=False,
                strict_clauses=[],
                dropped_clauses=[],
            ),
        )

        monkeypatch.setattr(
            ingestion_multimodal,
            "_search_nomic_children",
            lambda *args, **kwargs: [
                {
                    "parent_id": parent_id_text,
                    "embedding_family": "nomic",
                    "modality": "text",
                    "text": "refund terms",
                    "_distance": 0.2,
                    "_retrieval_channel": "nomic",
                    "_rank": 0,
                }
            ],
        )
        monkeypatch.setattr(
            ingestion_multimodal,
            "_search_clip_children",
            lambda *args, **kwargs: [
                {
                    "parent_id": parent_id_image,
                    "embedding_family": "clip",
                    "modality": "image",
                    "text": "",
                    "_distance": 0.1,
                    "_retrieval_channel": "clip",
                    "_rank": 0,
                }
            ],
        )
        monkeypatch.setattr(
            ingestion_multimodal,
            "_load_parent_lookup",
            lambda file_filter=None, workspace_id=None: {
                parent_id_text: {
                    "parent_id": parent_id_text,
                    "modality": "text",
                    "text": "Refunds are available within thirty days.",
                    "source_path": "/tmp/refunds.txt",
                    "source_type": "txt",
                    "extra": "{}",
                },
                parent_id_image: {
                    "parent_id": parent_id_image,
                    "modality": "image",
                    "text": "",
                    "source_path": "/tmp/error.png",
                    "source_type": "image",
                    "page": None,
                    "image_index": 0,
                    "mime": "image/png",
                    "width": 640,
                    "height": 480,
                    "extra": '{"cached_path": "/tmp/error_cached.png"}',
                },
            },
        )

        results = ingestion_multimodal.search_multimodal("screenshot of refund error", top_k=5)

        assert len(results) == 2
        assert {row["parent_id"] for row in results} == {parent_id_text, parent_id_image}
        assert any('"matched_embedding_family": "nomic"' in row["extra"] for row in results)
        assert any('"matched_embedding_family": "clip"' in row["extra"] for row in results)

    def test_search_multimodal_retries_with_relaxed_filter(self, monkeypatch):
        parent_id_text = "parent-text"

        monkeypatch.setattr(
            ingestion_multimodal,
            "compile_multimodal_filter_plan",
            lambda query, file_filter=None, workspace_id=None: CompiledFilterPlan(
                text_query=query,
                strict_sql_filter="document_kind = 'document'",
                relaxed_sql_filter=None,
                should_try_relaxed=True,
                strict_clauses=[],
                dropped_clauses=[],
            ),
        )

        def fake_nomic(query, top_k, sql_filter):
            if sql_filter == "document_kind = 'document'":
                return []
            return [
                {
                    "parent_id": parent_id_text,
                    "embedding_family": "nomic",
                    "modality": "text",
                    "text": "brainstorm writing hooks",
                    "_distance": 0.1,
                    "_retrieval_channel": "nomic",
                    "_rank": 0,
                }
            ]

        monkeypatch.setattr(ingestion_multimodal, "_search_nomic_children", fake_nomic)
        monkeypatch.setattr(ingestion_multimodal, "_search_clip_children", lambda *args, **kwargs: [])
        monkeypatch.setattr(ingestion_multimodal, "_lexical_source_path_filter", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            ingestion_multimodal,
            "_load_parent_lookup",
            lambda file_filter=None, workspace_id=None: {
                parent_id_text: {
                    "parent_id": parent_id_text,
                    "modality": "text",
                    "text": "Brainstorm writing hooks and outlines.",
                    "source_path": "/tmp/writing_tips.md",
                    "source_type": "md",
                    "extra": "{}",
                },
            },
        )

        results = ingestion_multimodal.search_multimodal("give me writing ideas from my notes", top_k=5)
        assert len(results) == 1
        assert results[0]["parent_id"] == parent_id_text

    def test_search_multimodal_skips_unfiltered_retry_for_high_precision_filter(self, monkeypatch):
        attempts = []

        monkeypatch.setattr(
            ingestion_multimodal,
            "compile_multimodal_filter_plan",
            lambda query, file_filter=None, workspace_id=None: CompiledFilterPlan(
                text_query="list down writing tips",
                strict_sql_filter="title LIKE '%Writing tips%'",
                relaxed_sql_filter=None,
                should_try_relaxed=False,
                strict_clauses=[],
                dropped_clauses=[],
                allow_unfiltered_fallback=False,
            ),
        )

        def fake_pass(*, label, semantic_query, sql_filter, nomic_pool, clip_pool):
            attempts.append((label, sql_filter))
            return []

        monkeypatch.setattr(ingestion_multimodal, "_run_semantic_retrieval_pass", fake_pass)
        monkeypatch.setattr(ingestion_multimodal, "_lexical_source_path_filter", lambda *args, **kwargs: None)

        results = ingestion_multimodal.search_multimodal("list down writing tips", top_k=5)
        assert results == []
        assert attempts == [("strict", "title LIKE '%Writing tips%'")]

    def test_compile_multimodal_filter_fallbacks_for_common_metadata_queries(self, monkeypatch):
        monkeypatch.setattr("src.rag.query_filters._get_query_constructor", lambda: (_ for _ in ()).throw(RuntimeError("no llm")))

        _, sql_filter = compile_multimodal_filter("give me log files from yesterday")
        assert "document_kind = 'log'" in (sql_filter or "")
        assert "source_mtime_date =" in (sql_filter or "")

        _, sql_filter = compile_multimodal_filter("give me all the books I have by Franz Kafka")
        assert "document_kind = 'book'" in (sql_filter or "")
        assert "author = 'Franz Kafka'" in (sql_filter or "")

    def test_compile_multimodal_filter_plan_relaxes_low_confidence_document_kind(self, monkeypatch):
        monkeypatch.setattr("src.rag.query_filters._get_query_constructor", lambda: (_ for _ in ()).throw(RuntimeError("no llm")))

        plan = compile_multimodal_filter_plan("give me writing ideas from my notes")

        assert plan.strict_sql_filter == "document_kind = 'document'"
        assert plan.should_try_relaxed is True
        assert plan.relaxed_sql_filter is None

    def test_real_onnx_embedder_setup_if_available(self):
        import pytest

        ort = pytest.importorskip("onnxruntime")
        pytest.importorskip("transformers")

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in providers:
            pytest.skip("CUDAExecutionProvider is not available in this environment")

        model_dir = Config.MULTIMODAL_MODEL_DIR
        text_candidates = ("text_encoder.onnx", "text_model.onnx", "model_text.onnx")
        vision_candidates = ("vision_encoder.onnx", "image_encoder.onnx", "vision_model.onnx", "model_vision.onnx")

        def has_any(names):
            return any(os.path.exists(os.path.join(model_dir, name)) for name in names)

        tokenizer_candidates = (
            "tokenizer.json",
            os.path.join("tokenizer", "tokenizer.json"),
        )

        if not os.path.isdir(model_dir):
            pytest.skip(f"Model directory not found: {model_dir}")
        if not has_any(text_candidates):
            pytest.skip("Text ONNX model file not found in configured model directory")
        if not has_any(vision_candidates):
            pytest.skip("Vision ONNX model file not found in configured model directory")
        if not has_any(tokenizer_candidates):
            pytest.skip("Tokenizer files not found in configured model directory")

        multimodal_onnx._EMBEDDER_SINGLETON = None
        multimodal_onnx._EMBEDDER_ERROR = None

        embedder = multimodal_onnx.get_multimodal_embedder(force_refresh=True)
        if embedder is None:
            error = multimodal_onnx._EMBEDDER_ERROR
            message = str(error) if error else "unknown embedder initialization failure"
            if "CUDAExecutionProvider" in message or "cuDNN" in message or "cudnn" in message:
                pytest.skip(f"CUDA provider is advertised but not usable in this environment: {message}")
            pytest.fail(f"Expected real ONNX embedder to load, but initialization failed: {message}")
        vector = embedder.embed_text("hello world")
        assert isinstance(vector, list)
        assert len(vector) > 0
        assert all(isinstance(value, float) for value in vector)
