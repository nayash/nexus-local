from __future__ import annotations

import argparse
from typing import Sequence

from src.core.preflight import format_report, run_preflight


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nexus-local",
        description="Nexus Local CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the Flet desktop app")
    run_parser.set_defaults(handler=_handle_run)

    setup_parser = subparsers.add_parser("setup", help="Bootstrap local runtime dependencies")
    setup_parser.add_argument("--all", action="store_true", help="Run all bootstrap actions.")
    setup_parser.add_argument("--install-ollama", action="store_true", help="Install Ollama if missing.")
    setup_parser.add_argument("--start-ollama", action="store_true", help="Start Ollama service if not running.")
    setup_parser.add_argument("--pull-models", action="store_true", help="Pull required Ollama models.")
    setup_parser.add_argument("--install-pyodide", action="store_true", help="Install pyodide npm package.")
    setup_parser.add_argument("--build-docker-image", action="store_true", help="Build docker sandbox image.")
    setup_parser.add_argument("--download-onnx", action="store_true", help="Download multimodal ONNX assets.")
    setup_parser.add_argument(
        "--check-multimodal",
        action="store_true",
        help="Validate multimodal embedder initialization.",
    )
    setup_parser.set_defaults(handler=_handle_setup)

    doctor_parser = subparsers.add_parser("doctor", help="Run diagnostics without mutating the environment")
    doctor_parser.add_argument(
        "--check-multimodal",
        action="store_true",
        help="Validate multimodal embedder initialization.",
    )
    doctor_parser.set_defaults(handler=_handle_doctor)

    return parser


def _handle_run(_: argparse.Namespace) -> int:
    from src.ui.main import run_app

    run_app()
    return 0


def _handle_setup(args: argparse.Namespace) -> int:
    selected = any(
        (
            args.install_ollama,
            args.start_ollama,
            args.pull_models,
            args.install_pyodide,
            args.build_docker_image,
            args.download_onnx,
            args.check_multimodal,
        )
    )
    run_all = args.all or not selected
    report = run_preflight(
        install_ollama=args.install_ollama or run_all,
        start_ollama=args.start_ollama or run_all,
        pull_models=args.pull_models or run_all,
        install_pyodide=args.install_pyodide or run_all,
        build_docker_image=args.build_docker_image or run_all,
        download_onnx=args.download_onnx or run_all,
        check_multimodal_embedder=args.check_multimodal or run_all,
        migrate_legacy_data=True,
    )
    print(format_report(report))
    return 0 if report.core_ready else 1


def _handle_doctor(args: argparse.Namespace) -> int:
    report = run_preflight(
        install_ollama=False,
        start_ollama=False,
        pull_models=False,
        install_pyodide=False,
        build_docker_image=False,
        download_onnx=False,
        check_multimodal_embedder=args.check_multimodal,
        migrate_legacy_data=True,
    )
    print(format_report(report))
    return 0 if report.core_ready else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

