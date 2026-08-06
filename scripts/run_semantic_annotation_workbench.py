"""Run the private SAG annotation workbench on the local loopback interface."""

import argparse
import threading
import webbrowser

from ragflow_style_pipeline.sag_annotation_server import create_workbench_server
from ragflow_style_pipeline.sag_annotation_workbench import AnnotationStore, AnnotationStoreError


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the loopback-only private semantic annotation workbench."
    )
    parser.add_argument("--input", required=True, help="Private in-progress A/B JSONL file")
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--port", type=int, default=0, help="Loopback port; 0 chooses an available port")
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        store = AnnotationStore(args.input, expected_annotator=args.annotator)
        server = create_workbench_server(store, port=args.port)
    except (AnnotationStoreError, OSError, ValueError) as exc:
        raise SystemExit("workbench_start_failed:" + str(exc)) from None
    bootstrap_url = (
        f"http://{server.expected_host}/?token={server.bootstrap_token}"
    )
    if not args.no_browser:
        threading.Timer(0.25, webbrowser.open, args=(bootstrap_url,)).start()
    print(
        f"SAG annotation workbench ready on loopback; records={store.summary()['records']}."
    )
    if args.no_browser:
        print("Open this private one-time bootstrap URL in a local browser:")
        print(bootstrap_url)
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
