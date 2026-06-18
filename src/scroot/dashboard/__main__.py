"""python -m scroot.dashboard - launch the review console."""
import sys


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Scroot Review Console",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Open http://localhost:7432 in your browser after starting.",
    )
    parser.add_argument("--port", type=int, default=7432)
    parser.add_argument("--store", default="./scroot_store.jsonl",
                        help="Path to JSONL feedback store")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--hosted", action="store_true",
                        help=argparse.SUPPRESS)  # Enterprise only
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn is required. Install with: pip install 'scroot[dashboard]'")
        sys.exit(1)

    from .server import create_app
    app = create_app(store_path=args.store, hosted=args.hosted)

    print("\n  ** SCROOT Review Console")
    print(f"  Store: {args.store}")
    print(f"  URL:   http://{args.host}:{args.port}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
