import sys
try:
    import fire
except ImportError:
    print("[ERROR] Install dependencies.", file=sys.stderr)
    sys.exit(1)
from src.ragcli import RAGCli


def main() -> None:
    """CLI entry point.

    Initializes the command-line interface using 'fire' to expose RAGCli
    functionalities and handles keyboard interruptions gracefully.
    """
    try:
        fire.Fire(RAGCli)
    except KeyboardInterrupt:
        print(
            "[WARNING] Interrupt the keyboard, finish the simulation.\n",
            file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
