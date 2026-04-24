import sys
try:
    import fire
except ImportError:
    print("[ERROR] Install dependencies.", file=sys.stderr)
    sys.exit(1)
from src.ragcli import RAGCli


def main() -> None:
    try:
        fire.Fire(RAGCli)
    except KeyboardInterrupt:
        print(
            "[WARNING] Interrupt the keyboard, finish the simulation.\n",
            file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
