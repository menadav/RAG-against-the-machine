import fire
import sys
from src.ragcli import RAGCli


def main() -> None:
    try:
        fire.Fire(RAGCli)
    except KeyboardInterrupt:
        print("[WARNING] Interrupt the keyboard, finish the simulation.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
