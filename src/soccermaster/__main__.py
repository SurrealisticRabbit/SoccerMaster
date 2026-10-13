"""Allow ``python -m soccermaster``."""

from soccermaster.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
