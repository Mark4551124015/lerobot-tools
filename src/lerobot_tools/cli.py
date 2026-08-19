"""Unified command line interface for :mod:`lerobot_tools`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch ``lerobot_tools`` subcommands without duplicating their argument parsers."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="lerobot_tools",
        description="LeRobot v3→v2.1 conversion and JPEG-in-LMDB cache utilities.",
    )
    subcommands = parser.add_subparsers(dest="command", metavar="COMMAND")
    subcommands.add_parser(
        "lmdb-build",
        add_help=False,
        help="Build JPEG-in-LMDB frame caches from LeRobot v2.1 videos.",
    )
    subcommands.add_parser(
        "convert",
        add_help=False,
        help="Convert task-sharded LeRobot v3 data to v2.1 episode files.",
    )
    parsed, remainder = parser.parse_known_args(arguments)
    if parsed.command is None:
        parser.print_help()
        return

    original_argv = sys.argv
    try:
        sys.argv = [f"{sys.argv[0]} {parsed.command}", *remainder]
        if parsed.command == "lmdb-build":
            from .build import main as build_main

            build_main()
        else:
            from .conversion import main as convert_main

            convert_main()
    finally:
        sys.argv = original_argv
