#!/usr/bin/env python3

from pathlib import Path


CLONE = Path(__file__).parent.parent.parent.name
# for cli parsers: give correct repo:
CLI_EPILOG = f"\nReport issues & feature requests here: https://github.com/GunnerLab/{CLONE}/issues\n"
