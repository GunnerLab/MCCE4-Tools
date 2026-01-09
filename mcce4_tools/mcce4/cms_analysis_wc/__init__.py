#!/usr/bin/env python3

import logging

from mcce4.constants import IONIZABLE_RES as IONIZABLES, ACIDIC_RES, BASIC_RES, POLAR_RES


APP_NAME = "ms_protonation"
MIN_OCC = 0.01

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s]: %(name)s:\n\t%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
