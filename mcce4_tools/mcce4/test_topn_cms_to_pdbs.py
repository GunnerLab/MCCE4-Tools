#!/usr/bin/env python

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

from mcce4.topn_cms_to_pdbs import tcms_parser
from mcce4.topn_cms_to_pdbs import APP_NAME
from mcce4.topn_cms_to_pdbs import IONIZABLE_RES
from mcce4.topn_cms_to_pdbs import N_TOP
from mcce4.topn_cms_to_pdbs import MIN_OCC

import pytest


class TestTcParser:

    def test_parser_default_values(self):
        """Parser returns ArgumentParser object with correct default values"""
        
        parser = tcms_parser()
    
        assert isinstance(parser, ArgumentParser)
        assert parser.prog == APP_NAME
        assert parser.formatter_class == RawDescriptionHelpFormatter
    
        # Check default values for all arguments
        defaults = {action.dest: action.default for action in parser._actions
                    if action.dest != "help"}
    
        assert defaults["mcce_dir"] == "./"
        assert defaults["ph"] == "7"
        assert defaults["eh"] == "0"
        assert defaults["n_top"] == N_TOP
        assert defaults["residue_kinds"] == IONIZABLE_RES
        assert defaults["min_occ"] == MIN_OCC
        assert defaults["wet"] is False
        assert defaults["overwrite"] is False

    def test_parser_argument_types(self):
        parser = tcms_parser()

        args = parser.parse_args([
            "-mcce_dir", "/path/to/dir",
            "-ph", "7.5",
            "-eh", "0.5",
            "-n_top", "10",
            "-residue_kinds", "ASP",
            "-min_occ", "0.01",
            "--wet"
        ])
        assert isinstance(args.mcce_dir, str)
        assert isinstance(args.ph, str)
        assert isinstance(args.eh, str)
        assert isinstance(args.n_top, int)
        assert isinstance(args.residue_kinds, str)
        assert isinstance(args.min_occ, float)
        assert isinstance(args.wet, bool)
    
    def test_residue_kinds_comparison_diff(self):
        parser = tcms_parser()
        args = parser.parse_args([
            "-residue_kinds", "ASP",
        ])
        # result set not empty:
        assert set(args.residue_kinds.split(",")).symmetric_difference(IONIZABLE_RES)

    def test_residue_kinds_comparison_same(self):
        parser = tcms_parser()

        args = parser.parse_args([
            "-residue_kinds", "ASP,GLU,ARG,HIS,LYS,CYS,TYR,NTR,CTR",
        ])
        # result set empty:
        assert not set(args.residue_kinds.split(",")).symmetric_difference(IONIZABLE_RES)


    def test_ph_eh_string_values(self):
        """Handling of string pH and eH values that need to be converted later"""
        parser = tcms_parser()
    
        # Test with integer pH/eH values
        args = parser.parse_args(["-ph", "5", "-eh", "2"])
        assert args.ph == "5"
        assert args.eh == "2"
        assert isinstance(args.ph, str)
        assert isinstance(args.eh, str)
    
        # Test with float pH/eH values
        args = parser.parse_args(["-ph", "7.5", "-eh", "-0.5"])
        assert args.ph == "7.5"
        assert args.eh == "-0.5"
        assert isinstance(args.ph, str)
        assert isinstance(args.eh, str)
    
        # Verify the values are not converted to numeric types by the parser
        # This allows for easier determination of int vs float later
        assert args.ph != 7.5
        assert args.eh != -0.5

    def test_parser_default_values(self):
        """Parser returns ArgumentParser object with correct default values"""
        parser = tcms_parser()
        args = parser.parse_args([])

        assert args.mcce_dir == "./"
        assert args.ph == "7"
        assert args.eh == "0"
        assert args.n_top == N_TOP
        assert args.residue_kinds == IONIZABLE_RES
        assert args.min_occ == MIN_OCC
        assert args.wet is False
