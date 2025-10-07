#!/usr/bin/env python3

"""
Module  cms_analysis_wc./tests/test_analysis.py
"""
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import numpy as np
import pandas as pd
import pytest

from mcce4.cms_analysis_wc import analysis as msa
# Needed for defaults in pipeline tests:
from mcce4.cms_analysis_wc import parameters as prm
# For rename/reorder tests:
from mcce4.cms_analysis_wc import IONIZABLES, ACIDIC_RES, BASIC_RES, POLAR_RES

# Fixtures for WeightedCorr
@pytest.fixture
def sample_series():
    x = pd.Series([1, 2, 3, 4, 5], name='X')
    y = pd.Series([5, 4, 3, 2, 1], name='Y')
    w = pd.Series([1, 1, 2, 1, 1], name='W')
    return x, y, w

@pytest.fixture
def sample_xyw_df(sample_series):
    x, y, w = sample_series
    return pd.DataFrame({'X': x, 'Y': y, 'W': w})

@pytest.fixture
def sample_df_wcol():
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [5, 4, 3, 2, 1],
        'C': [2, 3, 4, 5, 6],
        'Count': [1, 1, 2, 1, 1]
    })
    return df, 'Count'

@pytest.fixture
def sample_df_zero_var():
    df = pd.DataFrame({
        'A': [1, 1, 1, 1, 1],
        'B': [5, 4, 3, 2, 1],
        'Count': [1, 1, 2, 1, 1]
    })
    return df, 'Count'

# Tests for WeightedCorr Class
class TestWeightedCorr:

    def test_init_xyw(self, sample_xyw_df):
        wcorr = msa.WeightedCorr(xyw=sample_xyw_df, cutoff=0.1)
        assert wcorr.df is None
        assert wcorr.cutoff == 0.1
        pd.testing.assert_series_equal(pd.Series(wcorr.x), sample_xyw_df.iloc[:, 0], check_names=False)
        pd.testing.assert_series_equal(pd.Series(wcorr.y), sample_xyw_df.iloc[:, 1], check_names=False)
        pd.testing.assert_series_equal(pd.Series(wcorr.w), sample_xyw_df.iloc[:, 2], check_names=False)

    def test_init_series(self, sample_series):
        x, y, w = sample_series
        wcorr = msa.WeightedCorr(x=x, y=y, w=w, cutoff=0.05)
        assert wcorr.df is None
        assert wcorr.cutoff == 0.05
        pd.testing.assert_series_equal(pd.Series(wcorr.x), x, check_names=False)
        pd.testing.assert_series_equal(pd.Series(wcorr.y), y, check_names=False)
        pd.testing.assert_series_equal(pd.Series(wcorr.w), w, check_names=False)

    def test_init_df_wcol(self, sample_df_wcol):
        df, wcol = sample_df_wcol
        wcorr = msa.WeightedCorr(df=df, wcol=wcol, cutoff=0.0)
        assert wcorr.x is None
        assert wcorr.y is None
        assert wcorr.cutoff == 0.0
        pd.testing.assert_frame_equal(wcorr.df, df[['A', 'B', 'C']])
        pd.testing.assert_series_equal(wcorr.w, df[wcol])

    def test_init_missing_data(self):
        with pytest.raises(ValueError, match="No data supplied"):
            msa.WeightedCorr()

    def test_init_invalid_combination1(self, sample_xyw_df):
        with pytest.raises(ValueError, match="Incorrect arguments specified"):
            msa.WeightedCorr(xyw=sample_xyw_df, wcol="W")

    def test_init_invalid_combination2(self, sample_series):
        x, y, w = sample_series
        with pytest.raises(ValueError, match="Incorrect arguments specified"):
            msa.WeightedCorr(x=x, y=y, w=w, wcol="W")

    def test_init_invalid_types(self):
        with pytest.raises(TypeError, match="xyw should be a pd.DataFrame"):
            msa.WeightedCorr(xyw=[1, 2, 3])
        with pytest.raises(TypeError, match="x, y, w should be pd.Series"):
            msa.WeightedCorr(x=[1], y=[2], w=pd.Series([1]))
        with pytest.raises(ValueError, match="df should be a pd.DataFrame"):
            msa.WeightedCorr(df=[1, 2], wcol="W")
        with pytest.raises(ValueError, match="wcol should be a string"):
            msa.WeightedCorr(df=pd.DataFrame({'A': [1]}), wcol=1)

    def test_init_missing_wcol(self, sample_df_wcol):
        df, _ = sample_df_wcol
        with pytest.raises(KeyError, match="wcol not found"):
            msa.WeightedCorr(df=df, wcol="MissingCol")

    def test_init_dropna(self):
        x = pd.Series([1, 2, np.nan, 4, 5])
        y = pd.Series([5, np.nan, 3, 2, 1])
        w = pd.Series([1, 1, 2, 1, 1])
        wcorr = msa.WeightedCorr(x=x, y=y, w=w)
        assert len(wcorr.x) == 3 # NaN rows dropped

    def test_pearson_series(self, sample_series):
        x, y, w = sample_series
        wcorr = msa.WeightedCorr(x=x, y=y, w=w, cutoff=0.0)
        # Expected calculation (manual or reference):
        # Weighted mean x = (1*1 + 2*1 + 3*2 + 4*1 + 5*1) / 6 = 18 / 6 = 3
        # Weighted mean y = (5*1 + 4*1 + 3*2 + 2*1 + 1*1) / 6 = 18 / 6 = 3
        # Weighted cov(x,y) = sum(w*(x-mx)*(y-my)) = 1*(1-3)*(5-3) + 1*(2-3)*(4-3) + 2*(3-3)*(3-3) + 1*(4-3)*(2-3) + 1*(5-3)*(1-3)
        #                   = 1*(-2)*(2) + 1*(-1)*(1) + 2*(0)*(0) + 1*(1)*(-1) + 1*(2)*(-2)
        #                   = -4 - 1 + 0 - 1 - 4 = -10
        # Weighted var(x) = sum(w*(x-mx)^2) = 1*(-2)^2 + 1*(-1)^2 + 2*(0)^2 + 1*(1)^2 + 1*(2)^2
        #                 = 4 + 1 + 0 + 1 + 4 = 10
        # Weighted var(y) = sum(w*(y-my)^2) = 1*(2)^2 + 1*(1)^2 + 2*(0)^2 + 1*(-1)^2 + 1*(-2)^2
        #                 = 4 + 1 + 0 + 1 + 4 = 10
        # Weighted Pearson r = -10 / sqrt(10 * 10) = -10 / 10 = -1.0
        assert wcorr._pearson() == pytest.approx(-1.0)
        assert wcorr(method='pearson') == pytest.approx(-1.0)

    def test_spearman_series(self, sample_series):
        x, y, w = sample_series
        wcorr = msa.WeightedCorr(x=x, y=y, w=w, cutoff=0.0)
        # Spearman is Pearson on weighted ranks
        # Ranks x: [1, 2, 3, 4, 5] -> ranks [1, 2, 3, 4, 5]
        # Ranks y: [5, 4, 3, 2, 1] -> ranks [5, 4, 3, 2, 1] (reversed order)
        # Weighted ranks need calculation using _wrank
        # Since ranks are linear, expect Spearman to be same as Pearson here
        assert wcorr._spearman() == pytest.approx(-1.0)
        assert wcorr(method='spearman') == pytest.approx(-1.0)

    def test_pearson_df(self, sample_df_wcol):
        df, wcol = sample_df_wcol
        wcorr = msa.WeightedCorr(df=df, wcol=wcol, cutoff=0.0)
        result_df = wcorr(method='pearson')
        assert isinstance(result_df, pd.DataFrame)
        assert result_df.shape == (3, 3)
        assert result_df.loc['A', 'B'] == pytest.approx(-1.0) # A vs B should be -1
        assert result_df.loc['A', 'A'] == pytest.approx(1.0) # Diagonal should be 1
        assert result_df.loc['B', 'A'] == result_df.loc['A', 'B'] # Symmetry

    def test_spearman_df(self, sample_df_wcol):
        df, wcol = sample_df_wcol
        wcorr = msa.WeightedCorr(df=df, wcol=wcol, cutoff=0.0)
        result_df = wcorr(method='spearman')
        assert isinstance(result_df, pd.DataFrame)
        assert result_df.shape == (3, 3)
        assert result_df.loc['A', 'B'] == pytest.approx(-1.0) # A vs B ranks are perfectly reversed
        assert result_df.loc['A', 'A'] == pytest.approx(1.0) # Diagonal should be 1
        assert result_df.loc['B', 'A'] == result_df.loc['A', 'B'] # Symmetry

    def test_cutoff_series(self, sample_series):
        x, y, w = sample_series
        wcorr_low_cutoff = msa.WeightedCorr(x=x, y=y, w=w, cutoff=0.1)
        wcorr_high_cutoff = msa.WeightedCorr(x=x, y=y, w=w, cutoff=1.1)
        assert wcorr_low_cutoff(method='pearson') == pytest.approx(-1.0)
        assert wcorr_high_cutoff(method='pearson') == 0.0 # Below cutoff

    def test_cutoff_df(self, sample_df_wcol):
        df, wcol = sample_df_wcol
        # Calculate expected full matrix first
        wcorr_full = msa.WeightedCorr(df=df, wcol=wcol, cutoff=0.0)
        full_matrix = wcorr_full(method='pearson')

        # Apply cutoff - e.g., cutoff > abs(corr(A,C)) and abs(corr(B,C))
        # corr(A,C) = 1.0, corr(B,C) = -0.98... let's use cutoff 0.99
        cutoff = 0.99
        wcorr_cutoff = msa.WeightedCorr(df=df, wcol=wcol, cutoff=cutoff)
        result_df = wcorr_cutoff(method='pearson')

        # Check values against cutoff
        assert result_df.loc['A', 'B'] == pytest.approx(-1.0) # Above cutoff
        assert result_df.loc['A', 'C'] == pytest.approx(1.0) # Above cutoff
        assert result_df.loc['B', 'C'] == 0.0 # Below cutoff abs(-0.98...) < 0.99 -> 0

        # Check filtering of rows/cols where all off-diagonals are 0
        # In this case, C's off-diagonals are 1.0 and 0.0, so C is kept.
        # If corr(A,C) was also below cutoff, C row/col would be dropped.
        assert 'C' in result_df.index
        assert 'C' in result_df.columns

    def test_zero_variance(self, sample_df_zero_var):
        df, wcol = sample_df_zero_var
        wcorr = msa.WeightedCorr(df=df, wcol=wcol, cutoff=0.0)
        result_df = wcorr(method='pearson')
        # Correlation of A with itself should be NaN -> caught -> 0
        # Correlation of A with B should be NaN -> caught -> 0
        assert result_df.loc['A', 'A'] == 0.0
        assert result_df.loc['A', 'B'] == 0.0
        assert result_df.loc['B', 'A'] == 0.0
        assert result_df.loc['B', 'B'] == pytest.approx(1.0) # B has variance

    def test_invalid_method(self, sample_series):
        x, y, w = sample_series
        wcorr = msa.WeightedCorr(x=x, y=y, w=w)
        with pytest.raises(ValueError, match="`method` should be one of"):
            wcorr(method='invalid_method')

# Fixtures for Data Manipulation Functions
@pytest.fixture
def sample_top_df():
    return pd.DataFrame({
        'ASP_A0001_': [-1, 0, -1, 0],
        'GLU_A0002_': [0, -1, -1, 0],
        'LYS_B0010_': [1, 1, 0, 0],
        'NTR_A0001_': [1, 1, 1, 1],
        'SomeOther': [0, 0, 0, 0], # Should be ignored by choose_res_data if not in list
        'Count': [10, 5, 8, 2],
        'Energy': [-10.1, -9.5, -11.0, -8.0],
        'UID': ['s1','s2','s3','s4'] # Example extra column
    }).set_index('UID')

@pytest.fixture
def sample_fixed_resoi_crg_df():
    return pd.DataFrame({
        'Residue': ['HIS_A0005_', 'ARG_C0020_'],
        'Charge': [0.5, 1.0] # Example fixed charges
    })

# Tests for Data Manipulation Functions
def test_choose_res_data(sample_top_df):
    correl_resids = ['ASP_A0001_', 'LYS_B0010_']
    # Expect selected correl_resids + last two columns ('Count', 'Energy')
    # Note: The fixture has 3 trailing columns (Count, Energy, UID).
    # The function selects last *two* based on its code `df.columns[-3:-1].tolist()`
    # which actually selects 'Count' and 'Energy'. Let's assume UID is index.
    expected_cols = ['ASP_A0001_', 'LYS_B0010_', 'Count', 'Energy']
    result_df = msa.choose_res_data(sample_top_df, correl_resids)
    assert list(result_df.columns) == expected_cols
    assert result_df.shape[0] == sample_top_df.shape[0]
    pd.testing.assert_frame_equal(result_df, sample_top_df[expected_cols].reset_index(drop=True))

def test_add_fixed_resoi_crg_to_topdf(sample_top_df, sample_fixed_resoi_crg_df):
    # Test with cms_wc_format=True (default in pipeline)
    result_df_wc = msa.add_fixed_resoi_crg_to_topdf(sample_top_df, sample_fixed_resoi_crg_df, cms_wc_format=True)
    # Expected order: original cols up to 'Count', fixed cols, 'Count', 'Energy'
    # Original non-count/energy cols: ASP, GLU, LYS, NTR, SomeOther
    expected_cols_wc = ['ASP_A0001_', 'GLU_A0002_', 'LYS_B0010_', 'NTR_A0001_', 'SomeOther',
                        'HIS_A0005_', 'ARG_C0020_',
                        'Count', 'Energy'] # Assuming UID is index
    assert list(result_df_wc.columns) == expected_cols_wc
    assert result_df_wc['HIS_A0005_'].iloc[0] == 0.5 # Check fixed value broadcast
    assert result_df_wc['ARG_C0020_'].iloc[0] == 1.0
    assert result_df_wc.shape[0] == sample_top_df.shape[0]

    # Test with cms_wc_format=False
    # Need a df with 4 trailing columns for this format's index slicing
    sample_top_df_alt = sample_top_df.copy()
    sample_top_df_alt['Extra'] = 1
    result_df_nonwc = msa.add_fixed_resoi_crg_to_topdf(sample_top_df_alt, sample_fixed_resoi_crg_df, cms_wc_format=False)
    expected_cols_nonwc = ['ASP_A0001_', 'GLU_A0002_', 'LYS_B0010_', 'NTR_A0001_', 'SomeOther',
                           'HIS_A0005_', 'ARG_C0020_',
                           'Count', 'Energy', 'Extra'] # Assuming UID is index
    assert list(result_df_nonwc.columns) == expected_cols_nonwc


def test_add_fixed_resoi_crg_to_topdf_no_fixed(sample_top_df):
    result_df = msa.add_fixed_resoi_crg_to_topdf(sample_top_df, None)
    pd.testing.assert_frame_equal(result_df, sample_top_df)

    empty_fixed_df = pd.DataFrame({'Residue': [], 'Charge': []})
    result_df_empty = msa.add_fixed_resoi_crg_to_topdf(sample_top_df, empty_fixed_df)
    pd.testing.assert_frame_equal(result_df_empty, sample_top_df)


def test_rename_reorder_df_cols():
    # Create a df that looks like output of choose_res_data
    input_df = pd.DataFrame({
        'LYS_B0010_': [1, 1, 0, 0],  # Basic
        'GLU_A0002_': [0, -1, -1, 0], # Acidic
        'NTR_A0001_': [1, 1, 1, 1],  # Terminus (Other)
        'HIS_A0005_': [0, 1, 0, 1],  # Basic
        'ASP_A0001_': [-1, 0, -1, 0], # Acidic
        'TYR_C0030_': [0, 0, -1, -1], # Polar
        'CTR_C0050_': [-1, -1, -1, -1],# Terminus (Other)
        'Count': [10, 5, 8, 2],
        'Energy': [-10.1, -9.5, -11.0, -8.0] # Should be dropped
    })
    # Expected renaming: Chain + 1LCode + SeqNum
    # Expected order: Acidic (D,E), Polar (Y), Basic (H,K), Other (NTR, CTR)
    expected_renamed_cols = ['AD1', 'AE2', 'CY30', 'AH5', 'BK10', 'ANTR1', 'CCTR50', 'Count']
    expected_order = ['AD1', 'AE2', 'CY30', 'AH5', 'BK10', 'ANTR1', 'CCTR50', 'Count']

    result_df = msa.rename_reorder_df_cols(input_df)

    assert list(result_df.columns) == expected_order
    # Check if original columns were renamed correctly before reordering
    assert 'BK10' in result_df.columns # LYS_B0010_ -> BK10
    assert 'AE2' in result_df.columns  # GLU_A0002_ -> AE2
    assert 'ANTR1' in result_df.columns # NTR_A0001_ -> ANTR1
    assert 'Energy' not in result_df.columns # Energy col should be dropped


def test_combine_all_free_fixed_residues():
    free_df = pd.DataFrame({
        'Residue': ['ASP_A0001_', 'GLU_A0002_'],
        'AverageCharge': [-0.8, -0.2]
    })
    fixed_df = pd.DataFrame({
        'Residue': ['HIS_A0005_', 'ARG_C0020_'],
        'Charge': [0.5, 1.0]
    })
    result_df = msa.combine_all_free_fixed_residues(free_df, fixed_df)

    assert isinstance(result_df, pd.DataFrame)
    assert result_df.shape == (3, 4) # Transposed: 3 rows (cols+status), 4 cols (res)
    assert list(result_df.index) == ['AverageCharge', 'Charge', 'status']
    assert list(result_df.columns) == ['ASP_A0001_', 'GLU_A0002_', 'HIS_A0005_', 'ARG_C0020_']
    assert result_df.loc['status', 'ASP_A0001_'] == 'free'
    assert result_df.loc['status', 'HIS_A0005_'] == 'fixed'
    assert pd.isna(result_df.loc['Charge', 'ASP_A0001_']) # Charge should be NaN for free
    assert pd.isna(result_df.loc['AverageCharge', 'HIS_A0005_']) # AverageCharge should be NaN for fixed
    assert result_df.loc['AverageCharge', 'GLU_A0002_'] == -0.2
    assert result_df.loc['Charge', 'ARG_C0020_'] == 1.0

def test_combine_all_free_fixed_residues_one_empty():
    free_df = pd.DataFrame({
        'Residue': ['ASP_A0001_', 'GLU_A0002_'],
        'AverageCharge': [-0.8, -0.2]
    })
    result_df_no_fixed = msa.combine_all_free_fixed_residues(free_df, None)
    assert list(result_df_no_fixed.columns) == ['ASP_A0001_', 'GLU_A0002_']
    assert result_df_no_fixed.loc['status', 'ASP_A0001_'] == 'free'

    fixed_df = pd.DataFrame({
        'Residue': ['HIS_A0005_', 'ARG_C0020_'],
        'Charge': [0.5, 1.0]
    })
    result_df_no_free = msa.combine_all_free_fixed_residues(None, fixed_df)
    assert list(result_df_no_free.columns) == ['HIS_A0005_', 'ARG_C0020_']
    assert result_df_no_free.loc['status', 'HIS_A0005_'] == 'fixed'

def test_combine_all_free_fixed_residues_both_empty():
    result_df = msa.combine_all_free_fixed_residues(None, None)
    assert result_df is None # Or handle as empty df depending on desired behavior

def test_cluster_corr_matrix():
    # Simple correlation matrix where clustering is obvious
    corr_data = {
        'A': [1.0, 0.8, 0.1, 0.2],
        'B': [0.8, 1.0, 0.3, 0.0],
        'C': [0.1, 0.3, 1.0, 0.9],
        'D': [0.2, 0.0, 0.9, 1.0]
    }
    corr_df = pd.DataFrame(corr_data, index=['A', 'B', 'C', 'D'])
    # Expect A, B to cluster and C, D to cluster
    expected_order = ['A', 'B', 'C', 'D'] # or ['C', 'D', 'A', 'B']

    clustered_df = msa.cluster_corr_matrix(corr_df, n_clusters=2)

    assert list(clustered_df.columns) == expected_order or list(clustered_df.columns) == expected_order[::-1]
    assert list(clustered_df.index) == list(clustered_df.columns)
    # Check if values are preserved
    pd.testing.assert_frame_equal(clustered_df, corr_df.loc[clustered_df.index, clustered_df.columns])

def test_cluster_corr_matrix_min_clusters():
     corr_data = { 'A': [1.0, 0.1], 'B': [0.1, 1.0] }
     corr_df = pd.DataFrame(corr_data, index=['A', 'B'])
     # Should still run even if n_clusters < 3 is passed, defaults to 3 internally
     clustered_df = msa.cluster_corr_matrix(corr_df, n_clusters=1)
     assert list(clustered_df.columns) == ['A', 'B'] # Order shouldn't change much here


# --- Tests for CMSWC_Pipeline ---
# These require mocking heavily. We'll test the setup and flow logic.

@pytest.fixture
def default_main_params():
    # Mimic loading default params for pH 7, Eh 0
    return prm.params_main(ph="7", eh="0")

@pytest.fixture
def default_histo_params():
    return prm.params_histograms()

@pytest.fixture
@patch('mcce4.cms_analysis_wc.parameters.get_mcce_input_files')
@patch('pathlib.Path.exists')
@patch('pathlib.Path.mkdir')
@patch('pathlib.Path.resolve')
def pipeline_instance(mock_resolve, mock_mkdir, mock_exists, mock_get_inputs, default_main_params, default_histo_params):
    # Mock file system and param loading for basic instantiation
    mock_resolve.return_value = Path('/fake/mcce/dir')
    mock_exists.return_value = True # Assume paths exist initially
    mock_get_inputs.return_value = (Path('/fake/mcce/dir/head3.lst'), Path('/fake/mcce/dir/ms_out/pH7eH0ms.txt'))

    # Modify params slightly for testing
    main_p = default_main_params.copy()
    main_p['mcce_dir'] = '/fake/mcce/dir'
    main_p['output_dir'] = 'test_output' # Relative path
    main_p['correl_resids'] = ['ASP_A0001_', 'GLU_A0002_', 'LYS_B0010_']

    pipeline = msa.CMSWC_Pipeline(main_p, default_histo_params)
    return pipeline, main_p, default_histo_params, mock_get_inputs, mock_mkdir

class TestCMSWCPipeline:

    def test_init_and_setup(self, pipeline_instance):
        pipeline, main_p, _, mock_get_inputs, mock_mkdir = pipeline_instance

        assert pipeline.mcce_dir == Path('/fake/mcce/dir')
        assert pipeline.output_dir == Path('/fake/mcce/dir/test_output')
        assert pipeline.ph == "7"
        assert pipeline.eh == "0"
        assert pipeline.h3_fp == Path('/fake/mcce/dir/head3.lst')
        assert pipeline.msout_fp == Path('/fake/mcce/dir/ms_out/pH7eH0ms.txt')
        assert pipeline.residue_kinds == IONIZABLES # Default
        assert pipeline.correl_resids == ['ASP_A0001_', 'GLU_A0002_', 'LYS_B0010_']
        assert not pipeline.show_fig

        mock_get_inputs.assert_called_once_with(Path('/fake/mcce/dir'), "7", "0")
        # Check if output dir creation was attempted (if it didn't "exist")
        # Need to configure mock_exists for the output dir specifically
        # For now, assume mkdir was called if output dir is relative/doesn't exist
        # mock_mkdir.assert_called_once() # This depends on mock_exists config

    @patch('mcce4.cms_analysis_wc.analysis.MSout_np')
    @patch('mcce4.cms_analysis_wc.parameters.check_res_list')
    def test_load_data(self, mock_check_res, mock_msout_np, pipeline_instance):
        pipeline, _, _, _, _ = pipeline_instance
        mock_mc_instance = MagicMock()
        mock_msout_np.return_value = mock_mc_instance
        # Simulate check_res_list returning a valid list
        valid_resids = ['ASP_A0001_', 'GLU_A0002_']
        mock_check_res.return_value = valid_resids

        pipeline.load_data()

        mock_msout_np.assert_called_once_with(
            pipeline.h3_fp, pipeline.msout_fp, mc_load="crg", res_kinds=pipeline.residue_kinds
        )
        mock_mc_instance.get_uniq_ms.assert_called_once()
        mock_check_res.assert_called_once_with(
            pipeline.correl_resids, # Original list from params
            res_lst=pipeline.residue_kinds,
            conf_info=mock_mc_instance.conf_info
        )
        assert pipeline.mc == mock_mc_instance
        assert pipeline.correl_resids == valid_resids # Updated list

    @patch('mcce4.cms_analysis_wc.analysis.MSout_np')
    @patch('mcce4.cms_analysis_wc.parameters.check_res_list')
    def test_load_data_invalid_correl(self, mock_check_res, mock_msout_np, pipeline_instance):
        pipeline, _, _, _, _ = pipeline_instance
        mock_mc_instance = MagicMock()
        mock_msout_np.return_value = mock_mc_instance
        # Simulate check_res_list returning None or list < 2
        mock_check_res.return_value = ['ASP_A0001_'] # Too short

        pipeline.load_data()
        assert pipeline.correl_resids is None # Should be set to None

        mock_check_res.return_value = None # Empty
        pipeline.load_data()
        assert pipeline.correl_resids is None # Should be set to None


    @patch('mcce4.cms_analysis_wc.analysis.combine_all_free_fixed_residues')
    @patch('pandas.DataFrame.to_csv')
    def test_process_residue_charges(self, mock_to_csv, mock_combine, pipeline_instance):
        pipeline, _, _, _, _ = pipeline_instance
        # Mock the MSout_np instance and its methods
        pipeline.mc = MagicMock()
        mock_fixed_df = pd.DataFrame({'Residue': ['HIS_A0005_'], 'Charge': [0.5]})
        mock_free_df = pd.DataFrame({'Residue': ['ASP_A0001_'], 'AverageCharge': [-0.8]})
        mock_combined_df = pd.DataFrame({'status': ['free', 'fixed']}) # Dummy combined

        pipeline.mc.get_fixed_residues_df.return_value = mock_fixed_df
        pipeline.mc.get_free_res_aver_crg_df.return_value = mock_free_df
        pipeline.mc.get_fixed_res_of_interest_df.return_value = mock_fixed_df # Assume same for simplicity
        mock_combine.return_value = mock_combined_df

        pipeline.process_residue_charges()

        pipeline.mc.get_fixed_residues_df.assert_called_once()
        pipeline.mc.get_free_res_aver_crg_df.assert_called_once()
        mock_combine.assert_called_once_with(mock_free_df, mock_fixed_df)
        pipeline.mc.get_fixed_res_of_interest_df.assert_called_once()

        assert pipeline.fixed_resoi_crg_df is mock_fixed_df

        # Check CSV saving calls
        expected_calls = [
            call(pipeline.output_dir / pipeline.main_defaults["all_res_crg_csv"]), # Combined df
            call(pipeline.output_dir / pipeline.main_defaults["fixed_res_of_interest_csv"], index=False) # Fixed resoi df
        ]
        mock_to_csv.assert_has_calls(expected_calls, any_order=True)


    @patch('mcce4.cms_analysis_wc.analysis.add_fixed_resoi_crg_to_topdf')
    @patch('pandas.DataFrame.to_csv')
    def test_analyze_top_states(self, mock_to_csv, mock_add_fixed, pipeline_instance):
        pipeline, _, _, _, _ = pipeline_instance
        pipeline.mc = MagicMock()
        pipeline.fixed_resoi_crg_df = pd.DataFrame({'Residue': ['HIS_A0005_'], 'Charge': [0.5]}) # From previous step

        mock_top_cms_list = [[[-1, 0, 1], 10, -10.0]] # Dummy data structure
        mock_top_df_raw = pd.DataFrame({'ASP': [-1], 'GLU': [0], 'LYS': [1], 'Count': [10], 'Energy': [-10.0]})
        mock_top_df_with_fixed = pd.DataFrame({'ASP': [-1], 'HIS': [0.5], 'Count': [10]}) # Dummy combined

        pipeline.mc.get_topN_data.return_value = (mock_top_cms_list, None) # Ignore second return val
        pipeline.mc.top_cms_df.return_value = mock_top_df_raw
        mock_add_fixed.return_value = mock_top_df_with_fixed

        pipeline.analyze_top_states()

        n_top = int(pipeline.main_defaults["n_top"])
        min_occ = float(pipeline.main_defaults["min_occ"])
        pipeline.mc.get_topN_data.assert_called_once_with(N=n_top, min_occ=min_occ)
        pipeline.mc.top_cms_df.assert_called_once_with(mock_top_cms_list, cms_wc_format=True)
        mock_add_fixed.assert_called_once_with(mock_top_df_raw, pipeline.fixed_resoi_crg_df, cms_wc_format=True)

        assert pipeline.top_cms == mock_top_cms_list
        assert pipeline.top_df is mock_top_df_raw # Before adding fixed

        # Check CSV saving
        mock_to_csv.assert_called_once_with(
             pipeline.output_dir / pipeline.main_defaults["all_crg_count_resoi_csv"]
        )

    def test_analyze_top_states_no_states(self, pipeline_instance):
        pipeline, _, _, _, _ = pipeline_instance
        pipeline.mc = MagicMock()
        pipeline.mc.get_topN_data.return_value = ([], None) # No top states found

        pipeline.analyze_top_states()

        assert pipeline.top_cms == []
        assert pipeline.top_df is None # Should be None if no states

    @patch('mcce4.cms_analysis_wc.analysis.choose_res_data')
    @patch('mcce4.cms_analysis_wc.analysis.rename_reorder_df_cols')
    @patch('mcce4.cms_analysis_wc.analysis.WeightedCorr')
    @patch('mcce4.cms_analysis_wc.analysis.cluster_corr_matrix')
    @patch('mcce4.cms_analysis_wc.analysis.corr_heatmap')
    @patch('pandas.DataFrame.to_csv')
    def test_perform_correlation(self, mock_to_csv, mock_heatmap, mock_cluster, mock_wcorr_class, mock_rename, mock_choose, pipeline_instance):
        pipeline, _, _, _, _ = pipeline_instance
        # Setup pipeline state after analyze_top_states
        pipeline.correl_resids = ['ASP_A0001_', 'GLU_A0002_', 'LYS_B0010_', 'HIS_A0005_', 'ARG_C0020_', 'TYR_D0001_', 'CYS_E0001_'] # > 6 residues
        pipeline.top_df = pd.DataFrame({'ASP_A0001_': [-1], 'GLU_A0002_': [0], 'LYS_B0010_': [1], 'Count': [10]}) # Dummy

        mock_chosen_df = pd.DataFrame({'ASP_A0001_': [-1], 'Count': [10]})
        mock_renamed_df = pd.DataFrame({'AD1': [-1], 'Count': [10]})
        mock_corr_instance = MagicMock()
        mock_corr_matrix = pd.DataFrame({'AD1': [1.0, 0.5], 'AE2': [0.5, 1.0]}, index=['AD1', 'AE2']) # Dummy 7x7
        mock_corr_matrix = pd.DataFrame(np.random.rand(7,7), columns=pipeline.correl_resids, index=pipeline.correl_resids)
        mock_clustered_matrix = mock_corr_matrix.iloc[[1,0,2,3,4,5,6], [1,0,2,3,4,5,6]] # Dummy reorder

        mock_choose.return_value = mock_chosen_df
        mock_rename.return_value = mock_renamed_df
        mock_wcorr_class.return_value = mock_corr_instance # Instantiation returns mock
        mock_corr_instance.return_value = mock_corr_matrix # Calling instance returns matrix
        mock_cluster.return_value = mock_clustered_matrix

        pipeline.perform_correlation()

        mock_choose.assert_called_once_with(pipeline.top_df, pipeline.correl_resids)
        mock_to_csv.assert_called_once_with(
            pipeline.output_dir / pipeline.main_defaults["res_of_interest_data_csv"]
        )
        mock_rename.assert_called_once_with(mock_chosen_df)
        mock_wcorr_class.assert_called_once_with(
            df=mock_renamed_df, wcol="Count", cutoff=float(pipeline.main_defaults["corr_cutoff"])
        )
        mock_corr_instance.assert_called_once_with(method=pipeline.main_defaults["corr_method"])
        mock_cluster.assert_called_once_with(mock_corr_matrix, n_clusters=int(pipeline.main_defaults["n_clusters"]))
        mock_heatmap.assert_called_once_with(
            mock_clustered_matrix, # Should use clustered result
            out_dir=pipeline.output_dir,
            save_name=pipeline.main_defaults["corr_heatmap.save_name"],
            show=pipeline.show_fig,
            fig_size=eval(pipeline.main_defaults["corr_heatmap.fig_size"])
        )

    def test_perform_correlation_skip_no_resids(self, pipeline_instance):
         pipeline, _, _, _, _ = pipeline_instance
         pipeline.correl_resids = None
         pipeline.top_df = pd.DataFrame({'Count': [1]}) # Dummy
         # Use MagicMock to track calls easily
         msa.choose_res_data = MagicMock()
         pipeline.perform_correlation()
         msa.choose_res_data.assert_not_called()

    def test_perform_correlation_skip_no_top_df(self, pipeline_instance):
         pipeline, _, _, _, _ = pipeline_instance
         pipeline.correl_resids = ['Res1']
         pipeline.top_df = None
         msa.choose_res_data = MagicMock()
         pipeline.perform_correlation()
         msa.choose_res_data.assert_not_called()

    @patch('mcce4.cms_analysis_wc.analysis.energy_distribution')
    @patch('mcce4.cms_analysis_wc.analysis.crgms_energy_histogram')
    def test_generate_energy_plots(self, mock_crg_hist, mock_energy_dist, pipeline_instance):
        pipeline, _, histo_p, _, _ = pipeline_instance
        pipeline.mc = MagicMock()
        pipeline.mc.all_cms = np.array([[0, 0, 10, -10.0]]) # Dummy cms data
        pipeline.mc.all_ms = None # Assume loaded with crg only
        pipeline.mc.get_cms_energy_stats.return_value = (-15.0, -10.0, -5.0) # Emin, Eaver, Emax
        pipeline.mc.background_crg = 1 # Dummy background charge
        # Mock filtering to return some data for each bound type
        pipeline.mc.filter_cms_E_within_bounds.side_effect = lambda data, bounds: data if bounds[0] is not None or bounds[1] is not None else data # Simple mock

        pipeline.generate_energy_plots()

        # Check energy_distribution call
        mock_energy_dist.assert_called_once()
        call_args, call_kwargs = mock_energy_dist.call_args
        np.testing.assert_array_equal(call_args[0], pipeline.mc.all_cms)
        assert call_kwargs['kind'] == 'cms'
        assert call_kwargs['save_name'] == pipeline.output_dir / pipeline.main_defaults["energy_histogram.save_name"]

        # Check crgms_energy_histogram calls for each entry in histo_params
        assert mock_crg_hist.call_count == len(histo_p)

        # Example check for one histogram call (e.g., Emin bounds)
        expected_emin_bounds = (-15.0, -15.0 + 1.36)
        found_emin_call = False
        for call_args_list in mock_crg_hist.call_args_list:
             args, kwargs = call_args_list
             # Check if the filter was called with the expected bounds for this plot
             # This requires inspecting the filter mock calls or checking plot title/savename
             if kwargs['save_name'] == pipeline.output_dir / histo_p['charge_histogram1']['save_name']:
                 # Check that the filter was called appropriately before this plot call
                 # (Requires more intricate mocking of filter_cms_E_within_bounds)
                 # For now, just check the plot call itself
                 np.testing.assert_array_equal(args[0], pipeline.mc.all_cms) # Assuming filter returns original data
                 assert args[1] == pipeline.mc.background_crg
                 assert args[2] == histo_p['charge_histogram1']['title']
                 found_emin_call = True
                 break
        assert found_emin_call


    @patch.object(msa.CMSWC_Pipeline, 'load_data')
    @patch.object(msa.CMSWC_Pipeline, 'process_residue_charges')
    @patch.object(msa.CMSWC_Pipeline, 'generate_energy_plots')
    @patch.object(msa.CMSWC_Pipeline, 'analyze_top_states')
    @patch.object(msa.CMSWC_Pipeline, 'perform_correlation')
    def test_run_method_calls(self, mock_corr, mock_analyze, mock_plots, mock_process, mock_load, pipeline_instance):
        pipeline, _, _, _, _ = pipeline_instance

        pipeline.run()

        # Assert that each step was called once in order
        mock_load.assert_called_once()
        mock_process.assert_called_once()
        mock_plots.assert_called_once()
        mock_analyze.assert_called_once()
        mock_corr.assert_called_once()

        # Check call order if necessary (requires Mock's call_args_list or similar)
        # manager = Mock()
        # manager.attach_mock(mock_load, 'load')
        # ... attach others
        # expected_calls = [call.load(), call.process(), ...]
        # assert manager.mock_calls == expected_calls
