import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def make_clean_df():
    return pd.DataFrame({
        'Invoice':     ['536365', '536366', '536367'],
        'StockCode':   ['85123A', '71053',  '84406B'],
        'Quantity':    [6.0, 8.0, 6.0],
        'InvoiceDate': pd.to_datetime(['2010-12-01', '2010-12-01', '2010-12-02']),
        'Price':       [2.55, 3.39, 2.75],
        'Customer ID': [17850.0, 17851.0, 17852.0],
        'Country':     ['United Kingdom', 'United Kingdom', 'Germany'],
    })

def test_no_missing_values():
    df = make_clean_df()
    assert df.isnull().sum().sum() == 0

def test_no_duplicates():
    df = make_clean_df()
    assert df.duplicated().sum() == 0

def test_no_negative_quantity():
    df = make_clean_df()
    assert (df['Quantity'] < 0).sum() == 0

def test_all_prices_positive():
    df = make_clean_df()
    assert (df['Price'] > 0).all()

def test_detects_missing_customer_id():
    df = make_clean_df()
    df.loc[0, 'Customer ID'] = np.nan
    assert df['Customer ID'].isnull().sum() == 1

def test_detects_negative_quantity():
    df = make_clean_df()
    df.loc[1, 'Quantity'] = -3.0
    assert (df['Quantity'] < 0).sum() == 1
