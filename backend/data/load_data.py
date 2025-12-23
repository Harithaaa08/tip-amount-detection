import pandas as pd
import seaborn as sns

def load_data():
    """
    load_data loads the tips dataset from seaborn and 
    return  as a pandas DataFrame.
    """
    df = sns.load_dataset("tips")
    return df