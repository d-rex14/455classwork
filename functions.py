# EDA functions
def univariate(df):
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns

  df_results = pd.DataFrame(columns=["Data Type", "Count", "Missing", "Unique", "Mode", "Min", "Q1", "Median",
                                     "Q3", "Max", "Mean", "Std", "Skew", "Kurt"])

  for col in df.columns:
    df_results.loc[col, "Data Type"] = df[col].dtype
    df_results.loc[col, "Count"] = df[col].count()
    df_results.loc[col, "Missing"] = df[col].isna().sum()
    df_results.loc[col, "Unique"] = df[col].nunique()
    df_results.loc[col, "Mode"] = df[col].mode()[0]

    if df[col].dtype in ["int64", "float64"]:
      df_results.loc[col, "Min"] = df[col].min()
      df_results.loc[col, "Q1"] = df[col].quantile(0.25)
      df_results.loc[col, "Median"] = df[col].median()
      df_results.loc[col, "Q3"] = df[col].quantile(0.75)
      df_results.loc[col, "Max"] = df[col].max()
      df_results.loc[col, "Mean"] = df[col].mean()
      df_results.loc[col, "Std"] = df[col].std()
      df_results.loc[col, "Skew"] = df[col].skew()
      df_results.loc[col, "Kurt"] = df[col].kurt()

      # Check if column is NOT boolean 0/1
      unique_vals = set(df[col].dropna().unique())
      is_boolean = unique_vals.issubset({0, 1})
      
      if not is_boolean:
        # Create stacked plot: box plot on top, histogram with KDE underneath
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                        gridspec_kw={'height_ratios': [1, 2], 'hspace': 0.3})
        
        # Box plot on top
        sns.boxplot(data=df, y=col, ax=ax1)
        ax1.set_title(f'Box Plot and Distribution for {col}')
        ax1.set_xlabel('')
        ax1.set_ylabel(col)
        
        # Histogram with KDE overlay underneath
        sns.histplot(data=df, x=col, kde=True, ax=ax2)
        ax2.set_xlabel(col)
        ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    else:
      # Prepare for categorical plots
      plt.figure(figsize=(10, 6))
      ax = sns.countplot(data=df, x=col)
      plt.title(f'Count Plot for {col}')
      plt.xlabel(col)
      plt.ylabel('Count')
      plt.xticks(rotation=45, ha='right')
      
      # Add percentage labels above each bar
      total = len(df[col].dropna())
      for p in ax.patches:
        height = p.get_height()
        percentage = (height / total) * 100
        ax.text(p.get_x() + p.get_width() / 2., height,
                f'{percentage:.1f}%',
                ha='center', va='bottom')
      
      plt.tight_layout()
      plt.show()

  return df_results


def basic_wrangling(df):
  import pandas as pd

  # Drop columns where all values are different
  for col in df.columns:
    if df[col].nunique() == df[col].count() and not pd.api.types.is_numeric_dtype(df[col]):
      df.drop(columns=[col], inplace=True)

  return df

def bin_categories(df, pct=0.05, min_records=1):
  import pandas as pd

  # pct: threshold (e.g. 0.05 = 5%). Categories below this share are "small".
  # min_records: when every group is below pct, avoid binning if each group has at least this many records.
  for col in df.columns:
    if not pd.api.types.is_numeric_dtype(df[col]):
      total = df.shape[0]
      value_counts = df[col].value_counts()
      value_pcts = value_counts / total

      small_mask = value_pcts < pct
      small_cats = value_pcts[small_mask]
      if len(small_cats) == 0:
        continue

      # If every group is below pct, avoid binning entirely as long as each group has at least min_records
      if len(small_cats) == len(value_counts):
        if (value_counts >= min_records).all():
          continue

      # Only bin categories that are BOTH below pct AND have fewer than min_records
      small_cat_names = [
        c for c in small_cats.index
        if value_counts[c] < min_records
      ]
      if len(small_cat_names) == 0:
        continue

      df = df.copy()
      df.loc[df[col].isin(small_cat_names), col] = "Other"

      # If the new 'Other' bin doesn't make up pct of the dataset, drop those rows
      other_pct = (df[col] == "Other").sum() / len(df)
      if other_pct < pct:
        df = df[df[col] != "Other"].copy()

  return df