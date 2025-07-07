import pandas as pd

def load_dataset(file_path: str,column_names: list ) -> pd.DataFrame:

    df = pd.read_csv(file_path,names=column_names)
    return df



def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if not missing.empty:
        print("Missing Values Found:\n")
        print(missing)
    else:
        print("No missing values found!!")

    return missing


def drop_features(df: pd.DataFrame , cols_to_drop : list = None) -> pd.DataFrame:

    if cols_to_drop is None:
        cols_to_drop = ['num_outbound_cmds']

        if 'difficulty' in df.columns:
            cols_to_drop.append('difficulty')

    df = df.drop(columns=cols_to_drop , errors = 'ignore')
    
    
    print(f"Dropped Columns {cols_to_drop}")
    return df


def drop_constant_cols(df : pd.DataFrame) -> pd.DataFrame:

    const_cols = [col for col in df.columns if df[col].nunique() == 1]

    if const_cols:
        print("Dropping constant columns:" ,const_cols)
        return df.drop(columns=const_cols)
    else:
        print("No constant columns found")
        return df

'''

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:

    cat_cols = df.select_dtypes(include = ['category' , 'object']).columns.tolist()

    if 'label' in cat_cols:
        cat_cols.remove('label')

    print(f'categorical columns to encode: {cat_cols}')

    encoded_df = pd.get_dummies(df, columns=cat_cols , drop_first=True)
'''








