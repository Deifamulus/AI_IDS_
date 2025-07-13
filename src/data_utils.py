from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(df: pd.DataFrame , target_col: str='label' , test_size: float=0.2 , random_state: int=42) -> tuple:
    x = df.drop(columns=[target_col])
    y = df[target_col]

    x_train, x_test, y_train, y_test =   train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return x_train, x_test, y_train, y_test

