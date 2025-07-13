import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import joblib
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
ARTIFACTS_DIR = os.path.join(BASE_DIR,'artifacts')
os.makedirs(ARTIFACTS_DIR, exist_ok=True)



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
    

def reduce_rare_services(df: pd.DataFrame , threshold : int = 1000) -> pd.DataFrame:
    service_counts = df['service'].value_counts()
    rare_services = service_counts[service_counts < threshold].index.tolist()

    print(f"Reducing {len(rare_services)} rare services to 'rare'")

    df['service'] = df['service'].apply(lambda x: 'rare' if x in rare_services else x)

    return df    


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:

    cat_cols = df.select_dtypes(include = ['category' , 'object']).columns.tolist()

    if 'label' in cat_cols:
        cat_cols.remove('label')

    print(f'categorical columns to encode: {cat_cols}')

    encoded_df = pd.get_dummies(df, columns=cat_cols )

    return encoded_df


def scale_features(df: pd.DataFrame , save_path : str = None) -> pd.DataFrame:
    scaler = MinMaxScaler()

    features = df.drop(columns=['label'])
    labels = df['label'].reset_index(drop=True)

    
    num_cols = features.select_dtypes(include=['int64', 'float64']).columns

   
    scaled_array = scaler.fit_transform(features[num_cols])

    
    scaled_df = features.copy()
    scaled_df[num_cols] = scaled_array

    
    scaled_df['label'] = labels

    if save_path is None:
        save_path = os.path.join(ARTIFACTS_DIR, 'scaler.pkl')

    

    joblib.dump(scaler , save_path)
    print(f"scaler saved to {save_path}")

    print(f"Scaled numerical columns: {list(num_cols)}")
    return scaled_df


def encode_labels(df: pd.DataFrame , save_path: str = None) -> pd.DataFrame:
    encoder = LabelEncoder()
    df['label'] = encoder.fit_transform(df['label'])

    if save_path is None:
        save_path = os.path.join(ARTIFACTS_DIR, 'label_encoder.pkl')

    
    joblib.dump(encoder , save_path)

    print("Labels encoded and encoder saved to ",save_path)
    return df





















