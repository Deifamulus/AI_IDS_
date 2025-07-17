import os , sys 
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


from preprocessing import (
    load_dataset,
    check_missing_values,
    drop_features,
    drop_constant_cols,
    reduce_rare_services,
    encode_categorical_features,
    scale_features,
    encode_labels
)

data_path = '../data/raw/KDDTrain+.csv'
save_path = '../data/processed/processed_kdd.csv'

fields = pd.read_csv('../data/raw/Field Names.csv' , header=None)
columns = fields[0].tolist()
new_columns = columns + ['label' , 'difficulty']

def main():
    df = load_dataset(data_path, column_names=new_columns)

    check_missing_values(df)
    df = drop_features(df)
    df = drop_constant_cols(df)
    df = reduce_rare_services(df, threshold=1000)
    df = encode_categorical_features(df)
    print("Before scaling:", df.columns.tolist())
    assert 'label' in df.columns, "ERROR: 'label' column missing before scaling!"
    df = scale_features(df)      
    df = encode_labels(df) 

    os.makedirs(os.path.dirname(save_path) , exist_ok=True)
    df.to_csv(save_path , index=False)
    print(f"Processed data stored at {save_path}")

    print("Final Shape:", df.shape)
    print(df.head())

if __name__ == "__main__":
    main()




