from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import os

def split_data(df: pd.DataFrame , target_col: str='label' , test_size: float=0.2 , random_state: int=42) -> tuple:
    x = df.drop(columns=[target_col])
    y = df[target_col]

    x_train, x_test, y_train, y_test =   train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return x_train, x_test, y_train, y_test


def plot_leaderboard(csv_path='models/leaderboard.csv', save_path='models/leaderboard_plot.png'):
    
    if not os.path.exists(csv_path):
        print(f"[Warning] Leaderboard file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("[Warning] Leaderboard CSV is empty.")
        return

    
    df = df.sort_values("Best_F1_CV", ascending=True)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(df["Model"], df["Best_F1_CV"], color="skyblue")

    plt.xlabel("Best CV F1 Score")
    plt.title("Model Leaderboard - F1 Score Comparison")

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{width:.4f}", va="center")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Info] Leaderboard plot saved to {save_path}")
    plt.close()