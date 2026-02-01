import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

from model.HiddenMarkovModel import HiddenMarkovModel

INPUT_FILE = 'data/pamap2_master.pkl'

TRAIN_SUBS = [101, 102, 103, 104, 106, 107]
TEST_SUBS  = [105, 108]
KEEP_ACTIVITIES = [1, 2, 3, 4, 5, 6]

N_STATES = 6
N_MIXTURES = 4
DOWNSAMPLE_FACTOR = 4  

MODEL_FILENAME = 'unsupervised_hmm_pamap2.pkl'

def load_and_split_data(df, subjects, scaler=None, is_train=True):
    mask = df['subject_id'].isin(subjects)
    subset = df[mask].copy()

    subset = subset[subset['activity_id'].isin(KEEP_ACTIVITIES)].copy()

    subset = subset.iloc[::DOWNSAMPLE_FACTOR, :].reset_index(drop=True)

    meta_cols = ['timestamp', 'activity_id', 'subject_id']
    feature_cols = [c for c in subset.columns if c not in meta_cols]
    
    X = subset[feature_cols].values
    y = subset['activity_id'].values 

    if is_train:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError()
        X = scaler.transform(X)
        
    return X, y, scaler

def analyze_results(pred_states, true_labels):
    df_res = pd.DataFrame({'HMM_State': pred_states, 'True_Label': true_labels})
    
    df_res = df_res[df_res['True_Label'] != 0]

    mapping = df_res.groupby(['HMM_State'])['True_Label'].agg([
        ('Dominant_Activity', lambda x: x.mode()[0] if not x.mode().empty else -1),
        ('Count', 'count'),
        ('Confidence', lambda x: (x == x.mode()[0]).sum() / len(x) if not x.mode().empty else 0)
    ]).reset_index()
    
    return mapping

def main():
    full_df = pd.read_pickle(INPUT_FILE)

    X_train, y_train, scaler = load_and_split_data(full_df, TRAIN_SUBS, is_train=True)
    
    T_train, D = X_train.shape
    print(f"Train Data: {T_train} time steps, {D} dimensions")

    hmm = HiddenMarkovModel(
        num_states=N_STATES, 
        num_components_mixtures=[N_MIXTURES] * N_STATES, 
        data_dim=D
    )
    
    for i in range(N_STATES):
        rand_idx = np.random.choice(T_train, N_MIXTURES, replace=False)
        hmm.states[i].means = X_train[rand_idx]

    hmm.train(X_train, method='unsupervised', n_iter=100, tol=1e-4)

    print("\nTraining Complete.")
    print("Transition Matrix Sample:")
    print(tabulate(hmm.A[:5, :5], tablefmt="grid")) 

    hmm.save(MODEL_FILENAME)

    X_test, y_test, _ = load_and_split_data(full_df, TEST_SUBS, scaler=scaler, is_train=False)
    
    print(f"Running Viterbi Decoding on {len(X_test)} samples...")
    pred_states = hmm.predict_viterbi(X_test)

    analyze_results(pred_states, y_test)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    
    ax1.plot(pred_states, label='HMM State', color='blue', marker='o', markersize=2, linestyle='None')
    ax1.set_title("Predicted Hidden States (Unsupervised)")
    ax1.set_ylabel("State ID")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(y_test, label='True Label', color='red', marker='x', markersize=2, linestyle='None')
    ax2.set_title("Ground Truth Activity Labels")
    ax2.set_ylabel("Activity ID")
    ax2.set_xlabel("Time (downsampled)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()