import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from model.HiddenMarkovModel import HiddenMarkovModel

np.random.seed(9)

INPUT_FILE = 'data/pamap2_master.pkl'
MODEL_FILENAME = 'supervised_hmm_pamap2.pkl'

TRAIN_SUBJECTS = [101, 102, 103, 104, 106, 107]
TEST_SUBJECTS  = [105, 108]

DUMP_LABELS_LIST = [12, 13, 24, 16, 17, 0] 

ACTIVITY_MAP = {
    1:  "Lying",
    2:  "Sitting",
    3:  "Standing",
    4:  "Walking",
    5:  "Running",
    6:  "Cycling",
    7:  "Nordic walking",
    9:  "Watching TV",
    10: "Computer work",
    11: "Car driving",
    12: "Ascending stairs",
    13: "Descending stairs",
    16: "Vacuum cleaning",
    17: "Ironing",
    18: "Folding laundry",
    19: "House cleaning",
    20: "Playing soccer",
    24: "Rope jumping",
    0:  "Transient/Other"
}

N_MIXTURES = 5        
DOWNSAMPLE_FACTOR = 4 

def preprocess_data(df, subjects, scaler=None, is_train=True):
    # filter subjects
    mask = df['subject_id'].isin(subjects)
    subset = df[mask].copy()
    
    # filter activities (remove unwanted ones)
    subset = subset[~subset['activity_id'].isin(DUMP_LABELS_LIST)].copy()
    
    # downsample
    subset = subset.iloc[::DOWNSAMPLE_FACTOR, :].reset_index(drop=True)

    meta_cols = ['timestamp', 'activity_id', 'subject_id']
    feature_cols = [c for c in subset.columns if c not in meta_cols]
    
    X = subset[feature_cols].values
    y = subset['activity_id'].values.astype(int)

    if is_train:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        if scaler is None:
            raise ValueError()
        X = scaler.transform(X)
        
    return X, y, scaler

def evaluate_performance(y_true, y_pred, labels_list):

    acc = accuracy_score(y_true, y_pred)
    
    target_names = [ACTIVITY_MAP.get(lbl, str(lbl)) for lbl in labels_list]

    print(f"\n-> Model Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_true, y_pred, labels=labels_list, target_names=target_names))
    
    cm = confusion_matrix(y_true, y_pred, labels=labels_list)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, cmap=sns.cubehelix_palette(as_cmap=True), linewidths = .3, xticklabels = target_names, yticklabels = target_names)
    plt.title(f"Confusion Matrix (Supervised) - Accuracy: {acc:.2%}")
    plt.xlabel("Predicted Activity", fontsize=14)
    plt.ylabel("True Activity", fontsize=14)
    plt.tight_layout()
    plt.savefig(fname = './plots/confussion_matrix_supervised.png', dpi = 300)
    plt.show()
    

def main():
    full_df = pd.read_pickle(INPUT_FILE)

    X_train, y_train, scaler = preprocess_data(full_df, TRAIN_SUBJECTS, is_train=True)
    
    unique_activities = sorted(np.unique(y_train))

    N_STATES = len(unique_activities)

    act_to_state = {act: idx for idx, act in enumerate(unique_activities)}
    state_to_act = {idx: act for idx, act in enumerate(unique_activities)}
    
    y_train_mapped = np.array([act_to_state[act] for act in y_train])

    ########################################## Training ##########################################
    
    T_train, D = X_train.shape
    print(f"Training Data: {T_train} samples, {D} dimensions.")

    hmm = HiddenMarkovModel(
        num_states=N_STATES, 
        num_components_mixtures=[N_MIXTURES] * N_STATES, 
        data_dim=D
    )
    
    hmm.train(X_train, labels=y_train_mapped, method='supervised', n_iter=20)
    
    hmm.save(MODEL_FILENAME)
    
    ########################################## Testing ##########################################

    X_test, y_test, _ = preprocess_data(full_df, TEST_SUBJECTS, scaler=scaler, is_train=False)
    
    print(f"Prediction on {len(X_test)} samples")
    pred_states_indices = hmm.predict_viterbi(X_test)
    
    pred_labels = np.array([state_to_act.get(idx, 0) for idx in pred_states_indices])
    
    evaluate_performance(y_test, pred_labels, unique_activities)
    
    plt.figure(figsize=(16, 9))
    
    plt.plot(pred_labels, color='lightgray', label='Predicted', alpha=0.8, linewidth=1.5)
    plt.plot(y_test, color='r', label='Actual', linewidth=2, alpha=0.8)
    
    present_ids = sorted(list(set(unique_activities) | set(y_test)))
    present_names = [ACTIVITY_MAP.get(i, str(i)) for i in present_ids]
    
    plt.yticks(present_ids, present_names, fontsize=10)
    
    plt.title("Supervised GMM-HMM Prediction")
    plt.legend()
    plt.ylabel("Activity")
    plt.xlabel("Samples")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(fname = './plots/prediction_supervised.png', dpi = 300)
    plt.show()

if __name__ == "__main__":
    main()
