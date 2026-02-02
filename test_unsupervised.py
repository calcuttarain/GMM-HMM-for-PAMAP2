import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from model.HiddenMarkovModel import HiddenMarkovModel

DATA_FILE = 'data/pamap2_master.pkl'
MODEL_FILE = 'unsupervised_hmm_pamap2.pkl' 

TEST_SUBS = [105, 108]
KEEP_ACTIVITIES = [1, 2, 3, 4, 5, 6]
DOWNSAMPLE_FACTOR = 4 

ACTIVITY_MAP = {
    1: "Lying",
    2: "Sitting",
    3: "Standing",
    4: "Walking",
    5: "Running",
    6: "Cycling"
}

def load_test_data(filepath, subjects):
    df = pd.read_pickle(filepath)
    
    mask_sub = df['subject_id'].isin(subjects)
    
    mask_act = df['activity_id'].isin(KEEP_ACTIVITIES)
    
    subset = df[mask_sub & mask_act].copy()
    
    subset = subset.iloc[::DOWNSAMPLE_FACTOR, :].reset_index(drop=True)

    meta_cols = ['timestamp', 'activity_id', 'subject_id']
    feature_cols = [c for c in subset.columns if c not in meta_cols]
    
    X = subset[feature_cols].values
    y = subset['activity_id'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

def evaluate_classification(pred_states, true_labels):
    df_map = pd.DataFrame({'State': pred_states, 'Label': true_labels})
    
    state_to_activity = {}
    
    all_states = sorted(df_map['State'].unique())
    
    for state in all_states:
        subset = df_map[df_map['State'] == state]
        if len(subset) > 0:
            dominant_label = subset['Label'].mode()[0]
            count = len(subset)
            correct_count = (subset['Label'] == dominant_label).sum()
            purity = correct_count / count
            
            state_to_activity[state] = dominant_label
            
            act_name = ACTIVITY_MAP.get(dominant_label, str(dominant_label))
            print(f" {state:<5} | {act_name:<20} | {purity:.2f}")
        else:
            state_to_activity[state] = 0 

    pred_labels = np.array([state_to_activity.get(s, 0) for s in pred_states])
    
    acc = accuracy_score(true_labels, pred_labels)
    print(f"\n-> Accuracy: {acc*100:.2f}%")
    
    cm = confusion_matrix(true_labels, pred_labels)
    unique_labels = sorted(set(true_labels) | set(pred_labels))
    target_names = [ACTIVITY_MAP.get(lbl, str(lbl)) for lbl in unique_labels]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix (Accuracy: {acc:.2%})")
    plt.xlabel("Predicted Activity")
    plt.ylabel("True Activity")
    plt.tight_layout()
    plt.savefig(fname = './plots/confussion_matrix_unsupervised.png', dpi = 300)
    plt.show()
    
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=target_names, zero_division=0))

def main():
    hmm = HiddenMarkovModel.load(MODEL_FILE)

    X_test, y_test = load_test_data(DATA_FILE, TEST_SUBS)

    pred_states = hmm.predict_viterbi(X_test)
    
    evaluate_classification(pred_states, y_test)

if __name__ == "__main__":
    main()