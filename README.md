<h1 align="center">Human Activity Recognition with HMM-GMM</h1>

<p align="center">
  <strong>Authors:</strong><br>
  <a href="https://github.com/calcuttarain"><b>Tudor Pistol</b></a> & 
  <a href="https://github.com/teofilq"><b>Teofil Simiraș</b></a>
</p>

---

This project implements a Hidden Markov Model (HMM) with Gaussian Mixture Model (GMM) emissions to classify human physical activities using the PAMAP2 dataset. The entire pipeline, from mathematical modeling (Baum-Welch, Viterbi) to data processing, is implemented from scratch.

## Dataset Overview: PAMAP2

The project utilizes the **PAMAP2 Physical Activity Monitoring** dataset, available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring).

* **Hardware:** Three wireless Inertial Measurement Units (IMUs) placed on the dominant wrist, chest, and ankle, plus a heart rate monitor.
* **Data Spec:** IMUs record acceleration, gyroscope, and magnetometer data at 100Hz, while heart rate is recorded at ~9Hz.
* **Activities:** The dataset covers everyday actions (walking, lying, standing) and sport activities (running, cycling).

## Methodology

### 1. Preprocessing
Raw data files are merged and synchronized. Missing values caused by wireless data drops or frequency mismatches are handled via interpolation. Transient activities (labeled as ID 0) are filtered out to ensure analysis quality.

### 2. Model Architecture
The core system is a custom implementation of an HMM-GMM framework. Essentially, each activity is interpreted as a state in the Hidden Markov Model and assumed to be a Mixture of Gaussians:

* **Hidden States:** Represent specific physical activities.
* **Observations:** Continuous sensor data modeled by Gaussian Mixtures.
* **Algorithm:**
    * **Forward-Backward:** Used for state probability inference.
    * **Baum-Welch:** Implemented for unsupervised parameter estimation (EM algorithm).
    * **Viterbi:** Used to decode the most likely sequence of activities from sensor streams.

### 3. Learning Approaches
* **Supervised:** Initializes transition matrices and emission probabilities using ground-truth labels.
* **Unsupervised:** Learns activity patterns directly from unlabelled sensor data by optimizing log-likelihood.

## Bibliography

1. **Bilmes, J. A.** (1998). *A Gentle Tutorial of the EM Algorithm and its Application to Parameter Estimation for Gaussian Mixture and Hidden Markov Models* (TR-97-021). International Computer Science Institute (ICSI) & U.C. Berkeley.
2. **Rabiner, L. R.** (1989). *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition*. Proceedings of the IEEE, 77(2), 257-286.