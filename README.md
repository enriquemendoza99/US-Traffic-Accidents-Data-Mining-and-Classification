# US Traffic Accidents — Data Mining and Classification

A data mining pipeline applying clustering and classification techniques
to the US Accidents dataset (7.7 million records) to explore the
relationship between weather/time conditions and accident severity.

## Project Structure
data/

US_Accidents_March23.csv    — Dataset (not included, download separately)

accidents_project.py        — Main pipeline: preprocessing, clustering, classification

data_analysis.py            — Quick CSV inspection / sanity check script

requirements.txt            — Python dependencies

severity_distribution.png   — Output: severity class distribution

kmeans_cluster_sizes.png    — Output: K-Means cluster sizes

dbscan_cluster_sizes.png    — Output: DBSCAN cluster sizes

## How to Run

1. Download the [US Accidents dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
   and place `US_Accidents_March23.csv` inside a `data/` folder.

2. Install dependencies:
pip install -r requirements.txt

3. Quick sanity check (confirms the CSV loads correctly):
python data_analysis.py

4. Run the full pipeline:
python accidents_project.py

## Configuring Sample Sizes

At the top of `accidents_project.py`, row limits can be adjusted
independently per stage to balance speed against using the full dataset:
```python
MAX_ROWS_FOR_PREPROCESS = None        # None = full dataset (~7.7M rows)
MAX_ROWS_FOR_CLUSTERING = 1000        # KMeans sample size
MAX_ROWS_FOR_DBSCAN = 250000          # DBSCAN sample size
MAX_ROWS_FOR_CLASSIFICATION = None    # None = full dataset
```
Lower these for fast iteration during development; the full run on all
7.7 million rows takes roughly 70-90 minutes depending on hardware,
with the MLP training step being the most time-consuming part.

## Pipeline Stages

1. **Preprocessing:** Extracts hour/month from timestamps, encodes
   twilight indicators, imputes missing weather values, casts
   points-of-interest booleans to integers.
2. **K-Means Clustering (k=5):** Clusters on weather/time features,
   evaluated with silhouette score, Davies-Bouldin index, and mutual
   information with severity.
3. **DBSCAN Clustering:** Density-based clustering on a 250,000-row
   sample, evaluated the same way (excluding noise points).
4. **Decision Tree Classification:** Class-weighted training to handle
   severe class imbalance in the severity labels.
5. **MLP Classification:** Trained on a class-balanced (downsampled)
   training set for direct comparison against the Decision Tree.

## Key Findings

Both clustering algorithms showed weak association with severity
(mutual information under 0.01 bits for both), suggesting weather and
time conditions alone do not strongly determine accident severity.

The Decision Tree (class-weighted) and MLP (class-balanced) classifiers
illustrate a precision-recall tradeoff under class imbalance: the
Decision Tree achieves lower overall accuracy (29%) but much higher
recall on the rare, most-severe class (68%), while the MLP achieves
higher accuracy (66%) by leaning toward the majority class, with a
similar macro F1 score to the Decision Tree despite the large accuracy
gap. This highlights why accuracy alone is misleading on imbalanced data.
