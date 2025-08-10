This project analyzes prescription data for forecasting future drug usage volumes.

1. Graph and Cluster Construction

 • Built a drug co-occurrence graph based on joint prescriptions.

 • Strong connections were identified using the Jaccard coefficient.

 • Nodes were grouped into clusters (color-coded in the graph). Each cluster represents stable treatment schemes.

2. Forecasting Models

 • Tested different approaches, including Prophet without external regressors and Prophet with clusters as regressors.

 • The best results were achieved by Prophet with clusters: using cluster dynamics as external regressors reduced forecast errors according to MAE, MAPE, and WAPE metrics.

3. Results

 • The cluster-based model captures trends more accurately and better predicts peaks and drops compared to the model without clusters.

 • This approach enables consideration of complex treatment schemes rather than only aggregated totals.

4. Next Steps

 • Refine clustering: explore alternative community detection algorithms and tune Jaccard thresholds.

 • Feature engineering: incorporate additional clinical and temporal features (e.g., patient demographics, seasonal effects).
 • Hybrid models: test gradient boosting or neural networks with cluster dynamics as inputs.
 • Granularity: build separate forecasts per major therapeutic area and aggregate them for improved accuracy.
 • Continuous learning: implement a rolling retraining pipeline to adapt to new data in near real time.
