Project Goal: Build a model capable of accurately predicting residential property prices using the following features: Location, TotalSqft, Size, and number of Bathrooms. 

PHASE 1: Modeling with Outliers Retained
- Data was cleaned for missing values and categorical variables were encoded.
- Hyperparameter tuning was performed using GridSearchCV.
- Despite tuning, the performance of all three models plateaued.

Results from PHASE 1:
- Linear Regression (LR) : Train 80% of data score: 0.609 || Test 20% of data score: 0.555
- Support Vector Regression (SVR) : Train 80% of data score 0.500 || Test 20% of data score: 0.537
- Random Forest Regressor (RFR): Train 80% of data score: 0.952 || Test 20% of data score: 0.683

PHASE 1 Discussion:
- Based on these results, the Random Forest Regressor (RFR) showed the most promising performance. The next objective was therefore to prevent overfitting. To achieve this, outliers were removed to reduce noise and improve the model’s ability to generalize.

PHASE 2: Outlier Removal + Parameter Tunings
- Extreme values were removed using quantile-based filtering (1st–99th percentile).
- Models were retrained and tuned again.
- Outlier filtering from all numeric columns eliminated 427 rows, representing 3.2% of the data, to minimize noise and enhance the model’s ability to generalize.

Results from PHASE 2:
- Linear Regression (LR) : Train 80% of data score: 0.720 || Test 20% of data score: 0.633
- Support Vector Regression (SVR) : Train 80% of data score: 0.664 || Test 20% of data score: 0.611
- Random Forest Regressor (RFR): Train 80% of data score: 0.801 || Test 20% of data score: 0.623

PHASE 2 Discussion:
- The primary objective of Phase 2 was to reduce overfitting in the Random Forest Regressor (RFR). However, despite outlier removal and parameter tunings, RFR continued to exhibit clear overfitting (train-test gap of 0.178). Interestingly, the parameter tunings and outlier treatment significantly improved the performance of both Linear Regression (LR) and Support Vector Regression (SVR). Although both Linear Regression and Support Vector Regression showed mild overfitting, SVR demonstrated stronger generalization based on the train–test performance gap. Linear Regression had a train–test gap of 0.087, while SVR showed a smaller gap of 0.053, indicating more stable and reliable performance on unseen data. As a result, SVR emerged as the more suitable model for this dataset.

Key Insight
- SVR emerged as the most reliable model, showing the best balance between training and test performance after outlier removal and parameter tuning. While Random Forest achieved high training scores, it continued to overfit, and Linear Regression improved but still lagged slightly in generalization. 
- Outlier removal helped boost model stability and accuracy, but this step should only be performed with company's approval, as outliers may represent legitimate market variations.
- Overall, combining careful data cleaning, outlier management, and hyperparameter tuning led to a more robust and generalizable predictive model.
- There may be additional, more effective strategies to further enhance model performance that are yet to be explored. For example, creating new features or transforming existing ones can give the model more signal and improve performance.
