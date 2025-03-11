import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import time

# function used to categorize exam scores being tested into either passing or failing
def map_scores(value):
    if 1 <= value <= 2:
        return 0
    elif 3 <= value <= 5:
        return 1
    return value

start = time.time()
print(f"Start time: {start}")

total = 0
for testNumber in range(1000):
    # gets data
    full_data = pd.read_csv('data.csv')

    # seperates data for exams and for grade level
    score_columns = [col for col in full_data.columns if col.endswith('_Exam')]
    grade_level_columns = [col.replace('_Exam', '_Grade_Level') for col in score_columns]

    # lists all students who took exam in most recent year (labeled as grade_level = 2)
    test_masks = {col: (full_data[col.replace('_Exam', '_Grade_Level')] == 2) for col in score_columns}

    # substitutes average for exam as placeholder value for each exam if the exam is listed as 0 (which means the student did not take tha exam)
    for col in score_columns:
        non_test_mask = ~test_masks[col]
        mean_score = full_data.loc[non_test_mask & (full_data[col] != 0), col].mean()
        if pd.notna(mean_score):
            full_data.loc[non_test_mask & (full_data[col] == 0), col] = mean_score.astype(full_data[col].dtype)

    # uses function at beginning to turn exam score into pass/fail value (1 or 0)
    for col in score_columns:
        full_data[col] = full_data[col].apply(map_scores).astype(int)

    # reserves 20% of the data for testing
    total_samples = len(full_data)
    test_indices = np.random.choice(full_data.index, size=int(0.2 * total_samples), replace=False)
    full_data['is_test'] = False
    full_data.loc[test_indices, 'is_test'] = True

    # tracks correct predictions and total predictions to calculate accuracy at the end
    correct_predictions = 0
    total_predictions = 0

    for i, exam_col in enumerate(score_columns):
        grade_level_col = exam_col.replace('_Exam', '_Grade_Level')

        # identifies the most recent exams (grade_level = 2) for students being tested, also seperates students for training set
        test_mask = (full_data[grade_level_col] == 2) & (full_data['is_test'])
        train_mask = ~test_mask

        # prepares training and testing data
        x_train = full_data.loc[train_mask].drop(columns=score_columns + grade_level_columns + ['is_test'])
        y_train = full_data.loc[train_mask, exam_col].astype(int)

        
        x_test = full_data.loc[test_mask].drop(columns=score_columns + grade_level_columns + ['is_test'])
        y_test = full_data.loc[test_mask, exam_col].astype(int)
        
        # disregards any exams that dont have a data point in the test set (since it would be unnecessary to 
        # make accuracy predictions for it if its not used to make any predictions)
        if len(np.unique(y_train)) < 2 or test_mask.sum() == 0:
            continue

        # standardizes values to their corresponding z-score in training set
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        # initialized XG Boost model and provides it training/testing data
        xgb_model = XGBClassifier(eval_metric='logloss')
        xgb_model.fit(x_train, y_train)

        # uses model to make student predictions
        y_pred = xgb_model.predict(x_test)

        # adds number of correct/total predictions for this exam to overall counters
        correct_predictions += sum(y_pred == y_test.values)
        total_predictions += len(y_test)

        #print(f"Accuracy for {exam_col}: {sum(y_pred == y_test.values)} out of {len(y_test)}")

    # finds overall accuracy based on total prediction results
    overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    total += overall_accuracy
    print(f"\n{testNumber} - Overall Prediction Accuracy: {overall_accuracy:.2f}")

end = time.time()
print(f"End time: {end}")
print(f"Elapsed time: {end - start}")
print(f"Total accuracy: {total/1000}")