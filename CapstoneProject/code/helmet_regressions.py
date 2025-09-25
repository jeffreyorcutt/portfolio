import numpy as np
import pandas as pd
import optuna
import time
import datetime
from functools import wraps


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from joblib import dump, load

# Function to time the execution of a function
# This decorator came from https://stackoverflow.com/questions/1482208/how-to-time-a-function-in-python
def time_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        runtime = end_time - start_time
        formatted_runtime = str(datetime.timedelta(seconds=runtime))
        print(f"Function '{func.__name__} 'executed in {formatted_runtime}.")
        return result
    return wrapper

# used ChatGPT to help designing a pipeline in order to standardize the functions
# changed the code quite a bit to fit my needs
# each of the subsequent regressions was based off this model
# used https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html to figure out the
# requirements for the pipeline based on what I was trying to do...

# this function is used to train a logistic regression model with feature selection and hyperparameter tuning

# future work: move the list of predictors and transform keys out to an argument
# so that it can be used in the other models too
# would also like to make this more modular in terms of hardcoded literals too,
# currently have two versions of the models for subsetted data because of this...
# could go down to a maybe 2 functions instead of 6, with the right arguments
@time_function
def logistic_reg_model(df, use_stored_model=False, model_size=1.0):
    model_filename = './data/best_model.joblib'
    
    # Create copies of the input DataFrame.
    df_full = df.copy()
    df = df.copy()
    
    # Define full dataset predictors and target.
    X_full = df_full[['age', 'sex', 'time_category', 'prov', 
                      'drug_impairment_status', 'alcohol_status', 'cellphone_status', 'injp']]
    y_full = df_full['helmet_used']

    if not use_stored_model:
        # Convert 'prov' to categorical and reorder categories.
        df['prov'] = df['prov'].astype('category')
        provinces = list(df['prov'].cat.categories)
        if 'Bangkok' in provinces:
            provinces.remove('Bangkok')
        ordered_provinces = ['Bangkok'] + sorted(provinces)
        df['prov'] = df['prov'].cat.reorder_categories(ordered_provinces, ordered=True)
        
        # If model_size < 1.0, sample a stratified subset based on helmet_used.
        if model_size < 1.0:
            df = df.groupby('helmet_used', group_keys=False).apply(
                lambda x: x.sample(frac=model_size, random_state=42)
            )
            print(f'Sampled {model_size*100:.0f}% of the data stratified by helmet_used for training.')
        
        # Define features and target for training.
        X = df[['age', 'sex', 'time_category', 'prov', 
                'drug_impairment_status', 'alcohol_status', 'cellphone_status', 'injp']]
        y = df['helmet_used']

        # Build a preprocessor for numerical and categorical features.
        categorical_features = ['sex', 'time_category', 'prov', 
                                'drug_impairment_status', 'alcohol_status', 'cellphone_status', 'injp']
        numerical_features = ['age']
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first'), categorical_features)
            ]
        )

        # Build the pipeline with three steps:
        # 1. Preprocessing
        # 2. Automatic feature selection using SelectKBest (ANOVA F-test)
        # 3. Logistic Regression classifier
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('feature_selection', SelectKBest(score_func=f_classif)),
            ('classifier', LogisticRegression(solver='liblinear'))
        ])

        # Define hyperparameter grid.
        param_grid = {
            'feature_selection__k': [5, 10, 15, 30, 'all'],
            'classifier__C': [0.01, 0.1, 1, 10, 100]
        }

        # Define inner and outer cross-validation splits.
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Set up GridSearchCV for hyperparameter tuning using the inner CV.
        grid_search = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=4, verbose=2)

        # Perform nested cross-validation (outer CV) to get an unbiased estimate.
        nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring='roc_auc')
        print('Nested CV ROC AUC scores:', nested_scores)
        print('Mean Nested CV ROC AUC:', np.mean(nested_scores))

        # Fit the grid search on the (sampled) training data.
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        dump(best_model, model_filename)
        print(f'Saved the best model to disk as: {model_filename}')
    else:
        # Load the saved model.
        best_model = load(model_filename)
        print(f'Loaded stored model from disk: {model_filename}')

    # Evaluate the model on the full dataset.
    # Use a DataFrame selection rather than a list of column names.
    X_full = df_full[['age', 'sex', 'time_category', 'prov', 
                      'drug_impairment_status', 'alcohol_status', 'cellphone_status', 'injp']]
    y_full = df_full['helmet_used']
    y_proba_full = best_model.predict_proba(X_full)[:, 1]
    y_pred_full = best_model.predict(X_full)
    full_roc_auc = roc_auc_score(y_full, y_proba_full)
    full_accuracy = accuracy_score(y_full, y_pred_full)
    print('Logistic Regression Model Completed. ')
    print(f'Full dataset metrics: ROC AUC = {full_roc_auc:.4f}, Accuracy = {full_accuracy:.4f}')

    # Extract feature names and coefficients.
    selected_mask = best_model.named_steps['feature_selection'].get_support()
    all_feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
    selected_features = all_feature_names[selected_mask]
    coefficients = best_model.named_steps['classifier'].coef_[0]

    # Combine feature names and coefficients into a DataFrame.
    coef_df = pd.DataFrame({
        'feature': selected_features,
        'coefficient': coefficients
    })
    
    print(coef_df)
    
    return best_model

# this model is a random forest with complete grid search
# ChatGPT helped with the initial design of this model too, 
# but I had to modify it quite a bit to get it to work
@time_function
def grid_search_model(df, use_stored_model=False, model_size=1.0):

    model_filename = './data/best_rf_model.joblib'
    df_full = df.copy()

    
    if not use_stored_model:
        # If model_size < 1.0, sample a stratified subset based on helmet_used.
        if model_size < 1.0:
            df = df.groupby('helmet_used', group_keys=False).apply(
                lambda x: x.sample(frac=model_size, random_state=42)
            )
            print(f'Sampled {model_size*100:.0f}% of the data stratified by helmet_used for training.')
        
        # Define training predictors and target.
        X_train = df[['age', 'sex', 'time_category', 'prov', 
                      'drug_impairment_status', 'alcohol_status', 'cellphone_status', 'injp']]
        y_train = df['helmet_used']
        
        # Define features by type.
        numerical_features = ['age']
        categorical_features = ['sex', 'time_category', 'prov', 
                                'drug_impairment_status', 'alcohol_status', 'cellphone_status', 'injp']
        
        # Build a preprocessor: scale numerical features and one-hot encode categorical features.
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first'), categorical_features)
            ]
        )
        
        # Build the pipeline with the preprocessor and RandomForestClassifier.
        pipeline_rf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Set up hyperparameter grid for the Random Forest.
        param_grid_rf = {
            'classifier__n_estimators': [100, 200, 300, 500],
            'classifier__max_depth': [None, 5, 10, 15, 20],
            'classifier__min_samples_split': [2, 3, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2', None]
        }
        
        # Define inner and outer cross-validation splits.
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Set up GridSearchCV for hyperparameter tuning (inner CV).
        grid_search_rf = GridSearchCV(
            pipeline_rf, param_grid_rf, cv=inner_cv,
            scoring='roc_auc', n_jobs=4, verbose=2
        )
        print('Starting grid search for Random Forest...')
        
        # Optional: Evaluate the model using nested cross validation.
        nested_scores_rf = cross_val_score(
            grid_search_rf, X_train, y_train, cv=outer_cv,
            scoring='roc_auc', n_jobs=4, verbose=2
        )
        print('Nested CV ROC AUC scores for Random Forest:', nested_scores_rf)
        print('Mean Nested CV ROC AUC:', np.mean(nested_scores_rf))
        
        # Fit the grid search on the training data.
        grid_search_rf.fit(X_train, y_train)
        best_rf_model = grid_search_rf.best_estimator_
        print('Best hyperparameters found:', grid_search_rf.best_params_)
        
        dump(best_rf_model, model_filename)
        print(f'Saved the best model to disk as: {model_filename}')
    else:
        best_rf_model = load(model_filename)
        print(f'Loaded stored model from disk: {model_filename}')
    print('Grid Search Model Completed. ')
    # Evaluate the model on the full dataset.
    X_eval = df_full[['age', 'sex', 'time_category', 'prov', 
                      'drug_impairment_status', 'alcohol_status', 'cellphone_status', 'injp']]
    y_eval = df_full['helmet_used']
    y_proba_eval = best_rf_model.predict_proba(X_eval)[:, 1]
    y_pred_eval = best_rf_model.predict(X_eval)
    full_roc_auc = roc_auc_score(y_eval, y_proba_eval)
    full_accuracy = accuracy_score(y_eval, y_pred_eval)
    print(f'Full dataset metrics: ROC AUC = {full_roc_auc:.4f}, Accuracy = {full_accuracy:.4f}')
    
    # Extract feature importances from the best model.
    all_feature_names = best_rf_model.named_steps['preprocessor'].get_feature_names_out()
    importances = best_rf_model.named_steps['classifier'].feature_importances_
    
    # Combine feature names and importances into a DataFrame.
    importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    
    print('Feature importances:')
    print(importance_df)
    
    return best_rf_model

@time_function
def random_grid_model(df, use_stored_model=False, model_size=1.0):

    model_filename = './data/best_random_rf_model.joblib'
    
    # Create copies for evaluation and training.
    df_full = df.copy()
    df_train = df.copy()
        
    # Define evaluation predictors and target.
    X_full = df_full[['age', 'sex', 'time_category', 'prov', 
                      'drug_impairment_status', 'alcohol_status', 'cellphone_status', 'injp']]
    y_full = df_full['helmet_used']
    
    if not use_stored_model:
        # If model_size < 1.0, sample a stratified subset from training data.
        if model_size < 1.0:
            df_train = df_train.groupby('helmet_used', group_keys=False).apply(
                lambda x: x.sample(frac=model_size, random_state=42)
            )
            print(f'Sampled {model_size*100:.0f}% of the data stratified by helmet_used for training.')
        
        # Define training predictors and target.
        X_train = df_train[['age', 'sex', 'time_category', 'prov', 
                            'drug_impairment_status', 'alcohol_status', 'cellphone_status', 'injp']]
        y_train = df_train['helmet_used']
        
        # List features by type.
        numerical_features = ['age']
        categorical_features = ['sex', 'time_category', 'prov', 
                                'drug_impairment_status', 'alcohol_status', 'cellphone_status', 'injp']
        
        # Build a preprocessor: scale numerical features and one-hot encode categorical features.
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first'), categorical_features)
            ]
        )
        
        # Build the pipeline with the preprocessor and RandomForestClassifier.
        pipeline_rf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Set up hyperparameter grid for the Random Forest.
        param_grid_rf = {
            'classifier__n_estimators': [100, 200, 300, 500],
            'classifier__max_depth': [None, 5, 10, 15, 20],
            'classifier__min_samples_split': [2, 3, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2', None]
        }
        
        # Define inner and outer cross-validation splits.
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Set up RandomizedSearchCV for hyperparameter tuning (inner CV).
        random_search_rf = RandomizedSearchCV(
            pipeline_rf, param_grid_rf, n_iter=20, cv=inner_cv,
            scoring='roc_auc', n_jobs=4, verbose=1, random_state=42
        )
        print('Starting grid search for Random Forest...')
        
        # Evaluate the model using nested cross-validation.
        nested_scores_rf = cross_val_score(
            random_search_rf, X_train, y_train, cv=outer_cv,
            scoring='roc_auc', n_jobs=4, verbose=2
        )
        print(f'Nested CV ROC AUC scores for Random Forest: {nested_scores_rf}')
        print(f'Mean Nested CV ROC AUC: {np.mean(nested_scores_rf):.4f}')
        
        # Fit the randomized search on the training data.
        random_search_rf.fit(X_train, y_train)
        best_rf_model = random_search_rf.best_estimator_
        dump(best_rf_model, model_filename)
        print(f'Saved the best model to disk as: {model_filename}')
    else:
        best_rf_model = load(model_filename)
        print(f'Loaded stored model from disk: {model_filename}')
    
    # Evaluate the model on the full dataset.
    y_pred_full = best_rf_model.predict(X_full)
    y_proba_full = best_rf_model.predict_proba(X_full)[:, 1]
    full_roc_auc = roc_auc_score(y_full, y_proba_full)
    full_accuracy = accuracy_score(y_full, y_pred_full)
    print(f'Full dataset metrics: ROC AUC = {full_roc_auc:.4f}, Accuracy = {full_accuracy:.4f}')
    
    # Extract feature importances from the best model.
    all_feature_names = best_rf_model.named_steps['preprocessor'].get_feature_names_out()
    importances = best_rf_model.named_steps['classifier'].feature_importances_
    
    # Combine feature names and importances into a DataFrame.
    importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    print('Random Grid Search Model Completed. ')
    print('Feature importances:')
    print('\n' + str(importance_df))
    
    return best_rf_model
# this model is a random forest with bayesian hyperparameter tuning
# ChatGPT's earlier models were changed to Optuna for Bayesian tuning, 
# works the same as the other models but uses a different method for hyperparameter tuning
@time_function
def model_w_bayesian(df, use_stored_model = False, model_size = 1.0):
    model_filename = './data/best_bayesian_rf_model.joblib'
    # Create copies for full evaluation and training.
    df_full = df.copy()
    df_train = df.copy()
    
    # Define full dataset predictors and target.
    X_full = df_full[['age', 'sex', 'time_category', 'prov', 
                      'drug_impairment_status', 'alcohol_status', 'cellphone_status', 'injp']]
    y_full = df_full['helmet_used']
    
    if not use_stored_model:
        # If a subsample is desired, perform stratified sampling based on the target.
        if model_size < 1.0:
            df_train = df_train.groupby('helmet_used', group_keys=False).apply(
                lambda x: x.sample(frac=model_size, random_state=42)
            )
            print(f'Sampled {model_size*100:.0f}% of the data stratified by helmet_used.')
        
        # Define training predictors and target.
        X_train = df_train[['age', 'sex', 'time_category', 'prov', 
                            'drug_impairment_status', 'alcohol_status', 'cellphone_status', 'injp']]
        y_train = df_train['helmet_used']
        
        # Define the objective function for Optuna.
        def objective(trial):
            # Suggest hyperparameters from the search space.
            n_estimators = trial.suggest_categorical('n_estimators', [100, 200, 300, 500])
            max_depth = trial.suggest_categorical('max_depth', [None, 5, 10, 15, 20])
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            
            # Build the classifier with the suggested hyperparameters.
            classifier = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42,
                n_jobs=4
            )
            
            # Define the preprocessor: scale numerical features and one-hot encode categorical features.
            numerical_features = ['age']
            categorical_features = ['sex', 'time_category', 'prov', 
                                    'drug_impairment_status', 'alcohol_status', 'cellphone_status', 'injp']
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('cat', OneHotEncoder(drop='first'), categorical_features)
                ]
            )
            
            # Build the pipeline.
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', classifier)
            ])
            
            # Evaluate with 5-fold cross-validation using ROC AUC.
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=4)
            return scores.mean()
        
        # Create the Optuna study and optimize the objective function.
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        print('Best hyperparameters:', study.best_params)
        print('Best ROC AUC score:', study.best_value)
        
        # Build the best classifier using the best hyperparameters.
        best_params = study.best_params
        best_classifier = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            max_features=best_params['max_features'],
            random_state=42,
            n_jobs=-1
        )
        
        # Re-create the same preprocessor.
        numerical_features = ['age']
        categorical_features = ['sex', 'time_category', 'prov', 
                                'drug_impairment_status', 'alcohol_status', 'cellphone_status', 'injp']
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first'), categorical_features)
            ]
        )
        
        # Build the final pipeline and fit it on the (sampled) training dataset.
        best_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', best_classifier)
        ])
        best_pipeline.fit(X_train, y_train)
        
        # Save the best model to disk.
        dump(best_pipeline, model_filename)
        print(f'Saved the best model to disk as: {model_filename}')
    else:
        # Load the stored model.
        best_pipeline = load(model_filename)
        print(f'Loaded stored model from disk: {model_filename}')
        print('Optimal Model Pipeline Steps:', best_pipeline.steps)
    print('Bayesian Full Search Model Completed. ')
    # Evaluate and report metrics on the full dataset.
    y_proba_full = best_pipeline.predict_proba(X_full)[:, 1]
    y_pred_full = best_pipeline.predict(X_full)
    full_roc_auc = roc_auc_score(y_full, y_proba_full)
    full_accuracy = accuracy_score(y_full, y_pred_full)
    print(f'Full dataset metrics: ROC AUC = {full_roc_auc:.4f}, Accuracy = {full_accuracy:.4f}')
    
    # Compute the No Information Rate (NIR) on the training set.
    nir = df_train['helmet_used'].value_counts(normalize=True).max()
    print('No Information Rate Overall:', nir)
    
    return best_pipeline

@time_function
def random_grid_2023_model(df, use_stored_model=False, model_size=1.0):

    model_filename = './data/best_2023_random_rf_model.joblib'
    
    # Ensure the 'year' column exists.
    if 'year' not in df.columns:
        df.loc[:, 'year'] = df['adate'].dt.year

    # Filter the DataFrame for motorcycle accidents in 2023.
    df = df[df['year'] == 2023].copy()
    
    # Create separate copies for full evaluation and training.
    df_full = df.copy()
    df_train = df.copy()
        
    # Define evaluation predictors and target.
    X_full = df_full[['age', 'sex', 'time_category', 'prov', 
                      'drug_impairment_status', 'alcohol_status', 'cellphone_status', 'injp']]
    y_full = df_full['helmet_used']
    
    if not use_stored_model:
        # If model_size < 1.0, sample a stratified subset from training data.
        if model_size < 1.0:
            df_train = df_train.groupby('helmet_used', group_keys=False).apply(
                lambda x: x.sample(frac=model_size, random_state=42)
            )
            print(f'Sampled {model_size*100:.0f}% of the data stratified by helmet_used for training.')
        
        # Define training predictors and target.
        X_train = df_train[['age', 'sex', 'time_category', 'prov', 
                            'drug_impairment_status', 'alcohol_status', 'cellphone_status', 'injp']]
        y_train = df_train['helmet_used']
        
        # List features by type.
        numerical_features = ['age']
        categorical_features = ['sex', 'time_category', 'prov', 
                                'drug_impairment_status', 'alcohol_status', 'cellphone_status', 'injp']
        
        # Build a preprocessor: scale numerical features and one-hot encode categorical features.
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first'), categorical_features)
            ]
        )
        
        # Build the pipeline with the preprocessor and RandomForestClassifier.
        pipeline_rf = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Set up hyperparameter grid for the Random Forest.
        param_grid_rf = {
            'classifier__n_estimators': [100, 200, 300, 500],
            'classifier__max_depth': [None, 5, 10, 15, 20],
            'classifier__min_samples_split': [2, 3, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2', None]
        }
        
        # Define inner and outer cross-validation splits.
        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Set up RandomizedSearchCV for hyperparameter tuning (inner CV).
        random_search_rf = RandomizedSearchCV(
            pipeline_rf, param_grid_rf, n_iter=20, cv=inner_cv,
            scoring='roc_auc', n_jobs=4, verbose=1, random_state=42
        )
        print('Starting grid search for Random Forest...')
        
        # Evaluate the model using nested cross-validation.
        nested_scores_rf = cross_val_score(
            random_search_rf, X_train, y_train, cv=outer_cv,
            scoring='roc_auc', n_jobs=4, verbose=2
        )
        print(f'Nested CV ROC AUC scores for Random Forest: {nested_scores_rf}')
        print(f'Mean Nested CV ROC AUC: {np.mean(nested_scores_rf):.4f}')
        
        # Fit the randomized search on the training data.
        random_search_rf.fit(X_train, y_train)
        best_rf_model = random_search_rf.best_estimator_
        dump(best_rf_model, model_filename)
        print(f'Saved the best model to disk as: {model_filename}')
    else:
        best_rf_model = load(model_filename)
        print(f'Loaded stored model from disk: {model_filename}')
    
    # Evaluate the model on the full dataset.
    y_pred_full = best_rf_model.predict(X_full)
    y_proba_full = best_rf_model.predict_proba(X_full)[:, 1]
    full_roc_auc = roc_auc_score(y_full, y_proba_full)
    full_accuracy = accuracy_score(y_full, y_pred_full)
    print(f'Full dataset metrics: ROC AUC = {full_roc_auc:.4f}, Accuracy = {full_accuracy:.4f}')
    
    # Extract feature importances from the best model.
    all_feature_names = best_rf_model.named_steps['preprocessor'].get_feature_names_out()
    importances = best_rf_model.named_steps['classifier'].feature_importances_
    
    # Combine feature names and importances into a DataFrame.
    importance_df = pd.DataFrame({'feature': all_feature_names, 'importance': importances})
    importance_df = importance_df.sort_values(by='importance', ascending=False)
    print('Random Grid Search Model Completed. ')
    print('Feature importances:')
    print('\n' + str(importance_df))
    
    return best_rf_model

@time_function
def model_2023_w_bayesian(df, use_stored_model = False, model_size = 1.0):
    model_filename = './data/best_2023_bayesian_rf_model.joblib'
    
    # Ensure the 'year' column exists.
    if 'year' not in df.columns:
        df.loc[:, 'year'] = df['adate'].dt.year

    # Filter the DataFrame for motorcycle accidents in 2023.
    df = df[df['year'] == 2023].copy()
    
    # Create separate copies for full evaluation and training.
    df_full = df.copy()
    df_train = df.copy()
    
    # Map 'Helmet' to 1 and 'No Helmet' to 0.
    mapping = {'Helmet': 1, 'No Helmet': 0}
    df_full['helmet_used'] = df_full['helmet_status'].map(mapping)
    df_train['helmet_used'] = df_train['helmet_status'].map(mapping)
    
    # Define full dataset predictors and target.
    X_full = df_full[['age', 'sex', 'time_category', 'prov', 
                      'drug_impairment_status', 'alcohol_status', 'cellphone_status', 'injp']]
    y_full = df_full['helmet_used']
    
    if not use_stored_model:
        # If a subsample is desired, perform stratified sampling on the training set.
        if model_size < 1.0:
            df_train = df_train.groupby('helmet_used', group_keys=False).apply(
                lambda x: x.sample(frac=model_size, random_state=42)
            )
            print(f'Sampled {model_size*100:.0f}% of the data stratified by helmet_used.')
        
        # Define training predictors and target.
        X_train = df_train[['age', 'sex', 'time_category', 'prov', 
                            'drug_impairment_status', 'alcohol_status', 'cellphone_status', 'injp']]
        y_train = df_train['helmet_used']
        
        # Define the objective function for Optuna.
        def objective(trial):
            # Suggest hyperparameters.
            n_estimators = trial.suggest_categorical('n_estimators', [100, 200, 300, 500])
            max_depth = trial.suggest_categorical('max_depth', [None, 5, 10, 15, 20])
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            
            # Build the classifier.
            classifier = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42,
                n_jobs=4
            )
            
            # Define the preprocessor.
            numerical_features = ['age']
            categorical_features = ['sex', 'time_category', 'prov', 
                                    'drug_impairment_status', 'alcohol_status', 'cellphone_status', 'injp']
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('cat', OneHotEncoder(drop='first'), categorical_features)
                ]
            )
            
            # Build the pipeline.
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', classifier)
            ])
            
            # Evaluate with 5-fold cross-validation using ROC AUC.
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=4)
            return scores.mean()
        
        # Create the Optuna study and optimize the objective function.
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        print('Best hyperparameters:', study.best_params)
        print('Best ROC AUC score:', study.best_value)
        
        # Build the best classifier using the best hyperparameters.
        best_params = study.best_params
        best_classifier = RandomForestClassifier(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            min_samples_leaf=best_params['min_samples_leaf'],
            max_features=best_params['max_features'],
            random_state=42,
            n_jobs=-1
        )
        
        # Re-create the same preprocessor.
        numerical_features = ['age']
        categorical_features = ['sex', 'time_category', 'prov', 
                                'drug_impairment_status', 'alcohol_status', 'cellphone_status', 'injp']
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first'), categorical_features)
            ]
        )
        
        # Build the final pipeline and fit on the training dataset.
        best_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', best_classifier)
        ])
        best_pipeline.fit(X_train, y_train)
        
        # Save the best model to disk.
        dump(best_pipeline, model_filename)
        print(f'Saved the best model to disk as: {model_filename}')
    else:
        # Load the stored model.
        best_pipeline = load(model_filename)
        print(f'Loaded stored model from disk: {model_filename}')
        print('Optimal Model Pipeline Steps:', best_pipeline.steps)
    
    # Evaluate and report metrics on the full dataset.
    y_proba_full = best_pipeline.predict_proba(X_full)[:, 1]
    y_pred_full = best_pipeline.predict(X_full)
    full_roc_auc = roc_auc_score(y_full, y_proba_full)
    full_accuracy = accuracy_score(y_full, y_pred_full)
    print('Bayesian 2023 Search Model Completed. ')
    print(f'Full dataset metrics: ROC AUC = {full_roc_auc:.4f}, Accuracy = {full_accuracy:.4f}')
    
    # Compute the No Information Rate (NIR) for 2023 from the training set.
    nir_2023 = df_train['helmet_used'].value_counts(normalize=True).max()
    print('No Information Rate 2023:', nir_2023)
    
    return best_pipeline


def evaluate_model(model, df):
    '''
    Evaluate a given model on the full dataset.
    Computes ROC AUC and accuracy using the full dataset.
    '''
    # Create a copy of the data and generate the binary target
    df_eval = df.copy()
    df_eval['helmet_used'] = np.where(df_eval['helmet_status'] == 'Helmet', 1, 0)
    
    # Define the predictors and target as used in model training
    X_eval = df_eval[['age', 'sex', 'time_category', 'prov', 
                      'drug_impairment_status', 'alcohol_status', 'cellphone_status', 'year','injp']]
    y_eval = df_eval['helmet_used']
    
    # Get predictions and prediction probabilities from the model
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]
    
    # Compute evaluation metrics
    roc_auc = roc_auc_score(y_eval, y_proba)
    accuracy = accuracy_score(y_eval, y_pred)
    
    return {'ROC AUC': roc_auc, 'Accuracy': accuracy}

def compare_models(df, model_size=1.0, use_stored_model=False):
    '''
    Trains and evaluates four modeling approaches:
      1. Logistic Regression with feature selection via GridSearchCV.
      2. Random Forest tuned with GridSearchCV.
      3. Random Forest tuned with RandomizedSearchCV.
      4. Random Forest tuned with Bayesian optimization (Optuna).
      
    Returns a summary DataFrame with evaluation metrics and details for academic reporting.
    '''
    # Logistic Regression Model
    print('Training Logistic Regression Model...')
    lr_model = logistic_reg_model(df.copy(), use_stored_model=use_stored_model, model_size=model_size)
    lr_metrics = evaluate_model(lr_model, df)
    lr_details = ('Logistic Regression with automatic feature selection (SelectKBest using ANOVA F-test) '
                  'and hyperparameter tuning (GridSearchCV over C and k) with nested cross-validation.')

    # Random Forest Model via GridSearchCV
    print('Training Random Forest Model with GridSearchCV...')
    gs_rf_model = grid_search_model(df.copy(), use_stored_model=use_stored_model, model_size=model_size)
    gs_rf_metrics = evaluate_model(gs_rf_model, df)
    gs_rf_details = ('Random Forest model tuned using an exhaustive grid search (GridSearchCV) over several hyperparameters ')
    #                  'with nested cross-validation to obtain unbiased performance estimates.')

    # Random Forest Model via RandomizedSearchCV
    print('Training Random Forest Model with RandomizedSearchCV...')
    rs_rf_model = random_grid_model(df.copy(), use_stored_model=use_stored_model, model_size=model_size)
    rs_rf_metrics = evaluate_model(rs_rf_model, df)
    rs_rf_details = ('Random Forest model tuned using a randomized search (RandomizedSearchCV) with 20 iterations '
                     'of hyperparameter sampling and nested cross-validation for performance assessment.')

    # Random Forest Model via Bayesian Optimization (Optuna)
    print('Training Random Forest Model with Bayesian Optimization (Optuna)...')
    bayes_rf_model = model_w_bayesian(df.copy(), use_stored_model=use_stored_model, model_size=model_size)
    bayes_rf_metrics = evaluate_model(bayes_rf_model, df)
    bayes_rf_details = ('Random Forest model tuned using Bayesian optimization via Optuna, optimizing hyperparameters '
                        'over 50 trials to maximize ROC AUC.')

    # # Compile a summary DataFrame
    summary_df = pd.DataFrame({
        'Method': ['Logistic Regression', 'Grid Search RF', 'Randomized Search RF', 'Bayesian Optimization RF'],
        'ROC AUC': [lr_metrics['ROC AUC'], gs_rf_metrics['ROC AUC'], rs_rf_metrics['ROC AUC'], bayes_rf_metrics['ROC AUC']],
        'Accuracy': [lr_metrics['Accuracy'], gs_rf_metrics['Accuracy'], rs_rf_metrics['Accuracy'], bayes_rf_metrics['Accuracy']],
        'Details': [lr_details, gs_rf_details, rs_rf_details, bayes_rf_details] 
        })
    
    return summary_df