import os
import json
import zipfile
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from scipy.interpolate import interp1d
import joblib
from scipy.stats import norm
import gc
import lightgbm as lgb
import xgboost as xgb
matplotlib.use('Agg')

# -----------------------------------------------------------
# METHOD B - AI-ITACA Project - Chapter I 
# -----------------------------------------------------------
# PCA Regression Modeling with Random Forest, Gradient Boosting, LightGBM, and XGBoost
# The code generates PCA models and regression models for large spectral datasets, based on interpolation, PCA low-dimensionality processing, 
# and generation of regression models using: RandomForestRegressor, GradientBoostingRegressor, LGBMRegressor, and XGBRegressor.
#- INPUT: Directory containing spectral .txt files with headers including physical parameters.
#- custom_hyperparams: directory, sample_size (of all dataset), random_seed, output_dir, target_length (for interpolation), 
#variance_threshold (for PCA reduction), batch_size 
#- OUTPUT: PCA model, reference frequencies, physical parameters array, parameter names, valid files list, corrupted files list.
#  Gabriel Jaimes Illanes (gjaimes@cab.inta-csic.es)



def analyze_large_spectra(directory, sample_size=None, random_seed=42, output_dir="PCA_RF_Results_Large", 
                         target_length=20000, variance_threshold=0.95, batch_size=1000,
                         custom_hyperparams=None):
    """
    Main function for PCA analysis and regression modeling of large spectral datasets.
    """
    # SECTION 1: Data Preparation
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, "saved_models")
    temp_dir = os.path.join(output_dir, "temp_files")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    random.seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    if custom_hyperparams is None:
        custom_hyperparams = {
            'logn': {
                'RandomForest': {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
                'GradientBoosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 1.0},
                'XGBoost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 1.0},
                'LightGBM': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': -1, 'num_leaves': 31}
            },
            'tex': {
                'RandomForest': {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 1},
                'GradientBoosting': {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.8},
                'XGBoost': {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 7, 'subsample': 0.8},
                'LightGBM': {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 7, 'num_leaves': 50}
            },
            'velo': {
                'RandomForest': {'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1},
                'GradientBoosting': {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 4, 'subsample': 0.9},
                'XGBoost': {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 5, 'subsample': 0.9},
                'LightGBM': {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 5, 'num_leaves': 40}
            },
            'fwhm': {
                'RandomForest': {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
                'GradientBoosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 4, 'subsample': 1.0},
                'XGBoost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 1.0},
                'LightGBM': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'num_leaves': 31}
            }
        }
    if custom_hyperparams is None:
        custom_hyperparams = {
            'logn': {
                'RandomForest': {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
                'GradientBoosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 1.0},
                'XGBoost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 1.0},
                'LightGBM': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': -1, 'num_leaves': 31}
            },
            'tex': {
                'RandomForest': {'n_estimators': 200, 'max_depth': 20, 'min_samples_split': 5, 'min_samples_leaf': 1},
                'GradientBoosting': {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.8},
                'XGBoost': {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 7, 'subsample': 0.8},
                'LightGBM': {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 7, 'num_leaves': 50}
            },
            'velo': {
                'RandomForest': {'n_estimators': 150, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1},
                'GradientBoosting': {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 4, 'subsample': 0.9},
                'XGBoost': {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 5, 'subsample': 0.9},
                'LightGBM': {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 5, 'num_leaves': 40}
            },
            'fwhm': {
                'RandomForest': {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
                'GradientBoosting': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 4, 'subsample': 1.0},
                'XGBoost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'subsample': 1.0},
                'LightGBM': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'num_leaves': 31}
            }
        }
    
    all_files = [f for f in os.listdir(directory) if f.endswith(".txt")]
    
    if sample_size is not None and sample_size < len(all_files):
        print(f"\nRandomly sampling {sample_size} of {len(all_files)} spectra...")
        selected_files = random.sample(all_files, sample_size)
    else:
        print("\nUsing all available spectra...")
        selected_files = all_files
    
    reference_frequencies = None
    valid_files = []
    corrupted_files = []
    params = []
    reference_frequencies = None
    valid_files = []
    corrupted_files = []
    params = []
    
    print("\nFirst pass: Validating files and collecting parameters...")
    for filename in tqdm(selected_files, desc="Validating files"):
        filepath = os.path.join(directory, filename)
        
        try:
            with open(filepath, 'r') as file:
                lines = file.readlines()
            
            if len(lines) < 2:
                corrupted_files.append((filename, "File too short"))
                continue
                
            header = lines[0].strip()
            param_dict = {}
            
            for part in header.split():
                if '=' in part:
                    key, value = part.split('=')
                    key = key.strip()
                    value = value.strip("'")
                    
                    if key not in ['molecules', 'sourcesize']:
                        try:
                            param_dict[key] = float(value)
                        except ValueError:
                            param_dict[key] = value
            
            spectrum_data = []
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        freq = float(parts[0])
                        intensity = float(parts[1])
                        if not np.isfinite(freq) or not np.isfinite(intensity):
                            raise ValueError("Non-finite value found")
                        spectrum_data.append((freq, intensity))
                    except ValueError:
                        continue
            
            if not spectrum_data:
                corrupted_files.append((filename, "No valid data"))
                continue
                
            frequencies = np.array(spectrum_data)[:, 0]
            if reference_frequencies is None:
                min_freq = np.min(frequencies)
                max_freq = np.max(frequencies)
                reference_frequencies = np.linspace(min_freq, max_freq, target_length)
            
            params.append([
                param_dict.get('logn', np.nan),
                param_dict.get('tex', np.nan),
                param_dict.get('velo', np.nan),
                param_dict.get('fwhm', np.nan)
            ])
            valid_files.append(filename)
            
        except Exception as e:
            corrupted_files.append((filename, f"Processing error: {str(e)}"))
    
    if corrupted_files:
        print(f"\nCorrupted or discarded files ({len(corrupted_files)}):")
        with open(os.path.join(output_dir, "corrupted_files.txt"), 'w') as f:
            for filename, reason in corrupted_files:
                f.write(f"{filename}: {reason}\n")
    
    y = np.array(params)
    
    # SECTION 2: PCA Analysis
    print(f"\nSecond pass: Processing {len(valid_files)} spectra in batches of {batch_size}...")
    
    initial_components = min(100, target_length)
    ipca = IncrementalPCA(n_components=initial_components, batch_size=batch_size)
    scaler = StandardScaler()
    
    temp_scaled_file = os.path.join(temp_dir, "scaled_spectra.npy")
    temp_pca_file = os.path.join(temp_dir, "pca_results.npy")
    
    for f in [temp_scaled_file, temp_pca_file]:
        if os.path.exists(f):
            os.remove(f)
    
    for batch_start in tqdm(range(0, len(valid_files), batch_size), desc="Processing batches"):
        batch_end = min(batch_start + batch_size, len(valid_files))
        batch_files = valid_files[batch_start:batch_end]
        
        batch_data = []
        for filename in batch_files:
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()
            
            spectrum_data = []
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) >= 2:
                    freq = float(parts[0])
                    intensity = float(parts[1])
                    spectrum_data.append((freq, intensity))
            

            frequencies = np.array(spectrum_data)[:, 0]
            intensities = np.array(spectrum_data)[:, 1]
            interpolator = interp1d(frequencies, intensities, kind='linear',
                                 bounds_error=False, fill_value=0.0)
            interpolated_intensities = interpolator(reference_frequencies)
            batch_data.append(interpolated_intensities)
        
        X_batch = np.array(batch_data, dtype=np.float32)
        scaler.partial_fit(X_batch)
        X_batch_scaled = scaler.transform(X_batch)
        
        with open(temp_scaled_file, 'ab') as f:
            np.save(f, X_batch_scaled)
        
        ipca.partial_fit(X_batch_scaled)
        
        del X_batch, X_batch_scaled, batch_data
        gc.collect()
    
    cumulative_variance = np.cumsum(ipca.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    print(f"\nSelected {n_components} components to explain {variance_threshold*100:.1f}% variance")
    
    sigma_levels = {
        '2σ (95.45%)': norm.cdf(2) - norm.cdf(-2),
        '2.5σ (98.76%)': norm.cdf(2.5) - norm.cdf(-2.5),
        '3σ (99.73%)': norm.cdf(3) - norm.cdf(-3),
        '1.0 (100%)': 1.0
    }
    
    sigma_components = {}
    for name, level in sigma_levels.items():
        if level <= cumulative_variance[-1]:
            sigma_components[name] = np.argmax(cumulative_variance >= level) + 1
    
    print("\nNumber of components needed:")
    print(f"- For {variance_threshold*100:.1f}% variance: {n_components} components")
    for name, n in sigma_components.items():
        print(f"- For {name}: {n} components")
    
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    
    print("\nThird pass: Transforming data with final PCA...")
    
    with open(temp_scaled_file, 'rb') as f_scaled, \
         open(temp_pca_file, 'wb') as f_pca:
        
        ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        
        # First pass: Fit PCA completely
        f_scaled.seek(0)
        while True:
            try:
                X_batch_scaled = np.load(f_scaled)
                ipca.partial_fit(X_batch_scaled)
            except EOFError:
                break
        
        # Second pass: Transform data
        f_scaled.seek(0)
        while True:
            try:
                X_batch_scaled = np.load(f_scaled)
                X_batch_pca = ipca.transform(X_batch_scaled)
                np.save(f_pca, X_batch_pca)
                del X_batch_scaled, X_batch_pca
                gc.collect()
            except EOFError:
                break
    
    pca_results = []
    with open(temp_pca_file, 'rb') as f:
        while True:
            try:
                pca_results.append(np.load(f))
            except EOFError:
                break
    principal_components = np.vstack(pca_results)
    
    if os.path.exists(temp_scaled_file):
        os.remove(temp_scaled_file)
    if os.path.exists(temp_pca_file):
        os.remove(temp_pca_file)
    
    scaler_path = os.path.join(models_dir, "standard_scaler.save")
    pca_path = os.path.join(models_dir, "incremental_pca.save")
    joblib.dump(scaler, scaler_path)
    joblib.dump(ipca, pca_path)
    main_scaler_path = os.path.join(models_dir, "main_scaler.save")
    if not os.path.exists(main_scaler_path):
        joblib.dump(scaler, main_scaler_path)
    
    plt.figure(figsize=(12, 7))
    plt.plot(np.arange(1, len(cumulative_variance)+1), cumulative_variance, 'o-', label='Cumulative Variance')
    
    colors = ['r', 'g', 'b', 'm']
    for i, ((name, level), color) in enumerate(zip(sigma_levels.items(), colors)):
        if level <= cumulative_variance[-1]:
            n_comp = sigma_components[name]
            plt.axvline(x=n_comp, color=color, linestyle='--', alpha=0.7)
            plt.axhline(y=level, color=color, linestyle='--', alpha=0.7)
            plt.text(n_comp+5, level-0.05, f'{name}\n{n_comp} components', 
                   color=color, ha='left', va='top')
    
    plt.axvline(x=n_components, color='k', linestyle='--', label=f'Selected: {n_components} components')
    plt.axhline(y=variance_threshold, color='k', linestyle='--')
    
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by Principal Components')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, "variance_explained.png"))
    plt.close()
    
    n_components_to_plot = min(5, ipca.n_components_)
    plt.figure(figsize=(15, 10))
    
    for i in range(n_components_to_plot):
        plt.subplot(n_components_to_plot, 1, i+1)
        plt.plot(reference_frequencies, ipca.components_[i], label=f'Component {i+1}')
        
        top_10_idx = np.argsort(np.abs(ipca.components_[i]))[-10:]
        plt.scatter(reference_frequencies[top_10_idx], ipca.components_[i][top_10_idx], 
                  color='red', s=30, label='Top 10 frequencies' if i==0 else "")
        
        plt.ylabel(f'Loading C{i+1}')
        plt.grid(alpha=0.3)
        if i == n_components_to_plot-1:
            plt.xlabel('Frequency (Hz)')
        plt.legend()
    
    plt.suptitle('Principal Component Loadings')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "components_loadings.png"))
    plt.close()
    
    param_names = ['logn', 'tex', 'velo', 'fwhm']
    param_labels = ['log(n)', 'T_ex', 'V_los', 'FWHM']
    
    print("\nCalculating correlations with physical parameters...")
    for i, (param, label) in enumerate(zip(param_names, param_labels)):
        correlations = [np.corrcoef(principal_components[:, j], y[:, i])[0, 1] 
                      for j in range(min(10, ipca.n_components_))]
        
        with open(os.path.join(output_dir, f"correlations_{param}.txt"), 'w') as f:
            f.write(f"Correlations between principal components and {label}\n\n")
            for comp, corr in enumerate(correlations, 1):
                f.write(f"Component {comp}: {corr:.4f}\n")
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(correlations)+1), correlations)
        plt.title(f'Correlation between principal components and {label}')
        plt.xlabel('Principal component')
        plt.ylabel('Correlation')
        plt.grid()
        plt.savefig(os.path.join(output_dir, f"correlations_{param}.png"))
        plt.close()
        
        corr_threshold = 0.3
        significant_comps = [j for j, corr in enumerate(correlations) if abs(corr) > corr_threshold]
        
        if significant_comps:
            plt.figure(figsize=(12, 6))
            for comp in significant_comps:
                plt.plot(reference_frequencies, ipca.components_[comp], 
                        label=f'Component {comp+1} (corr={correlations[comp]:.2f})')
            
            plt.title(f'Loadings of components correlated with {label} (|corr|>{corr_threshold})')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Loading value')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"significant_loadings_{param}.png"))
            plt.close()
    
    # SECTION 3: Model Training
    print("\nSplitting data into training (35%), validation (35%) and test (30%) sets...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        principal_components, y, test_size=0.3, random_state=random_seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.5, random_state=random_seed
    )
    
    param_scalers = {}
    for i, param in enumerate(param_names):
        param_scaler = StandardScaler()
        y_train[:, i] = param_scaler.fit_transform(y_train[:, i].reshape(-1, 1)).flatten()
        y_val[:, i] = param_scaler.transform(y_val[:, i].reshape(-1, 1)).flatten()
        y_test[:, i] = param_scaler.transform(y_test[:, i].reshape(-1, 1)).flatten()
        param_scalers[param] = param_scaler

        scaler_filename = os.path.join(models_dir, f"{param}_scaler.save")
        joblib.dump(param_scaler, scaler_filename)
        print(f"Saved {param} scaler to: {scaler_filename}")
    

    def create_models_for_param(param_name):
        """Create models with custom hyperparameters for a specific parameter"""
        param_hyperparams = custom_hyperparams.get(param_name, {})
        
        models = {
            'RandomForest': RandomForestRegressor(
                random_state=random_seed,
                n_jobs=-1,
                **param_hyperparams.get('RandomForest', {})
            ),
            'GradientBoosting': GradientBoostingRegressor(
                random_state=random_seed,
                **param_hyperparams.get('GradientBoosting', {})
            ),
            'XGBoost': xgb.XGBRegressor(
                random_state=random_seed,
                n_jobs=-1,
                **param_hyperparams.get('XGBoost', {})
            ),
            'LightGBM': lgb.LGBMRegressor(
                random_state=random_seed,
                n_jobs=-1,
                **param_hyperparams.get('LightGBM', {})
            )
        }
        return models
    
    all_models = {}
    results = {}
    model_comparison = {}
    
    for i, (param, label) in enumerate(zip(param_names, param_labels)):
        print(f"\nTraining models for {label}...")
        
        y_train_param = y_train[:, i]
        y_val_param = y_val[:, i]
        y_test_param = y_test[:, i]
        
        models = create_models_for_param(param)
        
        param_results = {}
        model_scores = {}
        
        for model_name, model in models.items():
            print(f"  Training {model_name}...")
            
            try:
                model.fit(X_train, y_train_param)
                
                y_val_pred = model.predict(X_val)
                y_test_pred = model.predict(X_test)
                y_train_pred = model.predict(X_train)
                
                y_val_pred_orig = param_scalers[param].inverse_transform(y_val_pred.reshape(-1, 1)).flatten()
                y_test_pred_orig = param_scalers[param].inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
                y_train_pred_orig = param_scalers[param].inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
                
                y_val_actual_orig = param_scalers[param].inverse_transform(y_val_param.reshape(-1, 1)).flatten()
                y_test_actual_orig = param_scalers[param].inverse_transform(y_test_param.reshape(-1, 1)).flatten()
                y_train_actual_orig = param_scalers[param].inverse_transform(y_train_param.reshape(-1, 1)).flatten()
                
                train_mse = mean_squared_error(y_train_actual_orig, y_train_pred_orig)
                val_mse = mean_squared_error(y_val_actual_orig, y_val_pred_orig)
                test_mse = mean_squared_error(y_test_actual_orig, y_test_pred_orig)
                
                train_r2 = r2_score(y_train_actual_orig, y_train_pred_orig)
                val_r2 = r2_score(y_val_actual_orig, y_val_pred_orig)
                test_r2 = r2_score(y_test_actual_orig, y_test_pred_orig)
                
                model_results = {
                    'model': model,
                    'train_mse': train_mse,
                    'val_mse': val_mse,
                    'test_mse': test_mse,
                    'train_r2': train_r2,
                    'val_r2': val_r2,
                    'test_r2': test_r2,
                    'y_train': y_train_actual_orig,
                    'y_train_pred': y_train_pred_orig,
                    'y_val': y_val_actual_orig,
                    'y_val_pred': y_val_pred_orig,
                    'y_test': y_test_actual_orig,
                    'y_test_pred': y_test_pred_orig,
                    'hyperparameters': model.get_params()
                }
                
                param_results[model_name] = model_results
                model_scores[model_name] = test_r2  
                
                model_package = {
                    'model': model,
                    'metrics': {
                        'train_mse': train_mse,
                        'val_mse': val_mse,
                        'test_mse': test_mse,
                        'train_r2': train_r2,
                        'val_r2': val_r2,
                        'test_r2': test_r2,
                        'hyperparameters': model.get_params()
                    }
                }
                model_filename = os.path.join(models_dir, f"{param}_{model_name.lower().replace(' ', '_')}.save")
                joblib.dump(model_package, model_filename)
                print(f"    Saved {model_name} model and metrics to: {model_filename}")
                print(f"    {model_name} - Test R²: {test_r2:.4f}, Test MSE: {test_mse:.4f}")
                
            except Exception as e:
                print(f"    Error training {model_name}: {str(e)}")
                continue
        
        all_models[param] = models
        results[param] = param_results
        model_comparison[param] = model_scores
        
        with open(os.path.join(output_dir, f"model_comparison_{param}.txt"), 'w') as f:
            f.write(f"Model comparison for {label} prediction\n\n")
            f.write("Model\tTest R²\tTest MSE\tValidation R²\tValidation MSE\tHyperparameters\n")
            f.write("="*100 + "\n")
            for model_name, score in sorted(model_scores.items(), key=lambda x: x[1], reverse=True):
                test_mse = param_results[model_name]['test_mse']
                val_r2 = param_results[model_name]['val_r2']
                val_mse = param_results[model_name]['val_mse']
                hyperparams = str(param_results[model_name]['hyperparameters'])
                f.write(f"{model_name}\t{score:.4f}\t{test_mse:.4f}\t{val_r2:.4f}\t{val_mse:.4f}\t{hyperparams}\n")
    
    print("\nGenerating prediction plots for all models on test data...")
    for param, label in zip(param_names, param_labels):
        param_results = results[param]
        
        n_models = len(param_results)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 6), sharey=True)
        if n_models == 1:
            axes = [axes]
        colors = ['blue', 'green', 'orange', 'purple']
        for idx, (model_name, res) in enumerate(param_results.items()):
            ax = axes[idx]
            ax.scatter(res['y_test'], res['y_test_pred'], alpha=0.3, color=colors[idx], 
                      label=f'{model_name} (R²={res["test_r2"]:.3f})')
            min_val = min(np.min(res['y_test']), np.min(res['y_test_pred']))
            max_val = max(np.max(res['y_test']), np.max(res['y_test_pred']))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
            ax.set_xlabel(f'Actual {label}')
            ax.set_ylabel(f'Predicted {label}')
            ax.set_title(f'{model_name}\nTest R² = {res["test_r2"]:.3f}')
            ax.grid(alpha=0.3)
            ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"test_predictions_subplots_{param}.png"))
        plt.close()
        
        for model_name, res in param_results.items():
            plt.figure(figsize=(8, 6))
            plt.scatter(res['y_test'], res['y_test_pred'], alpha=0.5, 
                       label=f'{model_name} (R²={res["test_r2"]:.3f})')
            min_val = min(np.min(res['y_test']), np.min(res['y_test_pred']))
            max_val = max(np.max(res['y_test']), np.max(res['y_test_pred']))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
            plt.xlabel(f'Actual {label}')
            plt.ylabel(f'Predicted {label}')
            plt.title(f'{model_name} - Test Predictions for {label}\nR² = {res["test_r2"]:.3f}')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(output_dir, f"test_predictions_{param}_{model_name}.png"))
            plt.close()
    
    print("\nCreating model comparison summary...")
    with open(os.path.join(output_dir, "model_comparison_summary.txt"), 'w') as f:
        f.write("MODEL COMPARISON SUMMARY WITH CUSTOM HYPERPARAMETERS\n")
        f.write("=" * 80 + "\n\n")
        
        for param, label in zip(param_names, param_labels):
            f.write(f"Parameter: {label}\n")
            f.write("-" * 40 + "\n")
            
            scores = model_comparison[param]
            best_model = max(scores.items(), key=lambda x: x[1])
            
            f.write(f"Best model: {best_model[0]} (Test R² = {best_model[1]:.4f})\n\n")
            
            f.write("All models (sorted by Test R²):\n")
            for model_name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {model_name}: Test R² = {score:.4f}\n")
            
            f.write("\n")
    
    print("\nSaving all metrics to metrics_summary.csv ...")
    metrics_rows = []
    for param, label in zip(param_names, param_labels):
        param_results = results[param]
        for model_name, res in param_results.items():
            row = {
                'Parameter': label,
                'Model': model_name,
                'Train_MSE': res['train_mse'],
                'Val_MSE': res['val_mse'],
                'Test_MSE': res['test_mse'],
                'Train_R2': res['train_r2'],
                'Val_R2': res['val_r2'],
                'Test_R2': res['test_r2'],
                'Hyperparameters': str(res['hyperparameters'])
            }
            metrics_rows.append(row)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)
    print("Metrics saved to metrics_summary.csv")

    # SECTION 4: Results Packaging
    print("\nPackaging models for SpectralRegressor app compatibility...")

    freq_file = os.path.join(models_dir, "reference_frequencies.npy")
    np.save(freq_file, reference_frequencies)
    evr_file = os.path.join(models_dir, "explained_variance_ratio.npy")
    np.save(evr_file, ipca.explained_variance_ratio_)

    model_file_map = {}
    for param in param_names:
        model_file_map[param] = {}
        for fname in os.listdir(models_dir):
            if fname.startswith(param + "_") and fname.endswith(".save") and fname not in ["standard_scaler.save", "main_scaler.save", "incremental_pca.save", f"{param}_scaler.save"]:
                model_key = fname[len(param) + 1 : -5]  
                model_file_map[param][model_key] = fname

    param_scaler_map = {p: f"{p}_scaler.save" for p in param_names}

    metadata = {
        "description": "Packaged spectral regression models for SpectralRegressor Streamlit application",
        "version": 1,
        "target_length": int(len(reference_frequencies)),
        "n_components": int(ipca.n_components_),
        "variance_threshold": float(variance_threshold),
        "explained_variance_cumulative": float(np.cumsum(ipca.explained_variance_ratio_)[-1]),
        "paths": {
            "scaler": os.path.basename(scaler_path),
            "main_scaler": os.path.basename(main_scaler_path),
            "pca": os.path.basename(pca_path),
            "frequencies": os.path.basename(freq_file),
            "explained_variance_ratio": os.path.basename(evr_file),
            "param_scalers": param_scaler_map,
            "models": model_file_map
        },
        "param_names": param_names,
        "model_name_conventions": {
            "training_keys": ["RandomForest", "GradientBoosting", "LightGBM", "XGBoost"],
            "app_expected": ["Randomforest", "Gradientboosting", "Lightgbm", "Xgboost"],
            "note": "Loader should normalize names (lowercase) to match saved file keys"
        }
    }

    metadata_path = os.path.join(models_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as mdf:
        json.dump(metadata, mdf, indent=2)
    print(f"Metadata written to {metadata_path}")
    zip_output_path = os.path.join(output_dir, "models.zip")
    with zipfile.ZipFile(zip_output_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(models_dir):
            for f in files:
                if f.endswith(('.save', '.npy', '.json')):
                    full_path = os.path.join(root, f)
                    arcname = os.path.relpath(full_path, models_dir)  
                    zf.write(full_path, arcname=arcname)
        metrics_path = os.path.join(output_dir, "metrics_summary.csv")
        if os.path.exists(metrics_path):
            zf.write(metrics_path, arcname="metrics_summary.csv")
    print(f"Packaged ZIP for app: {zip_output_path}")
    
    print(f"\nAnalysis completed. Results saved in: {os.path.abspath(output_dir)}")
    
    return {
        'pca': ipca,
        'frequencies': reference_frequencies,
        'params': y,
        'param_names': param_names,
        'valid_files': valid_files,
        'corrupted_files': corrupted_files,
        'models': all_models,
        'param_scalers': param_scalers,
        'results': results,
        'model_comparison': model_comparison,
        'main_scaler': scaler,
        'n_components': n_components,
        'variance_threshold': variance_threshold,
        'sigma_components': sigma_components,
        'X_test': X_test,
        'y_test': y_test,
        'custom_hyperparams': custom_hyperparams
    }

if __name__ == "__main__":
    directory = r"D:\4.DATASETS\CLEANED_SPECTRA_4VAR_GUAPOS_COMBINED_2\ALL"
    sample_size = None
    target_length = 64607
    variance_threshold = 0.95
    batch_size = 1000
    output_dir = r"C:\1. AI - ITACA\25.Project Proposals\PROJECT_C\PREDICTIONS\Combined3\PCA_Results_Optimun_Dataset2_None_095_Hyperparams2_v4"

    custom_hyperparams = {
        'logn': {
            'RandomForest': {
                'max_depth': None,
                'min_samples_leaf': 2,
                'min_samples_split': 2,
                'n_estimators': 500
            },
            'GradientBoosting': {
                'learning_rate': 0.03,
                'max_depth': 9,
                'n_estimators': 500,
                'subsample': 0.6
            },
            'LightGBM': {
                'learning_rate': 0.03,
                'max_depth': -1,
                'n_estimators': 500,
                'num_leaves': 31,
                'subsample': 0.6
            },
            'XGBoost': {
                'colsample_bytree': 0.6,
                'learning_rate': 0.03,
                'max_depth': 9,
                'n_estimators': 500,
                'subsample': 0.6
            }
        },
        'Tex': {   # 👈 aquí están los valores anteriores
            'RandomForest': {
                'max_depth': 10,
                'max_features': 'sqrt',
                'min_samples_leaf': 1,
                'min_samples_split': 2,
                'n_estimators': 800
            },
            'GradientBoosting': {
                'learning_rate': 0.02,
                'max_depth': 3,
                'n_estimators': 900,
                'subsample': 0.55
            },
            'LightGBM': {
                'learning_rate': 0.1,
                'max_depth': 3,
                'n_estimators': 900,
                'num_leaves': 20,
                'subsample': 0.4
            },
            'XGBoost': {
                'colsample_bytree': 0.5,
                'learning_rate': 0.05,
                'max_depth': 3,
                'n_estimators': 900,
                'subsample': 0.55
            }
        },
        'Vlos': {
            'RandomForest': {
                'max_depth': None,
                'min_samples_leaf': 1,
                'min_samples_split': 2,
                'n_estimators': 300
            },
            'GradientBoosting': {
                'learning_rate': 0.01,
                'max_depth': 11,
                'n_estimators': 500,
                'subsample': 0.6
            },
            'LightGBM': {
                'learning_rate': 0.1,
                'max_depth': 9,
                'n_estimators': 500,
                'num_leaves': 31,
                'subsample': 0.6
            },
            'XGBoost': {
                'colsample_bytree': 0.9,
                'learning_rate': 0.03,
                'max_depth': 9,
                'n_estimators': 500,
                'subsample': 0.6
            }
        },
        'FWHM': {
            'RandomForest': {
                'max_depth': 20,
                'min_samples_leaf': 1,
                'min_samples_split': 7,
                'n_estimators': 200
            },
            'GradientBoosting': {
                'learning_rate': 0.01,
                'max_depth': 9,
                'n_estimators': 500,
                'subsample': 0.8
            },
            'LightGBM': {
                'learning_rate': 0.05,
                'max_depth': 7,
                'n_estimators': 200,
                'num_leaves': 50,
                'subsample': 0.6
            },
            'XGBoost': {
                'colsample_bytree': 1.0,
                'learning_rate': 0.03,
                'max_depth': 11,
                'n_estimators': 500,
                'subsample': 0.6
            }
        }

    }

    # Run analysis
    results = analyze_large_spectra(
        directory,
        sample_size=sample_size,
        target_length=target_length,
        variance_threshold=variance_threshold,
        batch_size=batch_size,
        output_dir=output_dir,
        custom_hyperparams=custom_hyperparams
    )