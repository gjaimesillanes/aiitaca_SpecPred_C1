import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from scipy.interpolate import interp1d
import pickle
from umap import UMAP
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import re

#-----------------------------------------------------------
#METHOD A - AI-ITACA Project - Chapter I
#-----------------------------------------------------------
#PCA and UMAP Analysis for Spectral Data and Prediction
#This code performs PCA and UMAP dimensionality reduction on large spectral datasets, interpolates spectra, and enables prediction and visualization of new spectra using trained models. It processes input files, extracts physical parameters, and generates visualizations for both training and prediction results.
#- INPUT: Directory containing spectral .txt files with headers including physical parameters.
#- Custom hyperparameters: directory, sample_size (number of spectra), random_seed, output_dir, target_length (for interpolation), variance_threshold (for PCA reduction), batch_size, n_neighbors (for UMAP)
#- OUTPUT: PCA and UMAP models, reference frequencies, physical parameters array, parameter names, valid files list, corrupted files list, prediction results, and visualization plots.


def extract_molecule_formula(header):
    pattern = r"molecules=['\"]([^,'\"]+)"
    match = re.search(pattern, header)
    if match:
        formula = match.group(1)
        if ',' in formula:
            formula = formula.split(',')[0]
        return formula
    return "Unknown"

def process_single_spectrum(filepath, reference_frequencies, target_length=64607):
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        if len(lines) < 2:
            raise ValueError("Empty file")
        header = lines[0].strip()
        param_dict = {}
        formula = extract_molecule_formula(header)
        filename = os.path.basename(filepath)
        for part in header.split():
            if '=' in part:
                try:
                    key, value = part.split('=')
                    key = key.strip()
                    value = value.strip("'")
                    if key in ['molecules', 'sourcesize']:
                        continue
                    try:
                        param_dict[key] = float(value)
                    except ValueError:
                        param_dict[key] = value
                except:
                    continue
        spectrum_data = []
        for line in lines[1:]:
            try:
                parts = line.strip().split()
                if len(parts) >= 2:
                    freq = float(parts[0])
                    intensity = float(parts[1])
                    if np.isfinite(freq) and np.isfinite(intensity):
                        spectrum_data.append([freq, intensity])
            except:
                continue
        if not spectrum_data:
            raise ValueError("No valid data points")
        spectrum_data = np.array(spectrum_data)
        interpolator = interp1d(spectrum_data[:, 0], spectrum_data[:, 1], 
                             kind='linear', bounds_error=False, fill_value=0.0)
        interpolated = interpolator(reference_frequencies)
        if not np.all(np.isfinite(interpolated)):
            raise ValueError("Invalid values after interpolation")
        params = [
            param_dict.get('logn', np.nan),
            param_dict.get('tex', np.nan),
            param_dict.get('velo', np.nan),
            param_dict.get('fwhm', np.nan)
        ]
        return interpolated, formula, params, filename
    except Exception as e:
        raise ValueError(f"Error processing file {filepath}: {str(e)}")

def safe_umap_transform(umap_model, training_data, new_data):
    combined_data = np.vstack([training_data, new_data])
    combined_embedding = umap_model.transform(combined_data)
    return combined_embedding[len(training_data):]

def predict_new_spectra(model_path, new_spectra_dir, output_dir="Predictions_Results_2"):
    # SECTION 1: Prediction and Visualization
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    os.makedirs(output_dir, exist_ok=True)
    new_files = [f for f in os.listdir(new_spectra_dir) if f.endswith(".txt")]
    new_spectra = []
    new_formulas = []
    new_params = []
    new_filenames = []
    for filename in tqdm(new_files):
        filepath = os.path.join(new_spectra_dir, filename)
        try:
            spectrum, formula, params, _ = process_single_spectrum(
                filepath, model['reference_frequencies'], model['target_length']
            )
            new_spectra.append(spectrum)
            new_formulas.append(formula)
            new_params.append(params)
            new_filenames.append(filename)
        except Exception as e:
            continue
    if not new_spectra:
        return
    X_new = np.array(new_spectra)
    y_new = np.array(new_params)
    formulas_new = np.array(new_formulas)
    filenames_new = np.array(new_filenames)
    valid_indices = ~np.isnan(y_new).any(axis=1)
    X_new = X_new[valid_indices]
    y_new = y_new[valid_indices]
    formulas_new = formulas_new[valid_indices]
    filenames_new = filenames_new[valid_indices]
    X_new_scaled = model['scaler'].transform(X_new)
    pca_components_new = model['pca'].transform(X_new_scaled)
    if 'X_pca_train' in model:
        umap_embedding_new = safe_umap_transform(
            model['umap'], 
            model['X_pca_train'], 
            pca_components_new
        )
    else:
        umap_embedding_new = model['umap'].transform(pca_components_new)
    predictions = {
        'X_new': X_new,
        'y_new': y_new,
        'formulas_new': formulas_new,
        'filenames_new': filenames_new,
        'pca_components_new': pca_components_new,
        'umap_embedding_new': umap_embedding_new,
        'model_info': {
            'model_path': model_path,
            'training_samples': model['sample_size'],
            'n_components': model['n_components']
        }
    }
    predictions_path = os.path.join(output_dir, "predictions_results.pkl")
    with open(predictions_path, 'wb') as f:
        pickle.dump(predictions, f)
    create_prediction_visualizations(
        model['embedding'], model['y'], model['formulas'],
        umap_embedding_new, y_new, formulas_new,
        model['reference_frequencies'], model['pca'].components_,
        model['cumulative_variance'], model['n_components'],
        output_dir
    )
    return predictions

def create_prediction_visualizations(embedding_train, y_train, formulas_train,
                                   embedding_new, y_new, formulas_new,
                                   reference_frequencies, pca_components,
                                   cumulative_variance, n_components,
                                   output_dir):
    # SECTION 2: Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'b-o')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.axvline(x=n_components, color='r', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Cumulative Explained Variance (Training)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "pca_cumulative_variance.png"))
    plt.close()
    plt.figure(figsize=(15, 10))
    for i in range(min(5, n_components)):
        plt.plot(reference_frequencies, pca_components[i], label=f'PC {i+1}')
    plt.xlabel('Frequency')
    plt.ylabel('Component Value')
    plt.title('First 5 Principal Components (Training)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "pca_components.png"))
    plt.close()
    param_names = ['logn', 'tex', 'velo', 'fwhm']
    param_labels = ['log(n)', 'T_ex (K)', 'Velocity (km/s)', 'FWHM (km/s)']
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.ravel()
    for i, (ax, param_name, param_label) in enumerate(zip(axes, param_names, param_labels)):
        sc_train = ax.scatter(embedding_train[:, 0], embedding_train[:, 1], 
                             c=y_train[:, i], cmap='viridis', alpha=0.4, s=8, label='Training')
        sc_new = ax.scatter(embedding_new[:, 0], embedding_new[:, 1], 
                           c=y_new[:, i], cmap='plasma', alpha=1.0, s=100, 
                           marker='X', edgecolors='red', linewidth=2, label='New Predictions')
        cbar = plt.colorbar(sc_train, ax=ax)
        cbar.set_label(param_label)
        ax.set_title(f'UMAP: {param_name} (Training + Predictions)')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.grid(True)
        ax.legend()
        for j in range(len(embedding_new)):
            ax.annotate(formulas_new[j], (embedding_new[j, 0], embedding_new[j, 1]),
                       fontsize=8, alpha=0.9, xytext=(5, 5), textcoords='offset points',
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "umap_predictions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    all_formulas = np.concatenate([formulas_train, formulas_new])
    unique_formulas = np.unique(all_formulas)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_formulas)))
    formula_to_color = {formula: color for formula, color in zip(unique_formulas, colors)}
    plt.figure(figsize=(14, 10))
    for formula in np.unique(formulas_train):
        mask = formulas_train == formula
        color = formula_to_color[formula]
        plt.scatter(embedding_train[mask, 0], embedding_train[mask, 1], 
                   color=color, alpha=0.4, s=15, label=f'{formula} (Train)')
    for formula in np.unique(formulas_new):
        mask = formulas_new == formula
        color = formula_to_color[formula]
        plt.scatter(embedding_new[mask, 0], embedding_new[mask, 1], 
                   color=color, marker='*', s=200, edgecolors='black', linewidth=2,
                   alpha=1.0, label=f'{formula} (New)')
    plt.title('UMAP: Molecular Formula (Training + Predictions)')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "umap_formula_predictions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    prediction_coords = []
    for i in range(len(embedding_new)):
        prediction_coords.append({
            'filename': filenames_new[i],
            'formula': formulas_new[i],
            'umap_x': embedding_new[i, 0],
            'umap_y': embedding_new[i, 1],
            'pca_components': pca_components_new[i].tolist(),
            'parameters': {
                'logn': y_new[i, 0],
                'tex': y_new[i, 1],
                'velo': y_new[i, 2],
                'fwhm': y_new[i, 3]
            }
        })
    coords_path = os.path.join(output_dir, "prediction_coordinates.csv")
    df_coords = pd.DataFrame(prediction_coords)
    df_coords.to_csv(coords_path, index=False)

def analyze_and_save_pca(directory, sample_size=None, random_seed=42, output_dir="PCA_Results_GUAPOS_Combined_2", 
                        target_length=64607, variance_threshold=0.95):
    # SECTION 3: PCA and UMAP Analysis
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    random.seed(random_seed)
    np.random.seed(random_seed)
    all_files = [f for f in os.listdir(directory) if f.endswith(".txt")]
    if sample_size is not None and sample_size < len(all_files):
        selected_files = random.sample(all_files, sample_size)
    else:
        selected_files = all_files
    data = []
    params = []
    formulas_list = []
    filenames_list = []
    reference_frequencies = None
    corrupted_files = []
    for filename in tqdm(selected_files):
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            if len(lines) < 2:
                corrupted_files.append((filename, "Empty file"))
                continue
            header = lines[0].strip()
            param_dict = {}
            formula = extract_molecule_formula(header)
            formulas_list.append(formula)
            filenames_list.append(filename)
            for part in header.split():
                if '=' in part:
                    try:
                        key, value = part.split('=')
                        key = key.strip()
                        value = value.strip("'")
                        if key in ['molecules', 'sourcesize']:
                            continue
                        try:
                            param_dict[key] = float(value)
                        except ValueError:
                            param_dict[key] = value
                    except:
                        continue
            spectrum_data = []
            for line in lines[1:]:
                try:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        freq = float(parts[0])
                        intensity = float(parts[1])
                        if np.isfinite(freq) and np.isfinite(intensity):
                            spectrum_data.append([freq, intensity])
                except:
                    continue
            if not spectrum_data:
                corrupted_files.append((filename, "No valid data points"))
                continue
            spectrum_data = np.array(spectrum_data)
            if reference_frequencies is None:
                min_freq = np.min(spectrum_data[:, 0])
                max_freq = np.max(spectrum_data[:, 0])
                reference_frequencies = np.linspace(min_freq, max_freq, target_length)
            interpolator = interp1d(spectrum_data[:, 0], spectrum_data[:, 1], 
                                 kind='linear', bounds_error=False, fill_value=0.0)
            interpolated = interpolator(reference_frequencies)
            if not np.all(np.isfinite(interpolated)):
                corrupted_files.append((filename, "Invalid values after interpolation"))
                continue
            data.append(interpolated)
            params.append([
                param_dict.get('logn', np.nan),
                param_dict.get('tex', np.nan),
                param_dict.get('velo', np.nan),
                param_dict.get('fwhm', np.nan)
            ])
        except Exception as e:
            corrupted_files.append((filename, f"Error: {str(e)}"))
            continue
    X = np.array(data)
    y = np.array(params)
    formulas_arr = np.array(formulas_list)
    filenames_arr = np.array(filenames_list)
    valid_indices = ~np.isnan(y).any(axis=1)
    X = X[valid_indices]
    y = y[valid_indices]
    formulas_arr = formulas_arr[valid_indices]
    filenames_arr = filenames_arr[valid_indices]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    batch_size = 500
    pca_full = IncrementalPCA(batch_size=batch_size)
    pca_full.fit(X_scaled)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'b-o')
    plt.axhline(y=variance_threshold, color='r', linestyle='--')
    plt.axvline(x=n_components, color='r', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Cumulative Explained Variance')
    plt.grid(True)
    cumulative_var_path = os.path.join(output_dir, "pca_cumulative_variance.png")
    plt.savefig(cumulative_var_path)
    plt.close()
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)
    X_pca_train = pca.transform(X_scaled)
    plt.figure(figsize=(15, 10))
    for i in range(min(5, n_components)):
        plt.plot(reference_frequencies, pca.components_[i], label=f'PC {i+1}')
    plt.xlabel('Frequency')
    plt.ylabel('Component Value')
    plt.title('First 5 Principal Components')
    plt.legend()
    plt.grid(True)
    pca_components_path = os.path.join(output_dir, "pca_components.png")
    plt.savefig(pca_components_path)
    plt.close()
    reducer = UMAP(
        n_neighbors=UMAP_CONFIG['n_neighbors'],
        min_dist=UMAP_CONFIG['min_dist'],
        n_components=UMAP_CONFIG['n_components'],
        metric=UMAP_CONFIG['metric'],
        random_state=UMAP_CONFIG['random_state']
    )
    embedding = reducer.fit_transform(principal_components)
    create_visualizations(embedding, y, formulas_arr, reference_frequencies, 
                         pca.components_, cumulative_variance, n_components,
                         output_dir, random_seed)
    model_name = f"pca_umap_model_ss{sample_size}_tl{target_length}_vt{variance_threshold}.pkl"
    model_path = os.path.join(output_dir, model_name)
    with open(model_path, 'wb') as f:
        pickle.dump({
            'X': principal_components,
            'X_pca_train': X_pca_train,
            'y': y,
            'formulas': formulas_arr,
            'filenames': filenames_arr,
            'pca': pca,
            'scaler': scaler,
            'umap': reducer,
            'params': ['logn', 'tex', 'velo', 'fwhm'],
            'variance_threshold': variance_threshold,
            'sample_size': sample_size,
            'target_length': target_length,
            'cumulative_variance': cumulative_variance,
            'reference_frequencies': reference_frequencies,
            'n_components': n_components,
            'embedding': embedding
        }, f)
    return model_path

def create_visualizations(embedding, y, formulas_arr, reference_frequencies, 
                         pca_components, cumulative_variance, n_components,
                         output_dir, random_seed):
    """Create all visualization plots"""
    
    # 1. PCA cumulative variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'b-o')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.axvline(x=n_components, color='r', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Cumulative Explained Variance')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "pca_cumulative_variance.png"))
    plt.close()
    
    # 2. PCA components
    plt.figure(figsize=(15, 10))
    for i in range(min(5, n_components)):
        plt.plot(reference_frequencies, pca_components[i], label=f'PC {i+1}')
    plt.xlabel('Frequency')
    plt.ylabel('Component Value')
    plt.title('First 5 Principal Components')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "pca_components.png"))
    plt.close()
    
    # 3. UMAP colored by parameters
    param_names = ['logn', 'tex', 'velo', 'fwhm']
    param_labels = ['log(n)', 'T_ex (K)', 'Velocity (km/s)', 'FWHM (km/s)']
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.ravel()
    
    for i, (ax, param_name, param_label) in enumerate(zip(axes, param_names, param_labels)):
        param_values = y[:, i]
        
        sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=param_values,
                       cmap='viridis', alpha=0.6, s=10)
        
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(param_label)
        
        ax.set_title(f'UMAP projection colored by {param_name}')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.grid(True)
        
        # Add formula labels for 10% random points
        n_samples = len(embedding)
        n_labels = max(1, int(n_samples * 0.1))
        random_indices = np.random.choice(n_samples, n_labels, replace=False)
        
        for idx in random_indices:
            ax.annotate(formulas_arr[idx], (embedding[idx, 0], embedding[idx, 1]),
                       fontsize=6, alpha=0.7, xytext=(2, 2), textcoords='offset points',
                       bbox=dict(boxstyle="round,pad=0.1", facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "umap_visualization.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 3D UMAP plots
    for i, (param_name, param_label) in enumerate(zip(param_names, param_labels)):
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        param_values = y[:, i]
        
        sc = ax.scatter(embedding[:, 0], embedding[:, 1], param_values,
                       c=param_values, cmap='viridis', alpha=0.6, s=20, depthshade=True)
        
        cbar = plt.colorbar(sc, ax=ax, shrink=0.5)
        cbar.set_label(param_label)
        
        ax.set_title(f'3D UMAP: {param_name} as Z-axis')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_zlabel(param_label)
        
        # Add formula labels
        n_samples = len(embedding)
        n_labels = max(1, int(n_samples * 0.1))
        random_indices = np.random.choice(n_samples, n_labels, replace=False)
        
        for idx in random_indices:
            ax.text(embedding[idx, 0], embedding[idx, 1], param_values[idx],
                   formulas_arr[idx], fontsize=6, alpha=0.8,
                   bbox=dict(boxstyle="round,pad=0.1", facecolor='yellow', alpha=0.5))
        
        ax.view_init(elev=20, azim=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"umap_3d_{param_name}.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 5. UMAP by formula - MODIFICADO: colores Ãºnicos para cada molÃ©cula
    unique_formulas = np.unique(formulas_arr)
    
    # Crear un mapa de colores Ãºnico para cada fÃ³rmula
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_formulas)))
    formula_to_color = {formula: color for formula, color in zip(unique_formulas, colors)}
    
    plt.figure(figsize=(12, 10))
    
    for formula in unique_formulas:
        mask = formulas_arr == formula
        color = formula_to_color[formula]
        plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                   color=color, label=formula, alpha=0.7, s=15)
    
    plt.title('UMAP projection colored by molecular formula')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "umap_by_formula.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":

    # =============================
    UMAP_CONFIG = {
        'n_neighbors': 15,         
        'min_dist': 1,          
        'n_components': 3,         
        'metric': 'euclidean',     
        'random_state': 42         
    }

    directory = r"/lustre/scratch-global/cab/gjaimes/2.DATASETS/CLEANED_SPECTRA_4VAR_GUAPOS_COMBINED_2/ALL"
    sample_size = 25000
    target_length = 64607
    variance_threshold = 0.95

    # Train the model
    model_path = analyze_and_save_pca(
        directory=directory,
        sample_size=sample_size,
        target_length=target_length,
        variance_threshold=variance_threshold,
        output_dir=r"/lustre/scratch-global/cab/gjaimes/2.DATASETS/PCA_Location_3D_Dataset2_25000_095_GENERIC_h11"
    )


