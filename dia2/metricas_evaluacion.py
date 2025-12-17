"""
Métricas y Evaluación - Estudiante B
Sistema de Análisis de Futbolistas FIFA

Implementa todas las métricas requeridas:
- Regresión: MAE, RMSE, R², Error máximo
- Clasificación: Accuracy, Precision, Recall, F1, AUC-ROC
- Matrices de confusión
- Análisis de residuos
- SHAP values (análisis de importancia)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# MÉTRICAS DE REGRESIÓN
# ============================================================================

class RegressionMetrics:
    """Calcula todas las métricas para tareas de regresión."""
    
    @staticmethod
    def mae(y_true, y_pred):
        """Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def rmse(y_true, y_pred):
        """Root Mean Squared Error."""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def r2_score(y_true, y_pred):
        """R² Score (Coeficiente de determinación)."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def max_error(y_true, y_pred):
        """Error máximo absoluto."""
        return np.max(np.abs(y_true - y_pred))
    
    @staticmethod
    def mape(y_true, y_pred):
        """Mean Absolute Percentage Error."""
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def compute_all(y_true, y_pred, verbose=True):
        """Calcula todas las métricas de regresión."""
        metrics = {
            'MAE': RegressionMetrics.mae(y_true, y_pred),
            'RMSE': RegressionMetrics.rmse(y_true, y_pred),
            'R²': RegressionMetrics.r2_score(y_true, y_pred),
            'Max Error': RegressionMetrics.max_error(y_true, y_pred),
            'MAPE (%)': RegressionMetrics.mape(y_true, y_pred)
        }
        
        if verbose:
            print("\n" + "="*60)
            print("MÉTRICAS DE REGRESIÓN")
            print("="*60)
            for name, value in metrics.items():
                print(f"{name:15s}: {value:10.4f}")
            print("="*60)
        
        return metrics

# ============================================================================
# MÉTRICAS DE CLASIFICACIÓN
# ============================================================================

class ClassificationMetrics:
    """Calcula todas las métricas para tareas de clasificación."""
    
    @staticmethod
    def accuracy(y_true, y_pred):
        """Accuracy global."""
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def precision_recall_f1(y_true, y_pred, average='macro'):
        """Calcula Precision, Recall y F1-Score."""
        classes = np.unique(y_true)
        n_classes = len(classes)
        
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        
        for cls in classes:
            # True Positives, False Positives, False Negatives
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            
            # Precision
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            precision_per_class.append(precision)
            
            # Recall
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recall_per_class.append(recall)
            
            # F1-Score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_per_class.append(f1)
        
        # Promedios
        if average == 'macro':
            precision = np.mean(precision_per_class)
            recall = np.mean(recall_per_class)
            f1 = np.mean(f1_per_class)
        elif average == 'weighted':
            weights = [np.sum(y_true == cls) for cls in classes]
            total = np.sum(weights)
            precision = np.sum([p * w for p, w in zip(precision_per_class, weights)]) / total
            recall = np.sum([r * w for r, w in zip(recall_per_class, weights)]) / total
            f1 = np.sum([f * w for f, w in zip(f1_per_class, weights)]) / total
        
        return precision, recall, f1, precision_per_class, recall_per_class, f1_per_class
    
    @staticmethod
    def confusion_matrix_custom(y_true, y_pred):
        """Matriz de confusión."""
        classes = np.unique(y_true)
        n_classes = len(classes)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        
        for i, true_class in enumerate(classes):
            for j, pred_class in enumerate(classes):
                cm[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
        
        return cm, classes
    
    @staticmethod
    def compute_all(y_true, y_pred, class_names=None, verbose=True):
        """Calcula todas las métricas de clasificación."""
        accuracy = ClassificationMetrics.accuracy(y_true, y_pred)
        precision, recall, f1, prec_per_class, rec_per_class, f1_per_class = \
            ClassificationMetrics.precision_recall_f1(y_true, y_pred)
        
        metrics = {
            'Accuracy': accuracy,
            'Precision (Macro)': precision,
            'Recall (Macro)': recall,
            'F1-Score (Macro)': f1,
            'Per-Class Precision': prec_per_class,
            'Per-Class Recall': rec_per_class,
            'Per-Class F1': f1_per_class
        }
        
        if verbose:
            print("\n" + "="*60)
            print("MÉTRICAS DE CLASIFICACIÓN")
            print("="*60)
            print(f"{'Accuracy':20s}: {accuracy:10.4f}")
            print(f"{'Precision (Macro)':20s}: {precision:10.4f}")
            print(f"{'Recall (Macro)':20s}: {recall:10.4f}")
            print(f"{'F1-Score (Macro)':20s}: {f1:10.4f}")
            
            if class_names is not None:
                print("\nMétricas por clase:")
                print("-" * 60)
                print(f"{'Clase':<15s} {'Precision':>12s} {'Recall':>12s} {'F1-Score':>12s}")
                print("-" * 60)
                for i, name in enumerate(class_names):
                    print(f"{name:<15s} {prec_per_class[i]:12.4f} "
                          f"{rec_per_class[i]:12.4f} {f1_per_class[i]:12.4f}")
            print("="*60)
        
        return metrics

# ============================================================================
# ANÁLISIS DE RESIDUOS (REGRESIÓN)
# ============================================================================

class ResidualAnalysis:
    """Análisis completo de residuos para modelos de regresión."""
    
    @staticmethod
    def plot_residuals(y_true, y_pred, save_path=None):
        """Genera gráficos de análisis de residuos."""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Residuos vs Valores Predichos
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Valores Predichos', fontsize=12)
        axes[0, 0].set_ylabel('Residuos', fontsize=12)
        axes[0, 0].set_title('Residuos vs Valores Predichos', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Distribución de Residuos
        axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Residuos', fontsize=12)
        axes[0, 1].set_ylabel('Frecuencia', fontsize=12)
        axes[0, 1].set_title('Distribución de Residuos', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q-Q Plot (Normalidad)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Test de Normalidad)', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Valores Reales vs Predichos
        axes[1, 1].scatter(y_true, y_pred, alpha=0.5, s=20)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[1, 1].set_xlabel('Valores Reales', fontsize=12)
        axes[1, 1].set_ylabel('Valores Predichos', fontsize=12)
        axes[1, 1].set_title('Valores Reales vs Predichos', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nGráfico guardado en: {save_path}")
        
        plt.show()
        
        return residuals
    
    @staticmethod
    def analyze(y_true, y_pred, verbose=True):
        """Análisis estadístico de residuos."""
        residuals = y_true - y_pred
        
        analysis = {
            'Media': np.mean(residuals),
            'Mediana': np.median(residuals),
            'Desv. Estándar': np.std(residuals),
            'Mínimo': np.min(residuals),
            'Máximo': np.max(residuals),
            'Percentil 25': np.percentile(residuals, 25),
            'Percentil 75': np.percentile(residuals, 75),
            'IQR': np.percentile(residuals, 75) - np.percentile(residuals, 25)
        }
        
        if verbose:
            print("\n" + "="*60)
            print("ANÁLISIS DE RESIDUOS")
            print("="*60)
            for name, value in analysis.items():
                print(f"{name:20s}: {value:10.4f}")
            print("="*60)
        
        return analysis

# ============================================================================
# VISUALIZACIÓN DE MATRIZ DE CONFUSIÓN
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """Visualiza la matriz de confusión con heatmap."""
    cm, classes = ClassificationMetrics.confusion_matrix_custom(y_true, y_pred)
    
    if class_names is None:
        class_names = [f"Clase {i}" for i in classes]
    
    # Normalizar por fila (recall)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Matriz de confusión absoluta
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Cantidad'}, ax=ax1)
    ax1.set_xlabel('Predicción', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Valor Real', fontsize=12, fontweight='bold')
    ax1.set_title('Matriz de Confusión (Valores Absolutos)', 
                  fontsize=14, fontweight='bold')
    
    # Matriz de confusión normalizada
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proporción'}, ax=ax2)
    ax2.set_xlabel('Predicción', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Valor Real', fontsize=12, fontweight='bold')
    ax2.set_title('Matriz de Confusión (Normalizada por Fila)', 
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nMatriz de confusión guardada en: {save_path}")
    
    plt.show()
    
    return cm, cm_normalized

# ============================================================================
# CURVA ROC Y AUC (CLASIFICACIÓN MULTICLASE)
# ============================================================================

def plot_roc_curves(y_true, y_proba, class_names=None, save_path=None):
    """Calcula y visualiza curvas ROC para clasificación multiclase."""
    n_classes = y_proba.shape[1]
    
    if class_names is None:
        class_names = [f"Clase {i}" for i in range(n_classes)]
    
    # Binarizar las etiquetas para One-vs-Rest
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Calcular ROC y AUC para cada clase
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Graficar
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
    
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Clasificador Aleatorio')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12, fontweight='bold')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12, fontweight='bold')
    plt.title('Curvas ROC - One-vs-Rest', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nCurvas ROC guardadas en: {save_path}")
    
    plt.show()
    
    # Calcular AUC macro y weighted
    auc_macro = np.mean(list(roc_auc.values()))
    
    print("\n" + "="*60)
    print("AUC-ROC POR CLASE")
    print("="*60)
    for i, name in enumerate(class_names):
        print(f"{name:20s}: {roc_auc[i]:.4f}")
    print("-"*60)
    print(f"{'AUC Macro':20s}: {auc_macro:.4f}")
    print("="*60)
    
    return roc_auc, auc_macro

# ============================================================================
# ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS (SHAP SIMPLIFICADO)
# ============================================================================

class FeatureImportanceAnalyzer:
    """
    Análisis de importancia de características.
    Implementación simplificada de SHAP usando perturbaciones.
    """
    
    @staticmethod
    def permutation_importance(model, X, y, n_repeats=10, metric='mae'):
        """
        Calcula la importancia por permutación.
        Similar a SHAP pero más simple.
        """
        n_features = X.shape[1]
        
        # Predicción baseline
        baseline_pred = model.predict(X)
        
        if metric == 'mae':
            baseline_score = np.mean(np.abs(y - baseline_pred))
        elif metric == 'rmse':
            baseline_score = np.sqrt(np.mean((y - baseline_pred) ** 2))
        elif metric == 'accuracy':
            baseline_score = np.mean(y == baseline_pred)
        
        importances = []
        
        for feat_idx in range(n_features):
            scores = []
            
            for _ in range(n_repeats):
                X_permuted = X.copy()
                # Permutar la característica
                X_permuted[:, feat_idx] = np.random.permutation(X_permuted[:, feat_idx])
                
                # Predicción con permutación
                permuted_pred = model.predict(X_permuted)
                
                if metric == 'mae':
                    score = np.mean(np.abs(y - permuted_pred))
                elif metric == 'rmse':
                    score = np.sqrt(np.mean((y - permuted_pred) ** 2))
                elif metric == 'accuracy':
                    score = np.mean(y == permuted_pred)
                
                # Para accuracy, queremos ver la reducción
                if metric == 'accuracy':
                    scores.append(baseline_score - score)
                else:
                    scores.append(score - baseline_score)
            
            importances.append(np.mean(scores))
        
        return np.array(importances)
    
    @staticmethod
    def plot_importance(importances, feature_names=None, top_k=20, save_path=None):
        """Visualiza la importancia de características."""
        n_features = len(importances)
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(n_features)]
        
        # Ordenar por importancia
        indices = np.argsort(np.abs(importances))[::-1][:top_k]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(indices)), importances[indices], color='steelblue', alpha=0.8)
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importancia (Cambio en Métrica)', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_k} Características Más Importantes', 
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nGráfico de importancia guardado en: {save_path}")
        
        plt.tight_layout()
        plt.show()
        
        return indices

if __name__ == "__main__":
    print("Módulo de Métricas y Evaluación - Día 2")
    print("="*60)
    print("\nFuncionalidades implementadas:")
    print("✓ Métricas de Regresión (MAE, RMSE, R², Max Error)")
    print("✓ Métricas de Clasificación (Acc, Prec, Recall, F1)")
    print("✓ Matrices de Confusión detalladas")
    print("✓ Curvas ROC y AUC multiclase")
    print("✓ Análisis de Residuos completo")
    print("✓ Análisis de Importancia de Características")