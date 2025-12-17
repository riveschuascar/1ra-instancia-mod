"""
Validación Cruzada - Estudiante C
Sistema de Análisis de Futbolistas FIFA

Implementa:
- K-Fold Cross Validation (k=5)
- Stratified K-Fold para clasificación
- Análisis de estabilidad del modelo
- Comparación de métricas entre folds
"""

import numpy as np
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt

# ============================================================================
# K-FOLD CROSS VALIDATION
# ============================================================================

class KFoldCV:
    """K-Fold Cross Validation para regresión."""
    
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X, y=None):
        """Genera los índices para cada fold."""
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            np.random.shuffle(indices)
        
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            yield train_indices, test_indices
            current = stop

# ============================================================================
# STRATIFIED K-FOLD CROSS VALIDATION
# ============================================================================

class StratifiedKFoldCV:
    """Stratified K-Fold Cross Validation para clasificación."""
    
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X, y):
        """Genera los índices estratificados para cada fold."""
        n_samples = X.shape[0]
        
        # Obtener clases únicas y sus conteos
        classes, y_indices = np.unique(y, return_inverse=True)
        n_classes = len(classes)
        
        # Contar muestras por clase
        class_counts = np.bincount(y_indices)
        
        # Verificar que hay suficientes muestras por clase
        if np.min(class_counts) < self.n_splits:
            raise ValueError(
                f"El número mínimo de muestras por clase ({np.min(class_counts)}) "
                f"es menor que n_splits={self.n_splits}"
            )
        
        # Crear índices por clase
        class_indices = [np.where(y_indices == i)[0] for i in range(n_classes)]
        
        # Mezclar si es necesario
        if self.shuffle:
            if self.random_state is not None:
                np.random.seed(self.random_state)
            for indices in class_indices:
                np.random.shuffle(indices)
        
        # Dividir cada clase en folds
        class_folds = []
        for indices in class_indices:
            n_class_samples = len(indices)
            fold_sizes = np.full(self.n_splits, n_class_samples // self.n_splits, dtype=int)
            fold_sizes[:n_class_samples % self.n_splits] += 1
            
            current = 0
            folds = []
            for fold_size in fold_sizes:
                start, stop = current, current + fold_size
                folds.append(indices[start:stop])
                current = stop
            class_folds.append(folds)
        
        # Combinar folds de todas las clases
        for fold_idx in range(self.n_splits):
            test_indices = np.concatenate([
                class_folds[cls_idx][fold_idx]
                for cls_idx in range(n_classes)
            ])
            train_indices = np.concatenate([
                np.concatenate([class_folds[cls_idx][i] for i in range(self.n_splits) if i != fold_idx])
                for cls_idx in range(n_classes)
            ])
            
            yield train_indices, test_indices

# ============================================================================
# CROSS VALIDATION SCORER
# ============================================================================

class CrossValidationScorer:
    """Evalúa modelos usando validación cruzada."""
    
    def __init__(self, model_class, model_params, task='regression'):
        """
        Args:
            model_class: Clase del modelo a entrenar
            model_params: Diccionario de parámetros para inicializar el modelo
            task: 'regression' o 'classification'
        """
        self.model_class = model_class
        self.model_params = model_params
        self.task = task
        self.fold_results = []
    
    def evaluate_regression(self, X_train, X_test, y_train, y_test):
        """Evalúa un fold para regresión."""
        # Crear y entrenar modelo
        model = self.model_class(**self.model_params)
        model.fit(X_train, y_train, verbose=False)
        
        # Predicciones
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Métricas
        from metricas_evaluacion import RegressionMetrics
        
        metrics = {
            'train': {
                'MAE': RegressionMetrics.mae(y_train, y_pred_train),
                'RMSE': RegressionMetrics.rmse(y_train, y_pred_train),
                'R²': RegressionMetrics.r2_score(y_train, y_pred_train)
            },
            'test': {
                'MAE': RegressionMetrics.mae(y_test, y_pred_test),
                'RMSE': RegressionMetrics.rmse(y_test, y_pred_test),
                'R²': RegressionMetrics.r2_score(y_test, y_pred_test)
            }
        }
        
        return metrics, model
    
    def evaluate_classification(self, X_train, X_test, y_train, y_test):
        """Evalúa un fold para clasificación."""
        # Crear y entrenar modelo
        model = self.model_class(**self.model_params)
        
        # One-hot encoding para entrenamiento
        n_classes = len(np.unique(y_train))
        y_train_onehot = np.zeros((len(y_train), n_classes))
        y_train_onehot[np.arange(len(y_train)), y_train] = 1
        
        model.fit(X_train, y_train_onehot, verbose=False)
        
        # Predicciones
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Métricas
        from metricas_evaluacion import ClassificationMetrics
        
        precision_train, recall_train, f1_train, _, _, _ = \
            ClassificationMetrics.precision_recall_f1(y_train, y_pred_train)
        precision_test, recall_test, f1_test, _, _, _ = \
            ClassificationMetrics.precision_recall_f1(y_test, y_pred_test)
        
        metrics = {
            'train': {
                'Accuracy': ClassificationMetrics.accuracy(y_train, y_pred_train),
                'Precision': precision_train,
                'Recall': recall_train,
                'F1': f1_train
            },
            'test': {
                'Accuracy': ClassificationMetrics.accuracy(y_test, y_pred_test),
                'Precision': precision_test,
                'Recall': recall_test,
                'F1': f1_test
            }
        }
        
        return metrics, model
    
    def cross_validate(self, X, y, n_splits=5, stratified=True, verbose=True):
        """
        Realiza validación cruzada completa.
        
        Returns:
            Dict con resultados agregados de todos los folds
        """
        self.fold_results = []
        
        # Seleccionar tipo de CV
        if self.task == 'classification' and stratified:
            cv = StratifiedKFoldCV(n_splits=n_splits, shuffle=True, random_state=42)
            splits = cv.split(X, y)
        else:
            cv = KFoldCV(n_splits=n_splits, shuffle=True, random_state=42)
            splits = cv.split(X, y)
        
        # Evaluar cada fold
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            if verbose:
                print(f"\nEvaluando Fold {fold_idx + 1}/{n_splits}...")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            if self.task == 'regression':
                metrics, model = self.evaluate_regression(X_train, X_test, y_train, y_test)
            else:
                metrics, model = self.evaluate_classification(X_train, X_test, y_train, y_test)
            
            self.fold_results.append({
                'fold': fold_idx + 1,
                'metrics': metrics,
                'model': model,
                'train_size': len(train_idx),
                'test_size': len(test_idx)
            })
            
            if verbose:
                self._print_fold_results(metrics, fold_idx + 1)
        
        # Calcular estadísticas agregadas
        summary = self._compute_summary()
        
        if verbose:
            self._print_summary(summary)
        
        return summary
    
    def _print_fold_results(self, metrics, fold_num):
        """Imprime resultados de un fold."""
        print(f"\nResultados Fold {fold_num}:")
        print("-" * 50)
        
        if self.task == 'regression':
            print(f"  Train: MAE={metrics['train']['MAE']:.4f}, "
                  f"RMSE={metrics['train']['RMSE']:.4f}, "
                  f"R²={metrics['train']['R²']:.4f}")
            print(f"  Test:  MAE={metrics['test']['MAE']:.4f}, "
                  f"RMSE={metrics['test']['RMSE']:.4f}, "
                  f"R²={metrics['test']['R²']:.4f}")
        else:
            print(f"  Train: Acc={metrics['train']['Accuracy']:.4f}, "
                  f"F1={metrics['train']['F1']:.4f}")
            print(f"  Test:  Acc={metrics['test']['Accuracy']:.4f}, "
                  f"F1={metrics['test']['F1']:.4f}")
    
    def _compute_summary(self):
        """Calcula estadísticas agregadas de todos los folds."""
        summary = {
            'train': {},
            'test': {},
            'std_train': {},
            'std_test': {}
        }
        
        # Obtener nombres de métricas
        metric_names = list(self.fold_results[0]['metrics']['train'].keys())
        
        # Calcular media y desviación estándar para cada métrica
        for metric in metric_names:
            train_values = [fold['metrics']['train'][metric] for fold in self.fold_results]
            test_values = [fold['metrics']['test'][metric] for fold in self.fold_results]
            
            summary['train'][metric] = np.mean(train_values)
            summary['test'][metric] = np.mean(test_values)
            summary['std_train'][metric] = np.std(train_values)
            summary['std_test'][metric] = np.std(test_values)
        
        return summary
    
    def _print_summary(self, summary):
        """Imprime resumen de validación cruzada."""
        print("\n" + "="*70)
        print("RESUMEN DE VALIDACIÓN CRUZADA")
        print("="*70)
        
        print(f"\nNúmero de folds: {len(self.fold_results)}")
        print(f"Tamaño promedio de train: {np.mean([f['train_size'] for f in self.fold_results]):.0f}")
        print(f"Tamaño promedio de test: {np.mean([f['test_size'] for f in self.fold_results]):.0f}")
        
        print("\nMétricas Promedio (Media ± Desv. Estándar):")
        print("-"*70)
        
        for metric in summary['train'].keys():
            train_mean = summary['train'][metric]
            train_std = summary['std_train'][metric]
            test_mean = summary['test'][metric]
            test_std = summary['std_test'][metric]
            
            print(f"{metric:15s}: Train = {train_mean:7.4f} ± {train_std:.4f}  |  "
                  f"Test = {test_mean:7.4f} ± {test_std:.4f}")
        
        print("="*70)
    
    def plot_cv_results(self, save_path=None):
        """Visualiza los resultados de validación cruzada."""
        if not self.fold_results:
            print("No hay resultados para graficar. Ejecuta cross_validate primero.")
            return
        
        metric_names = list(self.fold_results[0]['metrics']['train'].keys())
        n_metrics = len(metric_names)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        fold_numbers = [f['fold'] for f in self.fold_results]
        
        for idx, metric in enumerate(metric_names):
            train_values = [f['metrics']['train'][metric] for f in self.fold_results]
            test_values = [f['metrics']['test'][metric] for f in self.fold_results]
            
            x = np.arange(len(fold_numbers))
            width = 0.35
            
            axes[idx].bar(x - width/2, train_values, width, label='Train', 
                         alpha=0.8, color='steelblue')
            axes[idx].bar(x + width/2, test_values, width, label='Test', 
                         alpha=0.8, color='coral')
            
            axes[idx].set_xlabel('Fold', fontsize=12)
            axes[idx].set_ylabel(metric, fontsize=12)
            axes[idx].set_title(f'{metric} por Fold', fontsize=14, fontweight='bold')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(fold_numbers)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nGráfico guardado en: {save_path}")
        
        plt.show()

# ============================================================================
# GRID SEARCH CON CROSS VALIDATION
# ============================================================================

class GridSearchCV:
    """Grid Search con validación cruzada para encontrar mejores hiperparámetros."""
    
    def __init__(self, model_class, param_grid, task='regression', cv=5):
        """
        Args:
            model_class: Clase del modelo
            param_grid: Diccionario con parámetros a probar
            task: 'regression' o 'classification'
            cv: Número de folds
        """
        self.model_class = model_class
        self.param_grid = param_grid
        self.task = task
        self.cv = cv
        self.results = []
        self.best_params = None
        self.best_score = None
    
    def _generate_param_combinations(self):
        """Genera todas las combinaciones de parámetros."""
        import itertools
        
        keys = list(self.param_grid.keys())
        values = [self.param_grid[key] for key in keys]
        
        for combination in itertools.product(*values):
            yield dict(zip(keys, combination))
    
    def fit(self, X, y, verbose=True):
        """Busca los mejores hiperparámetros."""
        param_combinations = list(self._generate_param_combinations())
        n_combinations = len(param_combinations)
        
        if verbose:
            print(f"\nProbando {n_combinations} combinaciones de hiperparámetros...")
        
        for idx, params in enumerate(param_combinations):
            if verbose:
                print(f"\nCombinación {idx+1}/{n_combinations}: {params}")
            
            # Crear scorer con estos parámetros
            scorer = CrossValidationScorer(
                self.model_class,
                params,
                task=self.task
            )
            
            # Evaluar con CV
            summary = scorer.cross_validate(X, y, n_splits=self.cv, 
                                          stratified=(self.task=='classification'),
                                          verbose=False)
            
            # Guardar resultados
            score_metric = 'R²' if self.task == 'regression' else 'Accuracy'
            score = summary['test'][score_metric]
            
            self.results.append({
                'params': params,
                'score': score,
                'summary': summary
            })
            
            if verbose:
                print(f"  Score ({score_metric}): {score:.4f}")
            
            # Actualizar mejor score
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.best_params = params
        
        if verbose:
            print("\n" + "="*70)
            print("MEJORES HIPERPARÁMETROS ENCONTRADOS")
            print("="*70)
            print(f"Parámetros: {self.best_params}")
            print(f"Score: {self.best_score:.4f}")
            print("="*70)
        
        return self.best_params

if __name__ == "__main__":
    print("Módulo de Validación Cruzada - Día 2")
    print("="*60)
    print("\nFuncionalidades implementadas:")
    print("✓ K-Fold Cross Validation")
    print("✓ Stratified K-Fold para clasificación")
    print("✓ Grid Search con CV")
    print("✓ Análisis de estabilidad del modelo")
    print("✓ Visualización de resultados por fold")