
import numpy as np
import matplotlib.pyplot as plt


class Regresion:
    # constructor
    def __init__(self):
        # variable caracteristica
        self.__X = None
        # variable objetivo
        self.__y = None
        # data para test
        self.__X_testing = None 
        self.__y_testing = None
        # parametros del modelo
        self.__theta = None
        # historial para el descenso
        self.__history = None

    # metodo para cargar data
    def fit(self, x, y):
        m, n = x.shape
        # aniadir unidad de sesgo X0, columna de 1s
        self.__X = np.append(np.ones((m, 1)), x.reshape(m, -1), axis=1)
        # convertimos el vector y en matriz de mx1
        self.__y = y.reshape(-1, 1)
        # inicializamos parametros en 0
        self.__theta = np.zeros(n + 1)

    def split_test_stratified(self, test_size=0.2, bins=5, random_state=None):

        if self.__X is None or self.__y is None:
            raise ValueError("Primero cargue los datos con el método fit()")
        
        y = self.__y.flatten()
        m = len(y)
        test_samples = int(m * test_size)
        
        # Setear semilla
        if random_state is not None:
            np.random.seed(random_state)
        
        # Crear estratos basados en percentiles de y
        percentiles = np.linspace(0, 100, bins + 1)
        bins = np.percentile(y, percentiles)
        y_binned = np.digitize(y, bins[:-1])
        
        # Inicializar índices
        train_indices = []
        test_indices = []
        
        # Para cada estrato
        for stratum in np.unique(y_binned):
            stratum_indices = np.where(y_binned == stratum)[0]
            stratum_size = len(stratum_indices)
            stratum_test_size = int(stratum_size * test_size)
            
            np.random.shuffle(stratum_indices)
            
            test_indices.extend(stratum_indices[:stratum_test_size])
            train_indices.extend(stratum_indices[stratum_test_size:])
        
        # Mezclar los índices finales
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        
        # Crear splits
        self.__X_test = self.__X[test_indices]
        self.__y_test = self.__y[test_indices]
        self.__X = self.__X[train_indices]
        self.__y = self.__y[train_indices]

        
        # Asegurar dimensiones correctas
        self.__y = self.__y.reshape(-1, 1)
        self.__y_test = self.__y_test.reshape(-1, 1)
        

        
        print(f"División estratificada completada: {len(self.__y)} train, {len(self.__y_test)} test")
        print(f"Distribución en test: {np.histogram(self.__y_test, bins=bins)[0]}")
        # seleccion por correlacion

    def seleccion_por_correlacion(self, num, umbral=0.1):
        pass
        
    @property
    def get_X(self):
        return self.__X

    @property
    def get_y(self):
        return self.__y
    
    @property
    def get_X_test(self):
        return self.__X_testing

    @property
    def get_y_test(self):
        return self.__y_testing

    def get_param(self):
        return self.__theta

    # metodo para inicializar parametros
    def inicializar(self, t=None):
        m, n = self.__X.shape
        if t is None:
            self.__theta = np.zeros(n)
        else:
            self.__theta = t

    # normalizar datos
    def normalizar(self):
        u = self.__X[:, 1:].mean(0)
        desv = self.__X[:, 1:].std(0)
        self.__X[:, 1:] = (self.__X[:, 1:] - u) / desv

    # devolver funcion error: costo
    def get_j(self, theta):
        theta = theta.reshape(-1, 1)
        m = self.__X.shape[0]
        h = self.__X.dot(theta)
        error = h - self.__y
        j = 1 / (2 * m) * np.power(error, 2)
        return j.sum()

    # devolver gradiente
    def get_gradiente(self, theta):
        theta = theta.reshape(-1, 1)
        m = self.__X.shape[0]
        h = self.__X.dot(theta)
        error = h - self.__y
        t = 1 / m * self.__X.T.dot(error)
        return t.flatten()

    # implementacion del algoritmo del descenso de gradiente
    def descenso_de_gradiente(self, alpha, epsilon=10e-6, itera=None):
        js = []
        theta = self.__theta
        i = 0
        while True:
            js.append(self.get_j(theta))
            theta = theta - alpha * self.get_gradiente(theta)
            # si el costo actual menos costo anterior es menor a epsilon... fin
            if abs(self.get_j(theta) - js[-1]) < epsilon:
                break
            # si no... si iter no es none, verificamos si llegamo a la iteracion iter
            elif itera is not None:
                if i >= itera:
                    break
            i = i + 1
        print("Numero de iteraciones: ", i)
        print("Costo: ", js[-1])
        
        self.__theta = theta
        self.__history = np.array(js)
        print("Parametros: ", self.__theta)

    def descenso_gradiente_estocastico(self, alpha, epsilon=1e-6, max_iter=1000, random_state=None):
        """Implementación del descenso de gradiente estocástico (SGD)."""
        if random_state is not None:
            np.random.seed(random_state)

        m = self.__X.shape[0]
        js = []
        theta = self.__theta.copy()

        for epoch in range(max_iter):
            # Mezclar los datos en cada época
            indices = np.random.permutation(m)
            X_shuffled = self.__X[indices]
            y_shuffled = self.__y[indices]

            epoch_cost = 0

            for i in range(m):
                xi = X_shuffled[i].reshape(1, -1)  # Muestra i (1xn)
                yi = y_shuffled[i].reshape(1, 1)  # Etiqueta i (1x1)

                # Predicción y error para la muestra i
                error = xi.dot(theta) - yi
                grad = xi.T.dot(error).flatten()  # Gradiente para la muestra i

                # Actualizar theta
                theta -= alpha * grad

                # Calcular costo para la muestra i (opcional)
                epoch_cost += self.get_j(theta)

            js.append(epoch_cost / m)  # Costo promedio por época

            # Criterio de parada (opcional)
            if len(js) > 1 and abs(js[-1] - js[-2]) < epsilon:
                break

        self.__theta = theta
        self.__history = np.array(js)
        print(f"SGD completado en {epoch + 1} épocas. Costo final: {js[-1]:.4f}")

    def descenso_gradiente_mini_lotes(self, alpha, batch_size=32, epsilon=1e-6, max_iter=1000, random_state=None):
        """Implementación del descenso de gradiente por mini-lotes."""
        if random_state is not None:
            np.random.seed(random_state)

        m = self.__X.shape[0]
        js = []
        theta = self.__theta.copy()

        for epoch in range(max_iter):
            # Mezclar los datos en cada época
            indices = np.random.permutation(m)
            X_shuffled = self.__X[indices]
            y_shuffled = self.__y[indices]

            epoch_cost = 0

            for i in range(0, m, batch_size):
                # Obtener mini-lote
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Gradiente para el mini-lote
                error = X_batch.dot(theta) - y_batch
                grad = X_batch.T.dot(error).flatten() / len(X_batch)

                # Actualizar theta
                theta -= alpha * grad

                # Calcular costo para el mini-lote (opcional)
                epoch_cost += self.get_j(theta) * len(X_batch)

            js.append(epoch_cost / m)  # Costo promedio por época

            # Criterio de parada (opcional)
            if len(js) > 1 and abs(js[-1] - js[-2]) < epsilon:
                break

        self.__theta = theta
        self.__history = np.array(js)
        print(f"Mini-batch GD completado en {epoch + 1} épocas. Costo final: {js[-1]:.4f}")

    # implementacion de la ecuacion normal
    def get_ecu_norm(self):
        self.theta = (np.linalg.pinv(self.__X.T.dot(self.__X)).dot(self.__X.T).dot(self.__y))
        self.theta = self.theta.flatten()

    def predecir(self, x):
        if self.__theta is None:
            raise ValueError(
                "El modelo no ha sido entrenado. Ejecute descenso_de_gradiente() o get_ecu_norm() primero.")

        x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # Añadir columna de unos para el término de sesgo
        x = np.c_[np.ones(x.shape[0]), x]
        return x.dot(self.__theta)

    # Error medio cuadrado, mean square error (MSE)
    # útil para saber que tan cerca es la línea de ajuste de nuestra regresión a las observaciones
    def get_ECM(self, x=None, y=None):
        """Calcula el Error Cuadrático Medio (MSE) corregido."""
        if x is None or y is None:
            if hasattr(self, '_Regresion__X_test') and hasattr(self, '_Regresion__y_test'):
                x, y = self.__X_test, self.__y_test
            else:
                x, y = self.__X, self.__y

        # Asegurar que y es un array 2D (m, 1)
        y = y.reshape(-1, 1)

        # Predecir y asegurar dimensiones consistentes
        y_pred = x.dot(self.__theta).reshape(-1, 1)

        # Calcular MSE
        mse = np.mean((y_pred - y) ** 2)
        return mse

    def get_RECM(self, x=None, y=None):
        """Calcula la Raíz del Error Cuadrático Medio"""
        return np.sqrt(self.get_ECM(x, y))

    def get_r2(self, x=None, y=None):
        """Calcula el coeficiente de determinación R²"""
        if x is None or y is None:
            if self.__X_test is None or self.__y_test is None:
                x, y = self.__X, self.__y
            else:
                x, y = self.__X_test, self.__y_test

        y_pred = x.dot(self.__theta).reshape(-1, 1)  # Asegurar dimensión (m, 1)
        y = y.reshape(-1, 1)  # Asegurar dimensión (m, 1)

        ss_res = np.sum((y - y_pred) ** 2)  # Suma de cuadrados residual
        ss_tot = np.sum((y - np.mean(y)) ** 2)  # Suma de cuadrados total

        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0  # Evitar división por cero
        return r2

    def kfold_cross_validation(self, k=5, alpha=0.01, epsilon=1e-6, itera=100, random_state=None):
        if self.__X is None or self.__y is None:
            raise ValueError("Primero cargue los datos con el método fit()")

        if random_state is not None:
            np.random.seed(random_state)

        m = self.__X.shape[0]
        indices = np.arange(m)
        np.random.shuffle(indices)

        fold_size = m // k
        metrics = {
            'mse': np.zeros(k),
            'rmse': np.zeros(k),
            'r2': np.zeros(k)
        }

        for i in range(k):
            print(f"\nProcesando fold {i + 1}/{k}...")

            # Dividir en train y validation
            val_indices = indices[i * fold_size: (i + 1) * fold_size]
            train_indices = np.setdiff1d(indices, val_indices)

            X_train, y_train = self.__X[train_indices], self.__y[train_indices]
            X_val, y_val = self.__X[val_indices], self.__y[val_indices]

            # Guardar parámetros originales
            original_theta = self.__theta.copy()

            # Entrenar con el fold actual
            self.__X, self.__y = X_train, y_train
            self.descenso_de_gradiente(alpha=alpha, epsilon=epsilon, itera=itera)

            # Evaluar en el fold de validación
            y_pred = X_val.dot(self.__theta)

            # Calcular métricas
            mse = np.mean((y_pred - y_val) ** 2)
            rmse = np.sqrt(mse)
            ss_res = np.sum((y_val - y_pred) ** 2)
            ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
            r2 = 1 - (ss_res / ss_tot)

            metrics['mse'][i] = mse
            metrics['rmse'][i] = rmse
            metrics['r2'][i] = r2

            # Restaurar datos y parámetros originales
            self.__X = np.append(np.ones((m, 1)), self.__X[:, 1:], axis=1)
            self.__y = self.__y.reshape(-1, 1)
            self.__theta = original_theta

        # Calcular promedios
        avg_metrics = {key: np.mean(values) for key, values in metrics.items()}
        std_metrics = {f"{key}_std": np.std(values) for key, values in metrics.items()}

        print("\nResultados de K-Fold Cross-Validation:")
        print(f"MSE promedio: {avg_metrics['mse']:.4f} (±{std_metrics['mse_std']:.4f})")
        print(f"RMSE promedio: {avg_metrics['rmse']:.4f} (±{std_metrics['rmse_std']:.4f})")
        print(f"R² promedio: {avg_metrics['r2']:.4f} (±{std_metrics['r2_std']:.4f})")

        return {**avg_metrics, **std_metrics}
    
    def graficar_historial(self):
        fig1 = plt.figure()
        plt.plot(range(self.__history.size), self.__history)
        plt.grid()
        plt.xlabel("iteraciones")
        plt.ylabel(r"$J(\theta)$")
        plt.title("Evolución de costo en el descenso de Gradente")
        plt.show()

    def graficar_data(self, model=False):
        fig2 = plt.figure()
        if self.__X.shape[1] > 2:
            ax = fig2.add_subplot(projection='3d')
            ax.scatter(self.__X[:, 1], self.__X[:, 2], self.__y.flatten())
            if model:
                # calculamos los valores del plano para los puntos x e y
                xx1 = np.linspace(self.__X[:, 1].min(), self.__X[:, 1].max(), 100)
                xx2 = np.linspace(self.__X[:, 2].min(), self.__X[:, 2].max(), 100)
                xx1, xx2 = np.meshgrid(xx1, xx2)
                x1 = (self.__theta[1] * xx1)
                x2 = (self.__theta[2] * xx2)
                z = (x1 + x2 + self.__theta[0])
                ax.plot_surface(xx1, xx2, z, alpha=0.4, cmap='hot')
        else:
            plt.scatter(self.__X[:, 1], self.__y)
            if model:
                x = np.linspace(self.__X[:, 1].min(), self.__X[:, 1].max(), 100)
                plt.plot(x, self.__theta[0] + self.__theta[1] * x, c="red")
            plt.grid()
        plt.show()

    def graficar_superficie_costo(self, theta_indices=(0, 1), figsize=(10, 7)):
            """Visualización 3D de la superficie de costo"""
            if len(theta_indices) != 2:
                raise ValueError("Se deben especificar exactamente 2 índices de parámetros")
            
            i, j = theta_indices
            theta_i = np.linspace(self.__theta[i] - 3, self.__theta[i] + 3, 30)
            theta_j = np.linspace(self.__theta[j] - 3, self.__theta[j] + 3, 30)
            ti, tj = np.meshgrid(theta_i, theta_j)
            
            cost = np.zeros_like(ti)
            for k in range(ti.shape[0]):
                for l in range(ti.shape[1]):
                    temp_theta = self.__theta.copy()
                    temp_theta[i] = ti[k, l]
                    temp_theta[j] = tj[k, l]
                    cost[k, l] = self.get_j(temp_theta)
            
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            # Superficie
            surf = ax.plot_surface(ti, tj, cost, cmap='viridis', 
                                  alpha=0.8, linewidth=0, antialiased=True)
            
            # Punto óptimo
            ax.scatter([self.__theta[i]], [self.__theta[j]], [self.get_j(self.__theta)], 
                      c='red', s=100, label='Óptimo')
            
            # Configuración
            ax.set_xlabel(f'θ{i}')
            ax.set_ylabel(f'θ{j}')
            ax.set_zlabel('Costo J(θ)')
            ax.set_title(f'Superficie de Costo para θ{i} vs θ{j}')
            ax.legend()
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            plt.tight_layout()
            plt.show()
    
    def graficar_curvas_nivel(self, theta_indices=(0, 1), n_points=50, log_scale=True):

        if len(theta_indices) != 2:
            raise ValueError("theta_indices debe contener exactamente 2 índices")
            
        # Valores base para los parámetros no graficados
        base_theta = self.__theta.copy()
        
        # Rangos para los parámetros seleccionados
        t0_idx, t1_idx = theta_indices
        t0_range = np.linspace(base_theta[t0_idx] - 5, base_theta[t0_idx] + 5, n_points)
        t1_range = np.linspace(base_theta[t1_idx] - 5, base_theta[t1_idx] + 5, n_points)
        
        # Crear malla
        t0_mesh, t1_mesh = np.meshgrid(t0_range, t1_range)
        cost_values = np.zeros_like(t0_mesh)
        
        # Calcular costo para cada combinación
        for i in range(n_points):
            for j in range(n_points):
                test_theta = base_theta.copy()
                test_theta[t0_idx] = t0_mesh[i, j]
                test_theta[t1_idx] = t1_mesh[i, j]
                cost_values[i, j] = self.get_j(test_theta)
        
        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Curvas de nivel
        if log_scale:
            levels = np.logspace(np.log10(cost_values.min()), np.log10(cost_values.max()), 20)
            cs = ax.contour(t0_mesh, t1_mesh, cost_values, levels=levels, cmap='viridis')
        else:
            cs = ax.contour(t0_mesh, t1_mesh, cost_values, 20, cmap='viridis')
        
        # Punto óptimo
        ax.scatter(base_theta[t0_idx], base_theta[t1_idx], c='red', s=100)
        
        # Configuración
        ax.clabel(cs, inline=True, fontsize=10)
        ax.set_xlabel(f'θ{theta_indices[0]}')
        ax.set_ylabel(f'θ{theta_indices[1]}')
        ax.set_title(f'Curvas de Nivel para θ{theta_indices[0]} vs θ{theta_indices[1]}')
     #   ax.legend()
        ax.grid(True)
        plt.colorbar(cs, ax=ax, label='Costo J(θ)')
        plt.show()

