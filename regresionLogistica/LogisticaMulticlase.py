import numpy as np
# from scipy.optimize import fmin_bfgs  # ELIMINADO para usar DG

# Renombre de la clase: Clasificador Binario de Regresión Logística con Theta^2 y DG/L2
class RegresionLogisticaBinaria: 
    """
    Implementación de Regresión Logística Binaria con pesos positivos (theta^2)
    y optimización por Descenso de Gradiente CON Regularización L2.
    """
    
    # constructor
    def __init__(self):
        self.__X = None # Característica con sesgo (X0=1)
        self.__y = None # Variable objetivo (0 o 1)
        self.__theta = None # Parámetros del modelo (theta)
        self.__lambda = 0.0 # Parámetro de regularización L2

    def fit(self, x, y, lambda_param=0.0):
        """Carga los datos, prepara la matriz y define lambda."""
        m, n = x.shape
        # aniadir unidad de sesgo X0, columna de 1s
        self.__X = np.append(np.ones((m, 1)), x.reshape(m, -1), axis=1)
        self.__y = y.reshape(-1, 1)
        # Inicializamos parametros en 0.1, puede ser en 0
        self.__theta = np.ones(n + 1) * 0.1 
        self.__lambda = lambda_param

    # --- FUNCIÓN SIGMOIDE ---
    def sigmoid(self, z):
        """Función de activación sigmoide."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    # --- FUNCIÓN DE COSTO (ENTROPÍA CRUZADA CON REGULARIZACIÓN L2) ---
    def get_j(self, theta):
        """Función de costo de Entropía Cruzada para Regresión Logística."""
        theta = theta.reshape(-1, 1)
        m = self.__X.shape[0]
        
        # 1. Hipótesis (ESTÁNDAR: z = X * theta)
        z = self.__X.dot(theta)
        h = self.sigmoid(z)
        
        epsilon = 1e-15
        h = np.clip(h, epsilon, 1 - epsilon)
        
        # 2. Entropía cruzada estándar
        costo_log = -1/m * (self.__y.T.dot(np.log(h)) + (1-self.__y).T.dot(np.log(1-h)))
        
        # 3. Término de Regularización L2 (NO penalizar theta[0])
        theta_sin_sesgo = theta[1:]
        regularizacion = (self.__lambda / (2 * m)) * np.sum(np.square(theta_sin_sesgo))

        # 4. Costo Total
        j = costo_log.sum() + regularizacion
        return j

    # --- GRADIENTE (CON REGULARIZACIÓN L2) ---
    def get_gradiente(self, theta):
        """Calcula el gradiente de la función de costo con respecto a theta."""
        theta = theta.reshape(-1, 1)
        m = self.__X.shape[0]
        
        # 1. Error y Gradiente Estándar
        z = self.__X.dot(theta) # ESTÁNDAR: z = X * theta
        h = self.sigmoid(z)
        error = h - self.__y
        
        # Gradiente de Regresión Logística (dJ/dTheta, sin regla de la cadena de theta^2)
        grad_w = 1/m * self.__X.T.dot(error)
        
        # 2. Gradiente de Regularización L2
        grad_reg = (self.__lambda / m) * theta
        
        # 3. Ajuste (NO regularizar el término de sesgo grad_w[0])
        grad_final = grad_w + grad_reg
        grad_final[0] = grad_w[0] # Eliminar la regularización del sesgo
        
        return grad_final.flatten()
    
    # --- OPTIMIZACIÓN CON DESCENSO DE GRADIENTE ---
    def descenso_de_gradiente(self, alpha, num_iteraciones):
        """Optimización de parámetros usando el Descenso de Gradiente (GD)."""
        js = []
        theta = self.__theta.copy().reshape(-1, 1)
        
        for i in range(num_iteraciones):
            # 1. Calcular el gradiente con L2
            gradiente = self.get_gradiente(theta).reshape(-1, 1)
            
            # 2. Actualizar parámetros usando la Tasa de Aprendizaje (alpha)
            theta = theta - alpha * gradiente
            
            # 3. Registrar el costo
            costo = self.get_j(theta)
            js.append(costo)
            
            # Criterio de parada simple: si el costo se estanca
            if len(js) > 10 and abs(js[-1] - js[-10]) < 1e-6:
                break
                
        self.__theta = theta.flatten()
        self.__history = np.array(js)
        # print(f"DG finalizado en {i+1} iteraciones. Costo: {costo:.6f}") # Para seguimiento
    
    def get_param(self):
        """Retorna los parámetros theta del modelo."""
        return self.__theta

    def predecir(self, x_features):
        """Calcula la probabilidad P(y=1 | x) para las características de entrada."""
        if self.__theta is None:
            raise ValueError("El modelo binario no ha sido entrenado.")

        x = np.array(x_features)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        x_con_sesgo = np.c_[np.ones(x.shape[0]), x]
        
        # ESTÁNDAR: Multiplicamos X por theta (no cuadrado)
        z = x_con_sesgo.dot(self.__theta) 
        return self.sigmoid(z)


# Clase para manejar la clasificación multiclase (One-vs-Rest)
class RegresionLogisticaMulticlase:
    def __init__(self, num_clases, clasificador_binario=RegresionLogisticaBinaria):
        self.num_clases = num_clases
        self.clasificadores = []
        self.clasificador_binario = clasificador_binario 

    def fit(self, X_train, y_train, alpha, lambda_param, num_iteraciones):
        """
        Entrena K clasificadores binarios utilizando la estrategia One-vs-Rest (OvR).
        Acepta alpha, lambda y num_iteraciones como hiperparámetros de entrenamiento.
        """
        print("Iniciando entrenamiento Regresión Logística Multiclase (OvR) con DG...")
        self.clasificadores = []

        for k in range(self.num_clases):
            print(f"\nEntrenando clasificador OvR para Clase {k} (vs Resto)...")
            
            # 1. Crear Etiquetas Binarias (One-vs-Rest)
            y_k = (y_train == k).astype(int)
            
            # 2. Inicializar, Cargar datos y OPTIMIZAR
            model = self.clasificador_binario()
            # Cargar datos y parámetro lambda
            model.fit(X_train, y_k, lambda_param=lambda_param) 
            
            # Optimizar con Descenso de Gradiente usando alpha y num_iteraciones
            model.descenso_de_gradiente(alpha, num_iteraciones)
            
            self.clasificadores.append(model)
            
        print("\nEntrenamiento Multiclase Finalizado.")

    def predecir(self, X_test):
        """Predice las etiquetas de clase (0 a K-1) para X_test."""
        # ... (código predecir sin cambios) ...
        if not self.clasificadores:
            raise ValueError("El modelo OvR no ha sido entrenado.")
            
        predictions = []
        for model in self.clasificadores:
            prob_k = model.predecir(X_test).flatten()
            predictions.append(prob_k)

        predictions_matrix = np.column_stack(predictions)
        predicciones = np.argmax(predictions_matrix, axis=1) #
        
        return predicciones