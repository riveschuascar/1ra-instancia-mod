#!/usr/bin/env python3
"""
Ejemplo Simple - Modelo de Desarrollo de Jugador
=================================================

Este script muestra el uso básico del modelo para simular
el desarrollo de un jugador desde los 18 hasta los 35 años.
"""

from player_development_model import (
    PlayerDevelopmentModel,
    PlayerParameters,
    adaptive_training,
    plot_results
)
import matplotlib.pyplot as plt

def main():
    print("="*70)
    print("SIMULACIÓN DE DESARROLLO DE JUGADOR")
    print("="*70)
    
    # 1. Crear el modelo con parámetros por defecto
    print("\n1. Creando modelo con parámetros por defecto...")
    model = PlayerDevelopmentModel()
    
    # 2. Definir estado inicial del jugador
    print("2. Definiendo jugador inicial...")
    jugador_inicial = {
        'F': 0.60,  # Físico: 60% desarrollado
        'T': 0.50,  # Técnico: 50% desarrollado
        'M': 0.40,  # Mental: 40% desarrollado
        'A': 18.0   # Edad: 18 años
    }
    
    print(f"\n   Jugador: {jugador_inicial['A']:.0f} años")
    print(f"   - Físico:  {jugador_inicial['F']:.2f}")
    print(f"   - Técnico: {jugador_inicial['T']:.2f}")
    print(f"   - Mental:  {jugador_inicial['M']:.2f}")
    
    # 3. Configurar simulación
    print("\n3. Configurando simulación...")
    años_a_simular = 17  # De 18 a 35 años
    print(f"   Duración: {años_a_simular} años ({años_a_simular * 365} días)")
    print("   Estrategia: Entrenamiento adaptativo por edad")
    
    # 4. Ejecutar simulación
    print("\n4. Ejecutando simulación (esto puede tomar unos segundos)...")
    resultados = model.simulate(
        initial_state=jugador_inicial,
        training_schedule=adaptive_training,
        duration_days=años_a_simular * 365,
        dt=1.0  # Paso de 1 día
    )
    
    print("   ✓ Simulación completada!")
    
    # 5. Analizar resultados
    print("\n5. Analizando resultados...")
    
    # Encontrar pico de rendimiento
    idx_pico = resultados['Rating (R)'].idxmax()
    edad_pico = resultados.loc[idx_pico, 'Edad']
    rating_pico = resultados.loc[idx_pico, 'Rating (R)']
    
    # Estado final
    idx_final = len(resultados) - 1
    edad_final = resultados.loc[idx_final, 'Edad']
    rating_final = resultados.loc[idx_final, 'Rating (R)']
    
    print("\n" + "="*70)
    print("RESULTADOS DE LA SIMULACIÓN")
    print("="*70)
    
    print(f"\n📊 PICO DE RENDIMIENTO:")
    print(f"   Edad:     {edad_pico:.1f} años")
    print(f"   Rating:   {rating_pico:.1f}/100")
    print(f"   Físico:   {resultados.loc[idx_pico, 'Físico (F)']:.3f}")
    print(f"   Técnico:  {resultados.loc[idx_pico, 'Técnico (T)']:.3f}")
    print(f"   Mental:   {resultados.loc[idx_pico, 'Mental (M)']:.3f}")
    
    print(f"\n📈 ESTADO FINAL (edad {edad_final:.0f}):")
    print(f"   Rating:   {rating_final:.1f}/100")
    print(f"   Físico:   {resultados.loc[idx_final, 'Físico (F)']:.3f}")
    print(f"   Técnico:  {resultados.loc[idx_final, 'Técnico (T)']:.3f}")
    print(f"   Mental:   {resultados.loc[idx_final, 'Mental (M)']:.3f}")
    
    # Calcular duración del "prime" (rating > 85)
    prime_mask = resultados['Rating (R)'] >= 85
    if prime_mask.any():
        años_prime = prime_mask.sum() / 365
        edad_inicio_prime = resultados[prime_mask].iloc[0]['Edad']
        edad_fin_prime = resultados[prime_mask].iloc[-1]['Edad']
        
        print(f"\n⭐ PERÍODO PRIME (Rating ≥ 85):")
        print(f"   Duración:  {años_prime:.1f} años")
        print(f"   Desde:     {edad_inicio_prime:.1f} años")
        print(f"   Hasta:     {edad_fin_prime:.1f} años")
    
    print("\n" + "="*70)
    
    # 6. Generar gráficas
    print("\n6. Generando visualizaciones...")
    fig = plot_results(resultados, save_path='ejemplo_desarrollo.png')
    print("   ✓ Gráficas generadas: ejemplo_desarrollo.png")
    
    # 7. Guardar datos
    print("\n7. Guardando datos...")
    resultados.to_csv('ejemplo_desarrollo.csv', index=False)
    print("   ✓ Datos guardados: ejemplo_desarrollo.csv")
    
    print("\n" + "="*70)
    print("SIMULACIÓN COMPLETADA EXITOSAMENTE")
    print("="*70)
    
    # Mostrar algunos datos estadísticos adicionales
    print("\n📈 ESTADÍSTICAS ADICIONALES:")
    rating_promedio = resultados['Rating (R)'].mean()
    rating_mediana = resultados['Rating (R)'].median()
    rating_desv = resultados['Rating (R)'].std()
    
    print(f"   Rating promedio:  {rating_promedio:.1f}")
    print(f"   Rating mediano:   {rating_mediana:.1f}")
    print(f"   Desv. estándar:   {rating_desv:.1f}")
    
    # Tasa de cambio en el pico
    if idx_pico > 0 and idx_pico < len(resultados) - 1:
        ventana = 365  # 1 año
        idx_antes = max(0, idx_pico - ventana)
        idx_despues = min(len(resultados) - 1, idx_pico + ventana)
        
        mejora_antes = (resultados.loc[idx_pico, 'Rating (R)'] - 
                       resultados.loc[idx_antes, 'Rating (R)']) / (ventana / 365)
        declive_despues = (resultados.loc[idx_despues, 'Rating (R)'] - 
                          resultados.loc[idx_pico, 'Rating (R)']) / (ventana / 365)
        
        print(f"\n   Mejora antes del pico:     {mejora_antes:.2f} puntos/año")
        print(f"   Declive después del pico:  {declive_despues:.2f} puntos/año")
    
    print("\n" + "="*70)
    print("\n💡 TIP: Puedes modificar este script para:")
    print("   - Cambiar el estado inicial del jugador")
    print("   - Ajustar los parámetros del modelo")
    print("   - Probar diferentes estrategias de entrenamiento")
    print("   - Comparar múltiples escenarios")
    print("\n" + "="*70 + "\n")
    
    return resultados, model

if __name__ == "__main__":
    resultados, modelo = main()
    
    # Mostrar gráficas (comentar si se ejecuta sin interfaz gráfica)
    plt.show()
