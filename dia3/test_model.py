#!/usr/bin/env python3
"""
Script de Pruebas - Modelo de Desarrollo de Jugador
====================================================

Este script ejecuta una serie de pruebas para verificar que
el modelo funciona correctamente.
"""

import numpy as np
from player_development_model import (
    PlayerDevelopmentModel,
    PlayerParameters,
    constant_training,
    adaptive_training
)

def test_parameter_initialization():
    """Prueba 1: Inicialización de parámetros"""
    print("\n" + "="*70)
    print("PRUEBA 1: Inicialización de Parámetros")
    print("="*70)
    
    params = PlayerParameters()
    
    # Verificar valores por defecto
    assert params.alpha_F == 0.15, "Error en alpha_F"
    assert params.beta_F == 0.008, "Error en beta_F"
    assert params.A_opt == 27.0, "Error en A_opt"
    assert params.w_F + params.w_T + params.w_M == 1.0, "Pesos no suman 1"
    
    print("✓ Parámetros inicializados correctamente")
    print(f"  - Tasa aprendizaje físico: {params.alpha_F}")
    print(f"  - Tasa decaimiento físico: {params.beta_F}")
    print(f"  - Edad óptima: {params.A_opt}")
    print(f"  - Suma de pesos: {params.w_F + params.w_T + params.w_M}")

def test_age_factor():
    """Prueba 2: Factor de edad (curva gaussiana)"""
    print("\n" + "="*70)
    print("PRUEBA 2: Factor de Edad")
    print("="*70)
    
    model = PlayerDevelopmentModel()
    
    # Probar en edad óptima
    factor_opt = model.age_factor(27.0)
    assert abs(factor_opt - 1.0) < 0.01, "Factor en A_opt debería ser ~1.0"
    
    # Probar en edades alejadas
    factor_joven = model.age_factor(20.0)
    factor_viejo = model.age_factor(35.0)
    
    assert factor_joven < 1.0, "Factor para joven debe ser < 1.0"
    assert factor_viejo < 1.0, "Factor para viejo debe ser < 1.0"
    assert factor_joven < factor_opt, "Factor joven < factor óptimo"
    assert factor_viejo < factor_opt, "Factor viejo < factor óptimo"
    
    print("✓ Curva de edad funciona correctamente")
    print(f"  - Factor a 20 años: {factor_joven:.3f}")
    print(f"  - Factor a 27 años: {factor_opt:.3f}")
    print(f"  - Factor a 35 años: {factor_viejo:.3f}")

def test_derivatives():
    """Prueba 3: Cálculo de derivadas"""
    print("\n" + "="*70)
    print("PRUEBA 3: Cálculo de Derivadas")
    print("="*70)
    
    model = PlayerDevelopmentModel()
    
    # Estado de prueba
    F, T, M, A = 0.7, 0.6, 0.5, 25.0
    E_F, E_T, E_M = 0.8, 0.8, 0.6
    
    # Calcular derivadas
    dF = model.dF_dt(F, T, M, A, E_F)
    dT = model.dT_dt(F, T, M, A, E_T)
    dM = model.dM_dt(F, T, M, A, E_M)
    
    # Verificar que las derivadas tienen sentido
    assert dF > 0, "dF debería ser positivo (aprendizaje domina)"
    assert dT > 0, "dT debería ser positivo"
    assert dM > 0, "dM debería ser positivo"
    
    print("✓ Derivadas calculadas correctamente")
    print(f"  - dF/dt = {dF:.6f}")
    print(f"  - dT/dt = {dT:.6f}")
    print(f"  - dM/dt = {dM:.6f}")

def test_decay():
    """Prueba 4: Decaimiento por edad"""
    print("\n" + "="*70)
    print("PRUEBA 4: Decaimiento por Edad")
    print("="*70)
    
    model = PlayerDevelopmentModel()
    
    F, T, M = 0.9, 0.9, 0.9
    E_F, E_T, E_M = 0, 0, 0  # Sin entrenamiento
    
    # Antes de umbral de edad (no debería haber decaimiento significativo)
    dF_joven = model.dF_dt(F, T, M, 25.0, E_F)
    
    # Después de umbral de edad (debería haber decaimiento)
    dF_viejo = model.dF_dt(F, T, M, 35.0, E_F)
    dT_viejo = model.dT_dt(F, T, M, 35.0, E_T)
    dM_viejo = model.dM_dt(F, T, M, 38.0, E_M)
    
    assert dF_viejo < dF_joven, "Decaimiento físico debe ser mayor en edad avanzada"
    assert dF_viejo < 0, "Sin entrenamiento a 35 años, físico debe decaer"
    assert dT_viejo < 0, "Sin entrenamiento a 35 años, técnico debe decaer"
    assert dM_viejo < 0, "Sin entrenamiento a 38 años, mental debe decaer"
    
    print("✓ Decaimiento por edad funciona correctamente")
    print(f"  - dF/dt a 25 años (sin entrenamiento): {dF_joven:.6f}")
    print(f"  - dF/dt a 35 años (sin entrenamiento): {dF_viejo:.6f}")
    print(f"  - dT/dt a 35 años (sin entrenamiento): {dT_viejo:.6f}")
    print(f"  - dM/dt a 38 años (sin entrenamiento): {dM_viejo:.6f}")

def test_synergy():
    """Prueba 5: Efectos de sinergia"""
    print("\n" + "="*70)
    print("PRUEBA 5: Efectos de Sinergia")
    print("="*70)
    
    model = PlayerDevelopmentModel()
    
    # Caso 1: Sin atributo físico
    dT_sin_F = model.dT_dt(0.0, 0.5, 0.5, 25.0, 0.5)
    
    # Caso 2: Con atributo físico alto
    dT_con_F = model.dT_dt(0.9, 0.5, 0.5, 25.0, 0.5)
    
    # La sinergia debe hacer que dT sea mayor con F alto
    assert dT_con_F > dT_sin_F, "Sinergia físico→técnico no funciona"
    
    # Sinergia para mental
    dM_sin_FT = model.dM_dt(0.0, 0.0, 0.5, 25.0, 0.5)
    dM_con_FT = model.dM_dt(0.8, 0.8, 0.5, 25.0, 0.5)
    
    assert dM_con_FT > dM_sin_FT, "Sinergia (F+T)→mental no funciona"
    
    print("✓ Sinergias funcionan correctamente")
    print(f"  - dT/dt sin F: {dT_sin_F:.6f}")
    print(f"  - dT/dt con F alto: {dT_con_F:.6f}")
    print(f"  - Incremento por sinergia: {dT_con_F - dT_sin_F:.6f}")
    print(f"  - dM/dt sin F,T: {dM_sin_FT:.6f}")
    print(f"  - dM/dt con F,T altos: {dM_con_FT:.6f}")
    print(f"  - Incremento por sinergia: {dM_con_FT - dM_sin_FT:.6f}")

def test_rk4_step():
    """Prueba 6: Paso RK4"""
    print("\n" + "="*70)
    print("PRUEBA 6: Paso de Integración RK4")
    print("="*70)
    
    model = PlayerDevelopmentModel()
    
    # Estado inicial
    state = np.array([0.6, 0.5, 0.4, 70.0, 20.0])
    training = (0.8, 0.8, 0.6)
    
    # Ejecutar un paso
    new_state = model.rk4_step(state, dt=1.0, training=training)
    
    # Verificar que el estado cambió
    assert not np.array_equal(state, new_state), "Estado no cambió"
    
    # Verificar restricciones
    assert 0 <= new_state[0] <= 1, "F fuera de rango [0,1]"
    assert 0 <= new_state[1] <= 1, "T fuera de rango [0,1]"
    assert 0 <= new_state[2] <= 1, "M fuera de rango [0,1]"
    assert 40 <= new_state[3] <= 100, "R fuera de rango [40,100]"
    assert new_state[4] > state[4], "Edad no incrementó"
    
    print("✓ Paso RK4 ejecutado correctamente")
    print(f"  - Estado inicial: F={state[0]:.3f}, T={state[1]:.3f}, M={state[2]:.3f}")
    print(f"  - Estado final:   F={new_state[0]:.3f}, T={new_state[1]:.3f}, M={new_state[2]:.3f}")
    print(f"  - Edad: {state[4]:.3f} → {new_state[4]:.3f}")

def test_simulation():
    """Prueba 7: Simulación completa"""
    print("\n" + "="*70)
    print("PRUEBA 7: Simulación Completa")
    print("="*70)
    
    model = PlayerDevelopmentModel()
    
    initial_state = {
        'F': 0.5,
        'T': 0.5,
        'M': 0.4,
        'A': 20.0
    }
    
    # Simular 5 años
    results = model.simulate(
        initial_state=initial_state,
        training_schedule=constant_training(0.7, 0.7, 0.7),
        duration_days=5 * 365,
        dt=7.0  # Semanal para velocidad
    )
    
    # Verificaciones
    assert len(results) > 0, "Simulación no produjo resultados"
    assert 'Físico (F)' in results.columns, "Falta columna Físico"
    assert 'Técnico (T)' in results.columns, "Falta columna Técnico"
    assert 'Mental (M)' in results.columns, "Falta columna Mental"
    assert 'Rating (R)' in results.columns, "Falta columna Rating"
    assert 'Edad' in results.columns, "Falta columna Edad"
    
    # Verificar que la edad aumentó
    edad_inicial = results.iloc[0]['Edad']
    edad_final = results.iloc[-1]['Edad']
    assert edad_final > edad_inicial, "Edad no incrementó"
    assert abs(edad_final - edad_inicial - 5.0) < 0.1, "Incremento de edad incorrecto"
    
    print("✓ Simulación completada exitosamente")
    print(f"  - Número de pasos: {len(results)}")
    print(f"  - Edad inicial: {edad_inicial:.2f}")
    print(f"  - Edad final: {edad_final:.2f}")
    print(f"  - Rating inicial: {results.iloc[0]['Rating (R)']:.1f}")
    print(f"  - Rating final: {results.iloc[-1]['Rating (R)']:.1f}")

def test_rating_calculation():
    """Prueba 8: Cálculo del rating"""
    print("\n" + "="*70)
    print("PRUEBA 8: Cálculo del Rating")
    print("="*70)
    
    model = PlayerDevelopmentModel()
    
    # Casos extremos
    R_min = model.compute_rating(0.0, 0.0, 0.0)
    R_max = model.compute_rating(1.0, 1.0, 1.0)
    R_mid = model.compute_rating(0.5, 0.5, 0.5)
    
    assert abs(R_min - 40.0) < 0.01, "Rating mínimo debería ser 40"
    assert abs(R_max - 100.0) < 0.01, "Rating máximo debería ser 100"
    assert 40 < R_mid < 100, "Rating medio fuera de rango"
    
    # Verificar pesos
    R_F = model.compute_rating(1.0, 0.0, 0.0)  # Solo físico
    R_T = model.compute_rating(0.0, 1.0, 0.0)  # Solo técnico
    R_M = model.compute_rating(0.0, 0.0, 1.0)  # Solo mental
    
    # Técnico debería contribuir más (w_T = 0.40 es el mayor)
    assert R_T > R_F, "Peso técnico debería ser mayor que físico"
    assert R_T > R_M, "Peso técnico debería ser mayor que mental"
    
    print("✓ Cálculo de rating correcto")
    print(f"  - Rating mínimo (0,0,0): {R_min:.1f}")
    print(f"  - Rating máximo (1,1,1): {R_max:.1f}")
    print(f"  - Rating medio (0.5,0.5,0.5): {R_mid:.1f}")
    print(f"  - Rating solo F: {R_F:.1f}")
    print(f"  - Rating solo T: {R_T:.1f}")
    print(f"  - Rating solo M: {R_M:.1f}")

def test_training_schedules():
    """Prueba 9: Programas de entrenamiento"""
    print("\n" + "="*70)
    print("PRUEBA 9: Programas de Entrenamiento")
    print("="*70)
    
    # Constante
    training_const = constant_training(0.8, 0.7, 0.6)
    result_const = training_const(25.0)
    assert result_const == (0.8, 0.7, 0.6), "Entrenamiento constante no funciona"
    
    # Adaptativo
    result_joven = adaptive_training(22.0)
    result_prime = adaptive_training(27.0)
    result_veterano = adaptive_training(33.0)
    
    # Verificar que el entrenamiento físico disminuye con edad
    assert result_joven[0] >= result_veterano[0], "Entrenamiento físico debería disminuir con edad"
    # Verificar que el entrenamiento mental aumenta con edad
    assert result_veterano[2] >= result_joven[2], "Entrenamiento mental debería aumentar con edad"
    
    print("✓ Programas de entrenamiento funcionan correctamente")
    print(f"  - Constante: {result_const}")
    print(f"  - Adaptativo (22 años): {result_joven}")
    print(f"  - Adaptativo (27 años): {result_prime}")
    print(f"  - Adaptativo (33 años): {result_veterano}")

def test_peak_detection():
    """Prueba 10: Detección de pico de rendimiento"""
    print("\n" + "="*70)
    print("PRUEBA 10: Detección de Pico de Rendimiento")
    print("="*70)
    
    model = PlayerDevelopmentModel()
    
    initial_state = {
        'F': 0.6,
        'T': 0.5,
        'M': 0.4,
        'A': 20.0
    }
    
    results = model.simulate(
        initial_state=initial_state,
        training_schedule=adaptive_training,
        duration_days=15 * 365,
        dt=7.0
    )
    
    # Encontrar pico
    peak_idx = results['Rating (R)'].idxmax()
    peak_age = results.loc[peak_idx, 'Edad']
    peak_rating = results.loc[peak_idx, 'Rating (R)']
    
    # El pico debería estar cerca de A_opt (27 años)
    assert 23 <= peak_age <= 32, f"Pico a {peak_age:.1f} años parece inusual"
    assert peak_rating >= results.iloc[0]['Rating (R)'], "Pico debe ser mayor que inicial"
    
    print("✓ Pico detectado correctamente")
    print(f"  - Edad del pico: {peak_age:.1f} años")
    print(f"  - Rating del pico: {peak_rating:.1f}")
    print(f"  - F en pico: {results.loc[peak_idx, 'Físico (F)']:.3f}")
    print(f"  - T en pico: {results.loc[peak_idx, 'Técnico (T)']:.3f}")
    print(f"  - M en pico: {results.loc[peak_idx, 'Mental (M)']:.3f}")

def run_all_tests():
    """Ejecutar todas las pruebas"""
    print("\n" + "="*70)
    print("SUITE DE PRUEBAS - MODELO DE DESARROLLO DE JUGADOR")
    print("="*70)
    
    tests = [
        test_parameter_initialization,
        test_age_factor,
        test_derivatives,
        test_decay,
        test_synergy,
        test_rk4_step,
        test_simulation,
        test_rating_calculation,
        test_training_schedules,
        test_peak_detection
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ FALLÓ: {test_func.__name__}")
            print(f"   Error: {str(e)}")
            failed += 1
        except Exception as e:
            print(f"\n❌ ERROR: {test_func.__name__}")
            print(f"   Excepción: {str(e)}")
            failed += 1
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN DE PRUEBAS")
    print("="*70)
    print(f"✓ Pasadas: {passed}/{len(tests)}")
    print(f"❌ Fallidas: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ¡TODAS LAS PRUEBAS PASARON!")
    else:
        print(f"\n⚠️  {failed} prueba(s) fallaron. Revisa los errores arriba.")
    
    print("="*70 + "\n")
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
