
#Autores: Artemio Dias Guimarães e Fernanda Barbosa Santos
#23/10/2025

# SIMULADOR DE CONTROLE DIGITAL - TANQUE DE NÍVEL (PID)
# RESPOSTA AO DEGRAU E À RAMPA
#Objetivo: Simular a resposta de um sistema de controle de nível
#de um tanque (planta de 1ª ordem) a um controlador PID digital.

# Bloco 1: Importação de Bibliotecas

import numpy as np
import matplotlib.pyplot as plt
from control import tf, feedback, forced_response
from scipy import signal

# Bloco 2: Definição do Sistema Contínuo (Planta)
# PARÂMETROS REAIS DO TANQUE

K = 1855          # Ganho estático [m / (sinal de controle)]
tau = 3787        # Constante de tempo [s]

# Função de transferência contínua: Gp(s) = K / (τs + 1)
num_cont = [K]     # Numerador: K
den_cont = [tau, 1] # Denominador: τs + 1

Gp_cont = (num_cont, den_cont)

print(f"Função de transferência contínua: Gp(s) = {K} / ({tau}s + 1)")

# Bloco 3: Definição dos Parâmetros do Controlador (PID)
# CONTROLADOR PID COM GANHOS ESTABILIZADOS

Kp = 0.01       # Ganho proporcional
Ki = 0.000005    # Ganho integral
Kd = 0.0020      # Ganho derivativo

print(f"\nGanhos do PID:")
print(f"Kp = {Kp}")
print(f"Ki = {Ki}")
print(f"Kd = {Kd}")

# Bloco 4: Configuração da Simulação
# PERÍODOS DE AMOSTRAGEM

T_list = [126]
print(f"\nPeríodos de amostragem: {T_list} s")

# Bloco 5: Loop Principal de Simulação (por Período de Amostragem)

for T in T_list:
    print(f"\n" + "="*50)
    print(f"SIMULAÇÃO PARA T = {T}s")
    print("="*50)

    # Bloco 6: Discretização da Planta (ZOH) (signal.cont2discrete)
    #Usando ZOH poi ele modela realiticamente como atuador(via D/A)
    #"segura" o valor do sinal de controle durante o período T.

    Gz_disc = signal.cont2discrete(Gp_cont, T, method='zoh')
    num_disc, den_disc, _ = Gz_disc

    print(f"Planta discretizada (ZOH):")
    print(f"  Numerador: {num_disc[0]}")
    print(f"  Denominador: {den_disc}")

    # Criar função de transferência discreta para o control
    # Nota: num_disc[0] é usado porque a scipy retorna o numerador como [[...]]

    Gz = tf(num_disc[0], den_disc, T)

    # Bloco 7: Discretização do Controlador PID (Tustin/Misto)
    # Implementação do PID digital na forma posicional: C(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 - z^-1)
    # Os coeficientes 'b' são derivados de uma combinação de métodos:
    # - Integral: Regra Trapezoidal (Tustin/Bilinear)
    # - Derivativa: Diferença finita (Backward Euler)
    # Esta é uma forma digital robusta e comum do PID.

    b0 = Kp + Ki*T/2 + Kd/T
    b1 = -Kp + Ki*T/2 - 2*Kd/T
    b2 = Kd/T

    # O denominador [1, -1, 0] representa (1 - z^-1), que é o integrador digital necessário para a forma posicional.
    # Controlador PID discreto

    Cz = tf([b0, b1, b2], [1, -1, 0], T)

    print(f"Controlador PID discreto:")
    print(f"  Numerador: [{b0:.6f}, {b1:.6f}, {b2:.6f}]")
    print(f"  Denominador: [1, -1, 0]")

    # Bloco 8: Montagem da Malha Fechada Discreta
    # O sistema em malha fechada (sys_cl) é obtido pela realimentação
    # unitária (H=1) do produto do controlador (Cz) e da planta (Gz).
    # sys_cl = (Cz * Gz) / (1 + Cz * Gz)

    sys_cl = feedback(Cz * Gz, 1)

    # Bloco 9: Simulação da Resposta ao Degrau

    t_sim = 20000  # Tempo de simulação
    t = np.arange(0, t_sim, T)
    ref_degrau = np.ones_like(t)  # Degrau unitário = 1 metro

    # Resposta ao degrau

    t_degrau, y_degrau = forced_response(sys_cl, T=t, U=ref_degrau)

    # Esforço de controle para degrau

    t_u_degrau, u_degrau = forced_response(Cz, T=t, U=ref_degrau-y_degrau)

    # Bloco 10: Simulação da Resposta à Rampa
    t_rampa = np.arange(0, t_sim, T) # Vetor de tempo discreto

    # Rampa com inclinação suave: 0.0001 m/s (0.1 mm/s)

    ref_rampa = 0.0001 * t_rampa

    # Resposta à rampa

    t_rampa, y_rampa = forced_response(sys_cl, T=t_rampa, U=ref_rampa)

    # Esforço de controle para rampa

    t_u_rampa, u_rampa = forced_response(Cz, T=t_rampa, U=ref_rampa-y_rampa)

    # Bloco 11: Geração de Gráficos
    # GRÁFICO 1: RESPOSTA AO DEGRAU

    plt.figure(figsize=(12, 6))
    plt.plot(t_degrau, y_degrau, 'b-', linewidth=2, label=f'Saída - T={T}s')
    plt.plot(t_degrau, ref_degrau, 'r--', linewidth=2, alpha=0.7, label='Referência')
    plt.title(f'Resposta ao Degrau - Tanque de Nível (T={T}s)', fontsize=14, fontweight='bold')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Nível do Tanque (m)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 2.0)
    plt.tight_layout()
    plt.show()

    # GRÁFICO 2: RESPOSTA À RAMPA

    plt.figure(figsize=(12, 6))
    plt.plot(t_rampa, y_rampa, 'g-', linewidth=2, label=f'Saída - T={T}s')
    plt.plot(t_rampa, ref_rampa, 'r--', linewidth=2, alpha=0.7, label='Referência (rampa)')
    plt.title(f'Resposta à Rampa - Tanque de Nível (T={T}s)', fontsize=14, fontweight='bold')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Nível do Tanque (m)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # GRÁFICO 3: ESFORÇO DE CONTROLE (DEGRAU)

    plt.figure(figsize=(12, 6))
    plt.plot(t_u_degrau, u_degrau, 'r-', linewidth=2, label=f'Esforço de controle - T={T}s')
    plt.title(f'Esforço de Controle - Degrau (T={T}s)', fontsize=14, fontweight='bold')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Sinal de Controle')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # GRÁFICO 4: ESFORÇO DE CONTROLE (RAMPA)

    plt.figure(figsize=(12, 6))
    plt.plot(t_u_rampa, u_rampa, 'orange', linewidth=2, label=f'Esforço de controle - T={T}s')
    plt.title(f'Esforço de Controle - Rampa (T={T}s)', fontsize=14, fontweight='bold')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Sinal de Controle')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # GRÁFICO 5: ERRO DE RASTREAMENTO (DEGRAU)

    plt.figure(figsize=(12, 6))
    erro_degrau = ref_degrau - y_degrau
    plt.plot(t_degrau, erro_degrau, 'purple', linewidth=2, label=f'Erro - T={T}s')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Zero')
    plt.title(f'Erro de Rastreamento - Degrau (T={T}s)', fontsize=14, fontweight='bold')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Erro (m)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # GRÁFICO 6: ERRO DE RASTREAMENTO (RAMPA)

    plt.figure(figsize=(12, 6))
    erro_rampa = ref_rampa - y_rampa
    plt.plot(t_rampa, erro_rampa, 'brown', linewidth=2, label=f'Erro - T={T}s')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Zero')
    plt.title(f'Erro de Rastreamento - Rampa (T={T}s)', fontsize=14, fontweight='bold')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Erro (m)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Bloco 12: Cálculo de Métricas de Desempenho
    # --- Métricas para Degrau ---

    try:
        settling_time = None
        # Busca reversa pelas últimas 100 amostras (otimização)
        if len(y_degrau) > 10:
            for i in range(len(y_degrau)-1, max(0, len(y_degrau)-100), -1):
                if abs(y_degrau[i] - 1) <= 0.02:  # 2% do valor final
                    settling_time = t_degrau[i]
                    break

        # Sobressinal (Overshoot)

        overshoot = max(0, (np.max(y_degrau) - 1) * 100) if len(y_degrau) > 0 else 0

        print(f"Desempenho para DEGRAU (T = {T}s):")
        print(f"  - Valor final: {y_degrau[-1]:.4f} m")
        print(f"  - Erro estacionário: {abs(1 - y_degrau[-1]):.6f} m")
        print(f"  - Esforço máximo: {np.max(np.abs(u_degrau)):.6f}")
        if settling_time:
            print(f"  - Tempo de acomodação (2%): {settling_time:.0f} s")
        print(f"  - Sobresinal: {overshoot:.2f}%")

    except Exception as e:
        print(f"  - Erro no cálculo de métricas para degrau: {e}")

    # Cálculo de métricas de desempenho para rampa

    try:
        # Erro de velocidade (erro em regime permanente para rampa)
        erro_final_rampa = erro_rampa[-1]
        print(f"\nDesempenho para RAMPA (T = {T}s):")
        print(f"  - Erro final: {erro_final_rampa:.6f} m")
        print(f"  - Esforço máximo: {np.max(np.abs(u_rampa)):.6f}")

    except Exception as e:
        print(f"  - Erro no cálculo de métricas para rampa: {e}")

# Bloco 13: Resumo Final

print("\n" + "="*60)
print("RESUMO DA SIMULAÇÃO:")
print("="*60)
print(f"Total de gráficos gerados: {len(T_list) * 6}")
print("Método de discretização: ZOH (Zero Order Hold)")
