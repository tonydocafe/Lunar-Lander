# 🚀 DQN Lunar Lander - Explorando o Espaço com Inteligência Artificial! 🌕

Bem-vindo ao **DQN Lunar Lander**, um experimento onde ensinamos um agente a pousar suavemente em um planeta alienígena! 🌌🛸

---

## 🤖 Sobre o Projeto

Neste projeto, utilizamos **Deep Q-Networks (DQN)** para treinar um agente no ambiente *LunarLander-v3*, fornecido pelo Gymnasium. O objetivo? Fazer com que nossa nave pouse de forma segura sem explodir! 💥🔥

---

## 📦 Explicando o Script Linha por Linha

### 1️⃣ Importação das Bibliotecas
```python
import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
```
**gymnasium**: Fornece o ambiente de simulação.
**numpy**: Usado para operações matemáticas e manipulação de arrays.
**DQN**: O algoritmo de aprendizado por reforço baseado em Q-Learning profundo.
**evaluate_policy**: Função que avalia o desempenho do agente após o treinamento.

### 2️⃣ Criando o Ambiente
```python
env = gym.make("LunarLander-v3", render_mode="rgb_array")
```
**gym.make()** cria um ambiente de simulação baseado no jogo Lunar Lander.
**render_mode="rgb_array"** configura a renderização para capturar imagens.

### 3️⃣ Definição do Modelo DQN
```python
model = DQN(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=100_000,
    learning_starts=10_000,
    batch_size=64,
    gamma=0.99,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
    train_freq=4,
    target_update_interval=1000,
)
```
 **policy="MlpPolicy"**: Política de aprendizado baseada em redes neurais MLP (Multilayer Perceptron).
**env=env**: Associa o modelo ao ambiente.
**verbose=1**: Informações sobre o treinamento.
**learning_rate=1e-3**: Taxa de aprendizado.
**buffer_size=100_000**: Tamanho do buffer de replay.
**learning_starts=10_000**: Começa a aprender após 10.000 interações.
**batch_size=64**: Tamanho do lote para treinamento.
**gamma=0.99**: Fator de desconto para aprendizado futuro.
**exploration_fraction=0.1**: Fracionamento da exploração durante o treinamento.
**exploration_final_eps=0.05**: Limite inferior da exploração.
**train_freq=4**: Frequência de treinamento.
**target_update_interval=1000**: Intervalo para atualizar a rede-alvo.

### 4️⃣ Treinamento do Modelo
```python
print("Iniciando treinamento...")
model.learn(total_timesteps=int(200_000), progress_bar=True)
```
**model.learn()**: Treina o agente por 200.000 timesteps.
**progress_bar=True**: Exibe o progresso do treinamento.

### 5️⃣ Salvando o Modelo Treinado
```python
model.save("dqn_lunar")
print("Modelo salvo como dqn_lunar.zip")
```
- **model.save("dqn_lunar")**: Salva o modelo treinado em um arquivo ZIP.

### 6️⃣ Recarregando o Modelo
```python
del model
model = DQN.load("dqn_lunar", env=env)
print("Modelo carregado!")
```
- **del model**: Remove o modelo da memória.
- **DQN.load("dqn_lunar", env=env)**: Carrega o modelo salvo e associa ao ambiente.

### 7️⃣ Avaliação do Agente
```python
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Recompensa média: {mean_reward:.2f} ± {std_reward:.2f}")
```
- **evaluate_policy()**: Mede o desempenho do agente com 10 episódios de teste.
- **mean_reward**: Média das recompensas obtidas.
- **std_reward**: Desvio padrão das recompensas.

### 8️⃣ Executando o Agente Treinado
```python
obs, _ = env.reset()
rewards = []

for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    rewards.append(reward)
    env.render()
    
    if done:
        obs, _ = env.reset()
```
- **env.reset()**: Reinicia o ambiente.
- **model.predict(obs, deterministic=True)**: Escolhe a melhor ação baseada na política treinada.
- **env.step(action)**: Aplica a ação ao ambiente.
- **env.render()**: Renderiza a simulação.
- **if done:** Reinicia o ambiente ao final do episódio.

### 9️⃣ Fechando o Ambiente
```python
env.close()
```
- **env.close()**: Fecha a simulação e libera os recursos.

---

## 🚀 Como Rodar o Projeto?

### 1️⃣ Instale as dependências
```bash
pip install gymnasium numpy stable-baselines3
```

### 2️⃣ Execute o script
```bash
python dqn_lunar.py
```

### 3️⃣ Assista ao pouso (ou explosão)!
Se tudo der certo, seu agente pousará suavemente. Se não... bem, foguetes são difíceis de pilotar! 😆

---

## 🏆 Conclusão
Este README explica cada linha do script e como o Python interpreta os comandos. Agora, você pode entender e modificar o código para melhor desempenho ou criar novas versões! 🚀😃

Divirta-se e bons experimentos! 🧪

