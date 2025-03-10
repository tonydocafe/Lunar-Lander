# LunarLander DQN

Este projeto treina e visualiza um agente DQN (Deep Q-Network) jogando o ambiente **LunarLander-v3** do Gymnasium.

## Requisitos

Antes de executar os scripts, instale as dependências necessárias:

```bash
pip install gymnasium[box2d] stable-baselines3 pygame numpy matplotlib
```

## Treinamento do Modelo

O script `aprendizado.py` treina um agente DQN para jogar o LunarLander e salva o modelo treinado.

### Execução

```bash
python aprendizado.py
```

### O que o script faz:
1. Cria o ambiente `LunarLander-v3`.
2. Inicializa e treina o agente DQN por 200.000 passos.
3. Salva o modelo treinado como `dqn_lunar.zip`.
4. Avalia o desempenho do modelo.
5. Exibe um gráfico da recompensa acumulada ao longo do tempo.

## Visualização do Modelo Treinado

O script `visualize.py` carrega o modelo treinado e permite visualizar o agente jogando.

### Execução

```bash
python visualize.py
```

### O que o script faz:
1. Carrega o modelo `dqn_lunar.zip`.
2. Executa a simulação por 1000 passos.
3. Permite fechar a simulação clicando no botão **X** ou pressionando **Enter** no terminal.

## Estrutura do Projeto

```
.
├── aprendizado.py         # Script de treinamento do agente
├── visualização.py     # Script de visualização do agente treinado
├── dqn_lunar.zip    # Modelo treinado salvo
└── README.md        # Este arquivo
```

## Observações
- Certifique-se de que `dqn_lunar.zip` foi gerado antes de executar `visualize.py`.
- Se a janela não fechar ao clicar no **X**, pressione **Enter** no terminal para encerrar o programa corretamente.

## Autor
Desenvolvido por Tony Hudson Candido Junior.

