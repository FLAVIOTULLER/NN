# 🧠 NN - Neural Network Engine & Interactive Visualizer

Um projeto completo de **Rede Neural (Neural Network)** construído do zero, com visualizador interativo em tempo real para a web e implementação pura em Python.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-yellow.svg)
![HTML5](https://img.shields.io/badge/HTML5-Canvas-orange.svg)

---

## 🌟 Funcionalidades

### 1. 🖥️ Visualizador Interativo Web (`index.html`)
* **Visualização da Topologia da Rede**: Visualização dinâmica dos neurônios e pesos das conexões com feedback visual em tempo real.
* **Superfície de Decisão (Decision Boundary Heatmap)**: Gráfico 2D renderizado em Canvas mostrando as decisões da rede em tempo real.
* **Múltiplos Datasets**:
  * XOR problem
  * Círculos Concéntricos (Circles)
  * Duas Luas (Moons)
  * Espirais (Spiral)
* **Controles Interativos**:
  * Play, Pause, Step e Reset de treinamento.
  * Ajuste de taxa de aprendizado (Learning Rate).
  * Funções de ativação (Tanh, Sigmoid, ReLU).
  * Configuração de camadas ocultas e quantidade de neurônios.
* **Gráfico de Loss**: Curva de convergência em tempo real.

### 2. 🐍 Implementação em Python (`neural_network.py`)
* Camadas densas (`DenseLayer`) com pesos e biases inicializados aleatoriamente.
* Funções de ativação: Sigmoid, ReLU, Tanh e Softmax com suas respectivas derivadas.
* Funções de perda: MSE (Mean Squared Error) e Binary Cross-Entropy.
* Algoritmo de **Backpropagation** com otimização por Gradiente Descendente estocástico / mini-batch.
* Exemplos prontos de treinamento com saída no terminal.

---

## 🚀 Como Executar

### Visualizador Web:
1. Abra o arquivo `index.html` diretamente em qualquer navegador moderno (Chrome, Edge, Firefox).
2. Não requer instalação de dependências ou servidores!

### Script Python:
```bash
python neural_network.py
```

---

## 📦 Como Subir para o seu GitHub

1. Crie um repositório vazio chamado `NN` no seu GitHub: [github.com/new](https://github.com/new).
2. Execute no terminal:
```bash
git init
git add .
git commit -m "feat: initial commit - Neural Network project"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/NN.git
git push -u origin main
```

---

## 📄 Licença
Distribuído sob a licença MIT. Veja `LICENSE` para mais detalhes.
