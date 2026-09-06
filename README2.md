# Classificação Convolucional de Caracteres Manuscritos (EMNIST) - Letras 'V' vs 'F'

Pipeline de visão computacional e aprendizado profundo desenvolvido em **TensorFlow/Keras** para reconhecimento e classificação binária de caracteres manuscritos das letras **V** e **F** a partir de subconjuntos da base **EMNIST (Extended MNIST)**.

---

## 📌 Visão Geral do Projeto

O objetivo deste projeto é construir um classificador de alta precisão para distinguir grafias manuscritas de alta variabilidade entre os caracteres alfabéticos **'V'** e **'F'**. O pipeline abrange desde a extração e saneamento dos dados brutos até a correção de orientação geométrica, reamostragem tridimensional de canais e treinamento supervisionado com parada antecipada (*Early Stopping*).

### Destaques dos Resultados
* **Acurácia no Teste Independente:** **99,52%**
* **Função de Perda no Teste (Loss):** **0,0141**
* **Generalização:** Quase nula taxa de confusão residual entre traços oblíquos/angulares (`V`) e perpendiculares/ortogonais (`F`).

---

## 📊 Dataset e Pré-processamento

O conjunto de dados foi derivado do **EMNIST Balanced**:
* **Classes Selecionadas:**
  * **Classe 0 (`V`):** Unificação de maiúsculas e minúsculas nativa do EMNIST (código `31`).
  * **Classe 1 (`F`):** Mapeamento conjunto de maiúsculas (`15`) e minúsculas (`40`).
* **Tratamento de Orientação:** Correção da matriz original via transposição de eixos espaciais $(H, W)$ para adequação ao padrão visual canônico.
* **Formato de Entrada:** Redimensionamento bilinear de $28 \times 28 \times 1$ para **$32 \times 32$ pixels** e expansão para **3 canais de cor (RGB)** (`32, 32, 3`).
* **Normalização:** Escala de cinza normalizada no intervalo $[0.0, 1.0]$.

### Divisão Estratificada dos Dados
A base filtrada foi particionada preservando a proporção exata das classes em três conjuntos disjuntos:

| Partição | Proporção | Descrição |
| :--- | :---: | :--- |
| **Treinamento** | **70%** | Otimização dos gradientes e atualização de pesos |
| **Validação** | **15%** | Ajuste fino de hiperparâmetros e gatilho de Early Stopping |
| **Teste** | **15%** | Avaliação final cega e independente de desempenho |

---

## 🧠 Arquitetura da Rede Neural (CNN)

A topologia foi desenhada com extração hierárquica progressiva em 3 estágios convolucionais seguidos de subamostragem:

```text
Entrada: Tensor (32, 32, 3)
   │
   ├── Conv2D (32 filtros, kernel 3x3, ReLU, padding='same')
   │
   ├── Conv2D (64 filtros, kernel 3x3, ReLU, padding='same')
   ├── Conv2D (64 filtros, kernel 3x3, ReLU, padding='same')
   │
   ├── MaxPooling2D (pool_size 2x2)  --> Mapa reduzido para (16, 16, 64)
   │
   ├── Flatten                       --> Vetor unidimensional de 16.384 dimensões
   │
   └── Dense (2 neurônios, Softmax)  --> Probabilidades das classes ['V', 'F']
