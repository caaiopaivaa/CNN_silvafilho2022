# Classificação Convolucional de Caracteres Manuscritos (EMNIST) - Dígitos (1-5) e Letras (A-E)

Pipeline de visão computacional e aprendizado profundo desenvolvido em **TensorFlow/Keras** para reconhecimento óptico de caracteres manuscritos (OCR), categorizando um subconjunto de **10 classes**: dígitos `[1, 2, 3, 4, 5]` e caracteres alfabéticos `[A, B, C, D, E]`, extraídos da base **EMNIST (Extended MNIST)**.

---

## 📌 Visão Geral do Projeto

Este projeto implementa uma Rede Neural Convolucional (CNN) profunda para resolver o problema de classificação multiclasse em um conjunto foneticamente e morfologicamente diverso, mas livre das ambiguidades críticas do alfabeto completo (como `0/O` ou `1/l/I`). 

O pipeline integra extração bruta de tensores, saneamento de transposição espacial de eixos, interpolação para padrão de 3 canais de cor e treinamento supervisionado com avaliação cega em partição de teste independente.

### Principais Resultados
* **Acurácia no Teste (Test Accuracy):** **~98,0%**
* **Otimização:** Convergência assistida por *Early Stopping* com restauração automática do melhor conjunto de pesos (*best weights*).
* **Domínio de Classes:** 10 classes balanceadas (`1`, `2`, `3`, `4`, `5`, `A`, `B`, `C`, `D`, `E`).

---

## 📊 Dataset e Pré-processamento

Os dados são amostrados do split **EMNIST Balanced** e consolidados para filtragem das 10 classes alvo:

### Mapeamento das Classes
Para garantir representatividade real do manuscrito humano, foram consideradas grafias maiúsculas e minúsculas correspondentes:

| Índice Final | Caractere | Código Original EMNIST | Observações |
| :---: | :---: | :---: | :--- |
| **0** | `1` | `1` | Dígito numérico |
| **1** | `2` | `2` | Dígito numérico |
| **2** | `3` | `3` | Dígito numérico |
| **3** | `4` | `4` | Dígito numérico |
| **4** | `5` | `5` | Dígito numérico |
| **5** | `A` | `10` e `36` | Unificação de maiúscula (`A`) e minúscula (`a`) |
| **6** | `B` | `11` e `37` | Unificação de maiúscula (`B`) e minúscula (`b`) |
| **7** | `C` | `12` | Nativamente fundido no EMNIST Balanced (`C`/`c`) |
| **8** | `D` | `13` e `38` | Unificação de maiúscula (`D`) e minúscula (`d`) |
| **9** | `E` | `14` e `39` | Unificação de maiúscula (`E`) e minúscula (`e`) |

### Pipeline de Transformação
* **Correção Geométrica:** Transposição matricial dos eixos $(H, W)$ para corrigir a rotação padrão de 90° e espelhamento nativos do dataset EMNIST.
* **Resolução e Espaço de Cor:** Reamostragem bilinear de $28 \times 28 \times 1$ para **$32 \times 32$ pixels com 3 canais (RGB)** (`32, 32, 3`).
* **Normalização:** Conversão dos valores de intensidade de pixel para ponto flutuante no intervalo $[0.0, 1.0]$.

### Divisão Estratificada dos Dados
A partição preserva a proporcionalidade estrita entre as 10 classes:

| Partição | Proporção | Finalidade |
| :--- | :---: | :--- |
| **Treinamento** | **70%** | Atualização dos parâmetros via retropropagação (backpropagation) |
| **Validação** | **15%** | Avaliação intra-época e critério de parada prematura (*patience*) |
| **Teste** | **15%** | Verificação cega final do poder de generalização do modelo |

---

## 🧠 Arquitetura da Rede Neural (CNN)

A rede foi projetada para extração progressiva de padrões visuais (bordas, curvas e cruzamentos) através de um bloco convolucional de 4 estágios seguido de subamostragem:

```text
Entrada: Imagem (32, 32, 3)
   │
   ├── Conv2D (32 filtros, kernel 3x3, ativação ReLU, padding='same')
   │
   ├── Conv2D (64 filtros, kernel 3x3, ativação ReLU, padding='same')
   ├── Conv2D (64 filtros, kernel 3x3, ativação ReLU, padding='same')
   ├── Conv2D (64 filtros, kernel 3x3, ativação ReLU, padding='same')
   │
   ├── MaxPooling2D (pool_size 2x2)  --> Redução para mapa de (16, 16, 64)
   │
   ├── Flatten                       --> Vetor linear de 16.384 características
   │
   └── Dense (10 neurônios, Softmax) --> Distribuição de probabilidade sobre as 10 classes
