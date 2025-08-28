# 📊 Telecom X – Parte 2: Previsão de Evasão de Clientes (Churn)  
 
## 📌 Descrição do Projeto
 Este projeto é a parte 2 de uma análise exploratória dos dados onde nessa etapa iremos construir modelos de machine learning para previsão de evasão de clientes de uma empresa de telecomunicações. Utilizamos técnicas avançadas de análise de dados, machine learning e balanceamento de dados para identificar os principais fatores e padrões que contribuem para evasão de clientes. 

 ## 🛠 Tecnologias Utilizadas
* Python 3.x
* Pandas, NumPy (manipulação e análise de dados)
* Matplotlib
* Scikit-learn

 ## 🤖 Modelos de Machine learning
  * **DummyRegressor (modelo baseline)**: Serve como referência, pois faz previsões simples sem realmente aprender com os dados. Ele mostra o desempenho mínimo esperado e ajuda a avaliar se outros modelos realmente agregam valor.
    
  * **Decision Tree**: Modelo fácil de interpretar, pois segue uma lógica de “se-então”(fluxograma). Pode capturar relações não lineares, mas tende a sofrer com overfitting se não houver regularização.
    
  * **Regressão Logística**: Um modelo estatístico clássico para classificação binária. É simples, interpretável e eficiente, mas pode ter limitações em capturar relações complexas entre variáveis.
    
  * **Random Forest**: Conjunto de várias árvores de decisão (Ensemble), reduzindo o risco de overfitting e melhorando a robustez. Costuma ter bom desempenho em problemas de classificação como churn, pois equilibra interpretabilidade e capacidade preditiva, capturando padrões mais complexos nos dados.


## 📊 Análise dos Modelos Treinados  

### Estratégia para lidar com o desbalanceamento entre as classes da variável alvo (Churn)

O dataset apresentava um desbalanceamento significativo entre clientes que deram churn e os que permaneceram. Para lidar com esse problema, foi utilizada a estratégia `class_weight="balanced"`, que ajusta automaticamente os pesos das classes de acordo com sua frequência.

Essa abordagem foi escolhida por alguns motivos:

**Preserva os dados originais**: diferentemente de técnicas como SMOTE, não há criação artificial de novas amostras, evitando risco de introduzir ruído.

**Integração direta ao modelo**: o ajuste é feito internamente no algoritmo de classificação, garantindo simplicidade e eficiência.

**Equilíbrio no aprendizado**: permite que o modelo atribua maior importância à classe minoritária, reduzindo a tendência de prever apenas a classe majoritária.

Em resumo, o uso de class_weight="balanced" proporcionou uma forma prática e confiável de lidar com o desbalanceamento, mantendo a integridade dos dados e evitando vieses no processo de modelagem.  


![distribuicao churn](visualizations/distribuicao_churn.png)

--- 

### Estratégia para lidar com a multicolinearidade 
O VIF (Variance Inflation Factor) foi utilizado no projeto para identificar e mitigar problemas de multicolinearidade entre as variáveis independentes. A multicolinearidade ocorre quando duas ou mais variáveis explicativas estão altamente correlacionadas entre si, o que pode distorcer os coeficientes do modelo, dificultando a interpretação dos resultados e reduzindo a robustez das estimativas. Dessa forma, foi possível avaliar quais variáveis apresentavam redundância de informação e tomar decisões mais conscientes sobre manter ou remover variáveis no modelo. 

Optei por utilizar essa técnica para não prejudicar o modelo de regressão logística visto que para os modelos de Árvore que também foram utilizados, a multicolinearidade tem um impacto menor mas ainda sim pode haver benefícios.   

| Faixa de VIF          | Interpretação                   |
|-----------------------|---------------------------------|
| VIF ≈ 1               | Sem multicolinearidade          |
| 1 < VIF < 5           | Baixa (aceitável)               |
| 5 ≤ VIF < 10          | Moderada (acompanhar)           |
| VIF ≥ 10              | Alta (atenção!)                 |
| VIF = ∞ (infinito)    | Multicolinearidade perfeita ⚠️ |


* Exemplo de VIF gerado:  

| Variável                           | VIF        |
|-----------------------------------|-----------|
| const                              | 31.308484 |
| Charges.Monthly                    | 22.356986 |
| InternetService_Fiber optic        | 7.554042  |
| tenure                             | 2.784158  |
| Contract_Two year                  | 2.610640  |
| StreamingMovies_Yes                | 2.417216  |
| StreamingTV_Yes                    | 2.400758  |
| PaymentMethod_Electronic check     | 1.973409  |
| TechSupport_Yes                    | 1.850009  |
| PaymentMethod_Mailed check         | 1.840793  |
| OnlineSecurity_Yes                 | 1.810495  |
| DeviceProtection_Yes               | 1.784591  |
| OnlineBackup_Yes                   | 1.708249  |
| MultipleLines_Yes                  | 1.630113  |
| Contract_One year                  | 1.620312  |
| PhoneService_Yes                   | 1.609719  |
| PaymentMethod_Credit card (automatic) | 1.560356 |
| Partner_Yes                        | 1.462143  |
| Dependents_Yes                     | 1.383601  |
| PaperlessBilling_Yes               | 1.208148  |
| SeniorCitizen                      | 1.153343  |
| gender_Male                        | 1.001830  |

--- 
### Matriz de confusão e métricas de avaliação 

A matriz de confusão é importante porque permite avaliar detalhadamente o desempenho de um modelo de classificação, mostrando quantos casos foram corretamente ou incorretamente classificados para cada classe. Ela fornece insights sobre acertos, erros e desequilíbrios entre classes, complementando métricas como acurácia, precisão, recall e F1-score. 

O foco principal no projeto são em clientes que abandonam o serviço (Churn = 1) porém como naturalmente são dados desbalanceados (a proporção de clientes que abandonam é menor do que a proporção de clientes ativos) 

![](visualizations/confusion_matrix_logistic.png) 
  

--- 
![](visualizations/confusion_matrix_dt.png)  

--- 
![](visualizations/confusion_matrix_rf.png)   

--- 
![Curva ROC - Regressão logística](visualizations/roc_curve_logistic_r.png) 

--- 
![Curva ROC - Decision tree](visualizations/roc_curve_dt.png)  

--- 
![Curva ROC - Random Forest](visualizations/roc_curve_rf.png)  

## Análise de Importância das Variáveis 

![permutation importance - Regressão Logística](visualizations/permutation_importance_logistic.png)   

--- 

![permutation importance - Random Forest](visualizations/permutation_importance_rf.png)    

## Desempenho dos Modelos

Os modelos com melhor desempenho foram: Random Forest e Regressão Logística, estes apresentaram desempenho consistente na previsão de clientes propensos a churn:

**Recall(métrica principal neste projeto)**: alta capacidade de identificar clientes que realmente cancelariam, permitindo ações preventivas mais eficazes.

**Precisão e F1-score**: equilíbrio adequado entre falsos positivos e falsos negativos.

**AUC-ROC**: [ex.: 0.84], indicando bom poder discriminativo do modelo.

Estes modelos mostraram-se robustos e interpretáveis, sendo possível extrair insights das variáveis mais importantes para o negócio.  

## Próximos passos 
* Otimizar hiperparâmetros
* Treinar outros modelos de machine learning possíveis como XGBoost com foco na melhora de performance preditiva.
* Feature engineering
