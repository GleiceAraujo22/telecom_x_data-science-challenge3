# 📊 Telecom X – Parte 2: Previsão de Evasão de Clientes (Churn)  
--- 
## 📌 Descrição do Projeto
 Este projeto é a parte 2 de uma análise exploratória dos dados onde nessa etapa iremos construir modelos de machine learning para previsão de evasão de clientes de uma empresa de telecomunicações. Utilizamos técnicas avançadas de análise de dados, machine learning e balanceamento de dados para identificar os principais fatores e padrões que contribuem para evasão de clientes. 

 ## Tecnologias Utilizadas
* Python 3.x
* Pandas, NumPy (manipulação e análise de dados)
* Matplotlib
* Scikit-learn

 ## 🤖 Modelos de Machine learning
  * DummyRegressor (modelo baseline)
  * Decision Tree
  * Regressão Logística
  * Random Forest

## Análise dos Modelos Treinados 

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
