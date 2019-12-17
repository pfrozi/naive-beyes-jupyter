# Laboratório 03 - Método de Naive Bayes

## I. Análise inicial

1. **Para esse dataset, qual o percentual de emails de treinamento que correspondem a spams? É a maioria ou a minoria?**  
   Maioria, 71% de spams (3675 spams de um total de 5175).

2. **Observe as palavras mais freqüentes em emails de treinamento do tipo spam e do tipo não spam. Elas diferem significativamente? Dentre essas palavras, cite algumas que você acha que, de fato, são bastante características de spams, e algumas que são características de emails legítimos. Note que é esse tipo de regularidade/padrão de palavras em emails de cada tipo (spam ou não-spam) que será explorado pelo algoritmo de treinamento de Naive Bayes.**  
   Sim. Elas diferem principalmente porque as palavras com frequência zero na classe NÃO-SPAM parecem ser bem aleatórias, além de não fazer sentido para o contexto da Aaron.

   Algumas palavras que podem ser inferidas como spam e não-spam:

```
SPAM
Palavra --pheromone---
Palavra --paypal---
Palavra --gambling---
Palavra --excitement---
Palavra --herpes---
Palavra --drugs---

NÃO-SPAM
Palavra --aaron---
Palavra --ethic---
Palavra --aluminium---
Palavra --recruits---
>>> A maior parte dos nomes próprios, que sugerem a apresentação do e-mail.
```

## II. Classificação sem Laplace Smoothing

3. **Acione, agora, apenas a opção para mostrar palavras desconhecidas (isto é, de freqüência zero) dentre os spams. Explique por que você acha que algumas dessas palavras específicas não aparecem em spams, embora apareçam no conjunto de treinamento de emails dessa empresa em particular;**  
   Utilizando os exemplos anteriores podemos ver algumas palavras que dificilmente apareceriam nas instâncias classificadas como SPAM. Uma das mais óbvias é --aaron---, que corresponde ao próprio nome da empresa. É muito provável que a maior parte das mensagens que não são spams contenham o nome da empresa no seu corpo.

4. **Acione, agora, apenas a opção para mostrar palavras desconhecidas (isto é, de freqüência zero) dentre não-spams (isto é, emails legítimos). Explique por que você acha que algumas dessas palavras não aparecem em emails legítimos, embora apareçam no conjunto de treinamento de emails completo (o qual inclui tanto spams quanto não-spams);**  
   Ao menos que tenham mensagens específicas em que os funcionários utilizavam o e-mail profissional para assuntos pessoais, palavras como --drugs--- ou --herpes--- não apareceriam em e-mail's legítimos.

**Desligue tanto a opção para mostrar palavras desconhecidas dentre spams e dentre não spams; rode novamente o algoritmo no dataset “desbalanceado”. Responda:**

```
(Nao utilizando Laplace smoothing)
Carregando dataset desbalanceado
	Dataset possui 1500 spams de um total de 5172 emails (percentual de spams: 29.00%)
Fazendo predicoes no conjunto de teste...
Acuracia: 37.0000%	(37 of 100)
```

5. **Qual a acurácia do classificador? É alta ou baixa? Por que você acha que isso aconteceu?**  
   37.0000%. É muito baixa. Provavelmente ocorreu porque temos palavras que ocorrem com muita frequência nos SPAMs ou NÃO-SPAMs, mas possuem uma frequência zero na classe oposta.

**Agora rode o classificador no outro dataset (dataset “balanceado”). Responda:**

```
(Nao utilizando Laplace smoothing)
Carregando dataset balanceado
	Dataset possui 3675 spams de um total de 5175 emails (percentual de spams: 71.00%)
Fazendo predicoes no conjunto de teste...
Acuracia: 18.0000%	(18 of 100)
```

6. **Qual a acurácia do classificador? É alta ou baixa? Houve diferença significativa em relação à acurácia quando analisando o dataset desbalanceado? O que isso indica sobre a importância de se lidar com palavras desconhecidas, independente do grau de balanceamento do dataset?**  
   18.0000%. É péssimo. Novamente, a frequência zero das palavras em cada uma das classes impactou diretamente na acurácia da classificação. No caso do dataset balanceado vemos um impacto ainda maior, pois temos mais palavras com frequência 0% em NÃO-SPAM sendo consideradas no treinamento (no dataset balanceado temos 71.00% de spam).

## III. Classificação com Laplace Smoothing

**Acione o uso da técnica de Laplace Smoothing; isto é, altere o parâmetro use_laplace_smoothing para True. Responda:**

7. **Para o dataset “desbalanceado”, relate a acurácia do classificador quando usando como parâmetro de Laplace Smoothing (isto é, variável laplace_smoothing) os seguinte valores: 0.00001, 0.001, 1.0, 3.0, e 10.0. O que acontece quando se aumenta essa constante? Qual valor resulta na melhor acurácia?**  
   Para todos os valores de K, a acurácia se mostrou maior do que todas as execuções realizadas sem o Laplace Smoothing, pois ele evita a frequência zero no produtório das probabilidades. Como podemos ver nas execuções realizadas com o Laplace Smoothing habilitado, a acurácia vai se deteriorando a medida que o valor de K cresce. Isto ocorre porque K é um valor que é adicionado ao numerador da frequência da palavra. Logo, K's muito grandes não refletem uma frequência que deveria ser próxima à ZERO.

```
Execuções:

(Utilizando Laplace smoothing 0.00100)
Carregando dataset desbalanceado
	Dataset possui 1500 spams de um total de 5172 emails (percentual de spams: 29.00%)
Fazendo predicoes no conjunto de teste...
Acuracia: 96.0000%	(96 of 100)

(Utilizando Laplace smoothing 1.00000)
Carregando dataset desbalanceado
	Dataset possui 1500 spams de um total de 5172 emails (percentual de spams: 29.00%)
Fazendo predicoes no conjunto de teste...
Acuracia: 95.0000%	(95 of 100)

(Utilizando Laplace smoothing 3.00000)
Carregando dataset desbalanceado
	Dataset possui 1500 spams de um total de 5172 emails (percentual de spams: 29.00%)
Fazendo predicoes no conjunto de teste...
Acuracia: 93.0000%	(93 of 100)

(Utilizando Laplace smoothing 10.00000)
Carregando dataset desbalanceado
	Dataset possui 1500 spams de um total de 5172 emails (percentual de spams: 29.00%)
Fazendo predicoes no conjunto de teste...
Acuracia: 75.0000%	(75 of 100)
```

8. **Para o dataset “balanceado”, relate a acurácia do classificador quando usando como parâmetro de Laplace Smoothing (variável laplace_smoothing) os seguinte valores: 0.00001, 0.001, 1.0, 3.0, e 10.0. O que acontece quando se aumenta essa constante? Qual valor resulta na melhor acurácia? Isso sugere que o uso da técnica de Laplace Smoothing (a qual introduz observações fictícias de palavras em um email, quando a palavra é desconhecida) afeta de maneira igual datasets que são balanceados e desbalanceados?**  
   Ocorreu uma acurácia de 100% para todos os valores de K testados. Isto sugere que para datasets balanceados temos uma acurácia maior em relação aos valores de K testados. Uma hipótese para esta diferença é que em um dataset balanceado temos uma quantidade muito maior de palavras de emails SPAM com frequência zero nas instâncias NÃO_SPAM. Logo, como estamos utilizando o Laplace Smoothing, estas palavras com frequência zero em NÃO_SPAM impactam positivamente na classificação de novas instâncias.

```
Execuções:

(Utilizando Laplace smoothing 0.00001)
Carregando dataset balanceado
	Dataset possui 3675 spams de um total de 5175 emails (percentual de spams: 71.00%)
Fazendo predicoes no conjunto de teste...
Acuracia: 100.0000%	(100 of 100)

(Utilizando Laplace smoothing 0.00100)
Carregando dataset balanceado
	Dataset possui 3675 spams de um total de 5175 emails (percentual de spams: 71.00%)
Fazendo predicoes no conjunto de teste...
Acuracia: 100.0000%	(100 of 100)

(Utilizando Laplace smoothing 1.00000)
Carregando dataset balanceado
	Dataset possui 3675 spams de um total de 5175 emails (percentual de spams: 71.00%)
Fazendo predicoes no conjunto de teste...
Acuracia: 100.0000%	(100 of 100)

(Utilizando Laplace smoothing 3.00000)
Carregando dataset balanceado
	Dataset possui 3675 spams de um total de 5175 emails (percentual de spams: 71.00%)
Fazendo predicoes no conjunto de teste...
Acuracia: 100.0000%	(100 of 100)

(Utilizando Laplace smoothing 10.00000)
Carregando dataset balanceado
	Dataset possui 3675 spams de um total de 5175 emails (percentual de spams: 71.00%)
Fazendo predicoes no conjunto de teste...
Acuracia: 100.0000%	(100 of 100)
```
