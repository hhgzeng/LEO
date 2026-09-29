# Large language model-based evolutionary optimizer: Reasoning with elitism

## ARTICLE INFO

**Communicated by:** X. Yan  
**Keywords:** Large language models, Evolutionary Optimizers, Multi-objective optimization, Aerodynamic Design

## ABSTRACT

Large Language Models (LLMs) have demonstrated remarkable reasoning abilities, prompting interest in their application as black-box optimizers. This paper asserts that LLMs possess the capability for zero-shot optimization across diverse scenarios, including multi-objective and high-dimensional problems. We introduce a novel population-based method for numerical optimization using LLMs called Large Language-Model-Based Evolutionary Optimizer (LEO). Our hypothesis is supported through numerical examples, spanning benchmark and industrial engineering problems such as supersonic nozzle shape optimization, heat transfer, and windfarm layout optimization. We compare our method to several gradient-based and gradient-free optimization approaches. While LLMs yield comparable results to state-of-the-art methods, their imaginative nature and propensity to hallucinate demand careful handling. We provide practical guidelines for obtaining reliable answers from LLMs and discuss method limitations and potential research directions.

## 1. Introduction

The advent of Large Language Models (LLMs) has sparked a revolution in generative Artificial Intelligence (AI) research [1–3]. Since the introduction of transformer model [4], the generative AI research has seen a surge of activity and every subsequent generation of LLM models have become exceedingly more capable than the previous ones. For example, the first decoder-only model developed by OpenAI in 2018 was called Generative Pre-Trained transformers (GPT) based on the transformer architecture, was capable of processing textual data alone [2], while OpenAI’s latest GPT-4 model released in 2023 is multimodal, i.e., capable of dealing with natural language, code, as well as images [5]. Several studies since then have shown that Large Language Models (LLMs), such as GPT-4, possess strong reasoning ability [6,7]. Studies have also shown that a LLM’s performance can be further improved by techniques such as in-context learning [8], chain-of-thought prompting [9], and tree-of-thought prompting [10,11].

Some examples that highlight the generalization capability of LLM models are: (a) Gato [12], a generalist multi-modal agent based on a LLM capable of performing several tasks. (b) Eureka [13], a human-level reward design algorithm using LLMs, is a gradient-free in-context learning approach to reinforcement learning for robotic applications. (c) Voyager [14], a LLM-powered AI agent, has shown the ability to conduct autonomous exploration, skill acquisition, and discovery in an open-ended Minecraft world without human intervention. To test the reasoning and generalization ability of new generative AI models, Srivastava et al. published a suite of benchmarks containing 204 problems called the Beyond the Imitation Game benchmark (BIG-bench) [15].

This reasoning and generalization ability of LLMs has sparked interest in exploring use of LLM models as AI agents, particularly for applications in science and technology. Bran et al. [16] developed an autonomous AI agent called ChemCrow for computational chemistry research. This AI agent has demonstrated remarkable ability to accomplish tasks across organic synthesis, drug discovery, and material design. Similarly, Boiko et al. [17] presents an autonomous AI agent based on an LLM for chemical engineering research. Blanchard et al. [18] use masked LLMs for automating genetic mutations for molecules for application in drug likeness and synthesizability. Zhang et al. [19] present AutoML-GPT, an AI agent that acts as a bridge between various AI models as well as dynamically trains other AI models with optimized hyperparameters. Stella et al. [20] show that generative AI models can accelerate robot design process at conceptual as well as technical level. They further propose a human-AI co-design strategy for the same. Zheng et al. [21] explores the generative ability of GPT4 as a black-box optimizer to navigate architecture search space, making it an effective agent for neural architecture search. Singh et al. [22] study the utility of LLMs as task planners for robotic applications. Jablonka et al. [23] show that a fine-tuned GPT-3 model can outperform many other ML models, particularly in the low-data limit, for predictive Chemistry.

A common thread which passes through the studies mentioned above is the ability of LLMs to find an optimal solution for complex multi-objective optimization problems at hand. This has attracted a great deal of attention from the scientific community. Several studies have been published exploring the ability of LLMs to work as black-box optimizers. Melamed et al. [24] presents an automatic prompt optimization framework called PROPANE. In a method called InstructZero, Chen et al. optimize a low-dimensional soft-prompt to an open-source LLM, which in turn generates the prompt for the black-box LLM, which then performs a zero-shot evaluation [25]. The soft prompt is optimized using a Bayesian optimization method. Deepmind released a general hyperparameter optimization framework called Optformer based on the transformer architecture [26]. The idea of generation of optimized prompts automatically is also explored in Zhou et al. [27]. Similarly, Pryzant et al. [28] explores incorporating gradient descent into automatic prompt optimization. Chen et al. [29] introduces a discrete prompt-optimization framework incorporating human-designed feedback rules. We also see some examples of using LLM within a Reinforcement Learning (RL) framework for optimization [13,30,31].

While the examples mentioned so far focused on optimized prompt generation, a few studies have also explored using LLMs for mathematical optimization directly. Liu et al. [32] propose LLM-based Evolutionary Algorithm (LMEA), in which a LLM is responsible for the selection of the parent solution, mutation, cross-over, and generation of a new solution. Guo et al. [33] conducts an assessment of the various optimization abilities of LLM. Their study concludes that LLMs can perform optimization, including gradient descent, hill-climbing, grid search and black-box optimization well, particularly when the sample sizes are small. Pluhacek et al. [34] presents a strategy for using LLM for swarm intelligence-based optimization algorithms. Liu et al. [35] proposes LLM-based multi-objective optimization method, where LLM serves as black-box search operator for Multi-Objective Evolutionary Algorithms (MOEA). Liu et al. [36] proposes using LLMs for optimization algorithm evolution in a framework called AEL (which stands for Algorithm Evolution using LLMs). Liu et al. [37] adopt automatic hill-climbing process using language models as black box optimizers for vision-language models. They show that LLMs utilize implicit gradient direction for more efficient search. Optimization by PROmpting (OPRO) framework from Google Deepmind generates new solutions autoregressively and is seen to outperform human-level prompts [38]. We also see examples of using LLM for mathematical operations and optimization; for example, deep learning for symbolic mathematics [39], LLM for symbolic regression [40,41], using transformers for linear algebra, including matrix operations and eigen-decomposition [42] etc. In a framework called OptiMUS, LLM are used for formulating and solving Mixed-Integer Linear Programming (MILP) problems [43]. Zhang et al. [44] use LLMs for hyperparameter optimization. Romera-Paredes et al. [45] introduce an evolutionary procedure called FunSearch (short for searching in function space), where a pretrained LLM model is paired with a systematic evaluator for efficient function space search.

The literature cited here demonstrates beyond doubt the ability of LLMs to perform numerical optimization. However, the following questions remain unanswered:

• Can it be proved that LLMs develop an understanding of the objective function landscape and tune their search directions by leveraging this understanding? This issue is pivotal to employing LLMs for black-box optimization, and a positive response would solidify the credibility of LLMs as black-box optimizers.

• Is it possible to devise a reliable mechanism to ensure consistent performance from LLMs without relying on the ‘temperature’ parameter to induce randomness? It is well known that LLMs are prone to hallucinations, repetitive output, and can succumb to ‘mode collapse’ when used autoregressively [46]. Therefore, a foolproof framework is essential to render LLMs a feasible method for black-box optimization.

In this paper, our goal is to showcase the ability of LLMs to perform zero shot black-box optimization. We prove that LLMs showcase reasoning abilities and develop an understanding of the objective function landscape over the course of optimization. Via judicious balance between exploration and exploitation strategies, elitism as guardrails and engineering workarounds, we develop a robust framework to solve various optimization problems that hold practical relevance for the industry. We call this method a Large Language-model based Evolutionary Optimizer (LEO). The main contributions of this paper are as follows:

1. We introduce a novel population-based, parameter-free optimization approach in which an LLM is used to generate new candidate solutions or perturb existing solutions that perform exploration and exploitation of the design space. We employ elitism-based guardrails to retain best candidate solution in every optimization iterations. In addition, the hybrid optimization framework is assisted with engineering workarounds to overcome the limitations of LLMs.

2. We present distinguishing features of our method compared to other auto-regressive, evolutionary, or population-based methods using LLMs for black-box optimization, such as in Liu et al. [32], Yang et al. [38], Guo et al. [33], Liu et al. [35] (Section 2). A detailed comparison of LEO with existing approaches in the open literature is made in Table 1 that highlights contrasting features and similarities.

3. We solve several single and multi-objective benchmark optimization problems, as well as explore the ability of LLMs to solve high-dimensional problems. Additionally, we demonstrate the application of our method to several engineering problems such as aerodynamic shape optimization, heat transfer, and windfarm layout optimization (Section 3).

4. To quantify the merits of our approach, we compare our method against the state-of-the art methods for optimization, for both gradient-based and gradient-free (Section 3).

5. We provide evidence of LLM’s ability to reason and perform numerical optimization with the help of two tests (Section 4).

In Section 2, we provide a detailed description of LEO. In Section 3, we highlight the performance of LEO for various benchmark optimization problems as well as engineering applications. This is followed by the Section 4, wherein we discuss the reasoning ability of LLMs to sample better candidate points that result in accelerated convergence followed by challenges associated with LEO and recommendations to overcome them. Lastly, in Section 5, we highlight conclusions.

## 2. Methodology

### 2.1. Motivation towards populated-based approach

We begin this section to provide a motivation behind perusing a population-based approach for solving complex non-convex optimization problem. While non-population-based or gradient-based methods are preferred for their quick turn-around time towards convergence, the final solutions are likely to get trapped in local optima for non-convex problems. In this section, we setup a quick experiment to demonstrate this idea via the LLM-assisted optimization framework without a population-based strategy to generate a new candidate solution for every optimization iteration. To further elaborate, the framework for this toy experiment uses LLM to generate a new candidate solution given the history of the number of previously generated solutions, $n_{\mathrm{hist}}$. This approach is strictly not a population-based strategy but rather an adaptive way to generate new candidate solutions that is likely to minimize a function value. The prompt corresponding to this test is mentioned in Table 10 in Appendix A. Throughout this work, we use the openAI GPT-3.5 Turbo 0613 (with occasional use of Llama 3.1-70B for comparisons) model for all our evaluations.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/c8ae4f5015205732053ad47ac817fa37cad9d8e5a8da9b06bdfdd6e45cc843ff.jpg)
*(a)*

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/698c290a526f94cf68070cc46f7e66a16560b05d628641a2560a89633fbe93ff.jpg)
*(b)*

*Fig. 1. Convergence towards optimal nose cone shape with increasing number of decision variables as context.*

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/bdf2374cc0a7c77a28bb3681f525de2439da7a425d1928aed72501e37605998f.jpg)
*(a)*

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/b03860cb35e14274303e3c08aecd51a811f860012a31a82ae364d51ee0206e60.jpg)
*(b)*

*Fig. 2. Performance of the LLM-based search algorithm (without a population) described in Section 2.1 for the 2D Rosenbrock function.*

#### Test problem 1: Nose-cone shape optimization

**Problem statement:** Obtain optimal nose-cone shape via minimizing drag coefficient.

The problem under consideration is the popular nose-cone shape optimization. This toy problem presents itself as a single objective, single variable convex optimization problem, where the objective is to determine the optimal decision variable (i.e., power-law index, $n$ or $n_{\mathrm{pl}}$) that results in minimal drag coefficient body. Please refer to Appendix A.2 in the appendix for further information. Fig. 1(a) shows the variation of the function value (i.e., drag coefficient) with optimization iterations for various $n_{\mathrm{hist}}$ values, where a remarkable observation is made. It is found that the LLM struggles to find the optimal solution when $n_{\mathrm{hist}} = 1$. This is not surprising and points to the lack of context in the limited number of previously generated candidate solutions. However, for higher $n_{\mathrm{hist}}$ values, the optimization framework quickly aligns itself with the optimal value. This is also true for the corresponding decision variable value $n$ in Fig. 1(b). This experiment signifies that the LLM-based hybrid optimization framework struggles to locate the optimal solution when limited context in terms of the history of previous candidate points has been passed to it in every optimization iteration. This is true for even a simple convex optimization, as described above.

#### Test problem 2: 2D Rosenbrock benchmark optimization function

**Problem statement:** Obtain minima of the 2D Rosenbrock function

We now extend the above approach to solve a two-dimensional (2D) benchmark optimization function, specifically the Rosenbrock function, using $n_{\mathrm{hist}} = 10$. The global optima for this function is i.e., $f_{\min} = 0$, corresponding to $x_{\min}, y_{\min} = (1, 1)$. Fig. 2(a) shows the variation of the objective function value, $f$, with optimization iterations. It is clear that the hybrid approach, despite being fed the history of the last 10 candidate points, suffers from premature convergence. This can be confirmed from Fig. 2(b) where the decision variables $x, y$ are seen to converge to a local optimal solution. This failure signifies that a single trajectory of past candidate points alone is not sufficient to converge to the global optimal solution, as they might be points around the basins of a local optimal solution. It must be remarked that while Yang et al. [38] allude to this phenomenon as optimization stability resulting from sensitivity to prompt, we are of the opinion that this is rather a misnomer and stems purely from a lack of context and richness of information from the decision variable space. Besides, our work provides a compelling demonstration of this logic.

From the above two experiments, we can see that a hybrid LLM-based optimization approach without the rich information spanning the decision variable space (i.e., exploration) is unlikely to be useful for the LLM to generate good candidate points around the present candidate solutions (i.e., exploitation). While several other methods, exemplified by Liu et al. [32], Yang et al. [38], Guo et al. [33], incorporate the concept of exploration versus exploitation during solution search in an auto-regressive manner, they rely on adjusting the temperature of the LLM to toggle between these two modes. However, this approach, although effective in simpler scenarios, has limitations. Specifically, a higher temperature introduces greater randomness into the LLM’s output, potentially hindering its ability to learn the true distribution of solutions. Conversely, an excessively low temperature can lead to mode collapse—a situation where the model consistently predicts the same (incorrect) solution. This phenomenon is particularly associated with auto-regressive LLMs, as highlighted by Hopkins et al. [46]. Besides, the use of temperature as a user-defined parameter (unlike the self-adaptive mechanism introduced by Liu et al. [32]) on top of an already complex optimization method may diminish its efficiency or user friendliness. In the same vein, we will also illustrate how an existing evolutionary optimization framework utilizing user-defined parameters such as probability of crossover and probability of mutation can be simply replaced by using LLM, without the introduction of temperature.

Table 1 highlights differences between the proposed method (LEO) and other contemporary LLM-based numerical optimization methods, particularly [32–36,38] on the basis of the optimization strategy, the exploration strategy, LLM models used, and the problem as well as the objective function complexity. What mainly sets LEO apart compared to the other methods are the following three points:

1. A unique explore–exploit strategy to scan the objective function landscape and arrive at the global optima. Unlike the other methods, in LEO the underlying LLM is tasked with coming up with bounds on decision variables for exploit and explore actions based on the in-context examples and historical data. We later show in the manuscript that over the course of the optimization, the LLM learns the optimal distribution for the explore and exploit pools.

2. In contrast to other methods, the LLM is tasked with creating a meta-model in its ‘mind’. Thus, the notion of distance, i.e., in terms of ‘near by’ regions and ‘far away’ regions for constructing exploit and explore pools respectively is constructed by LLM based on the in-context and the historical data. This seems to work well and indicates a mathematical understanding of the natural language by LLM.

3. We develop elitism-based guardrails to correct the course of optimization and subsequently continue in-context learning of the LLM in the next iterations. This enhances the performance of the LLM. We demonstrate that the central duty of optimization is still performed by LLMs, and not by elitism based guardrails in Section 4.1.3.

In the following section, we propose a novel approach that uses LLM to generate points that maintain a fine balance between exploration and exploitation, to arrive at the global optimal solution. This is unlike the existing approaches, where population based evolutionary optimization strategies like genetic algorithms employ mutation and crossover to perturb the parent solution for offspring.

### 2.2. Large language-model based evolutionary optimizer (LEO)

In this section we present a novel optimization strategy called Large Language-model-based Evolutionary Optimizer (LEO) which is aimed to address the shortcomings discussed in the earlier section. The LLM-assisted optimizer is designed to sample better candidate solutions from parent solutions which results in faster convergence towards global optimal solution. In addition, strategies such as elitism-based guardrails and engineering workarounds allow for a robust and problem agnostic framework.

**Parent/initial solution generation:** Being a population-based strategy, LEO generates a pool of Parent solutions as initial guess at optimization iteration $\mathrm{iter} = 0$. There are a total of $N_{\mathrm{pop}}$ solutions generated, each for explore and exploit pool. The $2N_{\mathrm{pop}}$ parent solutions generated at this point are evaluated based on their cost function to render us the fitness score (see Fig. 3).

**Candidate solution generation:** At this stage, we generate candidate solutions from the Parent (or initial) solutions. To generate new candidate solutions that allow for a balance between exploitation and exploration, we invoke the LLM via two different prompts, i.e., shown in Tables 7 and 8, to finally create exploit and explore pools of candidate solutions. These pools of candidate solutions further undergo function evaluations for their respective fitness scores and are appended to the solutions from previous generations, resulting in $2N_{\mathrm{pop}}$ solutions in each pool.

**Elitism as guardrails:** To keep track of the best solutions at every optimization iteration, we invoke the elitism principle i.e., we keep our promising candidate solution in the design space by taking them directly to the next optimization iteration. This is achieved in two steps. Firstly, we employ the Port and Filter operation to move the $N_{\mathrm{port}}$ number of best solutions from the explore pool to the exploit pool (and remove the $N_{\mathrm{port}}$ worst solutions from the exploit pool). This allows the exploitation search restricted to good candidate solutions. Secondly, the $2N_{\mathrm{pop}}$ solutions in each pool are sorted in order of their function values, and only the top $N_{\mathrm{pop}}$ solutions are retained in each pool. This step is summarized in the form of a pseudo code in Algorithm 1, where the output generated at the end are the pool of explore and exploit candidate solutions.

**Termination criterion:** The above steps are repeated until total iterations (i.e., maximum generations or optimization iterations) $\mathrm{maxIters}$ is reached. For typical optimization exercise, $\mathrm{maxIters}$ depends on factors such as population size $\mathrm{popSize}$, number of variables $\mathrm{numVars}$, etc. As we will show in the following sections, for most problems with $\mathrm{numVars}=2$ and $\mathrm{popSize} = 8\text{--}10$, the $\mathrm{maxIters}=30$ is sufficient for LEO to converge.

#### Algorithm 1: LEO algorithm

```text
Algorithm main()
1  Input: popSize, maxIters, numVars, prompt_init, jitter_explore, jitter_exploit, solToPort, objFun(.)
2  Minimise: minFunVal
3  x_init ← random(popSize, numVars), x_explore ← x_init, x_exploit ← x_init ;
4  y_init ← objFun(x_init), y_explore ← y_init, y_exploit ← y_init ;
5  iter ← 1 ;
   while iter ≤ maxIters do
       prompt_explore ← PromptCreate(prompt_init, jitter_explore, x_explore) ;
       prompt_exploit ← PromptCreate(prompt_init, jitter_exploit, x_exploit) ;
       x̃_explore, x̃_exploit ← LLM(prompt_explore), LLM(prompt_exploit) ;
       ỹ_explore, ỹ_exploit ← objFun(x̃_explore), objFun(x̃_exploit) ;
       x_explore, y_explore ← UpdatePopulation(x_explore, y_explore, x̃_explore, ỹ_explore) ;
       x_exploit, y_exploit ← UpdatePopulation(x_exploit, y_exploit, x̃_exploit, ỹ_exploit) ;
       x_exploit, y_exploit ← PortFilter(x_explore, y_explore, x_exploit, y_exploit, solToPort) ;
       x_exploit, y_exploit ← sort([x_exploit, y_exploit]) ;
       x_explore, y_explore ← sort([x_explore, y_explore]) ;
       iter ← iter + 1
   end
6  minFunVal ← y_exploit[0] ;
7  return minFunVal

Procedure PromptCreate(prompt, jitter, x)
1  x_jitter ← x + jitter ;
2  prompt_update ← Insert x_jitter in prompt ;
3  return prompt_update ;

Procedure UpdatePopulation(x, y, x̃, ỹ)
1  x ← concat(x, x̃) ;
2  y ← concat(y, ỹ) ;
3  return x, y ;

Procedure PortFilter(x_explore, y_explore, x_exploit, y_exploit, numPort)
1  x_top, y_top ← sort([x_explore, y_explore])[0:numPort] ; /* Sort w.r.t y_explore */
2  x_exploit, y_exploit ← sort([x_exploit, y_exploit]) ;     /* Sort w.r.t y_exploit */
3  x_exploit[-numPort:] ← x_top ;
4  y_exploit[-numPort:] ← y_top ;
5  return x_exploit, y_exploit ;
```

### 3. Numerical results

In this section, we evaluate our proposed optimization strategy via a series of test cases, ranging from simple benchmark problems to engineering applications. These tests are classified into four categories: (a) simple benchmark function optimization problems; (b) multi-objective optimization problems; (c) high-dimensional benchmark optimization problems; and (d) industry-relevant engineering optimization problems. This allows us to scrutinize the approach for a range of problems of different complexity. In order to demonstrate the efficiency of the proposed method, in each experiments, we compare LEO with a few benchmark algorithms.

#### Table 1: Comparison of LEO with other contemporary LLM-based optimization methods

| Method | Optimization Strategy | Exploration Strategy | LLM | MO |
| :--- | :--- | :--- | :--- | :---: |
| **Hybrid swarm intelligence optimization algorithm** [34] | Swarm-based meta-heuristic algorithm generation with LLM. | Depends on underlying swarm-based optimization method | GPT-4 | Yes |
| **Multi-Objective Evolutionary Algorithms (MOEA)** [36] | 1. LLM used as black-box search operator for decomposition-based Multi-objective evolutionary algorithm (MOEA/D); 2. Design an explicit linear white-box operator with randomness (MOEA/D-LO) to model LLM | 1. In-context learning of LLM with temperature-induced randomness; 2. Explicit weighted Linear Operator with randomness for promoting exploration | GPT-3.5 Turbo | Yes |
| **LLM-driven Evolutionary Algorithm (LMEA)** [32] | 1. LMEA instructs LLM to select parent solutions from a population of candidate solutions; 2. LMEA performs crossover and mutation to generate offsprings | 1. LLM is used as a black-box model to generate new samples; 2. Temperature of LLMs (automatically adjusted) is used for encouraging exploration | GPT-3.5 Turbo (0613 version) | No |
| **Optimization by PROmpting (OPRO)** [38] | 1. Meta prompt for generating new samples of candidate points; 2. In-context learning (i.e. historical values of candidate points with scores) | 1. LLM temperature is used for managing exploration–exploitation tradeoff; 2. Multiple solutions generated to ensure stability | PaLM 2-L-IT, PaLM 2-L, GPT-3.5 Turbo, GPT-4 | No |
| **LLMs as black box optimizers** [33] | 1. LLMs are used for four optimization tasks: Gradient-descent, Hill-climbing, Grid-search, Black-box; 2. LLMs are guided by iterative prompting with Chain of Thoughts (CoT) method. | 1. Different strategy for each task; 2. Temperature fixed at 0.8; 3. Uncertainty metric defined to quantify randomness | GPT-3.5 Turbo (0613 version) | No |
| **Algorithm Evolution using LLM (AEL)** [36] | 1. AEL treats each algorithm as individual, and utilizes an evolutionary framework to evolve new algorithms; 2. Crossover and mutation are performed with prompt engineering | 1. Crossover ensures exploration | GPT-3.5 Turbo, GPT-4 | Yes |
| **LEO (ours)** | 1. LLMs are tasked with generating parameter values within bounds to perform exploration and exploitation (using prompt engineering); 2. The notion of distance, i.e., ‘near by’ regions and ‘far away’ regions is determined by LLMs based on the historical data; 3. Elitism based guardrails are used to correct the course of optimization and for further in-context training | 1. Bounds on the decision variables are decided by the LLM based on the historical values; 2. The prompt instructs the LLM to explore ‘far away’ regions for exploration and ‘near by’ regions for exploit; 3. Based on in-context data, the LLM dynamically determines a better direction for search via the explore pool (refer Section 4.1.3) | GPT-3.5 Turbo, Llama 3.1 | Yes |

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/97b9095d13e0d5a6f9654e3270ac6af6acf6282d7308b987b8919390d30820e4.jpg)

*Fig. 3. Schematic behind the working principle of LEO. The central figure illustrate the flowchart behind the intermediate steps involved from initial guess to final optimal solution whereas the supporting 3D plots along the edges underscore the distribution of initial, explore and exploit pool of points.*

#### Table 2: 2D Benchmark test functions with their equations, bounds, and global minima

| 2D Benchmark Objective Functions | Equation | Global Minima | Bounds |
| :--- | :--- | :--- | :---: |
| **ScaledSphere** | $f(x, y) = x^2 + y^4$ | $f(0., 0.) = 0.$ | $-1 \le x, y \le 4$ |
| **Himmelblau** | $f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2$ | $f(3.0, 2.0) = 0.0$; $f(-2.805118, 3.131312) = 0$; $f(-3.779310, -3.283186) = 0$; $f(3.584428, -1.848126) = 0$ | $-4 \le x, y \le 4$ |
| **Rosenbrock2D** | $f(x, y) = 100(y - x^2)^2 + (1 - x)^2$ | $f(1., 1.) = 0.$ | $0 \le x, y \le 2$ |
| **Sphere2D** | $f(x, y) = x^2 + y^2$ | $f(0., 0.) = 0.$ | $-1 \le x, y \le 1$ |
| **Beale** | $f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2$ | $f(3., 0.5) = 0.$ | $0 \le x, y \le 5$ |
| **Goldstein-Price** | $\begin{aligned} f(x, y) = &\left[1 + (x + y + 1)^2(19 - 14x + 3x^2 - 14y + 6xy + 3y^2)\right] \\ &\times \left[30 + (2x - 3y)^2(18 - 32x + 12x^2 + 48y - 36xy + 27y^2)\right] \end{aligned}$ | $f(0., -1) = 3.$ | $-2 \le x, y \le 2$ |

### 3.1. Benchmark optimization functions

We begin our experiments by recognizing the need to evaluate the merits of LEO on a range of test cases against established optimization methods. Consequently, in this section, we compare the proposed optimization algorithm (LEO) with various gradient-based and gradient-free optimization algorithms for 6 different 2D benchmark optimization problems. The corresponding equations describing the benchmark optimization function, decision variable bounds, as well as the decision variables corresponding to the global optima are shown in Table 2.

#### 3.1.1. Quantitative evaluation

Among the gradient-based optimization algorithms, we choose Stochastic Gradient Descent (SGD) [47], Adam [48], and L-BFGS-B [49] algorithms, whereas among the gradient-free optimization algorithms, we employ Simulated Annealing (SA) [50], Covariance matrix adaptation evolution strategy, CMA-ES [51], and Constrained Optimization by Linear Approximation, COBYLA [52] to compare against LEO.

**Problem formulation:** We investigate the performance of the aforementioned state-of-the-art (SOTA) optimization methods on the 6 different benchmark optimization functions mentioned in Table 2. Each experiment using an optimization method for a particular benchmark function is repeated 100 times using as many different random number seeds to obtain the statistically significant number. Table 3 reports the median values from these experiments at the end of 1000 function evaluations.

**Analysis:** It is found that L-BFGS-B and CMA-ES are the best performing SOTA gradient-based and gradient-free optimization methods, respectively achieving the global optima in all the benchmark optimization functions. However, it is interesting to note that LEO (using ChatGPT as well as Llama 3.1-70B model) also attains the global minima, albeit with a slightly lower precision when compared across the 100 independent runs. This confirms that LEO is indeed able to render robust solutions to complex optimization problem irrespective of the LLM being used as the workhorse. Moreover, and more importantly, LEO perform almost at par with the established SOTA methods like L-BFGS and CMA-ES while outperforming methods like SGD, SA, COBYLA and even Adam, especially on the Goldstein-Price problem.

#### 3.1.2. Qualitative evaluation

To further demonstrate the efficiency of LEO, we present a qualitative visualization of the convergence behavior for all the benchmark optimization functions. In Fig. 4 we show the gradual variation of the solutions in the exploit pool by plotting the solution with the lowest function value in exploit pool $f_{\min}$ with optimization iterations for a particular representative case. Every solution is color coded based on the optimization iteration at which it was generated. It is found that while sampling of random initial solution results in $f_{\min}$ being quite distant from the respective global optimal solutions, as optimization progresses, LEO is able to quickly navigate towards the global optimal solution. It must be noted that the convergence rate is rather high at the start of the optimization and later tapers off near the global optimal solution depending upon the topology of the functional value space (will be quantitatively demonstrated later in following sections). This is indicative of the fact that the rate of convergence at the start is heavily influenced by solutions obtained from exploration pool while towards the end, the success of the approach is largely determined by the solutions obtained from exploitation pool.

#### 3.1.3. Computational cost

To enable a fair comparison across multiple gradient-based and gradient-free optimization methods, it is imperative to consider the number of function evaluations in place of the total number of optimization iterations. This necessitates treating the total number of optimization iterations as a variable. Consider the case where the gradient descent method for a problem with $n_{\mathrm{var}}$ or $n_{\mathrm{dim}}$ variables (or dimensions) requires $n_{\mathrm{dim}} + 1$ number of function evaluations per optimization iteration compared to a population-based approach with population size $n_{\mathrm{pop}}$ that warrants $n_{\mathrm{pop}}$ number of function evaluations per optimization iterations. Considering this, in the present work, we choose to keep the same number of function evaluation calls (i.e., 1000), or 30 optimization iterations for LEO.

For the above set of experiments reported in Table 3, optimization algorithms like CMA-ES and LBFGS are extremely fast. For instance, 100 randomly seeded experiments take less than a minute across all 6 benchmark functions. On the contrary, LEO (with ChatGPT-3.5 as the LLM), however, takes approximately 3 h for all six functions to be repeated using 100 random seeds. We attribute this high computational time by LEO to predominantly 2 factors. Firstly, time consumed by the LLM function call: from our experiments, each call to the LLM with a $\sim 1000$ token prompt takes anywhere between 15–20 s for a response from the OpenAI servers. This is already quite time-prohibitive. Secondly, with the advent of open-source LLM models such as Llama 3.1-70B model, one could improve the inference speed with ideas like FlashAttention [53,54], model quantization [55,56]. However, such approaches are beyond the scope of the present work and shall be part of future work.

### 3.2. Multi-objective optimization problem

In the previous section, we demonstrated the merits of LEO in tackling single objective optimization problems for 2D standard benchmark optimization functions. In this section, we further explore the capabilities of LEO in tackling multi-objective optimization problems. The objective of this section is twofold: (a) Demonstrate that LLM can be used as plug-and-play or modular feature in the existing state-of-the art evolutionary optimization approaches, (b) demonstrate the ability of the resulting framework (LEO-modular) to minimize multiple conflicting objectives.

#### Table 3: Comparison of the median values for the minima obtained using various benchmarks and LEO (exploit pool) at the end of 1000 function evaluations

| 2D Benchmark Objective Functions | Global minima | SGD | SGD ($\nu=0.5$) | SGD ($\nu=0.9$) | Adam ($\beta_1=0.1$) | Adam ($\beta_1=0.5$) | Adam ($\beta_1=0.9$) | L-BFGS-B | SA ($T=0.1$) | SA ($T=1$) | SA ($T=10$) | CMA-ES | COBYLA | LEO (ChatGPT-3.5) | LEO (Llama 3.1-70B) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ScaledSphere** | 0.0000 | 0.0039 | 0.0008 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Himmelblau** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0003 | 0.0003 | 0.0002 | 0.0000 | 0.0000 | 0.0086 | 0.0000 |
| **Rosenbrock** | 0.0000 | 0.0362 | 0.0479 | 0.0406 | 0.0986 | 0.0192 | 0.0155 | 0.0000 | 0.0001 | 0.0002 | 0.0005 | 0.0000 | 0.0965 | 0.0012 | 0.00234 |
| **Sphere** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **Beale** | 0.0000 | 0.1330 | 3.2964 | 4.6201 | 1.1810 | 0.1995 | 0.9322 | 0.0000 | 0.0000 | 0.0000 | 0.0002 | 0.0000 | 0.0071 | 0.0156 | 0.0000 |
| **GoldsteinPrice** | 3.0000 | 72.5447 | 54.4256 | 30.3447 | 39.4401 | 30.0070 | 30.0000 | 3.0000 | 30.0038 | 30.0047 | 30.0044 | 3.0000 | 60.1236 | 3.0023 | 3.0000 |

*Note: For SGD with momentum, $\nu$ represents the momentum coefficient. For Adam, $\beta_2 = 0.999$. For Simulated Annealing (SA), exponential temperature decay is used with initial temperature $T$.*

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/0f8cbae74ebe52c9dddd3e7ad3879380df9248a14927af85bc9b84e86901685c.jpg)

*Fig. 4. 2D benchmark functions and their convergence plots using LEO optimization.*

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/14c2c2d1dc096936585ac416d83aa43e155dda3865a2cb94dcd837e042ff3018.jpg)

*Fig. 5. Original and the LEO-modular NSGA-II framework.*

#### 3.2.1. Problem setup

Multiple objectives are ubiquitous in many practical engineering domain. The need to minimize multiple conflicting constraints gives rise to a family of optimal solutions known as the Pareto optimal solution or Pareto optimal front. In this section, we show that the LLMs can indeed be used as a plug-and-play feature in existing optimization frameworks and perform the role of generation of good candidate solutions that lead to the correct Pareto Optimal front. To demonstrate this idea, we adopt Non-Dominated Sorting Genetic Algorithm (NSGA) II [57].

In Fig. 5 we show the original NSGA II and the proposed modular feature of the LLM assisted NSGA II algorithm (or LEO-modular). As can be seen, the idea in LEO-modular is simply to replace the modules responsible for generating/perturbing the candidate solutions, i.e., crossover and mutation. Instead, we now employ Chat GPT 3.5-Turbo 0613 model to do the same task of generating/perturbing candidate solutions using prompts shown in Table 9. The crossover parameter employed in NSGA II algorithm generates offspring by switching parts of the same chromosome among two parents. This operation retains the best of both parents chromosomes and often leads to candidate solutions or offsprings in the near vicinity to the parents solutions, encouraging exploitation. The mutation operator in binary coded genetic algorithm randomly flips the binary encoded bits of chromosomes based on certain mutation probability. This often results in significant jump in the parent solution, thus promoting exploration. The numerical experiments using LEO in the previous section has already demonstrated evidence of generating solutions that exhibits attributes of exploration and exploitation. We now demonstrate proof of concept that LLM-based exploration and exploitation works equally well within the NSGA II framework.

We now elaborate further to describe how the exploration/exploitation of design variables are performed in LEO-modular. In order to fit the scope of the original NSGA II algorithm for exploitation/exploration, we generate two offsprings using LLM via two parents as context at a given time. This must be contrasted with the approach described in Section 2.2, where we generate the entire population of solutions at a time. To demonstrate the merits of the present approach we undertake two test cases named after the researchers of the seminal work [58]: ZDT1 and ZDT3. The equations describing the multiple objective functions $f_1, f_2$ are listed in Table 4, while also highlighting the complexity associated with the two functions. We employ these test experiments as a two-variable problem, i.e., $n_{\mathrm{dim}} = 2$, in Table 4. The ZDT1 test was undertaken using a population size of 10, whereas the ZDT3 test was performed using a population size of 30. Higher population size for the ZDT3 test allows for the crisp capture of the segmented Pareto optimal front.

#### 3.2.2. Evaluation

To enable a fair comparison across the two aforementioned approaches, every input parameter including population size, objective functions, number of optimization iterations etc., are kept same. Both these test cases were run up to a total of 40 generations or optimization iterations. To quantitatively demonstrate the effectiveness of the two approaches, we compare the Pareto optimal front obtained from both LEO-modular and NSGA-II in Fig. 6. For the sake of comparison, we also present the solutions obtained from yet another implementation of the NSGA II algorithm using the open-source Pymoo [59] library, which we will treat as the reference solution. Pymoo-based solutions were generated using a much larger population size $N_{\mathrm{pop}} = 200$, and the optimization was carried out for a total of 100 generations. Consequently, the solutions would exhibit a dense Pareto optimal front. It can be seen that the Pareto optimal front obtained from LEO-modular is in agreement with the Pareto optimal front obtained from the traditional NSGA II algorithm. Fig. 6(b) shows the Pareto optimal front for the ZDT3 test case, where a discontinuous front has been obtained. It is again found that the solutions from both approaches are in excellent agreement despite the presence of discontinuities. These observations bears a remarkable significance, i.e., that the LLM-based candidate solutions are able to perform exploration and exploitation that distinctly identify the Pareto optimal front without any temperature-based parameter. This is especially significant considering that a low population size and relatively low number of generations were employed to get the final Pareto optimal front. In the view of the authors, this is the first ever instance where LLM have been used to perturb parent solutions to generate offsprings purely based on prompts and should be viewed differently than using LLM to generate offsprings via crossover and mutation.

#### Table 4: Equations and complexities of ZDT1 and ZDT3 functions

| Function | Equations | Complexity |
| :--- | :--- | :--- |
| **ZDT1** | $\begin{aligned} f_1(\mathbf{x}) &= x_1 \\ f_2(\mathbf{x}) &= g\left(1 - \sqrt{f_1/g}\right) \\ g &= 1 + \frac{9}{n-1} \sum_{i=2}^{n} x_i \end{aligned}$ | Non-uniformity in the search space and a convex Pareto-optimal front. Requires algorithms to maintain diversity. |
| **ZDT3** | $\begin{aligned} f_1(\mathbf{x}) &= x_1 \\ f_2(\mathbf{x}) &= g\left(1 - \sqrt{f_1/g} - (f_1/g)\sin(10\pi f_1)\right) \\ g &= 1 + \frac{9}{n-1} \sum_{i=2}^{n} x_i \end{aligned}$ | Disconnected Pareto-optimal front segments introduce additional complexity. Maintaining population diversity across gaps is challenging. |

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/b8765f23906cecaedb10089124c38d42b297afeac122140fb208fbde93d3af99.jpg)
*(a) test case: ZDT1*

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/545d47588f7b2eeb60913557f0be46b622953c60a0df3ef8b876482aaa757589.jpg)
*(b) test case: ZDT3*

*Fig. 6. Comparison of the Pareto optimal front obtained from the present approach with the NSGA II algorithm. The ZDT1 test was undertaken using a population size of $N_{\mathrm{pop}}=10$, whereas the ZDT3 was performed using a higher population size of $N_{\mathrm{pop}}=30$ to crisply capture the segmented Pareto optimal front. Pymoo-based reference solutions were generated using $N_{\mathrm{pop}} = 200$, and $N_{\mathrm{maxGen}} = 100$.*

### 3.3. High-dimensional problem

#### 3.3.1. Problem setup

We extend the scrutiny of LEO to high-dimensional problems by considering the $n$-dimensional ($n_{\mathrm{dim}}$) form of the Rosenbrock function as given below:

$$
f(\mathbf{x}) = f(x_0, x_1, \dots, x_{n-1}) = \sum_{i=0}^{n_{\mathrm{dim}}-1} \left[ 100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2 \right], \quad -\infty < x_i < \infty, \; \forall i \tag{1}
$$

This function has a global minimum of 0 as follows:

$$
f\Big(\underbrace{1, \dots, 1}_{n\text{ times}}\Big) = 0
$$

The global optima for the Rosenbrock function, $f_{\min} = 0$, corresponds to decision variable $\mathbf{x}_{\mathrm{var}} = (1, \dots, 1)$. While LEO is able to accurately predict this optima, it can be argued that ‘1’ may be one of the most common numbers encountered by the LLM during its training phase (LLMs are trained on a large variety of text data, and the number 1 can be considered to be widely occurring across multiple contexts and in various instances. Thus because of the autoregressive nature of LLMs, the next token predicted is very much influenced by the occurrence of a particular token in the training corpus), causing the LLM to predict this number simply because of its training history. This may result in the LLM predicting the correct solution by the virtue of it being a ‘trivial’ solution. To address this issue, we modify the Rosenbrock function, to have the minima shifted to an arbitrary location, as shown below:

$$
f(\mathbf{x}) = f(x_0, x_1, \dots, x_{n-1}) = \sum_{i=0}^{n_{\mathrm{dim}}-1} \left[ 100\left((x_{i+1} - a) - (x_i - a)^2\right)^2 + (1 - (x_i - a))^2 \right], \quad -\infty < x_i < \infty, \; \forall i \tag{2}
$$

This function has a global minimum of $f_{\min} = 0$ as follows:

$$
f\Big(\underbrace{1 + a, \dots, 1 + a}_{n\text{ times}}\Big) = 0
$$

with $a$ randomly selected as 0.2913. It is important to note that performing the aforementioned exercise, naturally forces LEO to thoroughly search for better candidate solutions rather than returning a trivial one.

#### 3.3.2. Evaluation

To present the convergence pattern obtained from LEO on the $n$-dimensional Rosenbrock problem with increasing dimensionality, we present the distribution of the mean objective function value in the exploit pool plotted with respect to the number of optimization iterations. We notice that as the dimensionality increases, the cost function value also increases. This exercise has been repeated using LEO-based on Llama 3.1-70B model, as shown in Appendix in Appendix A.1. This is not surprising and rather aligns well with the intuition that with higher dimensionality of the problem, the population size as well as the number of optimization iterations should also be increased. Nevertheless, the intention behind performing this exercise is to convey the consistency of the convergence rate for high-dimensional problem when limited to certain number of optimization iterations.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/46ee23c5ac8baa5240aa7fb0738039bd42f74b26160e352394f2fa85b3766da4.jpg)

*Fig. 7. Convergence pattern in the $f_{\min}$ obtained from exploit pool for the Rosenbrock function with different dimensions. The experiments were undertaken by varying $n_{\mathrm{dim}}$ from 2 all the way up to 25 dimensions, giving a total 7 instances of the functions with increasing dimensionality, i.e. 2, 4, 6, 8, 10, 20, 25. Each optimization problem is run for $N_{\mathrm{maxGen}} = 100$ optimization iterations, while performing 30 such independent runs for each problem to get statistically significant results.*

### 3.4. Engineering applications

In this section, we examine the utility of LEO for practical problems with industrial relevance. We have selected three applications for this purpose: (a) nozzle shape optimization for supersonic flow (b) 1D steady-state heat transfer problem; (c) wind-farm layout optimization. In the following subsections, we discuss the results obtained with LEO. The details of each of the problem setups are provided in Appendix A.

#### 3.4.1. Nozzle shape optimization

**Objective function and Shape Parameterization:** We begin by showcasing the utility of LEO for industrial-scale optimization problems, specifically the nozzle shape optimization for minimum radial velocity at the exit. The nozzle shape is represented using a Cubic Bezier curve. The length of the nozzle along with the inlet and outlet radii are kept constant, resulting in an effective 4-dimensional ($n_{\mathrm{var}} = 4$) problem. The radial velocity at the exit of the nozzle is computed using the area-Mach Number relation for high-speed compressible flows (please refer to Appendix A.3 in the Appendix).

**Evaluation:** Fig. 8(a) shows the juxtaposition of the initial nozzle shapes that are fed to both explore and exploit pools. Fig. 8(b) presents the final nozzle shapes from the explore pool after 30 optimization iterations, with minor variations as expected. The optimal shape obtained from the exploit pool with the least radial velocity at the nozzle exit is shown in Fig. 8(c). It is noted that the nozzle shape obtained from the final explore and exploit pool clearly demonstrated bell shaped contours, which augers well with the expected shapes derived using isentropic flow theory [60]. This indicates that LEO is able to refine its search towards obtaining optimal shapes that are consistent with expected theory, albeit for a 4D problem. Lastly, Fig. 13 (in Appendix A.3 in the Appendix) shows the convergence history in terms of minimum as well as average function value $f_{x_1, x_2}$ in both explore and exploit pool with optimization iterations. It is clearly observed that LEO is able to rapidly converge to optimal solution within a few optimization iterations.

#### 3.4.2. One-dimensional heat transfer problem

**Objective function:** In this section, we aim to find the solutions to 1D steady-state heat transfer problem in a domain discretized using $n_{\mathrm{dim}}$ points using LEO. Heat transfer is a fundamental phenomenon in many industrial applications. While robust numerical methods like finite difference coupled with iterative approaches allow us to compute the discretized solution to the governing equations, the same can be viewed as an optimization problem, i.e., temperature distribution that results in minimization of the residual loss of the underlying governing equation (please refer to Appendix A.4 in the Appendix). All the test cases in this section are solved using a population size of 10 for a total of 100 optimization iterations.

**Evaluation:** Fig. 8(d) shows the steady-state temperature distribution obtained by LEO and its comparison with the exact solution for $n_{\mathrm{dim}} = 2$. While $n_{\mathrm{dim}} = 2$ results in a rather jagged temperature profile, it is clear that the solution obtained from LEO closely resembles with the exact solution. This signifies that a solution for a low-dimensional problem can result in solutions that align with underlying physics. Fig. 8(e) shows the steady-state temperature profile for $n_{\mathrm{dim}} = 4$ where minor observable differences with the exact temperature profile are noted. Similarly, Fig. 8(f) shows a stark contrast between solutions obtained from LEO and their comparison with the exact temperature profile. This is not surprising and rather points to the observation that has been made in Section 3.3. Consequently, it can be remarked that while the present approach performs remarkably well in yielding solutions grounded in physical laws for low-dimensional problems, the nature of the solution starts to deviate significantly for problems with higher degrees of freedom.

#### 3.4.3. Windfarm layout optimization

**Objective function:** In this section we investigate LEO for windfarm layout optimization that aims to enhance the efficiency and output of the wind farms by manipulating the position of the wind turbines. We opt for three test cases with 2, 4, and 8 turbines with the domain. This results in 4, 8, and 16 decision variables, respectively, for the optimization problem as each turbine is uniquely specified in a 2D domain with $(x_{\mathrm{var}}, y_{\mathrm{var}})$ coordinates. We will treat the solution generated using Scipy’s SLSQP method as the reference solution. For details related to this problem please refer to Appendix A.5 in Appendix A.

**Evaluation:** Fig. 8(g)–(i) shows the kernel density estimation (KDE) for the optimal wind turbine locations obtained using LEO considering 30 independent runs. It is noted that LEO is able to produce layouts with a distribution that is in agreement with what is obtained with conventional Scipy’s SLSQP method (see Figs. 8(j)–(l)). This is especially true for smaller number of turbines i.e., $n_{\mathrm{dim}}$ up to 8, whereas some observable differences are noted for the case with $n_{\mathrm{dim}} = 16$. In addition to the distribution of the optimal turbine location, Fig. 14 (in Appendix A.4 in the Appendix) shows the Annual Energy Production (AEP) value with iterations for a representative example in each of the three cases. These figures demonstrate the optimization trajectory of the promising solutions generated using LEO. This clearly showcases that LEO can be applied seamlessly across multitude of test cases of industrial scale.

## 4. Discussion

As alluded to in the earlier sections, one of the novel aspects of LEO is its ability to reason towards generating better candidate solutions over the course of optimization. In this section, we undertake experiments to demonstrate the reasoning capabilities of LEO backed by strong evidence. We show that LLMs inherently possess attributes of reasoning that, when assisted by an elitism criterion, allow for a hybrid framework that renders faster convergence towards a global optimal solution for highly non-convex optimization problems. We then list out various challenges that are encountered while using LLMs as optimizers. This is followed by enumerating various engineering tweaks and strategies that were devised in conjunction with LLMs that allow us to circumvent such issues. In our experience, these tweaks have been instrumental in getting the best optimization performance out of the LLMs. Finally, we provide some future directions towards improving the performance of LEO for high dimensional problems.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/ac75fb9a57a4343052f08bfb6a33cdee414786655a4c7dc8816f4316ed2fcc7f.jpg)

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/5239b8df4ec5163bf71372156a32890d02b4de58217b7cd5468b287df6008e81.jpg)

*Fig. 8. Solutions obtained from LEO and its comparison. Top row: (a) Initial pool of solutions, (b) explore pool of solution, and (c) exploit pool of solutions obtained from LEO. Second row: Heat equation optimization for increasing number of variables. Bottom two rows: Density distribution of the optimal turbine location from all independent runs with increasing dimensionality.*

### 4.1. Proof of LLM reasoning capabilities

The elitism approach that we adopt in this paper, i.e., the explore–exploit-port-filter strategy, naturally prompts the question, ‘‘Does LLM’s reasoning indeed result in better candidate solutions, resulting in faster convergence to global optimal solution, or would a LLM tasked with the generation of random exploit and explore solutions result in better convergence?’’ To systematically address this issue, we designed the following two experiments: (a) In the first experiment, we examine the performance of LLM assisted by guardrails (i.e., LEO) and compare with the same framework where LLM are used to randomly generate explore and exploit solutions without guardrails (LEO-Rnd) (b) In the second experiment, we undertake additional statistical evaluations to exhibit the reasoning component of the LLM in arriving at the final optimal solution. We discuss our findings in the next two sub-sections.

#### 4.1.1. Experiment 1: LLM solely as solution generator

In this experiment, we designate the LLM as the role of only a random solution generator (LEO-Rnd), i.e., random perturbation of candidate points, without any historical context or guardrails such as elitism (please refer Tables 7 and 8 to understand how the prompts have been setup). The context in question is the information about candidate solutions and their respective function values from previous optimization iterations. In doing so, the burden of finding the optimal solution falls completely on the `port_and_filter` function. This exercise also allows us to prove the contrary: LLMs, when provided with the context of parent solutions and tasked with the generation of new candidate solutions such that the cost function value is reduced, result in faster convergence (refer to Table 6). The purpose of this experiment is to explicitly demonstrate the superiority of LLMs to generate better candidate points using optimization history (LEO), as opposed to simply using LLM to exhaustively generate solutions in an explore and exploit pool that might locate the global optimal solution (LEO-Rnd).

#### 4.1.2. Experiment 1: Evaluation

We show experiments using both LEO and LEO-Rnd in optimizing the 2D Goldstein-Price function. This test was performed for 100 optimization iterations using 10 different random number seeds. Table 5 highlights the differences between the two methods. Table 6 shows the results obtained from this experiment, where it is found that LEO outperforms LEO-Rnd in terms of mean and median convergence with optimization iterations. It must be noted that LEO is able to accurately identify the global optimal solution, whereas this is not reflected in solutions obtained from LEO-Rnd. In addition, Figs. 9(a) and 9(b) show the convergence plots of LEO and LEO-Rnd for both the mean and the median, computed across 10 experiments. It can be observed that at the start of the optimization, the rate of convergence exhibited by LEO is much higher than that of LEO-Rnd, which shows that it very quickly navigates from the poor initial guess to the global optima. On the contrary, LEO-Rnd shows signs of plateau, signifying that the LLM-based random solution generation without context and guardrails fails to locate the global optima, despite the `port_and_filter` function serving as a ‘soft’ optimization filter. We argue that this experiment is a clear demonstration of the ability of LLMs to build a topological relation of the decision variable with function space, with which it adaptively generates better candidate points, that aid in much faster convergence, thereby exhibiting reasoning capabilities.

#### Table 5: Difference between the underlying approach adopted for LEO and LEO-Rnd

| Method | Task | Context | Guardrails | Explore Prompt | Exploit Prompt |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LEO** | Generate Solutions at $k^{\text{th}}$ iteration such that function value is reduced | Solutions and function values at $(k - 1)^{\text{th}}$ iteration | Elitism | Refer Table 7 | Refer Table 8 |
| **LEO-Rnd** | Generate solutions at $k^{\text{th}}$ iteration | None | None | Refer Table 7 | Refer Table 8 |

#### Table 6: Comparison of numerical accuracy of LEO and LEO-Rnd methods for 2D Goldstein Price problem

| Method | Mean ± std | Median |
| :--- | :---: | :---: |
| **LEO** | $3.008 \pm 0.019$ | 3.000 |
| **LEO-Rnd** | $5.275 \pm 1.718$ | 5.407 |

#### 4.1.3. Experiment 2: Diminishing variance of the explore pool

A question of paramount importance in this study is to clearly demonstrate the reasoning ability of LLM to yield better candidate solutions. An affirmative answer to such an evaluation will form the backbone of the present optimization approach. To this extent, we ask the following question: if, in the present strategy, LEO progressively shows signs of diminishing variance in the explore pool, does it amount to reasoning? This is not a trivial proposition, as solutions are always exported from the explore pool to the exploit pool. So naturally, it is expected to have diminishing variance in the exploit pool and not the other way around. Additionally, if the explore pool progressively aligns in the direction of the global optimal solution, it naturally results in the generation of better candidate solutions in the exploit pool due to the use of the `port_and_filter` function. While several papers, for example [32,33,38], use the temperature to control the explore and exploit behavior, we entrust the responsibility of generating explore and exploit pools to the model itself.

#### 4.1.4. Experiment 2: Evaluation

Fig. 9 presents the results for the Goldstein-Price problem. Specifically, Figs. 9(c) and 9(d) illustrate the variation of the Kernel Density Function (KDF) for the explore pools corresponding to the two variables in the LEO method. Similarly, Figs. 9(e) and 9(f) depict the KDF for the same two variables, but this time using the LEO-Rnd method. Upon analyzing these figures, a clear trend emerges: the mean of the distribution associated with the explore pool in the LEO method converges towards the solutions, accompanied by a diminishing variance. This observation confirms that the model effectively reasons for and extracts ‘better’ points from regions characterized by lower objective function values. In contrast, the LEO-Rnd method does not exhibit a similar trend. This discrepancy can be attributed to the random selection of the explore pool. Notably, this finding underscores an important point: relying solely on temperature as a means to control exploration may not yield optimal results, as it primarily introduces randomness. On the other hand, the elitism-based guardrails effectively harness the reasoning capability of LLMs to generate ‘better’ points.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/0a43f91ac3e8be25ab2fa344fab177ac881a1991b894c7b46773454c23cf28b1.jpg)
*(a) Exploit pool (best): mean*

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/6f7d4462b229013e851922ea8f6d7a5d0849db20c552e5579ec28bea81567c95.jpg)
*(b) Exploit pool (best): median*

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/bbb1b7015bc39d83cb911e575c92ddf8551ae3e683b44f5334a4f3945ce5a78c.jpg)
*(c) LEO (var1)*

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/839c0998c5ba81b47a3a567407705a59bae11de50553d38f251528a8f1016f66.jpg)
*(d) LEO (var2)*

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/8db7e9fb33ec7d75c58d8012278c13c5afeaece2944e7ebde5e94cef58add484.jpg)
*(e) LEO-Rnd (var1)*

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/897e2b2cc3b6fc85b16da7e27124eb0a7fe118f703b59dfc4df0d7e4b10d01c1.jpg)
*(f) LEO-Rnd (var2)*

*Fig. 9. Plots demonstrating reasoning by LEO. Comparison of convergence behavior depicted by LEO and LEO-Rnd in terms of (a) mean and (b) median with optimization iterations. Comparison of kernel density functions for solutions belonging to explore pool (every 5th iteration) obtained from (c) LEO (var1), (d) LEO (var2), (e) LEO-Rnd (var1), and (f) LEO-Rnd (var2).*

### 4.2. Challenges involved with LLM based optimizers

We now discuss some practical challenges we have encountered during our exploration of LLMs in the context of optimization:

1. **Reproducibility of exact performances:** We can fix a random seed to start the optimization from the same point. However, we cannot curtail the randomness. If we fix the temperature to zero, we can ensure repeatability, but at the cost of reduced performance due to poor exploration. Alternatively, a higher temperature value would improve the model’s performance at the cost of reduced repeatability. Further, as argued previously, a very high temperature value can result in diminished performance of the method.

2. **LLM hallucinations:** LLMs sometimes produce numbers such as 1.23456 (i.e., some common sequences) or common integers such as 0 or 1, which are not backed by any rationale. This issue has been reported in [46]. This sometimes leads to the model getting stuck at local minima.

3. **Mode collapse problem:** In spite of a higher temperature value, there are times at which a LLM keeps producing the same output, often a trivial number such as 1.0000 or 0.0000. This stems from the model-collapse issue pertaining to the auto-regressive nature of the algorithm, as explained in [46].

4. **The impact of increasing dimensions on model accuracy:** From benchmark problems to engineering applications, we consistently observe that LEO produces erroneous solutions for high-dimensional cases. Further, the uncertainty in the solutions increases with an increase in dimensionality.

5. **Computational expenses of large language models:** As this technology is still in its infancy, there are currently high costs associated with querying GPT-3.5 Turbo, which is used in this study. More advanced models, such as the GPT-4, are even more expensive. However, as the technology evolves towards maturity, it is anticipated that the costs associated with LLMs will decrease, concurrently improving their efficacy.

6. **Limitations in addressing saddle points:** This method may not be able to identify and solve saddle point problems, because the explore–exploit strategy is based on prompts instructing the user to minimize the objective function value.

### 4.3. Remedies

Some of the remedies that would be helpful to mitigate the above mentioned challenges are as follows:

1. Exact reproducibility is still unsolved, but a straightforward way to extract consistent performance is to follow an ensemble approach. The mean of several runs converges to the global optima as the number of samples increases.

2. To address issues such as mode collapse and hallucinated numbers, we add ‘jitter’ to the prompt. This means a small amount of perturbation added to the examples in the prompts seems to solve both of these issues.

3. Assigning the identity of an optimization researcher at the very beginning of the prompt seems to improve the performance.

The detailed prompts used for the explore and exploit operations are presented in Appendix B.

### 4.4. Improving accuracy for high-dimensional problems

A possible remedy for the high-dimensional problem could be the stochastic sampling suggested in Hu et al. [61]. Herein, a subset of dimensions, randomly chosen, are handled at a time by LEO. The batch size becomes a hyperparameter and would depend on the context-window length of the LLM. Another possible approach to tackling the high-dimensional problems could be an agent based approach. Here, a number of LEO agents can work on a smaller number of dimensions and communicate their outputs through a message-passing interface or graph-based framework. A successful solution strategy for the high-dimensional problems could have a huge impact on the field of scientific machine learning, and would be the topic of future research.

## 5. Conclusions

This paper presents a population-based optimization method based on LLMs called Large Language-model-based Evolutionary Optimizer (LEO). We present a diverse set of benchmark test cases, spanning from elementary examples to multi-objective and high-dimensional numerical optimization problems. Furthermore, we illustrate the practical application of this method to industrial optimization problems, including shape optimization, heat transfer, and windfarm layout optimization.

Several key conclusions can be drawn based on the obtained results:

1. We assert that LLMs exhibit the capacity to reason and execute zero-shot optimization. The explore–exploit strategy, as proposed in this paper, effectively harnesses this inherent ability and consistently performs on par with, if not surpassing, some of the leading methods in numerical optimization.

2. We deduce that while LLMs are capable of optimization, achieving consistent and improved performance necessitates a series of engineering adjustments. LLMs tend to exhibit creativity and a propensity for hallucinations. Consequently, prompt engineering interventions become crucial to eliciting consistent behavior from these models.

3. Our observations reveal that although LLMs excel in low-dimensional problems, they require a greater number of iterations to converge to a global optimum in high-dimensional scenarios. This phenomenon is reflected in a gradual reduction in accuracy as the dimensionality increases for the same fixed number of iterations, a trend also observed in the engineering examples.

4. LEO offers several advantages over conventional methods for low-dimensional problems, such as a parameter-free method, ease of implementation, suitability for multi-objective optimization problems with conflicting objectives, as well as problems with multiple minima, automatic step-size selection, etc.

5. We have demonstrated that LLM-based exploration and exploitation offers a straightforward way to yield ‘effects’ of crossover and mutation akin to evolutionary strategies such as the popular NSGA II approach. This enables a seamless way to perturb candidate solutions that distinctively retain the identity of the Pareto optimal front.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/8b86dea2ff05e57c3f02ea5a71a05f74a253292818814b3ecad46f3cdb032898.jpg)

*Fig. 10. Convergence pattern in the $f_{\min}$ for the Rosenbrock function with different dimensionality obtained using LEO (with Llama 3.1-70B model as the LLM) for the exploit pool.*

While the method still requires overcoming a few hurdles to make it suitable for engineering applications, such as tackling high-dimensional problems and the high costs associated with running LLMs, as discussed previously, it still offers the following advantages in its current form:

1. **Parameter-Free Method:** Our approach operates without the need for any user-defined parameters. This inherent flexibility renders it suitable for a wide spectrum of problems, eliminating the burden of fine-tuning specific parameters.

2. **Decoupling Exploration and Exploitation:** Unlike other LLM-based optimization methods that rely on temperature-based exploration, our approach treats exploration and exploitation as distinct operations. This, when coupled with a rich context provided to the LLM, results in a superior distribution of candidate values. Notably, our experiments demonstrate that this approach leads to faster convergence compared to random point selection (governed by temperature) while maintaining other settings.

3. **Unrestricted step size (learning rate):** Unlike other gradient-based optimizers, there is no restriction on step size (learning rate), since it is automatically inferred based on the samples presented in the prompt. This addresses issues associated with too small as well as too large learning rates, such as getting stuck in local minima and loss of stability.

4. **Optimizing for multi-objective problems:** It is suitable for multi-objective optimization problems, as well as for problems with multiple global minima, as evident from the test cases presented in this study.

5. **A modular approach:** Owing to its simplicity, it can be seamlessly incorporated into any existing population-based optimization framework. This vastly improves the usability of this method.

The scope of this paper is limited to exploring the present capabilities of LLMs, specifically GPT-3.5 Turbo, in the context of optimization. The effects of LLM-specific parameters such as model size, model architecture, and modality (single-modal vs. multi-modal) on the abilities of the models to perform numerical optimization will be explored in future work. Additionally, our forthcoming work will address strategies aimed at mitigating the curse of dimensionality.

## CRediT authorship contribution statement

**Shuvayan Brahmachary:** Writing – review & editing, Visualization, Validation, Methodology, Investigation, Formal analysis, Conceptualization. **Subodh M. Joshi:** Writing – review & editing, Writing – original draft, Project administration, Methodology, Conceptualization. **Aniruddha Panda:** Software, Resources, Methodology, Formal analysis. **Kaushik Koneripalli:** Writing – review & editing, Methodology, Formal analysis, Data curation. **Arun Kumar Sagotra:** Validation, Investigation, Formal analysis. **Harshil Patel:** Validation, Formal analysis. **Ankush Sharma:** Software, Resources. **Ameya D. Jagtap:** Project administration, Investigation. **Kaushic Kalyanaraman:** Supervision, Project administration, Funding acquisition, Conceptualization.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgments

The authors from Shell India Markets Pvt. Ltd. would like to acknowledge the financial support from Shell, United States.

### Appendix A. Problem formulation

### A.1. LEO: $n$-dimensional rosenbrock function

The $n$-dimensional Rosenbrock function was attempted using LEO with Llama 3.1-70B LLM as the workhorse. Fig. 10 shows the convergence pattern in the $f_{\min}$ of the exploit pool of solutions at every optimization iterations. It can be found that the convergence trend shows consistent pattern, as also seen previously in Fig. 7. This observation is indicative of the fact that LEO is LLM agnostic and renders robust performance across multiple problems.

### A.2. Nose cone shape optimization

The nose cone shape optimization has been a well-studied problem in the aerospace industry since the early 1950’s. Researchers have adopted various theoretical approaches, such as the Newtonian method, Taylor–Maccoll theory, the Fay–Riddel stagnation point heat transfer approach, etc., to design nose cone shapes with the objective of minimizing drag coefficients, ballistic coefficients, heat transfer, etc. [62]. In the present work, we represent the shape of the nose cone body having a fineness (or length to diameter) ratio of $L / D = 1$ using a simple power-law curve as follows:

$$
y = a x^{n_{\mathrm{pl}}}, \quad 0.001 \le n_{\mathrm{pl}} \le 1 \tag{3}
$$

where $n_{\mathrm{pl}}$ is the decision variable that determines the shape of the body (see Fig. 11). In addition, we employ Newtonian theory to compute the drag coefficient $C_d$.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/6e4d6952d83e83757e7d626c2ee724fef8367482e2ca2c717c4f3d3018d8c5be.jpg)

*Fig. 11. Juxtaposition of typical nose cones represented using a simple power law body.*

### A.3. Nozzle shape optimization

The nozzle shape design is yet another key application in the aerospace industry. For instance, nozzle shape design is an integral component in the design of ramjet/scramjet engines for high thrust. In the present study, the two-dimensional (2D) nozzle is designed with the objective to maximize the streamwise velocity or minimize the radial velocity [63]. For any given nozzle shape, the radial velocity or the streamwise velocity can be easily computed using a simple algebraic equation such as the area-Mach number relation, invoking the simple quasi-1D flow assumptions for high-speed compressible flows. The 2D nozzle in consideration is parameterized using a cubic Bezier curve as follows:

$$
\begin{aligned}
x(t) &= (1 - t)^3 x_0 + 3(1 - t)^2 t x_1 + 3(1 - t)t^2 x_2 + t^3 x_3, \\
y(t) &= (1 - t)^3 y_0 + 3(1 - t)^2 t y_1 + 3(1 - t)t^2 y_2 + t^3 y_3, \quad t \in [0, 1].
\end{aligned}
$$

where the control points $(x_0, y_0)$ and $(x_3, y_3)$ are fixed in space (for fixed inlet radius, $r_i$, outlet radii, $r_o$, and length, $l$), whereas the remaining control points $(x_1, y_1)$ and $(x_2, y_2)$ are variables (see Fig. 12). This results in a 4D problem.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/6d710fae85734fe4542541e1fe42381a2f00394c54a9edbc5bde1928f3028899.jpg)

*Fig. 12. Schematic of the nozzle shape representation and associated variable.*

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/1b38974ccc44a9ac26470ecf467880d5f4855692cbb4fc6719ab062e713b7301.jpg)
*(a) Exploit pool*

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/ef14e567cd9d1180c68da23b30155f0922e0a637a61a65649f0ab38b4fcd71d6.jpg)
*(b) Explore pool*

*Fig. 13. Convergence of the minimum and mean function values with optimization iterations for exploit and explore pools.*

### A.4. One-dimensional heat transfer problem

We attempt to solve a 1D steady-state heat transfer equation as shown below:

$$
\frac{d^2 T}{dx^2} + \frac{q_x}{k} = 0 \quad \Big|\quad T_L = 0, \; T_R = 1 \tag{4}
$$

where $T$ is the non-dimensional temperature distribution and $q/k$ is the internal heat generation per unit volume. In order to solve the steady-state temperature distribution that satisfies the above equation, we impose the Dirichlet boundary condition at the extreme ends of the 1D computational domain, as shown above. The domain of length $L$ is discretized using $n_{\mathrm{dim}}$ elements, which are varied across multiple test cases in this present experiment. This results in an equal elemental spacing of $\Delta x = L / n_{\mathrm{dim}}$. We adopt second order central differencing for the second order term, followed by a constant value for $q/k = 10$. This results in a residual loss term (or objective function) as follows:

$$
\frac{T_{i-1} - 2T_i + T_{i+1}}{\Delta x^2} + \frac{q_x}{k} = 0 \tag{5}
$$

While there are established iterative methods that allow us to compute the steady-state temperature distribution across the 1D computational domain, we frame this problem as an optimization task. This results in an approach wherein we determine the discretized temperature field that admit solution to Eq. (5), i.e., minimization of loss $\sim 0$.

### A.5. Wind farm layout optimization

Wind farm layout optimization is a critical endeavor in the renewable energy sector. This optimization task is aimed at enhancing the efficiency and output of the wind farms by manipulating the position of the wind turbines. The objective function in this case is the annual energy production (AEP), which is the amount of electricity that a wind farm can generate in one year. AEP is influenced by wind conditions such as wind direction and speed. We consider 72 wind directions ranging from $0^\circ$ to $360^\circ$ with a step size of $5^\circ$, and we consider a random wind speed whose value is centered around $8\text{ m/s}$ with a standard deviation of $0.5\text{ m/s}$. The rotor diameter of the turbine is taken to be $126.0\text{ m}$ and the domain is considered to be of size $1000\text{ m} \times 1000\text{ m}$.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/aeb83d9527da18c1bae8f236656f9c22fc1401f9acbd3b7a782401b50851f354.jpg)
*(a) Wind-turbines=2*

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/2d026af5584261a31d03edbc5ca2e38accca76c14ffe4e8e3c3a8b0cded5eb4c.jpg)
*(b) Wind-turbines=4*

![image](https://cdn-mineru.openxlab.org.cn/result/2026-09-29/68a9ade0-ae55-44c6-80d2-05beb421a298/abfd8edc48bec7da1e70bc3d1205ece5f82276ba566fd8962608a862633c9896.jpg)
*(c) Wind-turbines=8*

*Fig. 14. Windfarm layout optimization loss curves for increasing dimensionality of the problem.*

## Appendix B. Prompts

See Tables 7–10.

### Table 7: Explore prompt for LEO and LEO-Rnd methods for Goldstein Price problem

| Prompt Type | LEO | LEO-Rnd |
| :--- | :--- | :--- |
| **Explore Prompt** | You are an optimization researcher tasked to minimize the value of loss. Current candidate solutions for N=2 variables in the order var1, var2 with their respective function loss in csv format are: [var1, var2, loss; -0.018822,-0.692810,74.271916; 0.320219,-1.629214,3430.938028; 0.886970,-1.846776,3507.116813; -1.149739,0.407518,4330.723936; 1.958779,-0.219516,5479.393434; -0.489178,0.695195,22743.401171; -1.957878,0.271882,91001.711544; -0.095039,1.589113,130107.754358; -0.664592,1.331668,169337.055368; -1.622905,1.526054,353181.485985]. You have to look at the above points and think of the (min, max) values for each variable that might reduce or minimize the loss. With these limits of (min, max) values for each variable, you must provide exactly 10 new, but completely different and scattered sets of values from the above ones, to explore away regions for minimizing the function loss. Generate the result like a csv file with 10 rows and 2 columns, where each row represents a candidate and each column represents a variable. The response must only contain these numerical values in the csv format without column headers. Do not provide additional text or explanation. Strictly, provide only the variable values as floating point numbers with precision format %.6f. | You must provide exactly 10 new, but completely different and scattered sets of values within all variables, in between -2 and 2. Generate the result like a csv file with 10 rows and 2 columns, where each row represents a candidate and each column represents a variable. The response must only contain these numerical values in the csv format without column headers. Do not provide additional text or explanation. Strictly, provide only the variable values as floating point numbers with precision format %.6f. For example (with column headers): 0.8166, 0.1868; 0.6788, 0.0772; ...; 0.1853, 0.3935 |

### Table 8: Exploit prompt for LEO and LEO-Rnd methods for Goldstein Price problem

| Prompt Type | LEO | LEO-Rnd |
| :--- | :--- | :--- |
| **Exploit prompt** | You are an optimization researcher tasked to minimize the value of loss. Current best candidate solution for N=2 variables in the order var1, var2, with their respective function loss in csv format are: [var1, var2, loss; 1.8859, 0.2995, 101.5838]. Please provide exactly 10 new but different candidates of values for all variables, in between -2 and 2, to exploit close by regions for minimizing the function loss. Generate the result like a csv file with 10 rows and 2 columns, where each row represents a candidate and each column represents a variable. The response must only contain these numerical values in the csv format without column headers and row index. Do not provide additional text or explanation. Strictly, provide only the variable values as floating point numbers with precision format %.6f. For example (with column headers): 0.8166, 0.1868; 0.6788, 0.0772; ...; 0.1853, 0.3935 | Please provide exactly 10 new but different candidates of values for all variables, in between -2 and 2 to exploit close by regions. Generate the result like a csv file with 10 rows and 2 columns, where each row represents a candidate and each column represents a variable. The response must only contain these numerical values in the csv format without column headers and row index. Do not provide additional text or explanation. Strictly, provide only the variable values as floating point numbers with precision format %.6f. For example (with column headers): 0.8166, 0.1868; 0.6788, 0.0772; ...; 0.1853, 0.3935 |

### Table 9: Exploit and explore prompts for NSGA II with LEO-modular

| Prompt Type | LEO-modular |
| :--- | :--- |
| **Exploit Prompt** | You are an intelligent assistant who can understand great technical details. You will help me minimize two functions by taking cues from given information. Current candidate solutions (a, b) with their function loss are: Candidate 1: a=0.2536, b=0.7346, function loss f1=0.2536, function loss f2=6.2219; Candidate 2: a=0.3953, b=0.1976, function loss f1=0.3953, function loss f2=1.7307. Please provide exactly 2 new but different pairs of values for 0<'a'<1 and 0<'b'<1 to exploit closeby regions for minimizing the function loss1 and function loss2. Take cues from the above points. The response should contain 4 values, corresponding to 2 pairs in the format: 'a1, b1, a2, b2, ..., an, bn'. Provide only these 4 numerical values in order as python [list], separated by commas, with no additional text or explanation. |
| **Explore Prompt** | You are an intelligent assistant who can understand great technical details. You will help me minimize two functions by taking cues from given information. Current candidate solutions (a, b) with their function loss are: Candidate 1: a=0.3953, b=0.1976, function Loss f1=0.3953, function Loss f2 =1.7307; Candidate 2: a=0.1061, b=0.9373, function Loss f1=0.1061, function Loss f2 =8.4352. Please provide exactly 2 new but different pairs of values for 0<'a'<1 and 0<'b'<1 to explore away regions for minimizing the function loss1 and function loss2. The response should contain 4 values, corresponding to 2 pairs in the format: 'a1, b1, a2, b2, ..., an, bn'. Provide only these 4 numerical values in order, separated by commas, with no additional text or explanation. |

### Table 10: Prompts for testing reasoning abilities of LEO for Rosenbrock 2D function and nose cone shape optimization problem

| Problem | Prompt |
| :--- | :--- |
| **Nosecone design** | You are an optimization researcher tasked to minimize the value of function loss. Current best candidate solution for 1 variable (n) with their respective function loss are: candidate: n=0.90000; loss: 0.37450. Give me a new (n) value that satisfies the following: (a) new n is different from all above, (b) has a function value lower than the above function values and (c) result in a rapid convergence towards the value of n that results in lowest function value. Do not write code or any explanation. The output must end with just a numerical value for next variable. |
| **Rosenbrock 2D** | You are an optimization researcher tasked to minimize the value of function loss with two input variables n1 and n2. Current best candidate solution for 2 variables (n1, n2) with their respective function loss are: input: n1=1.00, n2=0.300, loss:49.00. Give me a new (n1, n2) pair value that satisfies the following: (a) new n1 and n2 is different from all above, (b) has a function value lower than the above function values and (c) result in a rapid convergence towards the value of n1 and n2 that results in lowest function value. Do not write code or any explanation. The output must end with two numerical values for (n1, n2) only. |

## Data availability

Data will be made available on request.

## References

[1] Bonan Min, Hayley Ross, Elior Sulem, Amir Pouran Ben Veyseh, Thien Huu Nguyen, Oscar Sainz, Eneko Agirre, Ilana Heintz, Dan Roth, Recent advances in natural language processing via large pre-trained language models: A survey, ACM Comput. Surv. (ISSN: 0360-0300) 56 (2) (2023) <http://dx.doi.org/10.1145/3605943>.  
[2] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, Ji-Rong Wen, A survey of large language models, 2023, arXiv preprint arXiv:Arxiv-2303.18223.  
[3] Yiheng Liu, Tianle Han, Siyuan Ma, Jiayue Zhang, Yuanyuan Yang, Jiaming Tian, Hao He, Antong Li, Mengshen He, Zhengliang Liu, Zihao Wu, Lin Zhao, Dajiang Zhu, Xiang Li, Ning Qiang, Dingang Shen, Tianming Liu, Bao Ge, Summary of ChatGPT-related research and perspective towards the future of large language models, Meta-Radiology (ISSN: 2950-1628) 1 (2) (2023) 100017, <http://dx.doi.org/10.1016/j.metrad.2023.100017>, URL <https://www.sciencedirect.com/science/article/pii/S2950162823000176>.  
[4] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin, Attention is all you need, in: I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, R. Garnett (Eds.), in: Advances in Neural Information Processing Systems, vol. 30, Curran Associates, Inc., 2017, pp. 6000–6010, URL <https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>.  
[5] OpenAI, Josh Achiam, et al., GPT-4 technical report, 2023, arXiv preprint arXiv:Arxiv-2303.08774.  
[6] Jie Huang, Kevin Chen-Chuan Chang, Towards reasoning in large language models: A survey, 2023, arXiv preprint arXiv:Arxiv-2212.10403.  
[7] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, Yusuke Iwasawa, Large language models are zero-shot reasoners, 2023, arXiv preprint arXiv:Arxiv-2205.11916.  
[8] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D. Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei, Language models are few-shot learners, in: H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, H. Lin (Eds.), in: Advances in Neural Information Processing Systems, vol. 33, Curran Associates, Inc., 2020, pp. 1877–1901, URL <https://proceedings.neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf>.  
[9] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, Denny Zhou, Chain-of-thought prompting elicits reasoning in large language models, 2023, arXiv preprint arXiv:2201.11903.  
[10] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, Karthik Narasimhan, Tree of thoughts: Deliberate problem solving with large language models, Adv. Neural Inf. Process. Syst. 36 (2024).  
[11] Jieyi Long, Large language model guided tree-of-thought, 2023, arXiv preprint arXiv:2305.08291.  
[12] Scott Reed, Konrad Zolna, Emilio Parisotto, Sergio Gomez Colmenarejo, Alexander Novikov, Gabriel Barth-Maron, Mai Gimenez, Yury Sulsky, Jackie Kay, Jost Tobias Springenberg, Tom Eccles, Jake Bruce, Ali Razavi, Ashley Edwards, Nicolas Heess, Yutian Chen, Raia Hadsell, Oriol Vinyals, Mahyar Bordbar, Nando de Freitas, A generalist agent, 2022, arXiv preprint arXiv:2205.06175.  
[13] Yecheng Jason Ma, William Liang, Guanzhi Wang, De-An Huang, Osbert Bastani, Dinesh Jayaraman, Yuke Zhu, Linxi Fan, Anima Anandkumar, Eureka: Human-level reward design via coding large language models, 2023, arXiv preprint arXiv:Arxiv-2310.12931.  
[14] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, Anima Anandkumar, Voyager: An open-ended embodied agent with large language models, 2023, arXiv preprint arXiv:Arxiv-2305.16291.  
[15] Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, et al., Beyond the imitation game: Quantifying and extrapolating the capabilities of language models, 2023, arXiv preprint arXiv:2206.04615.  
[16] Andres M. Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew D. White, Philippe Schwaller, ChemCrow: Augmenting large-language models with chemistry tools, 2023, arXiv preprint arXiv:Arxiv-2304.05376.  
[17] Daniil Boiko, Robert MacKnight, Ben Kline, Gabe Gomes, Autonomous chemical research with large language models, Nature 624 (2023) 570–578, <http://dx.doi.org/10.1038/s41586-023-06792-0>.  
[18] Andrew E. Blanchard, Mayanka Chandra Shekar, Shang Gao, John Gounley, Isaac Lyngaas, Jens Glaser, Debsindhu Bhowmik, Automating genetic algorithm mutations for molecules using a masked language model, IEEE Trans. Evol. Comput. 26 (4) (2022) 793–799, <http://dx.doi.org/10.1109/TEVC.2022.3144045>.  
[19] Shujian Zhang, Chengyue Gong, Lemeng Wu, Xingchao Liu, Mingyuan Zhou, AutoML-GPT: Automatic machine learning with GPT, 2023, arXiv preprint arXiv:2305.02499.  
[20] Francesco Stella, Cosimo Della Santina, Josie Hughes, How can LLMs transform the robotic design process? Nat. Mach. Intell. (ISSN: 2522-5839) 5 (6) (2023) 561–564, <http://dx.doi.org/10.1038/s42256-023-00669-7>.  
[21] Mingkai Zheng, Xiu Su, Shan You, Fei Wang, Chen Qian, Chang Xu, Samuel Albanie, Can GPT-4 perform neural architecture search? 2023, arXiv preprint arXiv:2304.10970.  
[22] Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, Animesh Garg, ProgPrompt: Generating situated robot task plans using large language models, 2022, arXiv preprint arXiv:Arxiv-2209.11302.  
[23] Kevin Maik Jablonka, Philippe Schwaller, Andres Ortega-Guerrero, Berend Smit, Leveraging large language models for predictive chemistry, Nat. Mach. Intell. (ISSN: 2522-5839) 6 (2) (2024) 161–169, <http://dx.doi.org/10.1038/s42256-023-00788-1>.  
[24] Rimon Melamed, Lucas H. McCabe, Tanay Wakhare, Yejin Kim, H. Howie Huang, Enric Boix-Adsera, PROPANE: Prompt design as an inverse problem, 2023, arXiv preprint arXiv:2311.07064.  
[25] Lichang Chen, Jiuhai Chen, Tom Goldstein, Heng Huang, Tianyi Zhou, InstructZero: Efficient instruction optimization for black-box large language models, 2023, arXiv preprint arXiv:Arxiv-2306.03082.  
[26] Yutian Chen, Xingyou Song, Chansoo Lee, Zi Wang, Qiuyi Zhang, David Dohan, Kazuya Kawakami, Greg Kochanski, Arnaud Doucet, Marc’Aurelio Ranzato, Sagi Perel, Nando de Freitas, Towards learning universal hyperparameter optimizers with transformers, in: Neural Information Processing Systems (NeurIPS) 2022, 2022, pp. 32053–32068.  
[27] Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, Jimmy Ba, Large language models are human-level prompt engineers, 2023, arXiv preprint arXiv:Arxiv-2211.01910.  
[28] Reid Pryzant, Dan Iter, Jerry Li, Yin Tat Lee, Chenguang Zhu, Michael Zeng, Automatic prompt optimization with "Gradient Descent" and beam search, 2023, arXiv preprint arXiv:Arxiv-2305.03495.  
[29] Yongchao Chen, Jacob Arkin, Yilun Hao, Yang Zhang, Nicholas Roy, Chuchu Fan, Prompt optimization in multi-step tasks (PROMST): Integrating human feedback and preference alignment, 2024, arXiv preprint arXiv:2402.08702.  
[30] Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin, Pieter Abbeel, Aravind Srinivas, Igor Mordatch, Decision transformer: Reinforcement learning via sequence modeling, 2021, arXiv preprint arXiv:2106.01345.  
[31] Jane X. Wang, Zeb Kurth-Nelson, Dhruva Tirumala, Hubert Soyer, Joel Z. Leibo, Remi Munos, Charles Blundell, Dharshan Kumaran, Matt Botvinick, Learning to reinforcement learn, 2017, arXiv preprint arXiv:1611.05763.  
[32] Shengcai Liu, Caishun Chen, Xinghua Qu, Ke Tang, Yew-Soon Ong, Large language models as evolutionary optimizers, 2023, arXiv preprint arXiv:Arxiv-2310.19046.  
[33] Pei-Fu Guo, Ying-Hsuan Chen, Yun-Da Tsai, Shou-De Lin, Towards optimizing with large language models, 2023, arXiv preprint arXiv:Arxiv-2310.05204.  
[34] Michal Pluhacek, Anezka Kazikova, Tomas Kadavy, Adam Viktorin, Roman Senkerik, Leveraging large language models for the generation of novel metaheuristic optimization algorithms, in: Proceedings of the Companion Conference on Genetic and Evolutionary Computation, in: GECCO ’23 Companion, Association for Computing Machinery, New York, NY, USA, ISBN: 9798400701207, 2023, pp. 1812–1820, <http://dx.doi.org/10.1145/3583133.3596401>.  
[35] Fei Liu, Xi Lin, Zhenkun Wang, Shunyu Yao, Xialiang Tong, Mingxuan Yuan, Qingfu Zhang, Large language model for multi-objective evolutionary optimization, 2023, arXiv preprint arXiv:Arxiv-2310.12541.  
[36] Fei Liu, Xialiang Tong, Mingxuan Yuan, Qingfu Zhang, Algorithm evolution using large language model, 2023, arXiv preprint arXiv:Arxiv-2311.15249.  
[37] Shihong Liu, Zhiqiu Lin, Samuel Yu, Ryan Lee, Tiffany Ling, Deepak Pathak, Deva Ramanan, Language models as black-box optimizers for vision-language models, 2023, arXiv preprint arXiv:2309.05950.  
[38] Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, Xinyun Chen, Large language models as optimizers, 2023, arXiv preprint arXiv:2309.03409.  
[39] Guillaume Lample, François Charton, Deep learning for symbolic mathematics, 2019, arXiv preprint arXiv:1912.01412.  
[40] Mojtaba Valipour, Bowen You, Maysum Panju, Ali Ghodsi, SymbolicGPT: A generative transformer model for symbolic regression, 2021, arXiv preprint arXiv:2106.14131.  
[41] Vishesh Agarwal, Somak Aditya, Navin Goyal, Analyzing the nuances of transformers’ polynomial simplification abilities, 2021, arXiv preprint arXiv:2104.14095.  
[42] François Charton, Linear algebra with transformers, 2022, arXiv preprint arXiv:2112.01898.  
[43] Ali AhmadiTeshnizi, Wenzhi Gao, Madeleine Udell, OptiMUS: Optimization modeling using MIP solvers and large language models, 2023, arXiv preprint arXiv:Arxiv-2310.06116.  
[44] Michael Zhang, Nishkrit Desai, Juhan Bae, Jonathan Lorraine, Jimmy Ba, Using large language models for hyperparameter optimization, in: NeurIPS 2023 Foundation Models for Decision Making Workshop, 2023, URL <https://openreview.net/forum?id=FUdZ6HEOre>.  
[45] Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M. Pawan Kumar, Emilien Dupont, Francisco J.R. Ruiz, Jordan S. Ellenberg, Pengming Wang, Omar Fawzi, et al., Mathematical discoveries from program search with large language models, Nature (2023) 1–3.  
[46] Aspen K. Hopkins, Alex Renda, Michael Carbin, Can LLMs generate random numbers? Evaluating LLM sampling in controlled domains, in: ICML 2023 Workshop: Sampling and Optimization in Discrete Space, 2023, URL <https://openreview.net/forum?id=Vhh1K9LjVI>.  
[47] Jack Kiefer, Jacob Wolfowitz, Stochastic estimation of the maximum of a regression function, Ann. Math. Stat. (1952) 462–466.  
[48] Diederik P. Kingma, Jimmy Ba, Adam: A method for stochastic optimization, 2014, arXiv preprint arXiv:1412.6980.  
[49] Richard H. Byrd, Peihuang Lu, Jorge Nocedal, Ciyou Zhu, A limited memory algorithm for bound constrained optimization, SIAM J. Sci. Comput. 16 (5) (1995) 1190–1208.  
[50] Scott Kirkpatrick, C. Daniel Gelatt Jr., Mario P. Vecchi, Optimization by simulated annealing, Science 220 (4598) (1983) 671–680.  
[51] Nikolaus Hansen, Andreas Ostermeier, Adapting arbitrary normal mutation distributions in evolution strategies: The covariance matrix adaptation, in: Proceedings of IEEE International Conference on Evolutionary Computation, IEEE, 1996, pp. 312–317.  
[52] Michael J.D. Powell, A Direct Search Optimization Method That Models the Objective and Constraint Functions by Linear Interpolation, Springer, 1994.  
[53] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, Christopher Ré, Flashattention: Fast and memory-efficient exact attention with io-awareness, Adv. Neural Inf. Process. Syst. 35 (2022) 16344–16359.  
[54] Tri Dao, FlashAttention-2: Faster attention with better parallelism and work partitioning, 2023, arXiv:2307.08691.  
[55] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, Song Han, AWQ: Activation-aware weight quantization for LLM compression and acceleration, 2023, arXiv preprint arXiv:2306.00978.  
[56] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han, Smoothquant: Accurate and efficient post-training quantization for large language models, in: International Conference on Machine Learning, PMLR, 2023, pp. 38087–38099.  
[57] Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal, T.A.M.T. Meyarivan, A fast and elitist multiobjective genetic algorithm: NSGA-II, IEEE Trans. Evol. Comput. 6 (2) (2002) 182–197.  
[58] Eckart Zitzler, Kalyanmoy Deb, Lothar Thiele, Comparison of multiobjective evolutionary algorithms: Empirical results, Evol. Comput. 8 (2) (2000) 173–195.  
[59] J. Blank, K. Deb, pymoo: Multi-objective optimization in python, IEEE Access 8 (2020) 89497–89509.  
[60] G.V.R. Rao, Exhaust nozzle contour for optimum thrust, J. Jet Propuls. 28 (6) (1958) 377–382.  
[61] Zheyuan Hu, Khemraj Shukla, George Em Karniadakis, Kenji Kawaguchi, Tackling the curse of dimensionality with physics-informed neural networks, 2023, arXiv preprint arXiv:Arxiv-2307.12306.  
[62] Shuvayan Brahmachary, Ganesh Natarajan, Niranjan Sahoo, On maximum ballistic coefficient axisymmetric geometries in hypersonic flows, J. Spacecr. Rockets 55 (2) (2018) 518–522.  
[63] Eszter Dudás, Nicolas Suas-David, Shuvayan Brahmachary, Vinayak Kulkarni, Abdessamad Benidar, Samir Kassi, Christine Charles, Robert Georges, High-temperature hypersonic laval nozzle for non-LTE cavity ringdown spectroscopy, J. Chem. Phys. 152 (13) (2020).

## Author Biographies

**Shuvayan Brahmachary** received his Ph.D. in Fluid and Thermal Science stream of Mechanical Engineering Department from Indian Institute of Technology Guwahati in the year 2019. Following his Ph.D., he worked as a postdoctoral researcher in Department of Aeronautics and Astronautics, Kyushu University, Japan until 2021. Starting January 2022, he worked as postdoctoral research fellow in the Department of Informatics, Technical University of Munich, Germany. Presently, he is working as a scientific machine learning researcher in Shell, Bangalore. His research interest includes scientific machine learning, computational fluid dynamics, optimization.

**Subodh Joshi** received his Masters and Ph.D. degrees in Aerospace Engineering from Indian Institute of Technology Bombay (IIT Bombay), India in 2018. Following his Ph.D., Subodh worked as a postdoctoral researcher at IIT Bombay (2018-19), INRIA Bordeaux Sud Ouest Research Center, France (2019-20), and as a C. V. Raman Postdoctoral Fellow at IISc Bangalore, India (2020-22). Since 2022, Subodh is working as a Scientific Machine Learning Researcher at Shell Technology Center Bangalore. His research interests include Scientific Machine Learning, numerical methods (particularly higher-order accurate schemes) for conservation and balance laws and applications in computational physics, including aeroacoustics, fluid dynamics, industrial and multiphysics systems, and subsurface physics.

**Aniruddha Panda** has worked at the Shell Technology Center Bangalore and partly in Houston, for the last 5 years. He is currently in the role of Scientific Machine Learning Researcher, and prior to this worked extensively as High-Performance Computing researcher. He has a Ph.D. in computational science from Eindhoven University of Technology. His prior experience is in the areas of scientific computing, physics inspired neural networks, deep learning with applications in geophysics (seismic processing), edge computing, and renewables. In his current role, he is working in the areas of operator learning for subsurface flows, optimization for process engineering and the use of generative AI and agentic workflows for scientific applications.

**Kaushik Koneripalli** received his Master’s degree in Electric Engineering in Arizona State University in 2019. Following this, he worked as a Research Engineer in Siemens and a Computer Scientist at SRI International, both in the US, before joining the Scientific Machine Learning team in Shell Bangalore, India in 2023. His broad research interests are in deep representation learning with applications rooted in Computer Vision and NLP, with his current focus being on scaling graph ML for industrial applications.

**Arun Kumar Sagotra** received his Ph.D. in Materials Science and Engineering from University of New South Wales, Sydney in 2019. Following this, he worked as a Data Scientist in Hitachi Vantara and CSIRO in India and Australia respectively. He worked as a Scientific Machine Learning Researcher in Shell, India (2023-2024). Currently, he is working in Micron Technology. His Interest include Machine Learning, Computer Vision, Graph Neural Network, NLP and Materials Discovery.

**Harshil Patel** is a Scientific Machine Learning researcher at Shell Technology Center Bangalore. He has extensive experience working in the domains of natural language processing, machine vision, physics-inspired neural networks, and optimization during his 5 years of tenure at Shell. He has a Ph.D. in computational science from Eindhoven University of Technology, the Netherlands. He has a keen interest in fusing Artificial Intelligence with Physics to accelerate the scientific and engineering computational workflows.

**Ankush Kumar Sharma** received his Bachelor of Technology in Computer Science and Engineering from the National Institute of Technology, Hamirpur in 2013. He began his career at Samsung Research Institute, Noida as a software developer for 18 months before joining the Indian Army as an officer in the Corps of Signals in 2014, where he served until 2020. Ankush completed his Master of Technology in Software Systems (Data Analytics) from Birla Institute of Technology and Science, Pilani in 2020. Currently, he is working as a Scientific Machine Learning Researcher at Shell, India with a focus on creating end-to-end machine learning solutions. His research interests include natural language processing, geospatial analysis, knowledge graphs and generative AI exploration.

**Dr. Ameya D. Jagtap** is an Assistant Professor (tenure-track) in the Department of Aerospace Engineering at Worcester Polytechnic Institute (WPI), USA. Prior to WPI, he was an Assistant Professor of Applied Mathematics (Research) at Brown University for three and a half years. Dr. Jagtap holds a Ph.D. and a Master’s degree in Aerospace Engineering from the Indian Institute of Science, India, and conducted postdoctoral research at TIFR-CAM and Brown University. His research focuses at the intersection of scientific computing and machine learning algorithms, with broad applications in computational physics.

**Kaushic Kalyanaraman** is a Scientific Machine Learning researcher at Shell Technology Center Bangalore. Kaushic has extensive experience in Graph theory, Generative AI, Scientific ML as well as over 17 years of experience in Energy engineering and economics. He has a bachelors in Civil Engineering from National University of Singapore. In his current role, he is the founder, program manager and principal investigator of the Scientific Machine Learning Research group at Shell R&D. His area of interest is in advancing the Scientific Machine Learning and Generative AI for scientific discovery and advances in Energy Engineering and Economics.
