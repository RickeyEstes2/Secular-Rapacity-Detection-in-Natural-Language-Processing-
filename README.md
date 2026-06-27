While "secular rapacity" is not a standard, out-of-the-box benchmark in Natural Language Processing (NLP), it can be rigorously operationalized as a computational text analysis task. 

To detect it mathematically, we must deconstruct the phrase into computable variables:
1. **Rapacity:** Aggressively greedy, extractive, or predatory behavior (semantic polarity, extractive semantic roles, negative sentiment toward resources/victims).
2. **Secular:** This can mean either **non-religious/systemic** (e.g., corporate or economic greed rather than moral failing) or a **long-term, non-cyclical trend** (as in "secular trend" in finance/economics).

Implementing the detection of this concept requires a synthesis of linear algebra, probability theory, time-series analysis, graph theory, and information theory. Here is the computational mathematics behind how such a system is built.

---

### 1. Quantifying "Rapacity" (Linear Algebra & Topology)
To detect rapacity, the system must first understand the semantic space of greed, extraction, and predation. This is achieved using high-dimensional vector spaces.

*   **Word Embeddings & Vector Space Models:** Words are mapped to dense vectors in $\mathbb{R}^d$. Rapacious concepts (e.g., *monopolize, exploit, hoard, extract*) will cluster together. 
*   **Cosine Similarity:** To find if a text contains rapacious language, we calculate the geometric angle between the document vector and a "rapacity seed vector."
    $ \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} = \frac{\sum_{i=1}^{d} A_i B_i}{\sqrt{\sum_{i=1}^{d} A_i^2} \sqrt{\sum_{i=1}^{d} B_i^2}} $
*   **Transformer Attention Mathematics:** Modern models use self-attention to understand the *context* of rapacity (e.g., distinguishing "the market extracted value" from "the dentist extracted a tooth"). The attention mechanism computes the relevance of words using scaled dot-product attention:
    $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V $$
    Where $Q$ (Query), $K$ (Key), and $V$ (Value) are matrices. The softmax function ensures the model assigns high mathematical weight to the relationships between "actors" (corporations/individuals) and "resources" being extracted.

### 2. Isolating the "Secular" Context (Probability & Bayesian Inference)
If "secular" is defined as *non-religious or systemic economic behavior*, we must mathematically separate secular texts from moral, religious, or purely interpersonal texts. This is done using **Topic Modeling**.

*   **Latent Dirichlet Allocation (LDA):** LDA assumes documents are mixtures of topics, and topics are mixtures of words. It uses Dirichlet priors to model the probability distributions.
*   **The Generative Mathematics:** For a document $d$ and word $w$, the joint probability is modeled as:
    $$ P(w, z | \alpha, \beta) = P(w | z, \beta) P(z | d, \alpha) $$
    Where $z$ is the topic assignment, $\alpha$ is the document-topic Dirichlet prior, and $\beta$ is the topic-word Dirichlet prior.
*   **Variational Inference:** Because exact inference is computationally intractable, the system uses Variational Bayes to maximize the Evidence Lower Bound (ELBO), allowing the algorithm to isolate the "secular/economic" topic cluster from other clusters, ensuring we are only measuring rapacity within the correct systemic context.

### 3. Detecting "Secular" Trends (Time-Series & Stochastic Calculus)
If "secular" is defined as a *long-term, non-cyclical trend* (e.g., tracking the rise of corporate greed in financial reports over 50 years), we must apply time-series mathematics to the semantic outputs.

*   **Semantic Time-Series:** We convert the "rapacity scores" of documents into a continuous time-series signal $X_t$.
*   **Change-Point Detection (CUSUM Algorithm):** To detect when a "secular shift" toward rapacity begins, we use the Cumulative Sum (CUSUM) control chart, which is highly sensitive to long-term shifts in the mean of a distribution.
    $$ S_t = \max(0, S_{t-1} + X_t - \mu_0 - k) $$
    Where $S_t$ is the cumulative sum at time $t$, $\mu_0$ is the baseline historical average of rapacity, and $k$ is a slack parameter. If $S_t$ exceeds a mathematical threshold $h$, a secular trend shift is flagged.
*   **Autoregressive Modeling (ARIMA):** To ensure the detected rapacity is a "secular" (long-term) trend and not a seasonal/cyclical fluctuation, we apply differencing ($d$) in an ARIMA$(p,d,q)$ model to make the time-series stationary, mathematically stripping away short-term noise.

### 4. Mapping Extractive Relationships (Graph Theory & Matrix Factorization)
Rapacity inherently implies a relationship: an *actor* taking from a *victim/resource*. We can model this using Knowledge Graphs and Semantic Role Labeling (SRL).

*   **Adjacency Matrices:** We construct a directed graph $G = (V, E)$ where nodes $V$ are entities (corporations, governments, individuals) and edges $E$ represent extractive actions. This is represented as an adjacency matrix $A$.
*   **Eigenvector Centrality & PageRank:** To find the most "rapacious" actors (those who extract the most from highly connected resources), we calculate the eigenvector centrality. The centrality score $x_i$ of node $i$ is proportional to the sum of the scores of the nodes it connects to:
    $$ x_i = \frac{1}{\lambda} \sum_{j} A_{ij} x_j \implies \mathbf{A}\mathbf{x} = \lambda \mathbf{x} $$
    Where $\lambda$ is the principal eigenvalue. High eigenvector centrality in an extractive graph mathematically identifies systemic, secular predators.

### 5. Measuring Systemic Shifts (Information Theory)
Finally, to measure how the *overall distribution* of language in a corpus is shifting toward secular rapacity over time, we use Information Theory.

*   **Kullback-Leibler (KL) Divergence:** We compare the probability distribution of words in a historical baseline corpus ($Q$) to a modern corpus ($P$). KL divergence measures how much information is "lost" or how much the language has shifted.
    $$ D_{\text{KL}}(P \parallel Q) = \sum_{x \in \mathcal{X}} P(x) \log \left( \frac{P(x)}{Q(x)} \right) $$
*   **Jensen-Shannon (JS) Divergence:** Because KL divergence is asymmetric, JS divergence is often used to symmetrically measure the convergence of the two text distributions. A rising JS divergence score over decades, specifically localized to the "rapacity" vector subspace, provides mathematical proof of a secular shift in systemic behavior.

### Summary
To implement **secular rapacity detection**, a system does not look for a single keyword. Instead, it uses **linear algebra** to map the semantics of greed, **Bayesian probability** to isolate secular/economic contexts, **stochastic time-series math** to prove the trend is long-term rather than cyclical, **graph theory** to identify the predators and victims, and **information theory** to quantify the systemic shift in human language over time.
