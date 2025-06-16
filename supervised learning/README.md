- Supervised learning uses labeled data for training
- Therefore, when using supervised learning algorithms, you need to pay attention to whether your dataset has been processed (i.e., data cleaning and structuring, as well as labeling)
- Machine learning algorithm tasks are divided into two major categories: regression and classification
- Model categories are generally divided into discriminative and generative models
- Supervised learning algorithms generally include linear regression, and logistic regression derived from linear regression, Naive Bayes, Support Vector Machine (SVM), Tree Type, KNN, etc.


### Regression and Classification

Let's first look at the definitions:
- Classification is learning a function 𝑓: 𝑋→𝑌, where the output space 𝑌 is a discrete finite set
- Regression is learning a function 𝑓: 𝑋 → 𝑌, where the output space 𝑌 ⊆ 𝑅 is a subset of the continuous real domain

Let:
* $\mathcal{X}$: Input space (features)
* $\mathcal{Y}$: Output space (labels)

##### Classification
* **Goal**: Learn $f: \mathcal{X} \to \mathcal{Y}$, where
  $$
  \mathcal{Y} = \{1, 2, \dots, K\} \quad (K \ge 2)
  $$
  is a finite set (e.g., cat/dog/bird)
* **Loss function**: Most common is **0-1 loss**:
  $$
  L(f(x), y) = \begin{cases}
  0 & \text{if } f(x) = y \\
  1 & \text{otherwise}
  \end{cases}
  $$
  or softmax + cross-entropy loss.
* **Learning approach**: Learn $P(y \mid x)$ or decision boundary.

- Classification Goal: $f^*(x) = \arg\max_{y \in \mathcal{Y}} P(y \mid x)$

#####  Regression
* **Goal**: Learn $f: \mathcal{X} \to \mathbb{R}$, where

  $$
  \mathcal{Y} \subseteq \mathbb{R}
  $$
  is continuous, e.g., price, temperature, rent.
* **Loss function**: Usually uses Mean Squared Error (MSE) or Mean Absolute Error (MAE):
  $$
  L(f(x), y) = (f(x) - y)^2
  $$
* **Learning approach**: Learn expected value $\mathbb{E}[y \mid x]$ or fit function values.

- Regression Goal: $f^*(x) = \mathbb{E}[y \mid x]$

##### Visual Geometry

| Task | Data Distribution Visualization |
| --- | ------------------------- |
| Classification | Points of different classes distributed in feature space, model needs to draw lines/hyperplanes to separate them |
| Regression | Points distributed along some continuous curve, model needs to fit this curve |


Source: Bishop, "Pattern Recognition and Machine Learning":
Classification involves discrete labels, regression involves real-valued targets.

Hastie, Tibshirani, Friedman, "The Elements of Statistical Learning":
The goal of regression is to predict a continuous response; for classification, the response is categorical.

----

### Discriminative and Generative Models

- Discriminative models are when we care about "given an email, determine if it's spam" - we learn the joint probability $P(y \mid x)$, i.e., simultaneously modeling the generation mechanism of input x and output y

- Generative models are when we care about "how is a spam email written" - directly learn the conditional probability distribution 𝑃(𝑦∣𝑥), or directly learn the decision boundary 𝑓(𝑥)→𝑦, to achieve classification or regression tasks

##### Linear Regression
- Linear regression is the first supervised learning algorithm we learn. At first glance, this model seems very simple, but actually linear regression has existed for many years and has been used in various fields. It has derived many variants and is very complex. If you're learning machine learning for the first time, it's easy to be overwhelmed by this model. So if you go to an interview and the interviewer asks which model you're most familiar with, try not to say you're familiar with linear regression (🤓)

- Linear regression mainly includes the simplest Simple Linear Regression, Multiple Linear Regression derived from Simple, Ordinary Least Squares linear regression (OLS), Generalized Linear Models (GLM, e.g., a variant is Poisson Regression), linear regression with L1 (Ridge) and L2 (Lasso) regularization terms (OLS + L2 = Ridge regression), Recursive Least Squares linear regression for time series (RLS), nonlinear Kernel regression (Kernel methods), Principal Component Regression (PCR), Bayesian perspective linear regression, etc.

- Linear regression has both discriminative and generative versions (Gaussian Joint Modeling)

##### Logistic Regression
- Logistic regression can be loosely viewed as linear regression used for classification tasks

- Logistic regression includes the simplest standard logistic regression, logistic regression with L1 and L2 (same as above), Elastic Net logistic regression (L1 + L2 = Elastic Net), Softmax regression, Kernel logistic regression (same as above), Bayesian perspective logistic regression (same as above), online logistic regression (SGD, AdaGrad), etc.

- Logistic regression also has both discriminative and generative versions (Gaussian Discriminant Analysis, GDA is the generative variant of logistic regression)


**The characteristics of Linear Regression and Logistic regression are relatively strong interpretability, fast training, good accuracy, but they are relatively susceptible to outliers**

##### Naive Bayes
- Naive Bayes is a generative classification model. Why do we need to call it "Naive"? Because all features are mutually independent given that the class y is known

- Naive Bayes is not a single model. It models $P(x_i \mid y)$ based on conditional distribution assumptions. Conditional distributions are a large class of exponential family distributions. We can determine which distribution to use based on the data input type. For example, for classifying text word frequency, we use multinomial distribution (Multinomial NB), for image feature classification we use Gaussian distribution (Gaussian NB), Bernoulli NB, etc.

- The characteristics of Naive Bayes are fast training, assuming conditional independence, and being more robust when data is scarce

##### Support Vector Machine (SVM)
- SVM is a linear discriminative model that finds a hyperplane in the feature space that maximizes the geometric margin to optimally distinguish data of two categories


- **Margin**: refers to the distance from the hyperplane to the closest sample points
- **Support Vectors**: those sample points that are closest to the hyperplane and satisfy the constraint boundary $y^{(i)} (w^T x^{(i)} + b) = 1$
- SVM is essentially a **maximum margin classifier**

- In real data, it's often linearly inseparable or there's always noise, so we can't use hard margin, but introduce slack variables $\xi_i \ge 0$ (allowing some samples to cross the boundary)

- The primal problem of SVM is a convex optimization problem, more suitable for computing kernel functions
- SVM can be extended to nonlinear spaces by using kernel functions K(x,x′) to replace inner products

###### Kernel Method
- **Kernel methods** are a technique that defines kernel functions K(x,x′) to implicitly map original inputs to high-dimensional feature spaces, construct linear models in that space, without explicitly performing feature mapping

- Why do we need kernel methods? Much data is linearly inseparable in the original space, but may be linearly separable in higher-dimensional spaces. However, computation in high-dimensional spaces is very expensive or even impossible to compute. We introduce implicit mapping (computing inner products $\phi(x)^T \phi(x')$) and avoid directly constructing high-dimensional spaces (directly using $K(x, x')$)

- Mathematical definition: A function $K: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$ is a kernel function if and only if there exists a **Hilbert space** $\mathcal{H}$ and mapping $\phi: \mathcal{X} \rightarrow \mathcal{H}$, such that:
$$
K(x, x') = \langle \phi(x), \phi(x') \rangle_{\mathcal{H}}
$$

- Kernel functions usually have various kernels: linear kernel, polynomial kernel, Gaussian kernel (RBF kernel), sigmoid kernel, etc.

- An example is that the original dual form of SVM is: $f(x) = \text{sign}\left( \sum_{i=1}^m \alpha_i y^{(i)} \langle x^{(i)}, x \rangle + b \right)$
- After using kernel functions to replace inner products: $f(x) = \text{sign}\left( \sum_{i=1}^m \alpha_i y^{(i)} K(x^{(i)}, x) + b \right)$

- Common types of kernel methods include SVM with kernel, Kernel PCA, Kernel Ridge Regression, Kernel K-means, etc.

- Kernel Matrix (Gram Matrix): Given a set of samples $x^{(1)}, \dots, x^{(n)}$, define: $K_{ij} = K(x^{(i)}, x^{(j)})$, then $K \in \mathbb{R}^{n \times n}$ is a positive semi-definite matrix that must satisfy:
1. For any $\alpha \in \mathbb{R}^n$, we have $\alpha^T K \alpha \ge 0$ 
2. This is the core of Mercer's theorem: kernel functions must be positive definite kernels.


##### Covariance Matrix
- **Covariance matrix** characterizes how multiple variables vary together: the degree of correlation between each pair of variables is described by covariance, and the "joint correlation" of the entire group of variables constitutes the covariance matrix

- **Math Define**: Suppose you have $m$ samples, each sample is an $n$-dimensional vector (i.e., you have $n$ variables):
$$
X = 
\begin{bmatrix}
\text{---} (x^{(1)})^T \text{---} \\
\text{---} (x^{(2)})^T \text{---} \\
\vdots \\
\text{---} (x^{(m)})^T \text{---}
\end{bmatrix}
\in \mathbb{R}^{m \times n}
$$
Sample mean vector:
$$
\mu = \frac{1}{m} \sum_{i=1}^m x^{(i)} \in \mathbb{R}^n
$$
**Covariance matrix** $\Sigma \in \mathbb{R}^{n \times n}$ is defined as:
$$
\Sigma = \frac{1}{m} \sum_{i=1}^m (x^{(i)} - \mu)(x^{(i)} - \mu)^T
= \mathbb{E}\left[(x - \mu)(x - \mu)^T\right]
$$
Can also be written in matrix form (sample matrix as $X$, each row is a sample):
$$
\Sigma = \frac{1}{m} (X - \mu)^T (X - \mu)
$$

- We know that for $\Sigma_{ij}$, it represents the covariance between the $i$-th variable and the $j$-th variable:
$$
\Sigma_{ij} = \text{Cov}(x_i, x_j) = \mathbb{E}[(x_i - \mu_i)(x_j - \mu_j)]
$$
So:
* $\Sigma_{ii} = \text{Var}(x_i)$: variance of variable $x_i$
* $\Sigma_{ij} > 0$: positive correlation
* $\Sigma_{ij} < 0$: negative correlation
* $\Sigma_{ij} = 0$: uncorrelated (but not necessarily independent)

###### Visual Geometry
* The covariance matrix describes the "spread" of data points in various directions.
* Can be viewed as the **shape contour** of the data.
* If you plot a 2D data point cloud, the **eigenvectors** of the covariance matrix give the principal directions, and the **eigenvalues** give the variance (length) in that direction → Principal Component Analysis (PCA) selects the principal eigenvectors of the covariance matrix as projection directions.


- Covariance Matrix generally satisfies:
1. $\Sigma = \Sigma^{T}$ (symmetry)
2. $v^T \Sigma v \ge 0 \, \forall v$ (positive semi-definite)
3. $\lambda_i \ge 0$ (all eigenvalues non-negative)
4. Can be positive definite (if all samples are independent)
5. Can be diagonalized (orthogonal eigenvector diagonalization), etc.

- Application scenarios:
PCA
GDA
Gaussian Process
Multiple Variable Normal Distribution Model 
Whitening, etc.

For Example:
- Suppose We have 2 variables: 

| x₁  | x₂  |
| --- | --- |
| 1   | 2   |
| 2   | 4   |
| 3   | 6   |
| 4   | 8   |
x₂ = 2 * x₁ => linearly correlated

Compute Covariance Matrix：

$$
\Sigma = 
\begin{bmatrix}
\text{Var}(x_1) & \text{Cov}(x_1, x_2) \\
\text{Cov}(x_2, x_1) & \text{Var}(x_2)
\end{bmatrix}
= 
\begin{bmatrix}
1.67 & 3.33 \\
3.33 & 6.67
\end{bmatrix}
$$

###### Why can Covariance Matrix directly do Collaborative Filtering-based movie recommendation systems?
- Basic Idea: In recommendation systems, **if two movies have high rating covariance**, it means: if a user likes A, they are also likely to like B (or both rating trends are consistent). These movies are **directionally consistent** in the "user rating space", so we can use covariance to judge **similarity** between movies (covariance = similarity)

##### Tree Type
- Tree type algorithms can be widely applied to multiple fields, whether for regression and classification, or ranking and anomaly detection, etc.

- There are many tree-type algorithms, which we can roughly categorize into two types: basic trees (Simple tree) and ensemble trees (Ensemble tree)

- Basic trees mainly include Decision Tree, with variants like CART (Classification and Regression Trees)

- Linear Regression belongs to a type of Decision Tree

- Ensemble trees mainly include Random Forest (ensemble of multiple CART trees), Boosting Trees, where boosting trees mainly include GBDT (Gradient Boosting Decision Tree), XGBoost, LightGBM, CatBoost, AdaBoost, etc.


##### KNN
- Just a brief mention: KNN (k-nearest neighbors) algorithm is a non-parametric, distance-based classification or regression method. It belongs to lazy learning algorithms, commonly used in classification and regression tasks. It's suitable for small samples and nonlinear problems, but has low efficiency in large-scale data or high-dimensional spaces and needs optimization.

##### KD-Tree
- Also just a brief mention: KD-Tree is not an algorithm, but a special binary tree data structure
- Rigorous definition: **KD-Tree is a data structure that supports fast multi-dimensional space retrieval**, used for nearest neighbor search, range search, and other operations in $\mathbb{R}^k$
- KD-Tree is mainly used for KNN algorithms, facilitating fast nearest neighbor finding