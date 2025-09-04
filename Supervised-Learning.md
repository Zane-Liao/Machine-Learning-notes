# Introduction to Supervised Learning

- Supervised learning uses labeled data for training.
- Therefore, when using supervised learning algorithms, you should pay attention to whether your dataset has been processed (i.e., cleaned, structured, and labeled).
- Machine learning algorithm tasks are divided into two major categories: regression and classification.
- Model types are generally categorized as discriminant and generative.
- Supervised learning algorithms generally include linear regression, as well as its derivatives, logistic regression, Naive Bayes, Support Vector Machine (SVM), Tree Type, and KNN.
- Linear regression and ridge regression generally have closed-form solutions. In other words, with a closed-form solution, the optimal solution can be obtained using formulas.
- Logistic regression, SVM, decision trees, and clustering algorithms do not have closed-form solutions; we need to use iteration to find the optimal solution.
- Models that can directly obtain theoretical optimal solutions using convex optimization tools include linear regression, ridge regression, Lasso, logistic regression, and SVM.
- Non-convex optimization models: decision trees, K-means, etc.

[[#Regression]]
[[#Classification]]
[[#Linear Regression]]
[[#Logistic Regression]]
[[#Naive Bayes]]
[[#Support Vector Machine (SVM)]]
[[#Kernel Method]]
[[#Covariance Matrix]]
[[#Tree Type]]
[[#KNN]]
[[#KD-Tree]]

### Regression and Classification

Let's first look at the definitions:
- Classification is learning a function 𝑓: 𝑋→𝑌, where the output space 𝑌 is a discrete finite set
- Regression is learning a function 𝑓: 𝑋 → 𝑌, where the output space 𝑌 ⊆ 𝑅 is a subset of the field of continuous real numbers
Let:
* $\mathcal{X}$: input space (features)
* $\mathcal{Y}$: output space (labels)
##### Classification
* **Goal**: Learn $f: \mathcal{X} \to \mathcal{Y}$, where
$$
\mathcal{Y} = \{1, 2, \dots, K\} \quad (K \ge 2)
$$
is a finite set (e.g., cats/dogs/birds)
* **Loss Function**: The most common is **0-1 loss**:
$$
L(f(x), y) = \begin{cases}
0 & \text{if } f(x) = y \\
1 & \text{otherwise}
\end{cases}
$$
or softmax + cross-entropy loss.
* **Learning Method**: Learn $P(y \mid x)$ or the decision boundary.

- Classification Goal: $f^*(x) = \arg\max_{y \in \mathcal{Y}} P(y \mid x)$

##### Regression
* **Goal**: Learn $f: \mathcal{X} \to \mathbb{R}$, where

$$
\mathcal{Y} \subseteq \mathbb{R}
$$
is continuous, such as price, temperature, or rent.
* **Loss Function**: Typically, the mean squared error (MSE) or absolute error (MAE) is used:
$$
L(f(x), y) = (f(x) - y)^2
$$
* **Learning Method**: Learn the expected value $\mathbb{E}[y \mid x]$ or the fitted function value.

- Regression Goal: $f^*(x) = \mathbb{E}[y \mid x]$

##### Visual Geometry

| Task | Data Distribution Image |
| --- | ------------------------- |
| Classification | Various points are distributed in feature space, and the model needs to draw lines/hyperplanes to separate them. |
| Regression | Points are distributed along a continuous curve, and the model needs to fit this curve. |

Source: Bishop, "Pattern Recognition and Machine Learning":
Classification involves discrete labels, regression involves real-valued targets.

Hastie, Tibshirani, Friedman, "The Elements of Statistical Learning":
The goal of regression is to predict a continuous response; for classification, the response is categorical.

----

### Discriminant and Generative Models

- Discriminant models are concerned with "given an email, is it spam?" We learn the joint probability $P(y \mid x)$, which simultaneously models the input x and the output. The generation mechanism of y

- Generative models focus on the question "How is a spam email written?" and directly learn the conditional probability distribution 𝑃(𝑦|𝑥) or the decision boundary 𝑓(𝑥)→𝑦 to achieve classification or regression tasks.

##### Linear Regression
- Linear regression is the first supervised learning algorithm we learn. While it may seem simple at first glance, it's actually quite complex due to its long history and widespread use in various fields. Many variants have emerged, making it quite complex. If you're new to machine learning, it's easy to be overwhelmed by this model. So, if you're asked in an interview which model you're most familiar with, avoid saying you're familiar with linear regression. (🤓)

- Linear regression mainly includes the simplest Simple Linear Regression, the derived Multiple Linear Regression, Least Squares Linear Regression (OLS), and Generalized Linear Models (GLMs, for example, a variant is the Poisson Regression). Regression), linear regression with added L1 (Ridge) and L2 (Lasso) regularization terms (OLS + L2 = Ridge Regression), recursive least squares linear regression (RLS) for time series, nonlinear kernel regression (Kernel method), principal component regression (PCR), linear regression from a Bayesian perspective, etc.

- Linear regression has both discriminant and generative versions (Gaussian Joint Modeling)

##### Logistic Regression
- Logistic regression can be loosely considered linear regression for classification tasks.

- Logistic regression includes the simplest standardized logistic regression, logistic regression with added L1 and L2 terms (same as above), elastic net logistic regression (L1 + L2 = Elastic Net), softmax regression, kernel logistic regression (same as above), logistic regression from a Bayesian perspective (same as above), online logistic regression (SGD, AdaGrad), etc.

- Logistic regression also has both discriminant and generative versions (Gaussian Discriminant Analysis (GDA is a generative variant of logistic regression)

**Linear Regression and Logistic Regression are characterized by strong interpretability, fast training, and good accuracy, but are also susceptible to outliers**

##### Naive Bayes
- Naive Bayes is a generative classification model. Why is it called "Naive"? Because all features are independent given the known class y.

- Naive Bayes is not a single model. It models $P(xi \mid y)$ based on a conditional distribution assumption. The conditional distribution is a large family of exponential distributions. The distribution to use depends on the input data type. For example, for text word frequency classification, we use a multinomial distribution (NB), while for image feature classification, we use a Gaussian NB or Bernoulli NB.

- Naive Bayes is characterized by fast training, the assumption of conditional independence, and robustness when data is scarce.

##### Support Vector Machine (SVM)
- Support Vector Machine (SVM) is a linear discriminant model that finds a hyperplane in feature space that maximizes the geometric margin to optimally distinguish between two classes of data.

- Margin: The distance from the hyperplane to the closest sample point.
- Support Vectors: The sample points $y^{(i)} (w^T x^{(i)} + b) = 1$ that are closest to the hyperplane and satisfy the bounds.
- SVM is essentially a maximum margin classifier.

- Real data is often linearly inseparable or noisy, so we cannot use a hard margin. Instead, we introduce a slack variable $\xi_i \ge 0$ (allowing some samples to cross the bounds).

- The original SVM problem is a convex optimization problem, which is more suitable for computing kernel functions.
- SVM can be extended to nonlinear spaces by using the kernel function K(x,x′) instead of the inner product.

###### Kernel Method
- Kernel methods implicitly map the original input to a high-dimensional feature space by defining a kernel function K(x,x′), thereby constructing a linear model in this space without explicitly performing feature mapping.

- Why do we need kernel methods? Many data are linearly inseparable in the original space, but may be linearly separable in a higher-dimensional space. However, in a higher-dimensional space, the computational cost is very high or even impossible. We introduce an implicit mapping (calculating the inner product $\phi(x)^T \phi(x')$) and avoid directly constructing a high-dimensional space (directly using $K(x, x')$).

- Mathematical definition: A function $K: \mathcal{X} \times \mathcal{X} \rightarrow \mathbb{R}$ is a kernel function if and only if there exists a **Hilbert space** $\mathcal{H}$ and a mapping $\phi: \mathcal{X} \rightarrow \mathcal{H}$ such that:
$$
K(x, x') = \langle \phi(x), \phi(x') \rangle_{\mathcal{H}}
$$

- Kernel functions typically come in many forms, including linear kernels, polynomial kernels, Gaussian kernels (RBF kernels), and sigmoid kernels.

- An example is the original dual form of SVM: $f(x) = \text{sign}\left( \sum_{i=1}^m \alpha_i y^{(i)} \langle x^{(i)}, x \rangle + b \right)$
- After replacing the inner product with a kernel function: $f(x) = \text{sign}\left( \sum_{i=1}^m \alpha_i y^{(i)} K(x^{(i)}, x) + b \right)$

- Common types of kernel methods include SVM with kernel, Kernel PCA, Kernel Ridge Regression, and Kernel K-means.

- Given a set of samples, the kernel matrix (Gram Matrix) $x^{(1)}, \dots, x^{(n)}$, definition: $K_{ij} = K(x^{(i)}, x^{(j)})$ then $K \in \mathbb{R}^{n \times n}$ is a positive semidefinite matrix and must satisfy the following:
1. For any $\alpha \in \mathbb{R}^n$, $\alpha^T K \alpha \ge 0$
2. This is the core of Mercer's theorem: the kernel function must be positive definite.

##### Covariance Matrix
- **Covariance Matrix** describes how multiple variables change together: the degree of correlation between each pair of variables is described by covariance, and the "joint correlation" of the entire set of variables constitutes the covariance matrix

- **Math Define**: Suppose you have $m$ samples, each sample is an $n$-dimensional vector (that is, you have $n$ variables):
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
**Covariance Matrix** $\Sigma \in \mathbb{R}^{n \times n}$ is defined as:
$$
\Sigma = \frac{1}{m} \sum_{i=1}^m (x^{(i)} - \mu)(x^{(i)} - \mu)^T
= \mathbb{E}\left[(x - \mu)(x - \mu)^T\right]
$$
It can also be written in matrix form (the sample matrix is ​​$X$, with one sample per row):
$$
\Sigma = \frac{1}{m} (X - \mu)^T (X - \mu)
$$

- We know that for $\Sigma_{ij}$, it represents the relationship between the $i$th variable and the $j$th variable. Covariance of variables:
$$
\Sigma_{ij} = \text{Cov}(x_i, x_j) = \mathbb{E}[(x_i - \mu_i)(x_j - \mu_j)]
$$
So:
* $\Sigma_{ii} = \text{Var}(x_i)$: Variance of variable $x_i$
* $\Sigma_{ij} > 0$: Positive correlation
* $\Sigma_{ij} < 0$: Negative correlation
* $\Sigma_{ij} = 0$: Uncorrelated (but not necessarily independent)

###### Visual Geometry
* The covariance matrix describes the "scatter" of data points in all directions.
* It can be thought of as the **shape** of the data. * If you plot a 2D data point cloud, the **eigenvector** of the covariance matrix gives the principal direction, and the **eigenvalue** gives the variance (length) in that direction → Principal Component Analysis (PCA) selects the principal eigenvector of the covariance matrix as the projection direction.

- The Covariance Matrix generally satisfies
1. $\Sigma = \Sigma^{T}$ (symmetry)
2. $v^T \Sigma v \ge 0 \, \forall v$ (positive semidefinite)
3. $\lambda_i \ge 0$ (all eigenvalues ​​are non-negative)
4. Positive definite (if all samples are independent)
5. Diagonalizable (orthogonal eigenvectors are diagonalized), etc.
- Application Scenarios:
PCA
GDA
Gaussian Process
Multiple Variable Normal Distribution Model
Whitening et al.

For Example:
- Suppose we have 2 variables:

| x₁ | x₂ |
| --- | --- |
| 1 | 2 |
| 2 | 4 |
| 3 | 6 |
| 4 | 8 |
x₂ = 2 * x₁ => linear correlation

Compute the Covariance Matrix:

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
###### Why can the Covariance Matrix be used directly in a movie recommendation system based on collaborative filtering?
Basic Idea: In a recommendation system, if the covariance between the ratings of two movies is high, it means that if a user likes movie A, they are also likely to like movie B (or that their ratings have the same trend). These movies are moving in the same direction in the "user rating space," so we can use the covariance to determine the similarity between the movies (covariance = similarity).

##### Tree Type
- Tree algorithms are widely used in various fields, from regression and classification to ranking and anomaly detection.
- There are many tree-type algorithms, which can be roughly divided into two categories: simple trees and ensemble trees.
- The simplest tree is the decision tree, with variants like CART (Classification and Regression Tree).
- Decision trees can be used for regression tasks.
- Ensemble trees include Random Forest (an ensemble of multiple CART trees) and Boosted Trees. Boosted Trees include Gradient Boosting Decision Trees (GBDT), XGBoost, LightGBM, CatBoost, AdaBoost, etc.

##### KNN
- Simply put, the KNN (nearest neighbor) algorithm is a parameter-free, distance-based classification or regression method. It's a type of lazy learning algorithm and is commonly used in classification and regression tasks. It's suitable for small sample sizes and nonlinear problems, but it's inefficient in large-scale data or high-dimensional spaces and requires optimization.

##### KD-Tree
- Again, briefly, the KD-Tree is not an algorithm, but rather a special binary tree data structure.
- A strict definition: **KD-Tree is a data structure that supports fast multidimensional space retrieval**. It's used for nearest neighbor searches and range searches in $\mathbb{R}^k$.
- The KD-Tree is primarily used in the KNN algorithm to quickly find the nearest neighbor.