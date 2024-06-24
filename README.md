# SchrodingerFE

This package implements the resolution of the Schrödinger equation with finite elements. 

In 1D, the available methods are P1 finite elements and P2 finite elements.

In 2D, the available method is P1.

There are four methods to solve the Schrödinger equation: **FCI_full**, **FCI_sparse**,  **selected_CI_sparse**, **CDFCI_sparse**. All the methods aim at solving the ground state of the full configuration interaction (CI) eigenvalue problem,
$$H\pmb{c}=E_0 S\pmb{c},$$
with $H_{\pmb{is},\,\pmb{jt}} = \langle\Phi_{\pmb{is}}|\mathcal{H}|\Phi_{\pmb{jt}}\rangle$,  $S_{\pmb{is},\,\pmb{jt}} = \langle\Phi_{\pmb{is}}|\Phi_{\pmb{jt}}\rangle$, and $\Phi_{\pmb{is}}$ is a Slater determinant.

#### FCI_full & FCI_sparse
Both solve the eigenvalue problem by eigenvalue solvers, where *FCI_full* generates the Hamiltonian matrix directly, while *FCI_sparse* uses the matrix-free technique.

#### selected_CI_sparse
Use the selected CI algorithm in [![DOI](https://img.shields.io/badge/DOI-10.1016/j.jcp.2023.112251-blue)](https://doi.org/10.1016/j.jcp.2023.112251) to solve the eigenvalue problem. This method is only efficient for the Wigner localized systems, i.e., $`\mathcal{H}_{\alpha}=-\alpha T+V_{\rm{ee}}+V_{\rm{ext}}`$ with $\alpha\ll 1$. The algorithm includes two parts: initial guess construction and determinant selection.

**Initial guess construction:**
Consider the strongly correlated limit $\alpha\to0$ as the initial guess.
1. Solve the $\alpha=0$ limit problem 
```math
\min_{(\pmb{r}_1,\cdots,\pmb{r}_N)\in\Omega^N}\Big\{V_{\rm{ee}}(\pmb{r}_1,\cdots,\pmb{r}_N)+V_{\rm{ext}}(\pmb{r}_1,\cdots,\pmb{r}_N)\Big\},
```
to obtain the minimizer subset $U_{\rm s}$.

2. Select the Slater determinants corresponding to $U_{\rm s}$
```math
\begin{aligned}
				\mathcal{I}_{\delta} := \bigcup_{(\pmb{r}_1,\dots,\pmb{r}_N)\in U_{\rm s}}\Big\{ \pmb{is}\in\mathcal{I}~:~
				\text{there exists a permutation } \mathcal{P}\text{ of }\{1,\cdots,N\}
				\text{ such that } \\
				\max_{1\leq k\leq N} \big|\pmb{x}_{i_k}-\pmb{r}_{i_{\mathcal{P}(k)}}\big| \leq \delta
				\text{ with } \pmb{i} = (i_1,\cdots,i_N)\Big\}
				\qquad
			\end{aligned}
```
where $\mathcal{I}$ is the full CI Slater determinant set.

3. Generate the Hamiltonian within $`\mathcal{I}_{\delta}`$ and solve the corresponding eigenvalue problem to obtain the semi-classical limit initial state $`\pmb{c}_{\mathcal{I}_{\delta}}`$.

**Determinant selection:**
Aim to find the important determinant set $`\mathcal{J}\subset\mathcal{I}`$ adaptively. The corresponding approximation problem is 
```math
\min_{\pmb{c}|_{\mathcal{J}}\neq 0}f(\pmb{c}) \quad {\rm with}  \quad f(\pmb{c}) = \frac{\pmb{c}^\top H ^{(\alpha)} \pmb{c}}{\pmb{c}^\top S\pmb{c}}.
```
1. Find the connected set of a randomly selected subset $\mathcal{L}^{(k)} {\subset} \mathcal{J}^{(k)}$,
```math
\mathcal{L}^{(k)}_{\rm c} := \big\{\pmb{jt} :~ \big(H ^{(\alpha)}\big)_{\pmb{is},\pmb{jt}} \neq 0,~ \pmb{is} \in \mathcal{L}^{(k)}\big\},
```
and randomly select $`\mathcal{L}^{(k)}_{\rm s}{\subset}\mathcal{L}^{(k)}_{\rm c}`$ such that $`|\mathcal{L}^{(k)}_{\rm s}|= O(\mathfrak{k})`$. Then update $`\mathcal{J}^{(k+1)} = \mathcal{J}^{(k)} \cup \mathcal{K}^{(k)}`$,
```math
\mathcal{K}^{(k)} := \Big\{ \pmb{is}\in \mathcal{L}^{(k)}_{\rm s} :~ |\pmb{g}^{(k)}|_{\pmb{is}} \text{ is among the } \mathfrak{k} \text{ largest magnitudes of } \pmb{g}^{(k)}|_{\mathcal{L}^{(k)}_{\rm s}} \Big\}
```
with $`\pmb{g}^{(k)}:=\nabla f(\pmb{c}^{(k)})`$.

2. Update the CI state by compressed gradient $`\pmb{g}_{\rm s}^{(k)} := \pmb{g}^{(k)}|_{\mathcal{K}^{(k)}}`$,
```math
\pmb{c}^{(k+1)} = \pmb{c}^{(k)} + \beta^{(k)} \pmb{g}_{\rm s}^{(k)},
```
where $`\beta^{(k)} := \Big\{\beta\in\mathbb{R}:\min_\beta f\big(\pmb{c}^{(k)}+\beta \pmb{g}_{\rm s}^{(k)}\big)\Big\}`$.

3. Update the gradient by
$$\pmb{g}^{(k+1)} = \dfrac{2}{(\pmb{c}^{(k+1)})^\top S \pmb{c}^{(k+1)}}\pmb{b}^{(k+1)} 
			- \dfrac{2({\pmb{c}^{(k+1)}})^\top H^{(\alpha)} \pmb{c}^{(k+1)}}{\big(({\pmb{c}^{(k+1)}})^\top S \pmb{c}^{(k+1)}\big)^2}\pmb{d}^{(k+1)}\quad{\rm with}$$
$$\pmb{b}^{(k+1)}:= H^{(\alpha)}\pmb{c}^{(k+1)}= \pmb{b}^{(k)}+\beta^{(k)} H^{(\alpha)}\pmb{g}_s^{(k)}\quad{\rm and }\quad\pmb{d}^{(k+1)}:= S\pmb{c}^{(k+1)}= \pmb{d}^{(k)}+\beta^{(k)} S\pmb{g}_s^{(k)}.$$


#### CDFCI_sparse
Use the coordinate descent full CI (CD-FCI) algorithm in [![DOI](https://img.shields.io/badge/DOI-10.1021/acs.jctc.9b00138-blue)](https://doi.org/10.1021/acs.jctc.9b00138) to solve the eigenvalue problem. The initial guess is the restricted Hartree Fock approximation. The determinant selection process is similar with *selected_CI_sparse*, but $\mathcal{K}^{(k)}$ is selected within the connected set of $\mathcal{J}^{(k)}$.