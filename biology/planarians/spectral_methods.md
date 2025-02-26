# **Comparison of Spectral Analysis Methods for Planarian Data**

## **1. Introduction**
This report evaluates five different spectral decomposition techniques applied to the planarian bioelectric dataset:

1. **Fourier Transform**
2. **Standard Wavelet Transform**
3. **Standard Wavelet Transform with Fibonacci Scaling**
4. **Hyperbolic Wavelet Transform**
5. **Hyperbolic Wavelet Transform with Fibonacci Scaling**

The goal is to determine which method best captures **bistability, bioelectric anisotropies, and regenerative state transitions** in the dataset.

---

## **2. Methodology**

### **2.1 Fourier Transform**
The Fourier Transform (FT) decomposes signals into sinusoidal frequency components. It is computed as:
\[
F(k) = \int_{-\infty}^{\infty} f(x) e^{-2\pi i k x} dx
\]
FT provides global spectral information but lacks time localization, making it less effective for non-stationary signals such as **bioelectric wavefronts in morphogenesis**.

### **2.2 Standard Wavelet Transform**
The standard **wavelet transform** applies a localized basis function (wavelet) at different scales:
\[
W_f(a, b) = \int_{-\infty}^{\infty} f(x) \psi_{a, b}(x) dx
\]
where \( \psi_{a, b}(x) \) is a mother wavelet scaled and translated over time.

Unlike FT, **wavelets provide better time-frequency localization**, allowing us to observe **spatiotemporal evolution of bioelectric fields**.

### **2.3 Standard Wavelet with Fibonacci Scaling**
Instead of dyadic scaling (factor of 2), this method scales wavelets using Fibonacci numbers:
\[
S_j = F_j = F_{j-1} + F_{j-2}, \quad F_0 = 1, F_1 = 1
\]
This scaling aligns with **biological growth patterns**, making it suitable for analyzing **developmental morphogenetic signals**.

### **2.4 Hyperbolic Wavelet Transform**
The **Hyperbolic Wavelet Transform** introduces **anisotropic scaling** to better capture structured regularity in different directions:
\[
D_\alpha = \begin{bmatrix} \alpha & 0 \\ 0 & 2 - \alpha \end{bmatrix}
\]
This decomposition **better captures directional properties in regenerative tissue anisotropy**.

### **2.5 Hyperbolic Wavelet with Fibonacci Scaling**
Combining **anisotropic wavelets** with Fibonacci-based scaling allows for **non-linear biological growth constraints** to be incorporated into the spectral decomposition.

---

## **3. Results & Interpretation**

### **3.1 Time-Frequency Localization**
| Method | Time Resolution | Frequency Resolution | Biological Relevance |
|--------|---------------|------------------|-------------------|
| Fourier Transform | Poor | Excellent | Limited for bioelectric signals |
| Standard Wavelet | Good | Good | Detects wavefronts but lacks structured scaling |
| Standard Wavelet + Fibonacci | Good | Excellent | Captures growth patterns effectively |
| Hyperbolic Wavelet | Excellent | Good | Detects directional anisotropies |
| Hyperbolic + Fibonacci | Excellent | Excellent | Best method for capturing bioelectric evolution |

### **3.2 Free Energy & Entropy Analysis**
Entropy and free energy were computed for each method:

| Method | Free Energy | Entropy | Interpretation |
|--------|------------|---------|---------------|
| Fourier Transform | Low | High | Too much global information loss |
| Standard Wavelet | Medium | Medium | Useful but lacks directional encoding |
| Standard Wavelet + Fibonacci | High | Low | Optimal for biological scaling laws |
| Hyperbolic Wavelet | High | Medium | Captures anisotropies well |
| Hyperbolic + Fibonacci | Highest | Lowest | Best balance of detail and sparsity |

---

## **4. Conclusion & Recommendation**
Based on the findings:
- **Hyperbolic Wavelet Transform with Fibonacci Scaling is the most effective method** for analyzing planarian morphogenesis.
- It captures **both directional regularities and bioelectric growth patterns**.
- Standard wavelets **without structured scaling** miss crucial anisotropic effects.

