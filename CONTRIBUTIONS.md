# **Contributing to HLD-JAX**

## **How to Contribute**
We welcome contributions from the community to improve and expand HLD-JAX. Contributions can take various forms, including code enhancements, documentation improvements, and theoretical discussions. Below are the different ways you can contribute:

### **1️⃣ Reporting Issues**
If you encounter bugs, unexpected behavior, or have feature requests, please open an issue on our GitHub repository. When submitting an issue, try to provide:
- A clear description of the problem.
- Steps to reproduce the issue.
- Any relevant error messages or logs.

### **2️⃣ Submitting Pull Requests (PRs)**
To submit a pull request:
1. **Fork the Repository**: Click the "Fork" button on the GitHub page.
2. **Clone the Fork Locally**:
   ```bash
   git clone https://github.com/EBellache/HLD-JAX
   cd HLD-JAX
   ```
3. **Create a New Branch**:
   ```bash
   git checkout -b feature-name
   ```
4. **Make Your Changes**: Ensure code is well-commented and follows the project’s coding standards.
5. **Run Tests**: Before submitting, run all tests to ensure stability.
   ```bash
   pytest tests/
   ```
6. **Commit and Push**:
   ```bash
   git add .
   git commit -m "Description of changes"
   git push origin feature-name
   ```
7. **Submit a Pull Request**: Go to the original repository and create a PR.

### **3️⃣ Coding Guidelines**
- Follow **PEP 8** for Python styling.
- Use **JAX and NumPy** efficiently to optimize performance.
- Ensure **GPU compatibility** by leveraging `jit` and `pmap` when possible.
- Keep code **functional and modular** for better readability and testing.

### **4️⃣ Documentation Contributions**
- If you improve an existing feature or add a new one, update the relevant section in `README.md` and `docs/`.
- Submit additional examples or tutorial notebooks to the `examples/` directory.

### **5️⃣ Theoretical Contributions**
HLD-JAX is designed as a **general-purpose framework for holographic memory states**. We encourage discussions and theoretical expansions in:
- **Loop Quantum Gravity & Tetrads**
- **Gauge Field Projections & Solitonic Interactions**
- **Holographic Free Energy & Sparsification Techniques**
- **Applications to Biology, Material Science, and Physics**

If you have **new mathematical insights**, feel free to submit an issue or contribute to `docs/theory/` with an explanation.

### **6️⃣ Testing & Benchmarks**
- Always include test cases in `tests/` for new features.
- Performance benchmarks are encouraged, especially for **GPU execution speed comparisons**.

### **7️⃣ Community & Discussions**
- Join our **GitHub Discussions** to share ideas.
- Engage with other contributors and researchers.

---
🚀 **We appreciate your contributions to HLD-JAX!**

