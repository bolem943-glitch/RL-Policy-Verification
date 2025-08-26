# RL-Policy-Verification

This repository contains the implementation and experiments for my undergraduate thesis project.  
The work explores how **reinforcement learning (RL) policies** can be exported and formally verified using the **PRISM model checker**.

---

## 📌 Project Overview
- Implemented Q-learning agents in a custom **GridWorld** environment.  
- Exported deterministic and stochastic policies into **PRISM** models (DTMC/MDP).  
- Verified properties such as **goal reachability** and **safety** using PCTL queries.  
- Compared empirical training results with formal verification outcomes.

---

## 🛠️ Requirements
- Python 3.9+  
- NumPy  
- PRISM model checker (download from [http://www.prismmodelchecker.org](http://www.prismmodelchecker.org))  

Install Python dependencies:
```bash
pip install -r requirements.txt
