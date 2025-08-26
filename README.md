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

How to Run

Train an agent:

python main.py


Export the learned policy to PRISM:

python export_to_prism.py


Run verification in PRISM:

Open the generated .prism model file.

Execute provided queries (examples in queries.pctl).

📂 Repository Structure
├── agent.py              # Q-learning agent
├── environment.py        # GridWorld environment
├── export_to_prism.py    # Export policy to PRISM model
├── main.py               # Training and evaluation script
├── policy_model.prism    # Example PRISM model
├── sto_*                 # Stochastic versions of code
├── test/                 # Testing folder
└── README.md             # Project documentation

📊 Example Results

Deterministic policy converges faster but is less robust under noise.

Stochastic policies provide better safety guarantees but may require more training.

PRISM verification confirms theoretical guarantees of reachability and safety.

📄 License

This project is released under the MIT License.
