# evML
evML is a protocol for secure distributed compute networks with negligible entry costs. It requires nodes to prove, using their secure enclave, that they're computing results within a trustworthy environment. They can't cheat unless the node operator performs costly hardware tampering. evML then uses spot-checks to catch nodes submitting false outputs. When caught, the enclave’s unique identifier is blacklisted, making the attack cost a sunk investment. Therefore, honesty the only rational behavior. The worst-case analysis in @node-behaviour shows that honesty is optimal with a 5% computational overhead, assuming a hardware attack costs over \$2000. We conclude cheating is irrational within evML, though further empirical validation is warranted.

## Preliminary report
A preliminary report discussing the approach is provided. This work was led by Arbion Halili.

## Analysis code 
We provide analysis code in Python for the Markov Decision Process discussed in the preliminary report. To get started with this 

```bash
pip install pymdptoolbox
python analysis.py
```
You can modify the values in the file yourself to model other scenarios. 
