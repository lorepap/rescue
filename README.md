# EDTMSR
Enhanced Deep Traffic Matrix Super-Resolution Network


# Configuration
Code is tested with Python=3.8.13.
Install requirements with
```
pip -r install requirements/requirements.txt
```
Configure training settings for federated learning by editing _config/federated.yml_.

# Running
To run the EDTMSR code use the runner within the _runners_ folder:
```
python runners/federated_runner.py
```
You can choose between multi-client or single-client trainings. When multi-client is set to true, multiple sequential threads are thrown setting an increasing number of clients - from client_range[0] to client_range[1] (see _federated.yml_). When multi-client is false, a thread will be thrown training a single federated model with the number of clients set as _n_clients_.

# Plotting and results evaluation
Before plotting, launch the evaluation script to compute the metrics obtained by the inference process.
