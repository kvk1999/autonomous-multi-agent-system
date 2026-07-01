- [x] Add benchmark_concrete_metrics.py to generate synthetic points and time CPU (NumPy) vs GPU (PyTorch) haversine distance matrix

- [x] Run benchmark once locally and hardcode resulting times into app.py as “Concrete Metrics”


- [ ] Add generate_mock_apac_deliveries.py to generate mock_apac_deliveries.csv with 10,000 rows
- [ ] Wire app.py to load mock_apac_deliveries.csv and add Region Configuration + Monsoon Mode toggle
- [ ] Update optimization logic to apply monsoon travel-time congestion multiplier
- [ ] Update gemini_agent.py system prompt + simulated responses to be APAC/Zone-aware
- [ ] Smoke test: run streamlit app and verify UI renders, dataset loads, toggles work
