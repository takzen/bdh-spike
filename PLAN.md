### ETAP 1: Inicjalizacja Repozytorium i Środowiska

- [x] 1. Tworzymy lokalne repozytorium Git `bdh-spike` i inicjalizujemy strukturę katalogów (`bdh_spike/core`, `bdh_spike/plasticity`, `benchmarks`, `tests`).
- [x] 2. Konfigurujemy plik `pyproject.toml` z zależnościami: Python 3.13+ (`uv`), PyTorch 2.x, `snntorch`, `spikingjelly`, `pytest`, `ruff`.
- [x] 3. Wrzucamy do głównego katalogu przygotowany plik `AGENTS.md`, aby asystenci AI (Cursor, Claude Code) od razu trzymali się twardych reguł architektury.
- [x] 4. Publikujemy prywatne repozytorium na GitHub (`takzen/bdh-spike`) z angielskim opisem projektu.

### ETAP 2: Implementacja Rdzenia Neuromorficznego (BDH-PLIF)

- [ ] 5. Tworzymy plik `bdh_spike/core/neuron.py` z klasą neuronu `BDHSpikeCell` dziedziczącą po `torch.nn.Module`.
- [ ] 6. Implementujemy dynamikę potencjału błonowego PLIF: równanie zaniku (leaky decay), twardy reset (hard reset) oraz surogat gradientu (`fast_sigmoid`), który umożliwia wsteczną propagację na zwykłym GPU.
- [ ] 7. Dodajemy wektor sprzężenia rekurencyjnego BDH (`m_bdh`), który nieliniowo moduluje pobudliwość neuronów w kolejnych krokach czasowych.
- [ ] 8. Piszemy pierwsze testy jednostkowe w `tests/test_neurons.py` sprawdzające zachowanie tensora `[T, B, C]` oraz poprawność resetu potencjału.

### ETAP 3: Spike-Driven Attention (Mechanizm Uwagi bez Softmaxa)

- [ ] 9. Tworzymy plik `bdh_spike/core/attention.py` implementujący bezmnożeniową uwagę impulsową.
- [ ] 10. Zastępujemy klasyczny zmiennoprzecinkowy Softmax operacją rzadkiego maskowania asocjacyjnego (`Q_spike AND K_spike`) na binarnych impulsach.
- [ ] 11. Testujemy narzut pamięciowy i upewniamy się, że nie powstają wewnątrz żadne ciągłe tensory float.

### ETAP 4: Silnik Podwójnej Plastyczności (Rozwiązanie Catastrophic Forgetting)

- [ ] 12. Tworzymy moduł `bdh_spike/plasticity/stdp.py` z logiką uczenia online (3-Factor Hebbian Learning / STDP).
- [ ] 13. Rozdzielamy wagi na:
    - [ ] `W_slow` – wagi bazowe trenowane globalnym gradientem offline.
    - [ ] `W_fast` – wagi dynamiczne aktualizowane w locie w bloku `torch.no_grad()` na podstawie różnicy czasowej impulsów pre/post-synaptycznych.
- [ ] 14. Implementujemy moduł homeostazy w `bdh_spike/plasticity/homeostat.py`, który dynamicznie adaptuje próg napięcia `V_th`, chroniąc sieć przed „atakami padaczkowymi” (zbyt duża liczba wyładowań) lub uśpieniem (brak impulsów).

### ETAP 5: Złożenie Modelu i Pipeline Danych

- [ ] 15. Budujemy model nadrzędny `bdh_spike/models/bdh_spike_vit.py` (Vision-BDH-Spike) oraz wersję sekwencyjną dla danych ciągłych.
- [ ] 16. W pliku `bdh_spike/utils/encoders.py` tworzymy enkodery: zamianę surowych wektorów/pikseli na strumienie impulsów czasowych (Rate Encoding, Latency Encoding, Delta Encoding).

### ETAP 6: Metryki Energetyczne i Benchmarki

- [ ] 17. W `bdh_spike/neuromorphic/metrics.py` piszemy kalkulator SOPs (Synaptic Operations) vs FLOPs oraz miernik rzadkości wyładowań (Spike Sparsity Tracker).
- [ ] 18. Tworzymy benchmark `benchmarks/n_mnist_eval.py` testujący działanie na datasetach eventowych (N-MNIST / DVS-Gesture pobieranych automatycznie).
- [ ] 19. Tworzymy benchmark `benchmarks/continual_learning.py` (Split-Task Learning), wykazujący, że po douczeniu nowej klasy obiektów sieć nie zapomina starych dzięki wagom `W_fast`.

### ETAP 7: Diagnostyka i Wizualizacja

- [ ] 20. Piszemy skrypt `bdh_spike/utils/visualizer.py` generujący wykresy wyładowań (Spike Raster Plots) oraz przebiegi potencjału błonowego w czasie.
- [ ] 21. Przygotowujemy lekki podgląd telemetryczny (np. terminalowy HUD lub prosty eksport danych pod interfejs webowy).

### ETAP 8: Dokumentacja, README i Publikacja

- [ ] 22. Piszemy reprezentatywny plik `README.md` w suwerennym, taktycznym stylu Takzen: z matematycznym wyjaśnieniem BDH-PLIF, tabelą porównawczą SOPs vs FLOPs, wykresem rzadkości i instrukcją uruchomienia w 3 linijkach.
- [ ] 23. Wykonujemy pełny przebieg testów `pytest tests/` i formatowanie kodu (`ruff`).
- [ ] 24. Publikujemy oficjalne repozytorium jako pierwsze na świecie połączenie architektury BDH z paradygmatem SNN.
