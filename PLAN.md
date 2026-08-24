### ETAP 1: Inicjalizacja Repozytorium i Środowiska

1. Tworzymy lokalne repozytorium Git `bdh-spike` i inicjalizujemy strukturę katalogów (`bdh_spike/core`, `bdh_spike/plasticity`, `benchmarks`, `tests`).
2. Konfigurujemy plik `pyproject.toml` z zależnościami: Python 3.11+, PyTorch 2.x, `snntorch`, `spikingjelly`, `pytest`, `ruff`.
3. Wrzucamy do głównego katalogu przygotowany plik `AGENTS.md`, aby asystenci AI (Cursor, Claude Code) od razu trzymali się twardych reguł architektury.

### ETAP 2: Implementacja Rdzenia Neuromorficznego (BDH-PLIF)

4. Tworzymy plik `bdh_spike/core/neuron.py` z klasą neuronu `BDHSpikeCell` dziedziczącą po `torch.nn.Module`.
5. Implementujemy dynamikę potencjału błonowego PLIF: równanie zaniku (leaky decay), twardy reset (hard reset) oraz surogat gradientu (`fast_sigmoid`), który umożliwia wsteczną propagację na zwykłym GPU.
6. Dodajemy wektor sprzężenia rekurencyjnego BDH (`m_bdh`), który nieliniowo moduluje pobudliwość neuronów w kolejnych krokach czasowych.
7. Piszemy pierwsze testy jednostkowe w `tests/test_neurons.py` sprawdzające zachowanie tensora `[T, B, C]` oraz poprawność resetu potencjału.

### ETAP 3: Spike-Driven Attention (Mechanizm Uwagi bez Softmaxa)

8. Tworzymy plik `bdh_spike/core/attention.py` implementujący bezmnożeniową uwagę impulsową.
9. Zastępujemy klasyczny zmiennoprzecinkowy Softmax operacją rzadkiego maskowania asocjacyjnego (`Q_spike AND K_spike`) na binarnych impulsach.
10. Testujemy narzut pamięciowy i upewniamy się, że nie powstają wewnątrz żadne ciągłe tensory float.

### ETAP 4: Silnik Podwójnej Plastyczności (Rozwiązanie Catastrophic Forgetting)

11. Tworzymy moduł `bdh_spike/plasticity/stdp.py` z logiką uczenia online (3-Factor Hebbian Learning / STDP).
12. Rozdzielamy wagi na:
    - `W_slow` – wagi bazowe trenowane globalnym gradientem offline.
    - `W_fast` – wagi dynamiczne aktualizowane w locie w bloku `torch.no_grad()` na podstawie różnicy czasowej impulsów pre/post-synaptycznych.
13. Implementujemy moduł homeostazy w `bdh_spike/plasticity/homeostat.py`, który dynamicznie adaptuje próg napięcia `V_th`, chroniąc sieć przed „atakami padaczkowymi” (zbyt duża liczba wyładowań) lub uśpieniem (brak impulsów).

### ETAP 5: Złożenie Modelu i Pipeline Danych

14. Budujemy model nadrzędny `bdh_spike/models/bdh_spike_vit.py` (Vision-BDH-Spike) oraz wersję sekwencyjną dla danych ciągłych.
15. W pliku `bdh_spike/utils/encoders.py` tworzymy enkodery: zamianę surowych wektorów/pikseli na strumienie impulsów czasowych (Rate Encoding, Latency Encoding, Delta Encoding).

### ETAP 6: Metryki Energetyczne i Benchmarki

16. W `bdh_spike/neuromorphic/metrics.py` piszemy kalkulator SOPs (Synaptic Operations) vs FLOPs oraz miernik rzadkości wyładowań (Spike Sparsity Tracker).
17. Tworzymy benchmark `benchmarks/n_mnist_eval.py` testujący działanie na datasetach eventowych (N-MNIST / DVS-Gesture pobieranych automatycznie).
18. Tworzymy benchmark `benchmarks/continual_learning.py` (Split-Task Learning), wykazujący, że po douczeniu nowej klasy obiektów sieć nie zapomina starych dzięki wagom `W_fast`.

### ETAP 7: Diagnostyka i Wizualizacja

19. Piszemy skrypt `bdh_spike/utils/visualizer.py` generujący wykresy wyładowań (Spike Raster Plots) oraz przebiegi potencjału błonowego w czasie.
20. Przygotowujemy lekki podgląd telemetryczny (np. terminalowy HUD lub prosty eksport danych pod interfejs webowy).

### ETAP 8: Dokumentacja, README i Publikacja

21. Piszemy reprezentatywny plik `README.md` w suwerennym, taktycznym stylu Takzen: z matematycznym wyjaśnieniem BDH-PLIF, tabelą porównawczą SOPs vs FLOPs, wykresem rzadkości i instrukcją uruchomienia w 3 linijkach.
22. Wykonujemy pełny przebieg testów `pytest tests/` i formatowanie kodu (`ruff`).
23. Publikujemy oficjalne repozytorium jako pierwsze na świecie połączenie architektury BDH z paradygmatem SNN.
