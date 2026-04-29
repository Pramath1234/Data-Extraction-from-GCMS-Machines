# GC-MS Spectrum Peak Extraction — GCMS_Documents_PramathKP

This repository contains the full implementation, output data, manual verification tooling, and associated reports for a study on **automated peak extraction from GC-MS vector PDF spectra** using three algorithmic approaches, each implemented in Python and C++ (serial, OpenMP parallel, and MPI parallel).

---

## Project Overview

GC-MS (Gas Chromatography–Mass Spectrometry) instruments export spectra as vector PDF files in which:
- **Vertical line segments** represent ion fragment peaks
- **Numeric text labels** identify the corresponding m/z values
- **Horizontal line segments** define the x- and y-axes
- A **text string** near the top identifies the compound name

This project implements and benchmarks three algorithms for automatically extracting the m/z spectrum and relative abundance values from these vector PDFs, without OCR or rasterisation — working directly with the vector geometry and embedded text.

**Dataset:** 213 vector PDF spectra from the Combustion, Gasification and Propulsion Laboratory, Indian Institute of Science (IISc), Bengaluru.

---

## Repository Structure

```
GCMS_Documents_PramathKP/
│
├── Greedy/                        # Algorithm 1 — Greedy Peak Matching
│   ├── Greedy.py                  # Python implementation (multiprocessing)
│   ├── GreedySerial.cpp           # C++ serial implementation
│   ├── GreedyOMP.cpp              # C++ OpenMP parallel implementation
│   ├── GreedyMPI.cpp              # C++ MPI parallel implementation
│   └── Greedy.json                # Output: extracted spectra (230 entries)
│
├── DynamicProgramming/            # Algorithm 2 — DP-based Label–Peak Assignment
│   ├── DP.py                      # Python implementation (multiprocessing)
│   ├── DPSerial.cpp               # C++ serial implementation
│   ├── DPOMP.cpp                  # C++ OpenMP parallel implementation
│   ├── DPMPI.cpp                  # C++ MPI parallel implementation
│   └── DP.json                    # Output: extracted spectra (229 entries)
│
├── HungarianGraph/                # Algorithm 3 — Hungarian Algorithm / Graph Matching
│   ├── Graph.py                   # Python implementation (multiprocessing)
│   ├── GraphSerial.cpp            # C++ serial implementation
│   ├── GraphOMP.cpp               # C++ OpenMP parallel implementation
│   ├── GraphMPI.cpp               # C++ MPI parallel implementation
│   └── Graph.json                 # Output: extracted spectra (230 entries)
│
├── Manual_Verif/                  # Manual verification tooling
│   ├── ManualScripting.py         # CLI script: compares predicted vs actual RA
│   └── index.html                 # Web-based interactive verification tool
│
└── Report:Paper:PPT(s)/           # Reports, paper drafts, and presentations
    ├── GC_MS_Paper - FinalDraft.pdf
    ├── ReportGCMS.pdf
    ├── Automated Spectrum Peak Extraction - PPT.pdf
    └── Parallel Processing - PPT.pdf
```

---

## Algorithms

### Algorithm 1 — Greedy (`Greedy/`)
A **greedy nearest-neighbour** approach. For each candidate m/z label extracted from the PDF, the algorithm searches for the closest vertical peak segment (by x-position) within a dynamically computed tolerance. Labels are accepted if the vertical gap between the label centre and the peak tip falls within geometric bounds. Relative abundance is computed from the normalised peak height.

### Algorithm 2 — Dynamic Programming (`DynamicProgramming/`)
A **monotone sequence alignment** approach. Labels and peaks are both sorted by x-position, and a DP table is filled to find the minimum-cost monotone matching (with gap penalties for unmatched labels or peaks). Backtracking recovers the optimal label–peak assignment. Relative abundance is computed from the matched peak height.

### Algorithm 3 — Hungarian Graph (`HungarianGraph/`)
A **global optimal assignment** approach using the **Hungarian algorithm**. A cost matrix is built between all candidate labels and all detected peaks (with geometry-based validity checks and dummy columns for unmatched labels). The Hungarian algorithm finds the minimum-cost perfect matching. Relative abundance is computed from the assigned peak's normalised height.

---

## Implementations

Each algorithm is provided in four variants:

| Variant | Language | Parallelism | File suffix |
|---|---|---|---|
| Python | Python 3 | `concurrent.futures.ProcessPoolExecutor` (6 workers) | `.py` |
| Serial C++ | C++17 | None (single-threaded) | `Serial.cpp` |
| OMP C++ | C++17 | OpenMP (`#pragma omp parallel for`) | `OMP.cpp` |
| MPI C++ | C++17 | MPI (`MPI_Bcast`, `MPI_Send`/`Recv`) | `MPI.cpp` |

---

## Dependencies

### Python
```
pymupdf       (fitz)    # PDF vector parsing
pdfplumber               # supplementary PDF text extraction
pyinstrument             # profiling
scipy                    # linear_sum_assignment (Graph.py only)
numpy
```
Install with:
```bash
pip install pymupdf pdfplumber pyinstrument scipy numpy
```

### C++
| Library | Purpose | Install |
|---|---|---|
| [MuPDF](https://mupdf.com/) (`libmupdf`) | PDF parsing | `brew install mupdf` |
| [nlohmann/json](https://github.com/nlohmann/json) | JSON output | `brew install nlohmann-json` |
| libomp | OpenMP runtime (OMP variants) | `brew install libomp` |
| Open MPI | MPI runtime (MPI variants) | `brew install open-mpi` |

---

## Building the C++ Files

### Serial
```bash
g++ -std=c++17 -O2 \
  -I/usr/local/opt/mupdf/include \
  -I/usr/local/opt/nlohmann-json/include \
  -L/usr/local/opt/mupdf/lib -lmupdf \
  GreedySerial.cpp -o GreedySerial
```

### OpenMP
```bash
g++ -std=c++17 -O2 -Xpreprocessor -fopenmp \
  -I/usr/local/opt/mupdf/include \
  -I/usr/local/opt/nlohmann-json/include \
  -I/usr/local/opt/libomp/include \
  -L/usr/local/opt/mupdf/lib \
  -L/usr/local/opt/libomp/lib \
  -lmupdf -lomp \
  GreedyOMP.cpp -o GreedyOMP
```
Control thread count via environment variable:
```bash
OMP_NUM_THREADS=4 ./GreedyOMP
```

### MPI
```bash
mpicxx -std=c++17 -O2 \
  -I/usr/local/opt/mupdf/include \
  -I/usr/local/opt/nlohmann-json/include \
  -L/usr/local/opt/mupdf/lib -lmupdf \
  GreedyMPI.cpp -o GreedyMPI
```
Run with:
```bash
mpirun -np 4 ./GreedyMPI
```

> The same compile commands apply to `DP*` and `Graph*` files with their respective filenames.

---

## Running the Python Scripts

```bash
cd Greedy/
python3 Greedy.py        # processes FinalDataset, saves Greedy.json

cd ../DynamicProgramming/
python3 DP.py            # saves DP.json

cd ../HungarianGraph/
python3 Graph.py         # saves Graph.json
```

---

## Output Format

All implementations produce a JSON file with one entry per PDF:

```json
[
  {
    "file": "10001A.pdf",
    "chemical_name": "Lycopodan-8-one, 11,12-didehydro-5-hydroxy-15-methyl-, (5β,15R)-",
    "spectrum": ["41", "55", "77", "105", "160", "174", "191", "233"],
    "relative_abundance": [50.99, 26.54, 26.54, 10.21, 9.19, 100.0, 82.64, 24.50]
  },
  ...
]
```

| Field | Description |
|---|---|
| `file` | Source PDF filename |
| `chemical_name` | Compound name extracted from the PDF |
| `spectrum` | List of m/z values (as strings, in ascending x-order) |
| `relative_abundance` | Corresponding peak heights normalised to the tallest peak (0–100) |

---

## Manual Verification

The `Manual_Verif/` folder contains two tools for validating algorithm output against hand-measured values:

### CLI Tool (`ManualScripting.py`)
Loads a JSON output file, prompts for a filename, then asks for manually measured pixel heights and the maximum height `H`. Prints a side-by-side comparison of predicted vs actual relative abundance.

```bash
cd Manual_Verif/
python3 ManualScripting.py
# Enter the file name (e.g. 12038A.pdf):
# Enter the values for array A: 120 340 800 ...
# Enter the max height H: 800
```

### Web Tool (`index.html`)
An interactive browser-based verification interface. Open directly in any browser — no server required.

---

## Reports & Presentations

| File | Description |
|---|---|
| `GC_MS_Paper - FinalDraft.pdf` | Full research paper (IJDAR submission draft) |
| `ReportGCMS.pdf` | Project report |
| `Automated Spectrum Peak Extraction - PPT.pdf` | Algorithm overview presentation |
| `Parallel Processing - PPT.pdf` | Parallelism benchmarking presentation |

---

## Author

**Pramath K P**  
Indian Institute of Science (IISc), Bengaluru
