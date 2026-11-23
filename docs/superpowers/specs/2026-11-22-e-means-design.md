# Optional E-means hierarchy clustering

The user approved the preceding investigation and requested implementation with Superpowers.

## Design
Add `clustering_backend` choices `kmeans` (default), `e_means`, and `spherical_e_means`, plus positive finite `e_means_epochs` (default 5). Persist settings in hierarchy configuration; old artifacts inherit defaults and changing settings requires a rebuild. Apply the selected backend to initial construction, local maintenance, and fanout balancing. Use the CPU NumPy Faiss implementation with FP32 inputs, sorted IDs and the configured seed. Use final nearest-centroid assignments, then retain our normalized member means and deterministic maximum-size splitting. Detect fewer occupied labels explicitly. Load Faiss only for selected E-means operations and explain missing/old installations.

## Approaches
The approved recommendation was an optional backend rather than replacing the default. Euclidean E-means isolates optimizer differences; spherical E-means also changes the objective to align with cosine-based maintenance. Keep both selectable. GPU execution and automatic crossover thresholds are deferred until benchmarks establish their value.

## Validation
Test configuration, CLI persistence, missing dependencies, duplicate vectors, size bounds, seed repeatability, build/update/level propagation and old artifacts. Run actual Faiss E-means tests in a compatible isolated environment if available, and the existing hierarchy/description/retrieval regressions. Do not claim corpus-quality or speed improvements without real-corpus measurements.
