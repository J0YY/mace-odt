# MACE-ODT research specification bundle

Read `MACE_ODT_Research_Specification.md` for the standalone specification.
It uses standard Markdown with `$...$` and `$$...$$` LaTeX math.

Contents:
- `MACE_ODT_Research_Specification.md`: research design, mathematics, source mapping, experiments, and limitations.
- `RESEARCH_LOG.md`: dated decisions, hypotheses, and claim boundaries.
- `EXPERIMENTS.md`: executable experiment map with gates and required outputs.
- `PRELIMINARY_RESULTS.md`: real-checkpoint results and their claim boundaries.
- `ALGORITHM_ALIGNMENT.md`: exact mapping from Dooms Algorithms 1 through 3 to
  the tied mixed-order MACE implementation.
- `src/mace_odt`: the real-checkpoint audit and experiment implementation.
- `cluster`: Athena setup and Slurm entry points.
- `verify_mace_odt_math.py`: deterministic finite-dimensional mathematical checks using Python 3.10+ and NumPy.
- `mace_odt_math_checks.json`: actual report from the executed checks.
- `source_manifest.json`: source and output SHA-256 identities and numerical environment.

Run checks with:

```bash
python verify_mace_odt_math.py --output fresh_checks.json
```

Run without Python's `-O` flag: the tests use assertions.
The original 16 synthetic checks pass. The package now also contains the
real-checkpoint compiler, environment construction, native functional maps,
and their unit tests. Released checkpoint and dataset files remain external
and are identified by repository records and hashes.

Start the real-checkpoint work locally or on Athena with:

```bash
python -m pip install -e .
python -m mace_odt.cli.environment_audit --output results/environment.json
python -m mace_odt.cli.checkpoint_audit --model small --output results/checkpoint_small.json
python -m mace_odt.cli.path_quotient_audit --model small --output results/path_quotient_small.json
```

Athena entry points for the completed exactness and coefficient experiments are
under `cluster`. Run them in numerical order. The frozen held-out fidelity job
uses the official MACE-OFF23 test archive and a checked-in metadata manifest.

The checkpoint audit downloads the official MACE-OFF23 checkpoint through the
official MACE loader. The model has a separate academic software license.

The source manuscripts are not duplicated in the bundle. Their filenames,
anchors, publication identities, and hashes are documented for verification.
