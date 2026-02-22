# wf_rotate_back

Extract CI coefficients from an SHCI `wf*.dat` file and back-transform them from a rotated orbital basis to the original basis using an orbital rotation matrix.

## Build

```bash
cd tools/wf_rotate_back
make
```

## Rotation matrix format

A whitespace-separated square matrix (`n_orb x n_orb`).

- Row index = old orbital index (0-based internally)
- Column index = new orbital index
- Matrix element = `U(old, new)`

This matches SHCI natural-orbital rotation usage where columns are new orbitals expanded in the old basis.

## Run

```bash
./wf_rotate_back <wf.dat> <rot_matrix.txt> [options]
```

Options:
- `--state <i>`: state index to extract (default `0`)
- `--top-new <n>`: only keep top-`n` determinants in new basis (by `|CI|`)
- `--ci-cut <x>`: discard new-basis determinants with `|CI| < x`
- `--amp-cut <x>`: discard tiny minor amplitudes during back-transform (default `1e-12`)
- `--max-comb <n>`: max allowed combinations per spin sector `C(norb, nelec)` (default `200000`)
- `--out-prefix <p>`: output prefix (default `ci_rotated`)

## Output

- `<prefix>_new_basis.tsv`: extracted CI entries from `wf.dat`
- `<prefix>_old_basis.tsv`: back-transformed CI entries in original basis

Both files store occupied orbitals (1-based) for alpha/beta and the CI coefficient.
