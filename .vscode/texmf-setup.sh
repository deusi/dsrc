#!/usr/bin/env bash
# One-time: populate a project-local texmf tree with the acmart font packages
# (libertine, newtx/newtxmath, inconsolata/zi4, mweights) that the container's
# minimal TeX Live 2022 is missing.
#
# Run INSIDE the container so mktexlsr/updmap-user match the container's TeX:
#   apptainer exec --overlay /scratch/ab9738/dsrc/dsrc_gpu_env.ext3:ro \
#     /scratch/ab9738/dsrc/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
#     bash /scratch/ab9738/dsrc/.vscode/texmf-setup.sh
#
# Sources come from the frozen TeX Live 2022 archive so the font versions match
# the container's acmart.cls exactly (the current-CTAN tds ships renamed/
# incomplete TFMs and breaks LinLibertineT-tlf-t1). The build recipe in
# settings.json must export the same TEXMFHOME/TEXMFVAR/TEXMFCONFIG *and*
# prepend this tree to TEXMFDBS, or kpathsea returns relative TFM paths.
set -euo pipefail
REPO=/scratch/ab9738/dsrc
export TEXMFHOME=$REPO/.vscode/texmf
export TEXMFVAR=$REPO/.vscode/texmf-var
export TEXMFCONFIG=$REPO/.vscode/texmf-config
rm -rf "$TEXMFHOME" "$TEXMFVAR" "$TEXMFCONFIG"
mkdir -p "$TEXMFHOME" "$TEXMFVAR" "$TEXMFCONFIG"

arch=https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2022/tlnet-final/archive
work=$(mktemp -d)
for pkg in libertine inconsolata newtx mweights; do
  echo ">> $pkg (TL2022 archive)"
  curl -fLsS "$arch/$pkg.tar.xz" -o "$work/$pkg.tar.xz"
  rm -rf "$work/x" && mkdir -p "$work/x"
  tar xf "$work/$pkg.tar.xz" -C "$work/x"
  # These archives extract tex/ fonts/ (and tlpkg/, doc/) with no RELOC prefix.
  for d in tex fonts; do [ -d "$work/x/$d" ] && cp -a "$work/x/$d" "$TEXMFHOME"/; done
done
rm -rf "$work"

# Filename database for the tree (build recipe adds it to TEXMFDBS so this is used).
mktexlsr "$TEXMFHOME" >/dev/null 2>&1 || true

# Enable the font maps the packages shipped, into the writable user var tree.
shopt -s nullglob
for m in "$TEXMFHOME"/fonts/map/*/*/*.map; do
  updmap-user --enable Map="$(basename "$m")" >/dev/null 2>&1 || true
done
updmap-user >/dev/null 2>&1 || true

echo "OK — trees under $REPO/.vscode/  (texmf, texmf-var, texmf-config)"
kpsewhich libertine.sty newtxmath.sty zi4.sty mweights.sty LinLibertineT-tlf-t1.tfm || true
