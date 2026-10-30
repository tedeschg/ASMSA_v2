#!/usr/bin/env bash
set -e

echo "ASMSA v2 container"
echo "  python  : $(python --version 2>&1)"
echo "  torch   : $(python -c 'import torch; print(torch.__version__, "cuda:", torch.cuda.is_available())')"
echo "  gromacs : $(gmx --version 2>/dev/null | head -n1 || echo 'not found')"
echo

exec "$@"
