#!/usr/bin/env bash
# install_torchcodec_isolated.sh
# Install torchcodec into <sam3>/.torchcodec_env/ with --no-deps so the
# container's torch is never touched. Then verify it imports.
#
# Usage:
#   bash install_torchcodec_isolated.sh                 # auto-pick version
#   TC_VER="0.3.*"  bash install_torchcodec_isolated.sh # override
#   TC_FFMPEG=1     bash install_torchcodec_isolated.sh # use bundled FFmpeg
#
# Rollback:
#   rm -rf <sam3>/.torchcodec_env

set -euo pipefail

SAM3_DIR="$(cd "$(dirname "$0")" && pwd)"
TC_DIR="${SAM3_DIR}/.torchcodec_env"

# Auto-pick torchcodec version from torch major.minor
if [ -z "${TC_VER:-}" ]; then
    PY_MM=$(python3 -c "import torch; print('.'.join(torch.__version__.split('.')[:2]))")
    case "$PY_MM" in
        2.4) TC_VER="0.1.*";;
        2.5) TC_VER="0.2.*";;
        2.6) TC_VER="0.3.*";;
        2.7) TC_VER="0.4.*";;
        *)   TC_VER="";;   # let pip pick latest (--no-deps still protects torch)
    esac
fi

if [ "${TC_FFMPEG:-0}" = "1" ]; then
    PKG="torchcodec[ffmpeg]"
else
    PKG="torchcodec"
fi
if [ -n "$TC_VER" ]; then PKG="${PKG}==${TC_VER}"; fi

echo "torch:      $(python3 -c 'import torch; print(torch.__version__)')"
echo "installing: $PKG"
echo "target:     $TC_DIR"
echo

mkdir -p "$TC_DIR"
python3 -m pip install --target="$TC_DIR" --no-deps "$PKG"

echo
echo "─── import smoke test ───"
PYTHONPATH="$TC_DIR:${PYTHONPATH:-}" python3 -c "
import torchcodec
from torchcodec.decoders import VideoDecoder
print(f'OK — torchcodec {torchcodec.__version__}')
print(f'     path: {torchcodec.__file__}')
"

echo
echo "─── usage ───"
echo "export PYTHONPATH=$TC_DIR:\$PYTHONPATH"
echo "python3 benchmark.py --skip_encoder"
