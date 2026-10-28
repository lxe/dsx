#!/bin/bash
# Rebuild webui/dist from llama.cpp's web UI (tools/ui, MIT) at a pinned commit. Needs git, node 20+, npm.
# dsx-server embeds webui/dist at compile time and serves it at /. logo.svg replaces llama.cpp's logo and the app is
# named dsx; the full build (npm run build) also generates the favicons and app icons from the logo.
set -euo pipefail
COMMIT=${1:-547208d}
here="$(cd "$(dirname "$0")" && pwd)"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git clone -q --filter=blob:none --sparse https://github.com/ggml-org/llama.cpp "$tmp/llama.cpp"
cd "$tmp/llama.cpp"
git checkout -q "$COMMIT"
git sparse-checkout set --no-cone /LICENSE '/tools/ui/*'
cd tools/ui
cp "$here/logo.svg" src/lib/assets/logo.svg
sed -i "s|'Local AI chat interface powered by llama.cpp'|'Local DeepSeek chat with dsx (web UI from llama.cpp)'|" src/lib/constants/pwa.constants.ts
grep -q "chat with dsx" src/lib/constants/pwa.constants.ts
sed -i "s/|| 'llama-ui'/|| 'dsx'/" src/lib/constants/app.constants.ts
grep -q "|| 'dsx'" src/lib/constants/app.constants.ts
npm ci --no-audit --no-fund
npm run build
rm -rf "$here/dist"
cp -r dist "$here/dist"
cp ../../LICENSE "$here/LICENSE-llama.cpp"
echo "webui/dist rebuilt from llama.cpp $COMMIT"
