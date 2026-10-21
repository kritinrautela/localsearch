#!/bin/zsh
# One-time setup on a new Mac: Homebrew tools, Python packages, the EmbeddingGemma 2 model, config.toml, the login
# agent and LocalSearch.app. Safe to run again: finished steps are skipped.
set -e
cd ~/localsearch 2>/dev/null || { echo "This folder must be at ~/localsearch (your home folder > localsearch)."; exit 1; }

if ! command -v brew >/dev/null; then
    echo "Homebrew is missing. Install it from https://brew.sh, then run this again."
    exit 1
fi
for pkg in llama.cpp poppler uv; do
    brew list --versions $pkg >/dev/null || brew install $pkg
done

echo "Python packages..."
uv sync

# The model: ggml-org's EmbeddingGemma 2 GGUF on Hugging Face (about 865 MB for both files), checked by SHA-256.
mkdir -p models logs state
URL=https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/resolve/main
fetch() {
    local name=$1 sum=$2
    if [[ -f models/$name ]] && [[ "$(shasum -a 256 models/$name | cut -d' ' -f1)" == $sum ]]; then
        return
    fi
    echo "Downloading $name..."
    curl -L --fail --progress-bar -C - -o models/$name.part $URL/$name
    if [[ "$(shasum -a 256 models/$name.part | cut -d' ' -f1)" != $sum ]]; then
        echo "$name did not download correctly. Run this script again."
        mv models/$name.part ~/.Trash/ 2>/dev/null
        exit 1
    fi
    mv models/$name.part models/$name
}
fetch embeddinggemma-2-Q8_0.gguf 2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135
fetch mmproj-embeddinggemma-2-Q8_0.gguf c4a8a52691ecef40618438928bdf9e68379b854e24166f292592353db0aab64f

[[ -f config.toml ]] || cp config.example.toml config.toml

mkdir -p ~/Applications
ditto macos/LocalSearch.app ~/Applications/LocalSearch.app
tools/install_agent.sh

echo
echo "Done. Press Control-Option-Space to search. The first index builds in the background (menu bar icon shows"
echo "progress); allow the 'Python would like to access...' prompts so it can read your folders."
