#!/bin/bash
# install_and_run_syncthing_uv.sh
# Installs Syncthing + UV (Astral) and auto-starts Syncthing

set -e

echo "=== Setting up directories ==="
mkdir -p "$HOME/.local/bin"
mkdir -p "$HOME/.local/share/syncthing"

# Ensure ~/.local/bin is in PATH for this session
export PATH="$HOME/.local/bin:$PATH"

# ----------------------------
# Detect architecture
# ----------------------------
ARCH=$(uname -m)
case "$ARCH" in
    x86_64) ARCH="amd64" ;;
    aarch64) ARCH="arm64" ;;
    armv7l) ARCH="armv7" ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

echo "=== Installing Syncthing ($ARCH) ==="

# Get latest version
SYNCTHING_VERSION=$(curl -s https://api.github.com/repos/syncthing/syncthing/releases/latest \
    | grep '"tag_name":' | sed -E 's/.*"v([^"]+)".*/\1/')

echo "Downloading Syncthing v$SYNCTHING_VERSION..."

curl -L "https://github.com/syncthing/syncthing/releases/download/v$SYNCTHING_VERSION/syncthing-linux-$ARCH-v$SYNCTHING_VERSION.tar.gz" \
    -o /tmp/syncthing.tar.gz

# Extract and install
tar -xzf /tmp/syncthing.tar.gz -C /tmp
mv "/tmp/syncthing-linux-$ARCH-v$SYNCTHING_VERSION/syncthing" "$HOME/.local/bin/"
chmod +x "$HOME/.local/bin/syncthing"
rm -rf /tmp/syncthing*

echo "Syncthing installed."

# ----------------------------
# Install UV (Astral)
# ----------------------------
echo "=== Installing UV (Astral) ==="
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload PATH in case uv installer modified shell config
export PATH="$HOME/.local/bin:$PATH"

# ----------------------------
# Start Syncthing
# ----------------------------
echo "=== Starting Syncthing ==="

# Run Syncthing in background, log output
nohup syncthing > "$HOME/.local/share/syncthing/syncthing.log" 2>&1 &

sleep 2

echo "=== Done ==="
echo "Syncthing is now running in the background."
echo "Web UI: http://127.0.0.1:8384"
echo "Logs: $HOME/.local/share/syncthing/syncthing.log"
echo ""
echo "To stop Syncthing:"
echo "pkill syncthing"