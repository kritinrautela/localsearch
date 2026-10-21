#!/bin/zsh
# Start the search window (app.py) at login and keep it running. Undo: tools/install_agent.sh --remove
LABEL=com.kritin.localsearch
PLIST=~/Library/LaunchAgents/$LABEL.plist
if [[ "$1" == "--remove" ]]; then
    launchctl bootout gui/$(id -u)/$LABEL 2>/dev/null
    mv "$PLIST" ~/.Trash/ 2>/dev/null
    echo "removed $LABEL"
    exit 0
fi
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array><string>$HOME/localsearch/.venv/bin/python</string><string>$HOME/localsearch/app.py</string></array>
    <key>WorkingDirectory</key><string>$HOME/localsearch</string>
    <key>RunAtLoad</key><true/>
    <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
    <key>ProcessType</key><string>Interactive</string>
    <key>StandardOutPath</key><string>$HOME/localsearch/logs/app.log</string>
    <key>StandardErrorPath</key><string>$HOME/localsearch/logs/app.log</string>
</dict>
</plist>
PL
launchctl bootout gui/$(id -u)/$LABEL 2>/dev/null && sleep 2
launchctl bootstrap gui/$(id -u) "$PLIST" && echo "installed $LABEL: running now and at every login"
