# LocalSearch

Spotlight that understands what your files are *about*. Press a shortcut, type something like
"the rental agreement for the flat" or "photos at the beach", and it finds the document, photo or video, even
when you can't remember the file name.

Everything runs on your Mac. Your files, the search index and the AI model never leave it: the app blocks
itself from the internet. The only exception is the "Search the web" row, which opens your browser if you pick it.

Made by Kritin. Works on any Apple Silicon Mac (M1 to M4) with macOS 14 or newer. It looks best on macOS 26
Tahoe.

---

## What it does

- **Documents:** PDFs, Word, PowerPoint, Excel, CSV, web pages, text and Markdown notes, searched by meaning *and* by exact words or
  codes like an invoice number. Scanned PDFs and screenshots are read with Apple's text recognition.
- **Photos and videos:** describe the picture ("dog on the sofa", "birthday cake") and it finds it. A video opens
  at the second that matched.
- **Apps, calculator, settings:** type an app name (even with letters missing: `wtsp` finds WhatsApp), a sum
  like `12% of 340`, or a setting like `bluetooth`.
- **Clipboard history**, a preview of every result, and actions like Show in Finder or Copy.
- **Code search** with `code:` if you write code.

---

## Install (about 15 minutes, once)

You type a few commands into **Terminal** (press ⌘Space, type `Terminal`, press Return). Paste each command and
press Return.

**1. Install Homebrew** (the tool that installs the other tools). Skip this if you already have it.

```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

It asks for your Mac password (nothing shows while you type, that's normal). When it finishes, it prints two
"Next steps" commands starting with `echo` and `eval`. Run those too.

**2. Download LocalSearch.** The project is private, so Kritin has to add you on GitHub first
(github.com/kritinrautela/localsearch > Settings > Collaborators). Accept the invite email, then:

```sh
brew install gh
gh auth login
gh repo clone kritinrautela/localsearch ~/localsearch
```

For `gh auth login`, choose GitHub.com, HTTPS, and "Login with a web browser".

It must live at `~/localsearch` (a folder called `localsearch` in your home folder).

**3. Run the installer:**

```sh
~/localsearch/tools/install.sh
```

This installs the helper tools, downloads the AI model (about 865 MB, from Hugging Face, checked so a broken
download can't be used), and starts LocalSearch. It also starts it every time you log in.

**4. Allow access.** macOS asks "Python would like to access files in your Documents folder" (and Desktop,
Photos, etc.). Click **Allow** each time, or LocalSearch can't see those files.

**5. Plug in your Mac and let the first index run.** The magnifying-glass icon in the menu bar is LocalSearch. The
first index runs in the background, and you can keep using your Mac. Exact-word search works within minutes.
Searching by meaning, and photos and videos, need the Mac to be **plugged in** (the AI part pauses on battery), and
take an hour or more with a big photo library. Leaving it charging overnight is easiest.

---

## Use ⌘Space for it (recommended)

Until you change this, LocalSearch opens with **Control-Option-Space** (⌃⌥Space), because Spotlight owns ⌘Space.

To give ⌘Space to LocalSearch:

1. Open **System Settings > Keyboard > Keyboard Shortcuts…**, then **Spotlight**.
2. Untick **Show Spotlight search**. On macOS 26 Tahoe, also untick **Show Apps** if it is set to ⌘Space.
3. Click Done.

LocalSearch notices within 10 seconds and takes ⌘Space by itself. To undo, tick them again.

---

## How to search

| You want | Type |
| --- | --- |
| A document by what's in it | `bond refund letter from the landlord` |
| An exact word, code or number | `INV-2041` |
| Photos or videos | `photos beach sunset` or `video birthday` |
| Only PDFs | `kind:pdf tax return` |
| Recent files | `date:week notes` (also `date:today`, `date:month`, `date:2025`) |
| By file name only | `name: cv` |
| Things you copied | `clip: address` |
| Code | `code: playlist shuffle` |

**Keys:** ↑↓ to move, **Return** opens, **⌘Return** shows in Finder, **Space** previews (Quick Look),
**⌘C** copies the file, **⌘K** shows every action (Open With, Find Similar, Copy Path…), **Tab** switches
between All / Documents / Photos & Video / Apps / Clipboard, **Esc** closes. You can drag a result straight into
Mail, WhatsApp or a folder.

---

## Choose which folders it searches

Click the menu bar icon > **Settings…**. You can add or remove:

- folders for documents (starts with Documents and Desktop),
- folders for photos, video and music (starts with Movies, Pictures, your Photos library and Music),
- folders it must **never** look in, and file types to skip (for example `csv`).

Save, and it updates the index straight away. It then keeps itself up to date: changed files are picked up within
a minute, plus a full check once a night while the Mac is charging.

---

## Menu bar icon

- **Search**: opens the window.
- **Update index now**: re-checks every folder.
- **Pause indexing**: if the Mac feels busy.
- **Settings…**: folders, file types, which code editor to use.
- **Quit localsearch**: stops it until the next login. To start it again, open **LocalSearch** from
  Applications in your home folder (or press ⌘Space and type `localsearch`).

---

## Updating

When Kritin makes it better:

```sh
cd ~/localsearch && git pull && uv sync && launchctl kickstart -k gui/$(id -u)/com.kritin.localsearch
```

Your settings (`config.toml`) and your index are not touched by updates.

---

## If something is wrong

- **The shortcut does nothing:** check the menu bar icon is there. If not, open **LocalSearch** from
  `~/Applications`. Still nothing? Run `~/localsearch/tools/install.sh` again (it skips what's already done).
- **A folder's files never show up:** System Settings > Privacy & Security > Files and Folders (or Full Disk Access)
  and allow **python3.14** or **Python**, then use **Update index now**.
- **Photos aren't found, or only exact words work:** the AI part only runs while the Mac is plugged in. Leave it
  charging for a while.
- **Two windows open on ⌘Space:** Spotlight's or Tahoe's Show Apps shortcut is still on ⌘Space (see above).
- Logs are in `~/localsearch/logs/` if you need to send Kritin something. Only send the log files, nothing else.

## Uninstall

```sh
~/localsearch/tools/install_agent.sh --remove
```

Then drag `~/localsearch` and `~/Applications/LocalSearch.app` to the Bin. If you changed the Spotlight
shortcut, tick it again in System Settings.

---

<details>
<summary>For the curious: how it works</summary>

- `indexer.py` reads your folders into `index.sqlite`: text pieces with full-text search, plus EmbeddingGemma 2
  "meaning" vectors made on your Mac by `llama-server` (llama.cpp). Photos and video frames get vectors too.
- `search.py` mixes meaning and exact-word scores, and groups copies of a file (`report v2`, `report final`) into
  one result.
- `app.py` is the window (native macOS, PyObjC), `watch.py` keeps the index fresh, `codesearch.py` handles `code:`.
- `common.py` blocks every network connection except to the Mac itself (127.0.0.1).
- Settings live in `config.toml`, created from `config.example.toml` on install.

</details>
