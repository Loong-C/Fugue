# Remote Sync Notes

Updated: 2026-05-25.

The local `main` branch and GitHub `origin/main` are synchronized. This note records
the recovery steps for the ownership/authentication issue that appeared while publishing
the implementation work.

## Current State

- Remote: `https://github.com/Loong-C/Fugue.git`
- Local branch: `main`
- Remote branch: `origin/main`
- Local status at the time of writing: `main` and `origin/main` both point to the
  implementation branch tip.
- The generated corpora, style cache, and MIDI verification outputs are intentionally
  ignored by git through `.gitignore`.

## Future Push Command

For future pushes from a shell that sees the repository as a different owner, use:

```powershell
git -c safe.directory=F:/Personal/Code/Fugue push origin main
```

If Git reports dubious ownership again, keep the `safe.directory` option above or add
the repository once to the user's global Git config:

```powershell
git config --global --add safe.directory F:/Personal/Code/Fugue
```

## Authentication Blocker Resolved

The earlier push failure was caused by local authentication and ownership checks, not
by repository content:

- The Codex sandbox process could not access the user's GitHub keyring token.
- Plain `git push origin main` reaches GitHub, receives a 401 challenge, then waits for
  Git Credential Manager.
- `GIT_TERMINAL_PROMPT=0` with credential helpers disabled fails with
  `could not read Username for 'https://github.com'`.
- Device login with the global `socks5://127.0.0.1:7890` proxy fails because Git
  Credential Manager's .NET HTTP stack does not support the `socks5` proxy scheme.
- Device login with the proxy disabled reaches the local network policy and cannot open
  a direct socket to GitHub.
- The user's interactive PowerShell session did have a valid `gh` login; pushing with
  `git -c safe.directory=F:/Personal/Code/Fugue push origin main` resolved the remote
  sync once the dubious ownership check was bypassed.

## Recovery Options

Use one of these paths if a future checkout hits the same issue:

1. Install GitHub CLI and authenticate:

   ```powershell
   winget install GitHub.cli
   gh auth login
   git -c safe.directory=F:/Personal/Code/Fugue push origin main
   ```

2. Keep using Git Credential Manager, but set a proxy scheme it can use, then log in:

   ```powershell
   git config --global --unset-all http.proxy
   git config --global --unset-all https.proxy
   git config --global http.proxy http://127.0.0.1:7890
   git config --global https.proxy http://127.0.0.1:7890
   git credential-manager github login --device --url https://github.com --username Loong-C
   git -c safe.directory=F:/Personal/Code/Fugue push origin main
   ```

   This only works if the local proxy port accepts HTTP CONNECT traffic. If it is
   SOCKS-only, configure the proxy client to expose an HTTP proxy port first.

3. Use a personal access token through Git Credential Manager:

   ```powershell
   git credential-manager github login --pat --url https://github.com --username Loong-C
   git -c safe.directory=F:/Personal/Code/Fugue push origin main
   ```

## Verification Before Push

The current implementation should be checked before pushing if more edits are made:

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python scripts\verify_generator.py --variants 16
```
