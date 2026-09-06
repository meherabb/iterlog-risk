# Anonymity checklist

This repository is released alongside a paper under double-blind review. This document
records what was checked before release and what **you** (the person uploading this) still
need to verify on GitHub's side, since some identity leaks happen at the hosting layer, not
in the file contents.

## What has already been done in these files

- [x] No author names, initials, or usernames anywhere in source, comments, configs, or docs.
- [x] No email addresses, personal URLs, or lab/university names.
- [x] No affiliation strings (department, university, funder, grant number).
- [x] `LICENSE` uses a generic copyright holder (`The Authors`), not a named entity.
- [x] `CITATION.cff` lists `Anonymous` as the author and leaves the DOI/URL fields blank.
- [x] No links to the authors' other repositories, personal websites, or social profiles.
- [x] No absolute file paths from a development machine (e.g. `/home/username/...`) baked
      into configs, notebooks, or scripts — everything is relative to the repo root.
- [x] Notebook outputs are cleared before commit (see the `nbstripout` note below) so that
      cell-execution metadata, kernel names, and any inline `print()` of a machine hostname
      or username don't leak through.
- [x] No `.git` history bundled in the released archive — a fresh `git init` is expected on
      your end so no old commits with real names are dragged along.

## What you must still do yourself

1. **Use a fresh, anonymous hosting path.** A direct link to `github.com/<your-username>/<repo>`
   deanonymizes you the moment a reviewer clicks it, even if the *files* are clean. The two
   standard options at ICLR:
   - **Anonymous GitHub** (<https://anonymous.4open.science>): point it at your real repo and
     it serves a scrubbed, read-only mirror at an anonymous URL. This is the most common
     choice and is what most ICLR authors use — it also automatically strips `git blame`
     and commit-author metadata, which a plain upload does not.
   - **A throwaway organization/account** created solely for this submission, with no other
     activity, no profile photo, no bio, and no starred/followed repos that could be
     fingerprinted.
2. **Check your commit history before pushing**, not just the file contents:
   ```bash
   git log --all --format='%an <%ae>' | sort -u
   ```
   If this prints anything other than a placeholder identity, either squash to a single
   commit authored by a placeholder, or push these files as a fresh `git init` with no
   inherited history.
3. **Set a placeholder git identity** for this repo specifically before your first commit:
   ```bash
   git config user.name "Anonymous"
   git config user.email "anonymous@example.com"
   ```
4. **Double-check `environment.yml` / `requirements.txt`** if you regenerate them locally
   (e.g. via `conda env export` or `pip freeze`) — both commands can leak local package
   paths or a machine-specific channel URL. Prefer the versions already pinned in this
   release over a freshly-exported one.
5. **Scrub notebook metadata**, not just outputs, if you add new notebooks:
   ```bash
   pip install nbstripout
   nbstripout --install          # auto-strips on every commit, repo-wide
   ```
6. **Search the whole tree once more before your first push**, as a final gate:
   ```bash
   grep -RiE "(your real name|your university|your @email domain)" .
   ```
7. If this paper cites or extends your own prior work, make sure that citation is phrased in
   the third person in every doc file (`README.md`, `docs/*.md`) — "concurrent with prior
   work by X" rather than "in our earlier paper," which is what the main text already does
   and what this repository's docs mirror.

## Re-identification after acceptance

Once the paper is accepted and the review period ends, replace the placeholders in
`CITATION.cff`, `LICENSE`, and this file's copyright lines with the real author list, and
update the `README.md` citation section with the camera-ready BibTeX entry.
