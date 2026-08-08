# Releasing Prism

Prism publishes two independent release families from this one repository:

- **Model artifacts** — tag `prism-<lang>-<semver>` (e.g. `prism-no-0.2.4`,
  `prism-en-0.1.0`). A GitHub release carrying the packaged model tarballs.
  The library code is not involved (§1).
- **Library / bindings** — tag `v<semver>` (e.g. `v0.6.0`). Versions the
  Swift/C++/C/Java code. Produced by the **`release`** workflow (§2), which
  builds the binaries and creates the tag + release in one run.

Optional, and **not** part of any workflow: publishing the Java JAR to Maven
Central (`mvn -Prelease deploy`, see §5). Skip it unless you explicitly
intend a Maven Central release — the free publishing quota resets monthly.

---

## 1. Model release (`prism-<lang>-<semver>`)

1. Build and package the artifact(s) into `models/` — each precision as a
   `.tar.gz`, plus a `SHA256SUMS` file and (for convenience) a top-level
   copy of each `manifest.json`.
2. Verify integrity: `shasum -a 256 -c models/prism-<lang>-<ver>.SHA256SUMS`.
3. Create the release (`prism-*` tags trigger no CI):

   ```bash
   gh release create prism-<lang>-<ver> \
     --title "prism-<lang> <ver> — …" \
     --notes-file models/prism-<lang>-<ver>-release-notes.md \
     --latest=false --target main \
     models/prism-<lang>-<ver>.tar.gz \
     models/prism-<lang>-<ver>-fast.tar.gz \
     models/prism-<lang>-<ver>.SHA256SUMS \
     models/prism-<lang>-<ver>-manifest.json \
     models/prism-<lang>-<ver>-fast-manifest.json
   ```

   `--latest=false` keeps the "Latest" badge on the newest library release.
   Omit the `-fast` assets for single-precision models.
4. Mirror to Hugging Face — see §4.

> **Ordering constraint.** The library `release` workflow's Apple consumer
> test downloads a released model (the `consumer_model` input, default
> `prism-no-0.2.4`). That model release **must exist before** you run the
> library release for a version that depends on it.

---

## 2. Library release (`v<semver>`) — the `release` workflow

One workflow does the whole thing: it builds the Apple XCFramework and the
Java JAR **once**, pins the freshly-built XCFramework's SwiftPM checksum into
`Package.swift`, commits that, and creates the tag + GitHub release attaching
**the exact bytes it built**. Because nothing is rebuilt after the checksum
is taken, the byte-drift that used to break SwiftPM consumers cannot happen
(that was the whole hazard of the old manual flow — see §3).

### Prepare

Put the release commit on `main` with **every version string bumped** to the
target version — `java/pom.xml`, README, `docs/INTEGRATION.md`, the
`examples/`. Do **not** touch `Package.swift`'s `prismNativeReleaseURL` /
`prismNativeReleaseChecksum`; the workflow owns those two lines. Optionally
write release notes to `docs/release-notes/v<ver>.md` (else the release gets
auto-generated notes).

### Run

Actions → **release** → *Run workflow*, with:

- `version` — e.g. `0.6.1` (no leading `v`).
- `consumer_model` — usually the default; set it if this version needs a
  newer model for the Apple consumer test.
- `latest` — whether to mark it the repo's Latest release.
- `dry_run` — **run once with this checked first.** It builds and verifies
  the checksum and prints a summary, but commits/tags/publishes nothing.

The workflow then (when `dry_run` is off):

1. builds the five Apple slices → `PrismNative.xcframework.zip` (+ runs the
   third-party consumer test) and the four-platform JAR, in parallel;
2. checks preconditions: `main` hasn't moved, `pom.xml` version == `version`,
   tag `v<version>` doesn't exist;
3. verifies the built zip's `sha256` equals its recorded checksum and the
   build job's output, then pins that URL + checksum into `Package.swift`;
4. commits `chore: pin PrismNative URL + checksum for v<version>`, pushes it
   to `main`, and creates the tag + release attaching the XCFramework zip,
   its `.checksum`, and the JAR.

### Verify

Consistency is guaranteed by construction (the attached bytes are the ones
whose checksum was pinned), but confirm the published release:

```bash
gh release download vX.Y.Z --repo dmlux/Prism \
  --pattern "PrismNative.xcframework.zip" --dir /tmp/verify
diff <(sha256sum /tmp/verify/PrismNative.xcframework.zip | cut -d' ' -f1) \
     <(git show vX.Y.Z:Package.swift | grep -A1 prismNativeReleaseChecksum | grep -oE '[0-9a-f]{64}')
```

Then build a quickstart under `examples/` against the tag as a smoke test.

> **Notes.** The `release` job commits to `main` with the default
> `GITHUB_TOKEN`; if `main` is branch-protected against direct pushes, grant
> the token an exception or use the manual fallback. A `GITHUB_TOKEN` commit
> does not re-trigger workflows, so the pin commit starts no new run. The
> reusable `prism-native` / `java-natives` workflows can still be run
> standalone via *Run workflow* for ad-hoc builds; they no longer run on tag
> pushes and no longer attach anything — only `release` does.

---

## 3. Library release — manual fallback

If the `release` workflow can't be used (e.g. `main` is protected and the bot
can't push), the original manual flow still works. Its one hazard is that the
old design **rebuilt** the XCFramework on the tag, and ExecuTorch's build is
not reliably byte-identical (it broke `v0.4.0`/`v0.4.1`, and `v0.6.0` drifted
`f6f86519…` vs `8733f897…`), so the tagged `Package.swift` checksum could end
up not matching the attached zip.

1. **Dispatch** `prism-native` and `java-natives` on `main` (Actions → *Run
   workflow*). Wait for **both green** before continuing — never tag while a
   dispatch runs.
2. **Pin.** Download the `PrismNative.xcframework` artifact from the
   `prism-native` run, read `PrismNative.xcframework.zip.checksum`, and set
   `prismNativeReleaseURL` (→ the `vX.Y.Z` URL) and `prismNativeReleaseChecksum`
   in `Package.swift`. Commit.
3. **Tag atomically.** `gh release create vX.Y.Z --target main --title … --latest`
   — creates the tag *and* the release in one call, so the release exists
   before any CI attach step. (These workflows no longer attach on tag push,
   so upload the artifacts yourself: the JAR from `java-natives` and the
   XCFramework zip + checksum from the `prism-native` run.)
4. **Verify + reconcile.** Download the attached zip, `sha256sum` it, and
   confirm it equals the `Package.swift` pin. If it drifted, re-attach the
   **dispatch** artifact you pinned (it is proven good) with `--clobber` —
   do **not** move the tag or re-pin to the drifted bytes:

   ```bash
   gh release upload vX.Y.Z --repo dmlux/Prism \
     /path/to/dispatch/PrismNative.xcframework.zip \
     /path/to/dispatch/PrismNative.xcframework.zip.checksum --clobber
   ```

---

## 4. Hugging Face mirror

Model artifacts are mirrored to `dmlux/prism-<lang>` (e.g.
[`dmlux/prism-no`](https://huggingface.co/dmlux/prism-no),
[`dmlux/prism-en`](https://huggingface.co/dmlux/prism-en)):

```bash
# new language: hf repo create prism-<lang> --repo-type model
hf upload dmlux/prism-<lang> models/prism-<lang>-<ver> prism-<lang>-<ver>
hf upload dmlux/prism-<lang> models/prism-<lang>-<ver>-fast prism-<lang>-<ver>-fast
hf upload dmlux/prism-<lang> <card.md> README.md
```

Then confirm no `.DS_Store` slipped in (`HfApi().list_repo_files(...)`) and
that the card's version references and folder table point at the new version.
Older byte-identical versions may be pruned to save space, or kept for pinned
consumers.

---

## 5. Maven Central (optional, Java only)

The Java binding is also published to Maven Central as
[`io.github.dmlux:prism`](https://central.sonatype.com/artifact/io.github.dmlux/prism)
(since 0.2.0). A **separate manual step**, not triggered by any workflow, and
the free publishing quota resets monthly — only do it when you actually
intend a Maven release for the version.

Do it **after** the `vX.Y.Z` GitHub release exists, because the JAR published
to Central must embed the same per-platform natives the CI built.

### One-time setup on the release machine

- **Credentials.** A Central user token in `~/.m2/settings.xml` under
  `<server><id>central</id>…`. Generate at central.sonatype.com → *View
  Account → Generate User Token*; namespace `io.github.dmlux` is verified via
  GitHub login.
- **GPG.** Central requires signed artifacts. Key setup and this machine's
  quirks (`disable-ipv6` in `~/.gnupg/dirmngr.conf`, `export GPG_TTY=$(tty)`
  for pinentry) are in the publishing runbook. Pre-cache the passphrase:
  `echo test | gpg --clearsign -o /dev/null`.
- **Maven + JDK.** No system Maven; use a portable Apache Maven with
  `JAVA_HOME=$HOME/.sdkman/candidates/java/current` (Temurin 21).

### Deploy

1. **Embed the release natives** (the `release` build reads
   `java/src/main/resources/`, gitignored):

   ```bash
   gh release download vX.Y.Z --repo dmlux/Prism --pattern "*-all-platforms.jar" --dir /tmp
   unzip -o /tmp/prism-*-all-platforms.jar 'io/github/dmlux/prism/native/*' \
     -d java/src/main/resources/
   ```

2. **Dry run** (build + sign, no upload); confirm four natives:

   ```bash
   mvn -q -Prelease -DskipTests -Dgpg.skip=true -f java/pom.xml clean package
   unzip -l java/target/prism-*.jar | grep -c 'native/.*\(so\|dylib\)'   # expect 4
   ```

3. **Deploy** (signs and uploads; waits for Central to validate):

   ```bash
   mvn -Prelease -DskipTests -f java/pom.xml clean deploy
   ```

4. **Publish.** central.sonatype.com → *Publishing → Deployments* → **Publish**
   (deliberately manual — Central never deletes). Resolves on
   `repo.maven.apache.org` within ~15–60 min.

> **Pitfall.** A `deploy` whose output was piped through `head`/truncated can
> complete *invisibly* and still upload, leaving a duplicate *Validated*
> deployment. Harmless — **Drop** it.

---

## Quick checklist

**Model `prism-<lang>-<ver>`:** package → verify SHA → `gh release create`
(`--latest=false`) → HF mirror (§4).

**Library `vX.Y.Z`:**

- [ ] Model the consumer test downloads is released (§1 ordering constraint).
- [ ] Version bumps committed on `main` (pom, README, INTEGRATION, examples);
      `Package.swift` PrismNative pin left untouched.
- [ ] Optional: release notes at `docs/release-notes/vX.Y.Z.md`.
- [ ] Run the **release** workflow with `dry_run` checked; review the summary.
- [ ] Run it again with `dry_run` off.
- [ ] Verify the published XCFramework `sha256` == the tagged `Package.swift`
      pin, and build an `examples/` quickstart against the tag.
- [ ] (Optional, separate) Maven Central per §5.
