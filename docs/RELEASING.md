# Releasing Prism

Prism publishes two independent release families from this one repository:

- **Model artifacts** — tag `prism-<lang>-<semver>` (e.g. `prism-no-0.2.4`,
  `prism-en-0.1.0`). A GitHub release carrying the packaged model tarballs.
  The library code is not involved.
- **Library / bindings** — tag `v<semver>` (e.g. `v0.6.0`). Versions the
  Swift/C++/C/Java code. Two CI workflows fire on `v*` tags and attach the
  binary deliverables to the release.

The model releases are simple. The library release has one genuinely tricky
step — pinning the SwiftPM binary-target checksum — described in full below.

Optional, and **not** part of any workflow: publishing the Java JAR to Maven
Central (`mvn -Prelease deploy`, see §5). Skip it unless you explicitly
intend a Maven Central release — the free publishing quota resets monthly.

---

## 1. Model release (`prism-<lang>-<semver>`)

1. Build and package the artifact(s) into `models/` — each precision as a
   `.tar.gz`, plus a `SHA256SUMS` file and (for convenience) a top-level
   copy of each `manifest.json`.
2. Verify integrity: `shasum -a 256 -c models/prism-<lang>-<ver>.SHA256SUMS`.
3. Create the release (does not touch the library; `prism-*` tags trigger no
   CI):

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

   Use `--latest=false` so a model release never steals the "Latest" badge
   from the newest library release. Omit the `-fast` assets for models that
   ship a single precision.
4. Mirror to Hugging Face — see §4.

> **Ordering constraint.** The library CI's consumer test downloads a
> specific released model (currently `prism-no-0.2.4-fast`, hard-coded in
> `.github/workflows/prism-native.yml`). That model release **must exist
> before** you dispatch or tag the matching library version, or the CI fails.

---

## 2. Library release (`v<semver>`) — overview

The library ships one binary deliverable per platform, attached to the
GitHub release by CI:

- **`prism-<ver>-all-platforms.jar`** — `java-natives.yml`. Self-contained
  JAR with embedded natives for macOS and Linux.
- **`PrismNative.xcframework.zip`** (+ `.checksum`) — `prism-native.yml`.
  The Apple SwiftPM binary target for the stable C ABI.

Both workflows trigger on `workflow_dispatch` and on `v*` tag pushes.

The complication: SwiftPM pins a binary target by **URL + checksum**, and
those two values live in the tagged `Package.swift`. So `Package.swift` at
tag `v<ver>` must already contain the checksum of the `PrismNative.xcframework.zip`
that ends up attached to the `v<ver>` release. That is a chicken-and-egg the
release flow resolves by **building once on `main` first**, committing that
checksum, and only then tagging.

---

## 3. Library release — step by step

Assume the target version is `X.Y.Z` and the release commit (version bumps
across the tree, docs, examples) is already on `main`, with `Package.swift`
still pointing `prismNativeReleaseURL` / `prismNativeReleaseChecksum` at the
*previous* release. Confirm the model that CI downloads (§1 ordering
constraint) is already released.

### Step 1 — Dispatch both workflows on `main`

```bash
gh workflow run prism-native.yml --repo dmlux/Prism --ref main
gh workflow run java-natives.yml --repo dmlux/Prism --ref main
```

`prism-native` builds all five slices, assembles `PrismNative.xcframework`,
runs the third-party consumer test against the downloaded fast model, and
uploads `PrismNative.xcframework.zip` + `.checksum` as **workflow
artifacts** (a `main` dispatch does not attach to any release). It also warms
the compiler cache in `main` scope, which the later tag run restores. Wait
for both to go green (the macOS slice build is ~1–2 h).

### Step 2 — Pin the checksum into `Package.swift`

Download the checksum the dispatch run produced and edit `Package.swift`:

```bash
gh run download <prism-native-run-id> --repo dmlux/Prism \
  --name PrismNative.xcframework --dir /tmp/xcf
cat /tmp/xcf/PrismNative.xcframework.zip.checksum
```

- `prismNativeReleaseURL` →
  `https://github.com/dmlux/Prism/releases/download/vX.Y.Z/PrismNative.xcframework.zip`
- `prismNativeReleaseChecksum` → the value just printed

Commit it: `chore: pin PrismNative URL + checksum for vX.Y.Z`.

### Step 3 — Create the release + tag

```bash
gh release create vX.Y.Z --target <commit-from-step-2> \
  --title "Prism library X.Y.Z" --notes-file <notes> --latest
```

`gh release create` with a new tag creates the tag and pushes it, which
**re-triggers both workflows on the tag**. The tag runs rebuild (cache
restored from the `main` dispatch → normally byte-identical) and, because the
ref is now `refs/tags/v*`, attach the JAR, `PrismNative.xcframework.zip`, and
its `.checksum` to the `vX.Y.Z` release.

### Step 4 — Verify the pin matches the attached bytes (critical)

The tag re-trigger is the step that has bitten past releases: if the tag
build is **not** byte-identical to the `main` dispatch build (ExecuTorch has
shown a few-byte drift — this actually broke `v0.4.0`/`v0.4.1`), the checksum
committed in Step 2 no longer matches the attached zip, and every SwiftPM
consumer of the tag fails to resolve.

Always verify after the tag run finishes:

```bash
gh release download vX.Y.Z --repo dmlux/Prism \
  --pattern "PrismNative.xcframework.zip" --dir /tmp/verify
swift package compute-checksum /tmp/verify/PrismNative.xcframework.zip
# must equal prismNativeReleaseChecksum in Package.swift @ vX.Y.Z
```

If they match, the release is consistent — done.

If they differ, reconcile so the tag points at the checksum of the **actually
attached** bytes (the attach step is deliberately `--clobber`-free, so the
first-attached zip is authoritative and must not be replaced):

1. Update `Package.swift` `prismNativeReleaseChecksum` to the computed value.
2. Move the tag onto the corrected commit:

   ```bash
   git commit -am "fix: correct PrismNative checksum for vX.Y.Z"
   git tag -f vX.Y.Z && git push -f origin vX.Y.Z
   ```

   The re-pushed tag triggers CI again; its attach step now finds the zip
   already present and **fails loudly on that one step by design** — that is
   expected and harmless, because the authoritative zip is already attached
   and now matches the tagged `Package.swift`.

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
(since 0.2.0). This is a **separate manual step**, not triggered by any
tag or workflow, and the free publishing quota resets monthly — only do
it when you actually intend a Maven release for the version.

Do it **after** the `vX.Y.Z` GitHub release exists, because the JAR
published to Central must embed the same per-platform natives the CI
built.

### One-time setup on the release machine

- **Credentials.** A Central user token lives in `~/.m2/settings.xml`
  under `<server><id>central</id>…`. Generate it at
  central.sonatype.com → *View Account → Generate User Token*. The
  namespace `io.github.dmlux` is verified via GitHub login.
- **GPG.** Central requires every artifact signed. Key setup and this
  machine's quirks (`disable-ipv6` in `~/.gnupg/dirmngr.conf`,
  `export GPG_TTY=$(tty)` for pinentry) are in the publishing runbook.
  Pre-cache the passphrase in the agent before deploying:
  `echo test | gpg --clearsign -o /dev/null`.
- **Maven + JDK.** No system Maven is installed; use a portable
  Apache Maven with `JAVA_HOME=$HOME/.sdkman/candidates/java/current`
  (Temurin 21).

### Deploy

1. **Embed the release natives.** The `release` build reads natives from
   `java/src/main/resources/` (gitignored). Populate it from the JAR the
   `vX.Y.Z` release just got — so Central ships all four platforms, not
   only the local one:

   ```bash
   gh release download vX.Y.Z --repo dmlux/Prism --pattern "*-all-platforms.jar" --dir /tmp
   unzip -o /tmp/prism-*-all-platforms.jar 'io/github/dmlux/prism/native/*' \
     -d java/src/main/resources/
   ```

2. **Dry run** (build + sign, no upload) to confirm the JAR carries all
   four natives:

   ```bash
   mvn -q -Prelease -DskipTests -Dgpg.skip=true -f java/pom.xml clean package
   unzip -l java/target/prism-*.jar | grep -c 'native/.*\(so\|dylib\)'   # expect 4
   ```

3. **Deploy** (signs and uploads; the `central-publishing` plugin waits
   for Central to validate):

   ```bash
   mvn -Prelease -DskipTests -f java/pom.xml clean deploy
   ```

4. **Publish.** The upload lands in the portal as a *Validated*
   deployment. Open central.sonatype.com → *Publishing → Deployments*
   and click **Publish** (deliberately manual — Central never deletes a
   published version). It resolves on `repo.maven.apache.org` within
   ~15–60 min.

> **Pitfall.** If a `deploy` invocation's output was piped through
> `head`/truncated, the command can complete *invisibly* and still
> upload — leaving a second, identical *Validated* deployment in the
> portal. It is harmless; **Drop** the duplicate. A version can only be
> published once.

---

## Quick checklist (library `vX.Y.Z`)

- [ ] Model that CI downloads is released (§1 ordering constraint).
- [ ] Version bumps committed on `main`; `Package.swift` still on previous pin.
- [ ] `gh workflow run` both workflows on `main`; both green.
- [ ] Pin URL + checksum from the dispatch artifact into `Package.swift`; commit.
- [ ] `gh release create vX.Y.Z` at that commit (`--latest`).
- [ ] After tag CI: download the attached XCFramework, `compute-checksum`,
      confirm it equals the tagged `Package.swift` — reconcile + re-tag if not.
- [ ] (Optional, separate) Maven Central per §5 — embed the release
      natives, then `mvn -Prelease deploy`, then Publish in the portal.
