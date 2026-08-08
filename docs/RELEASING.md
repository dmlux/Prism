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
the compiler cache in `main` scope, which the later tag run restores.

> **Gate — do not tag until BOTH dispatches are green.** Two reasons:
> (1) the tag run only restores a warm cache if the `main` dispatch has
> already *finished* saving it, so tagging early forces a slow cold rebuild
> (and, for the JAR, wastes the warm-up entirely); (2) you must confirm both
> builds are actually healthy on `main` before cutting a public release —
> otherwise you tag and only then discover a broken build. Wait for both run
> conclusions to be `success` (macOS slices are ~1–2 h cold, minutes warm).
> Never start Step 2/3 while either dispatch is still running or red.

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
**re-triggers both workflows on the tag**. The tag runs rebuild and, because
the ref is now `refs/tags/v*`, attach the JAR, `PrismNative.xcframework.zip`,
and its `.checksum` to the `vX.Y.Z` release. The tag-run rebuild is *usually*
byte-identical to the `main` dispatch but **not reliably so** (see Step 4) —
which is why the pin was taken from the dispatch build, not this one.

### Step 4 — Verify the pin matches the attached bytes (critical)

The tag re-trigger is the step that has bitten every other release: the tag
build is often **not** byte-identical to the `main` dispatch build (ExecuTorch
has a nondeterministic few-byte drift — it broke `v0.4.0`/`v0.4.1`, and
`v0.6.0` drifted `f6f86519…` on `main` vs `8733f897…` on the tag). When it
drifts, the checksum committed in Step 2 no longer matches the zip the tag run
attached, and **every SwiftPM consumer of the tag fails to resolve**.

Always verify after the tag run finishes:

```bash
gh release download vX.Y.Z --repo dmlux/Prism \
  --pattern "PrismNative.xcframework.zip" --dir /tmp/verify
swift package compute-checksum /tmp/verify/PrismNative.xcframework.zip
# must equal prismNativeReleaseChecksum in Package.swift @ vX.Y.Z
```

If they match, the release is consistent — done.

If they differ, **do not move the tag and do not re-pin to the drifted
bytes.** The correct fix keeps the tag immutable and makes the *attached
bytes* equal the *already-committed* pin, by re-attaching the exact dispatch
build the pin came from (it is proven good — it passed the consumer test in
the dispatch run):

```bash
gh run download <prism-native-dispatch-run-id> --repo dmlux/Prism \
  --name PrismNative.xcframework --dir /tmp/xcf
gh release upload vX.Y.Z --repo dmlux/Prism \
  /tmp/xcf/PrismNative.xcframework.zip \
  /tmp/xcf/PrismNative.xcframework.zip.checksum --clobber
```

Then re-run the verification above; it now matches. No tag move, no CI
re-trigger. This is the **one sanctioned use of `--clobber`** on the
XCFramework asset: replacing the tag-run's drifted rebuild with the exact
bytes the tagged `Package.swift` already pins. (`v0.6.0` was reconciled this
way.)

> **Why this is clumsy — and the planned fix.** The tag-run rebuild is pure
> waste: it repeats a ~1 h build and can only *break* the release, never
> improve it. The clean design is a single `workflow_dispatch(version)` that
> builds once, commits the checksum, creates the tag+release, and attaches
> **the bytes it just built** — no second build, no drift window, no manual
> pin. Until that lands, the dispatch-then-tag dance above is the process.
> See §6.

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

## 6. Planned simplification (not yet implemented)

The current library flow has too many manual, fragile steps for what should
be one button. The root cause is narrow: SwiftPM pins the binary target by a
checksum that must live in the *tagged* `Package.swift`, so today we build on
`main` to learn the checksum, commit it, then tag — and the tag rebuild
(different bytes) is what forces the manual reconcile in Step 4.

A single tag-producing workflow removes all of it. One
`workflow_dispatch` with a `version` input would:

1. build the XCFramework and the JAR once (parallel jobs);
2. compute the XCFramework checksum;
3. write URL + checksum into `Package.swift`, bump version strings, commit;
4. create the tag + GitHub release at that commit (atomic);
5. attach **the exact bytes just built** (no rebuild) plus the JAR.

That eliminates the second build, the drift window, the manual checksum copy,
and the "both dispatches green" gate — the only human input becomes the
version number (optionally behind an Actions approval gate). The model
releases could fold into the same or a sibling `workflow_dispatch` that
packages `models/`, uploads the tarballs, and mirrors to Hugging Face.

Until this lands, follow §1–§5.

---

## Quick checklist (library `vX.Y.Z`)

- [ ] Model that CI downloads is released (§1 ordering constraint).
- [ ] Version bumps committed on `main`; `Package.swift` still on previous pin.
- [ ] `gh workflow run` both workflows on `main`; **wait for BOTH to be
      green before tagging** — never tag while a dispatch runs (§3 gate).
- [ ] Pin URL + checksum from the dispatch artifact into `Package.swift`; commit.
- [ ] `gh release create vX.Y.Z` at that commit (`--latest`).
- [ ] After tag CI: download the attached XCFramework, `compute-checksum`,
      confirm it equals the tagged `Package.swift`. If it drifted, re-attach
      the dispatch build with `--clobber` (§4) — do not move the tag.
- [ ] (Optional, separate) Maven Central per §5 — embed the release
      natives, then `mvn -Prelease deploy`, then Publish in the portal.
