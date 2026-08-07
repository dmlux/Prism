# Example texts

Original prose texts written specifically for the Prism repository, used as
checked-in fixtures for the segmentation parity tests, the tagger tests, and
the reproducible benchmark suite:

- `skarvholmen-bokmaal.txt` — Norwegian Bokmål short story (~4.6 kB)
- `fjellvatnet-nynorsk.txt` — Norwegian Nynorsk essay (~3.9 kB)
- `harbor-english.txt` — English short story (~4.2 kB); covers English
  abbreviations (`Mr.`, `Dr.`, `e.g.`, `cf.`, `a.m.`, `approx.`, `Nov.`), a
  missing space after punctuation (`water.Every`), a hyphen across a line
  break (`boat-\nhouse`), a URL, and an e-mail address

Both texts are original works created for this repository and are dedicated
to the public domain under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).
They may be copied, modified, and redistributed without attribution.

The texts deliberately contain the extraction artifacts the runtime
segmentation repairs on real e-book input: hard-wrapped lines inside
sentences, words hyphenated across line breaks (`bjørke-\nskogen`), and one
missing space after sentence punctuation per text (`havet.Et`, `vatnet.Etter`).
They also cover the token conventions the segmentation implements:
abbreviations (`ca.`, `kl.`, `nr.`, `f.eks.`, `bl.a.`, `dvs.`, `osv.`),
ordinals and dates (`17. mai`, `3. november`), decimal and clock numbers
(`2,3`, `12.30`), quotation dialogue (`«…»`), a URL, and an e-mail address.

The reference segmentation counts pinned in the C++, Swift, and Python test
suites come from the Python reference implementation
(`prism.data.segmentation.segment_pretokenized_sentences` with the Norwegian
abbreviation policy). The `*-subword-parity.json` files record, per sentence,
the subword IDs the reference Hugging Face tokenizer produces for the
`prism-no` vocabulary; the native byte-level BPE implementations must
reproduce them exactly. Regenerate them after changing a text:

```bash
python -m prism.tools.subword_parity_fixture \
  --text data/examples/skarvholmen-bokmaal.txt \
  --vocabulary models/prism-no-0.2.3/vocabulary.json \
  --output data/examples/skarvholmen-bokmaal-subword-parity.json
```

Everything else in this directory (for example locally kept larger texts)
stays untracked; see the repository `.gitignore`.
