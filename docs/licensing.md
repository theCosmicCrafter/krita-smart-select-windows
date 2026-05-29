# Licensing

Krita Smart Select is licensed under GPL-3.0-only.

Krita itself is GPLv3, and native plugins link against Krita libraries and use
Krita plugin APIs. This plugin therefore uses a GPL-compatible license for the
plugin code. Krita's licensing page states that distributed Krita plugins must
also be shared under the GNU GPL: <https://krita.org/en/about/license/>.

## Repository License

- Plugin code: GPL-3.0-only.
- Tool icon resource file: CC0-1.0 where marked by SPDX in the file.
- BiRefNet HR Matting model: MIT, downloaded from the pinned Hugging Face repo
  and revision documented in [models.md](models.md).

The model checkpoint and Hugging Face custom model code are runtime artifacts
downloaded into the user model cache. They are not committed into this
repository.

## Distribution Notes

A release package should include:

- the GPL license text from [../LICENSE](../LICENSE);
- this licensing summary;
- the plugin Python/C++ files;
- no model checkpoint files;
- no managed runtime venv.

When redistributing a package that includes third-party artifacts, include the
corresponding third-party license notices.
