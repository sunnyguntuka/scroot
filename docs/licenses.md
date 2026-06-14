# Third-party licenses

scroot is licensed [Apache-2.0](../LICENSE). This page lists the
licenses of scroot's runtime dependencies (core install plus the
`dashboard`, `database`, `security`, and `api` extras), generated with
[`pip-licenses`](https://github.com/raimon49/pip-licenses).

**No GPL, LGPL, or AGPL dependencies were found.** All dependencies are
MIT, BSD, Apache-2.0, ISC, MPL-2.0, or PSF-licensed - all compatible
with Apache-2.0 distribution.

To regenerate this table:

```bash
pip install -e ".[all,dashboard,database,security]"
pip install pip-licenses
pip-licenses --format=markdown --order=license --with-urls
```

| Name | Version | License | URL |
|---|---|---|---|
| distro | 1.9.0 | Apache Software License | https://github.com/python-distro/distro |
| huggingface_hub | 0.36.2 | Apache Software License | https://github.com/huggingface/huggingface_hub |
| openai | 2.41.1 | Apache Software License | https://github.com/openai/openai-python |
| requests | 2.34.2 | Apache Software License | https://github.com/psf/requests |
| safetensors | 0.8.0 | Apache Software License | https://github.com/huggingface/safetensors |
| sentence-transformers | 3.4.1 | Apache Software License | https://www.SBERT.net |
| tokenizers | 0.22.2 | Apache Software License | https://github.com/huggingface/tokenizers |
| transformers | 4.57.6 | Apache Software License | https://github.com/huggingface/transformers |
| sniffio | 1.3.1 | Apache Software License; MIT License | https://github.com/python-trio/sniffio |
| python-multipart | 0.0.32 | Apache-2.0 | https://github.com/Kludex/python-multipart |
| regex | 2026.5.9 | Apache-2.0 AND CNRI-Python | https://github.com/mrabarnett/mrab-regex |
| packaging | 26.2 | Apache-2.0 OR BSD-2-Clause | https://github.com/pypa/packaging |
| cryptography | 48.0.1 | Apache-2.0 OR BSD-3-Clause | https://github.com/pyca/cryptography |
| Jinja2 | 3.1.6 | BSD License | https://github.com/pallets/jinja/ |
| colorama | 0.4.6 | BSD License | https://github.com/tartley/colorama |
| httpx | 0.28.1 | BSD License | https://github.com/encode/httpx |
| mpmath | 1.3.0 | BSD License | http://mpmath.org/ |
| scipy | 1.17.1 | BSD License | https://scipy.org/ |
| sympy | 1.14.0 | BSD License | https://sympy.org |
| threadpoolctl | 3.6.0 | BSD License | https://github.com/joblib/threadpoolctl |
| Pygments | 2.20.0 | BSD-2-Clause | https://pygments.org |
| MarkupSafe | 3.0.3 | BSD-3-Clause | https://github.com/pallets/markupsafe/ |
| click | 8.4.1 | BSD-3-Clause | https://github.com/pallets/click/ |
| fsspec | 2026.4.0 | BSD-3-Clause | https://github.com/fsspec/filesystem_spec |
| httpcore | 1.0.9 | BSD-3-Clause | https://www.encode.io/httpcore/ |
| idna | 3.18 | BSD-3-Clause | https://github.com/kjd/idna |
| joblib | 1.5.3 | BSD-3-Clause | https://joblib.readthedocs.io |
| networkx | 3.6.1 | BSD-3-Clause | https://networkx.org/ |
| pycparser | 3.0 | BSD-3-Clause | https://github.com/eliben/pycparser |
| scikit-learn | 1.9.0 | BSD-3-Clause | https://scikit-learn.org |
| starlette | 1.2.1 | BSD-3-Clause | https://github.com/Kludex/starlette |
| torch | 2.12.0 | BSD-3-Clause | https://pytorch.org |
| uvicorn | 0.49.0 | BSD-3-Clause | https://uvicorn.dev/ |
| numpy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | https://numpy.org |
| shellingham | 1.5.4 | ISC License (ISCL) | https://github.com/sarugaku/shellingham |
| SQLAlchemy | 2.0.50 | MIT | https://www.sqlalchemy.org |
| annotated-doc | 0.0.4 | MIT | https://github.com/fastapi/annotated-doc |
| anyio | 4.13.0 | MIT | https://anyio.readthedocs.io/en/stable/versionhistory.html |
| cffi | 2.0.0 | MIT | https://cffi.readthedocs.io/en/latest/whatsnew.html |
| charset-normalizer | 3.4.7 | MIT | https://github.com/jawah/charset_normalizer/blob/master/CHANGELOG.md |
| fastapi | 0.136.3 | MIT | https://github.com/fastapi/fastapi |
| filelock | 3.29.2 | MIT | https://github.com/tox-dev/py-filelock |
| jiter | 0.15.0 | MIT | https://github.com/pydantic/jiter/ |
| narwhals | 2.22.1 | MIT | https://github.com/narwhals-dev/narwhals |
| pydantic | 2.13.4 | MIT | https://github.com/pydantic/pydantic |
| pydantic_core | 2.46.4 | MIT | https://github.com/pydantic |
| typer | 0.26.7 | MIT | https://github.com/fastapi/typer |
| typing-inspection | 0.4.2 | MIT | https://github.com/pydantic/typing-inspection |
| urllib3 | 2.7.0 | MIT | https://github.com/urllib3/urllib3/blob/main/CHANGES.rst |
| greenlet | 3.5.1 | MIT AND PSF-2.0 | https://greenlet.readthedocs.io |
| PyYAML | 6.0.3 | MIT License | https://pyyaml.org/ |
| annotated-types | 0.7.0 | MIT License | https://github.com/annotated-types/annotated-types |
| h11 | 0.16.0 | MIT License | https://github.com/python-hyper/h11 |
| markdown-it-py | 4.2.0 | MIT License | https://github.com/executablebooks/markdown-it-py |
| mdurl | 0.1.2 | MIT License | https://github.com/executablebooks/mdurl |
| rich | 15.0.0 | MIT License | https://github.com/Textualize/rich |
| pillow | 12.2.0 | MIT-CMU | https://python-pillow.github.io |
| tqdm | 4.68.2 | MPL-2.0 AND MIT | https://tqdm.github.io |
| certifi | 2026.5.20 | Mozilla Public License 2.0 (MPL 2.0) | https://github.com/certifi/python-certifi |
| typing_extensions | 4.15.0 | PSF-2.0 | https://github.com/python/typing_extensions |

## `scroot[local]` extra

The optional `local` extra (`pip install 'scroot[local]'`) adds:

| Name | License | URL |
|---|---|---|
| llama-cpp-python | MIT | https://github.com/abetlen/llama-cpp-python |
| huggingface-hub | Apache Software License | https://github.com/huggingface/huggingface_hub |

This extra is strictly opt-in and is not installed by `pip install scroot`.
