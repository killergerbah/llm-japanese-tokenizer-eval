# Third-party notices

This project is released under the BSD Zero Clause License (0BSD) (see
`LICENSE`). It depends on the packages below. All are permissively licensed;
**none impose copyleft obligations on this project's source code**.

## Runtime dependencies

| Package | Version used | License | Notes |
|---|---|---|---|
| fugashi | 1.5.2 | MIT | Cython wrapper for MeCab |
| MeCab (bundled in fugashi) | 0.996 | **BSD-3-Clause** | part of MeCab's GPL-2.0 / LGPL-2.1 / BSD-3-Clause tri-license; the BSD option is used |
| unidic-lite | 1.0.8 | MIT | packaging of the UniDic dictionary |
| UniDic (bundled in unidic-lite) | 2.1.2 | **BSD-3-Clause** | The UniDic Consortium |
| openai | 3.x | Apache-2.0 | OpenAI-compatible client |
| pandas | 3.x | BSD-3-Clause | |
| numpy | 2.x | BSD-3-Clause | |
| matplotlib | 3.11.x | PSF-based (matplotlib license) | BSD-compatible |
| seaborn | 0.13.x | BSD-3-Clause | |
| pyarrow | 25.x | Apache-2.0 | Parquet output |
| PyYAML | 6.x | MIT | |
| python-dotenv | 1.x | BSD-3-Clause | |
| tqdm | 4.x | MPL-2.0 AND MIT | MPL-2.0 is file-level copyleft; the dependency is used unmodified, so it does not affect this project |

## Optional dependencies

| Package | Extra | License |
|---|---|---|
| datasets | `[gold]` | Apache-2.0 |
| sudachipy | `[sudachi]` | Apache-2.0 |
| sudachidict-core | `[sudachi]` | Apache-2.0 |

## Why there is no copyleft from MeCab

MeCab is tri-licensed: the user may choose **GPL-2.0**, **LGPL-2.1**, or
**BSD-3-Clause**. The distribution used here (`fugashi` wheels) ships the
BSD-3-Clause text as `LICENSE.mecab`, so the BSD terms apply and there is no
copyleft obligation. Likewise, the bundled UniDic dictionary ships under
BSD-3-Clause (`LICENSE.unidic`). If you instead install the full `unidic`
package or a differently licensed dictionary, re-check its terms.

## Data is licensed separately

The benchmark **gold data is not covered by the 0BSD license**. It is derived
from Universal Dependencies treebanks, which carry Creative Commons licenses and
are **not committed to this repository** (see `.gitignore`). Licenses vary by
treebank and revision:

| Treebank | License |
|---|---|
| UD_Japanese-GSD | CC BY-SA 4.0 (the NonCommercial restriction was dropped in UD v2.5) |
| UD_Japanese-PUD | CC BY-SA 3.0 |
| UD_Japanese-BCCWJ | CC BY-NC-SA 4.0 (non-commercial) |
| KWDLC (Kyoto University Web Document Leads Corpus) | No declared license; the README states research use. Do not redistribute. |

If you distribute `gold.jsonl` or any other derived data, you must comply with
the license of the specific treebank and revision you used. CC BY-SA is
share-alike: derived annotations must be redistributed under the same license.
Always confirm the exact license in the pinned treebank revision's `LICENSE`/
`README`.

---

## MeCab license text (BSD-3-Clause)

```
Copyright (c) 2001-2008, Taku Kudo
Copyright (c) 2004-2008, Nippon Telegraph and Telephone Corporation
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above
   copyright notice, this list of conditions and the
   following disclaimer.

 * Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the
   following disclaimer in the documentation and/or other
   materials provided with the distribution.

 * Neither the name of the Nippon Telegraph and Telegraph Corporation
   nor the names of its contributors may be used to endorse or
   promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## UniDic license text (BSD-3-Clause)

```
Copyright (c) 2011-2017, The UniDic Consortium
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

 * Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the
   distribution.

 * Neither the name of the UniDic Consortium nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
