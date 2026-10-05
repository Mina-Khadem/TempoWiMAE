# Third-party licenses and attribution

TempoWiMAE is released under the Apache License 2.0 (see `LICENSE`). Parts of the
implementation are adapted from the following projects. Their notices are reproduced
here and must be kept with any redistribution.

| Component | Where it is used | Upstream project | License |
|---|---|---|---|
| Transformer block, multi-head attention with query/value bias, MLP, 3-D patch embedding, stochastic depth wrapper, truncated-normal initialisation, sinusoidal position helpers | `tempowimae_model.py` | [VideoMAE](https://github.com/MCG-NJU/VideoMAE) (`modeling_pretrain.py`, `modeling_finetune.py`), which builds on BEiT and timm | CC BY-NC 4.0 (VideoMAE repository; see the note below) |
| BEiT Transformer blocks (origin of the block structure used by VideoMAE) | same files | [microsoft/unilm/beit](https://github.com/microsoft/unilm/tree/master/beit) | MIT |
| `drop_path`, `to_2tuple`, `trunc_normal_` (imported at runtime) | `tempowimae_model.py` | [pytorch-image-models (timm)](https://github.com/huggingface/pytorch-image-models) | Apache License 2.0 |

## Note on the VideoMAE license

The VideoMAE repository states: "The majority of this project is released under the
CC-BY-NC 4.0 license as found in the LICENSE file. Portions of the project are
available under separate license terms: SlowFast and pytorch-image-models are licensed
under the Apache 2.0 license. BEiT is licensed under the MIT license." The Transformer
building blocks adapted here derive from the BEiT (MIT) and timm (Apache 2.0) code paths
that VideoMAE itself reuses. Before publishing this repository under Apache-2.0, the
author should confirm that the retained code falls under those permissive terms; the
copyright notices below are kept in all cases.

## microsoft/unilm (BEiT): MIT License

```
The MIT License (MIT)

Copyright (c) Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## pytorch-image-models (timm): Apache License 2.0

```
Copyright 2019 Ross Wightman

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

The full Apache License 2.0 text is in the root `LICENSE` file.

## VideoMAE: Creative Commons Attribution-NonCommercial 4.0 International

The VideoMAE license text is available at
https://github.com/MCG-NJU/VideoMAE/blob/main/LICENSE and
https://creativecommons.org/licenses/by-nc/4.0/legalcode.
