# Don't Look Only Once: Towards Multimodal Interactive Reasoning with Selective Visual Revisitation  

<p align="left">
    <a href='https://jiwanchung.github.io/' target='_blank'>Jiwan Chung<sup>*</sup></a>&emsp;
    <a href='https://junhyeok.kim/' target='_blank'>Junhyeok Kim<sup>*</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?user=w3hOuRoAAAAJ' target='_blank'>Siyeol Kim</a>&emsp;
    <a href='https://jaeyoung-l.github.io/' target='_blank'>Jaeyoung Lee</a>&emsp;
    <a href="https://scholar.google.com/citations?user=Og3gN_AAAAAJ" target='_blank'>Minsoo Kim</a>&emsp;
    <a href='https://mirlab.yonsei.ac.kr/' target='_blank'>Youngjae Yu</a>
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2505.18842-b31b1b.svg)](https://arxiv.org/abs/2505.18842) [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-kjunh/v1--7B-FFD21E)](https://huggingface.co/kjunh/v1-7B)

<p align="center">
  <img src="assets/figure.png">
</p>

## Installation
```bash
conda create -n v1 python=3.10 -y
conda activate v1
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## Demo

### Gradio Web UI
Highly Recommended as the copy tokens are displayed on image.

<p align="center">
  <img src="assets/demo.png">
</p>

```bash
python run_gradio.py
```

### Inference
```bash
python inference.py
```
The script uses a default image URL and text prompt. To use your own inputs, you can modify the `image` variable within the `messages` list and the `text` field for the user prompt.

## Coming Soon
- [x] Inference code
- [ ] Training data
- [ ] Evaluation code
- [ ] Training code


## Citation
If you find our work valuable, please cite:
```bibtex
@misc{chung2025dontlookoncemultimodal,
      title={Don't Look Only Once: Towards Multimodal Interactive Reasoning with Selective Visual Revisitation}, 
      author={Jiwan Chung and Junhyeok Kim and Siyeol Kim and Jaeyoung Lee and Min Soo Kim and Youngjae Yu},
      year={2025},
      eprint={2505.18842},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.18842}, 
}
```