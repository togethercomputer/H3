# Together README

WIP commit to test serving of 125M model.
Once it's ready, we can serve the larger models easily.

Important files:
* The Python serving script is in `H3/examples/serving_h3.py`.
* The bash serving script is in `serve.sh`.
* The Dockerfile is in `Together_dockerfile`.
* The config is in `local-cfg.yaml`.

By default, we expect the model checkpoint to be in `/home/user/.together/models/H3-125M/model.pt` inside the Docker container.

You can get it by running these commands:
```
git lfs install
git clone https://huggingface.co/danfu09/H3-125M
```

You may need to run these commands after install to install FlashAttention (we can add them to the Dockerfile):
```
git submodule init
git submodule update
cd flash-attention

git submodule init
git submodule update

pip install -e .
```