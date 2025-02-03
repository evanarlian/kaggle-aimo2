# vast.ai
My personal flow. I used vast because they have L4 GPUs.

Make new SSH key (just once) and copy to every lambda instances to allow github access. Make sure to add SSH key to github as well.
```bash
# just once
ssh-keygen -t ed25519 -f ~/.ssh/vastai -N ""

SSH_IP=
SSH_PORT=
scp -P $SSH_PORT ~/.ssh/vastai root@$SSH_IP:~/.ssh/id_ed25519
scp -P $SSH_PORT -r ~/.kaggle root@$SSH_IP:~
scp -P $SSH_PORT ~/.tmux.conf root@$SSH_IP:~
```

Inside vast vm now. It is running automatically inside tmux. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Repo and venv setup
```bash
git clone git@github.com:evanarlian/kaggle-aimo2.git
cd kaggle-aimo2
uv sync
# restart shell after this
```

Auth stuffs
```bash
# open vscode remote, super slick
# manually fill .env first, or copy from local machine
export $(cat .env | xargs)
git config --global credential.helper store
huggingface-cli login --token $HF_TOKEN --add-to-git-credential
wandb login $WANDB_TOKEN
```
