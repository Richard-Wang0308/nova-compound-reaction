export CUDA_VISIBLE_DEVICES=0,1    # make both GPUs visible for two-stage pipeline (PSICHIC on GPU 0, Boltz-2 on GPU 1)
python3 neurons/miner.py --wallet.name multisig-jjpes-atel --wallet.hotkey hot --logging.info
