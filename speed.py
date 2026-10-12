from pathlib import Path
from tqdm import tqdm
import time

import torch

from datasets.nuscenes import compile_data
from models.model import Model


def main():

    device = torch.device("cuda:0")

    ckpt_path = Path("test/first_20epochs/16.pt") # Mettere il checkpoint desiderato

    assert ckpt_path.exists(), \
        f"[Error] Checkpoint not found at {ckpt_path}"

    # Validation loader, batch size = 1d
    loader = compile_data(False, "trainval", "../data/nuscenes", "vehicle", batch_size=1, num_workers=0, drop_last=False)

    sample = next(iter(loader))
    images, seg, depths, intrinsics, extrinsics, labels = sample

    images = images.to(device)
    intrinsics = intrinsics.to(device)
    extrinsics = extrinsics.to(device)

    # Inizializzazione della rete
    network = Model([0], backbone="sucam", n_classes=1, loss_type="focal", use_seg=False, use_dep=False)

    network.load(torch.load(str(ckpt_path), map_location="cpu"))
    network.eval()

    with torch.no_grad():

        latency = []

        for _ in tqdm(range(1050)):

            torch.cuda.synchronize()

            start_time = time.perf_counter()

            network(images, intrinsics, extrinsics)  # Forward della rete: ciò che ci interessa 

            torch.cuda.synchronize()

            end_time = time.perf_counter()

            latency.append(end_time - start_time)

        # primi 50 = warm-up
        latency = latency[50:]

    latency = torch.tensor(latency).mean().item()

    print(
        "inference latency: {:.3f} ms, speed: {:.3f} fps".format(
            latency * 1000,
            1.0 / latency
        )
    )


if __name__ == "__main__":
    main()