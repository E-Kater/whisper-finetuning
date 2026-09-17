"""Анализ топологии GPU: PCIe vs NVLink."""

import subprocess
import json


def get_topology():
    """Запускает nvidia-smi topo -m и парсит вывод."""
    result = subprocess.run(
        ["nvidia-smi", "topo", "-m"],
        capture_output=True, text=True,
    )
    return result.stdout


def get_gpu_info():
    """Получает информацию о GPU."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
         "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    return result.stdout.strip()


def main():
    print("=== GPU Info ===")
    print(get_gpu_info())

    print("\n=== Topology ===")
    topo = get_topology()
    print(topo)

    # Анализ
    print("\n=== Analysis ===")
    if "PHB" in topo:
        print("Connection type: PHB (PCIe Host Bridge)")
        print("Bandwidth: ~16 GB/s (PCIe 3.0 x16)")
    if "NV" in topo and "NV1" in topo:
        print("NVLink detected: ~300 GB/s (A100)")
    elif "PHB" in topo:
        print("No NVLink. Communication goes through PCIe.")
        print("This limits bandwidth to ~16 GB/s.")

    # Сохраняем в JSON
    result = {
        "gpu_info": get_gpu_info(),
        "topology": topo,
    }
    with open("results/topology.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
