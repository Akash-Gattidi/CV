import os
import sys
import urllib.request

def download_with_progress(url: str, output_path: str):
    temp_path = output_path + ".part"
    headers = {"User-Agent": "Mozilla/5.0"}
    downloaded_bytes = 0

    if os.path.exists(temp_path):
        downloaded_bytes = os.path.getsize(temp_path)
        headers["Range"] = f"bytes={downloaded_bytes}-"
        print(f"Resuming {os.path.basename(output_path)} from {downloaded_bytes / (1024*1024):.2f} MB...")
    else:
        print(f"Starting download for {os.path.basename(output_path)}...")

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            content_length = response.headers.get("Content-Length")
            total_size = downloaded_bytes + int(content_length) if content_length else None
            mode = "ab" if downloaded_bytes > 0 and response.status == 206 else "wb"
            if mode == "wb":
                downloaded_bytes = 0

            chunk_size = 1024 * 1024  # 1 MB chunks
            last_print = 0

            with open(temp_path, mode) as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded_bytes += len(chunk)

                    # Print progress every 50 MB or at completion
                    if downloaded_bytes - last_print >= 50 * 1024 * 1024:
                        if total_size:
                            pct = (downloaded_bytes / total_size) * 100
                            print(f"[{pct:.1f}%] {downloaded_bytes / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB")
                        else:
                            print(f"{downloaded_bytes / (1024*1024):.1f} MB downloaded")
                        last_print = downloaded_bytes

        if os.path.exists(temp_path):
            os.replace(temp_path, output_path)
        print(f"Finished {os.path.basename(output_path)}: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

    except urllib.error.HTTPError as e:
        if e.code == 416:
            # Range already satisfied
            if os.path.exists(temp_path):
                os.replace(temp_path, output_path)
            print(f"{os.path.basename(output_path)} is already fully downloaded.")
        else:
            raise

def download_dust3r():
    local_dir = os.path.join("checkpoints", "dust3r_512")
    os.makedirs(local_dir, exist_ok=True)

    config_url = "https://huggingface.co/naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt/resolve/main/config.json"
    model_url = "https://huggingface.co/naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt/resolve/main/model.safetensors"

    config_path = os.path.join(local_dir, "config.json")
    model_path = os.path.join(local_dir, "model.safetensors")

    if not os.path.exists(config_path) or os.path.getsize(config_path) < 100:
        download_with_progress(config_url, config_path)
    else:
        print("config.json already present.")

    if not os.path.exists(model_path) or os.path.getsize(model_path) < 2000000000:
        download_with_progress(model_url, model_path)
    else:
        print(f"model.safetensors already complete ({os.path.getsize(model_path) / (1024*1024):.2f} MB).")

    print("\nDUSt3R checkpoint is completely downloaded and ready!")

if __name__ == "__main__":
    download_dust3r()
