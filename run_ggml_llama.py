import os
import sys
from subprocess import check_call, check_output
from urllib.request import urlopen


def download_url(url: str) -> None:
    fname = os.path.basename(url)
    if os.path.exists(fname):
        return
    with urlopen(url) as s, open(fname, "wb") as f:
        f.write(s.read())


def clone_llama_cpp(folder_name: str, branch: str = "b2074") -> None:
    if os.path.exists(folder_name):
        return
    check_call(
        [
            "git",
            "clone",
            "--branch",
            branch,
            "--depth",
            "1",
            "https://github.com/ggerganov/llama.cpp",
            folder_name,
        ]
    )


def compile_llama_cpp(folder_name: str) -> None:
    if os.path.exists(os.path.join(folder_name, "main")):
        return
    check_call(["make", "-C", folder_name, "main"])


def convert_model(folder_name: str, model_name: str) -> None:
    if model_name == "stories15M.pt":
        ctx = 388
    elif model_name == "stories110M.pt":
        ctx = 786
    else:
        ctx = 4096
    check_call(
        [
            sys.executable,
            f"{folder_name}/convert.py",
            "--outtype",
            "f32",
            "--vocab-dir",
            ".",
            model_name,
            "--ctx",
            str(ctx),
        ]
    )


def run_inference_on_cpu(folder_name: str) -> None:
    check_call(
        [
            f"{folder_name}/main",
            "-m",
            "ggml-model-f32.gguf",
            "--prompt",
            "Once upon a time",
            "-n",
            "1024",
            "--n-gpu-layers",
            "0",
        ]
    )


def main() -> None:
    download_url("https://github.com/karpathy/llama2.c/raw/master/tokenizer.model")
    download_url(
        "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt"
    )
    folder_name = "ggerganov_llama.cpp"
    model_name = "stories15M.pt"
    clone_llama_cpp(folder_name=folder_name)
    compile_llama_cpp(folder_name)
    convert_model(folder_name, model_name)
    run_inference_on_cpu(folder_name)


if __name__ == "__main__":
    main()
