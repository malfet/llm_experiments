import os
import sys
from typing import Optional
from urllib.request import urlopen


def check_call(args) -> None:
    from subprocess import PIPE, Popen, STDOUT

    with Popen(args, stdout=PIPE, stderr=STDOUT) as p:
        print(p.stdout.read().decode("utf-8"))
        p.communicate()
        if p.returncode != 0:
            raise RuntimeError("Process execution failed")


def download_url(url: str) -> None:
    fname = os.path.basename(url)
    if os.path.exists(fname):
        return
    with urlopen(url) as s, open(fname, "wb") as f:
        f.write(s.read())


def clone_llama_cpp(folder_name: str, branch: str = "b2774") -> None:
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


def convert_model(folder_name: str, model_name: str, dtype: str = "f32") -> None:
    if model_name == "stories15M.pt":
        ctx = 288
    elif model_name == "stories110M.pt":
        ctx = 768
    else:
        ctx = 4096
    check_call(
        [
            sys.executable,
            f"{folder_name}/convert.py",
            "--outtype",
            dtype,
            "--vocab-dir",
            ".",
            model_name,
            "--ctx",
            str(ctx),
        ]
    )


def run_inference_on_cpu(
    folder_name: str,
    dtype: str = "f32",
    prompt: str = "Once upon a time",
    seq_len: int = 1024,
    seed: Optional[int] = None,
) -> None:
    args = [
        f"{folder_name}/main",
        "-m",
        f"ggml-model-{dtype}.gguf",
        "--prompt",
        prompt,
        "--n-predict",
        str(seq_len),
        "--n-gpu-layers",
        "0",
    ]
    if seed is not None:
        args.extend(["-s", str(seed)])
    check_call(args)


def parse_args():
    from argparse import ArgumentParser

    parser = ArgumentParser("Comple, convert and run llama.cpp")
    parser.add_argument("--model-path", type=str, default="stories15M.pt")
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--llama-branch", type=str, default="b2074")
    parser.add_argument("--dtype", type=str, default="f32")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--seq-len", type=int, default=1024)
    # Do not attempt to parse CLI arguments if running inside notebook
    return parser.parse_args([] if hasattr(__builtins__, "__IPYTHON__") else None)


def main() -> None:
    download_url("https://github.com/karpathy/llama2.c/raw/master/tokenizer.model")
    download_url(
        "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.pt"
    )
    args = parse_args()
    folder_name = "ggerganov_llama.cpp"
    clone_llama_cpp(folder_name=folder_name, branch=args.llama_branch)
    compile_llama_cpp(folder_name)
    convert_model(folder_name, args.model_path, dtype=args.dtype)
    run_inference_on_cpu(
        folder_name,
        dtype=args.dtype,
        prompt=args.prompt,
        seq_len=args.seq_len,
        seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
