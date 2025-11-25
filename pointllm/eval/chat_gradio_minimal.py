import argparse
import logging
import os
import time

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch
from transformers import AutoTokenizer

from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.data import pc_norm, farthest_point_sample
from pointllm.model import PointLLMLlamaForCausalLM
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.utils import disable_torch_init

MAX_POINTS = 8192


def _resolve_device(device_pref: str) -> torch.device:
    """Pick the best available device based on the user preference."""
    if device_pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_pref == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_pref)


def init_model(args):
    """Load tokenizer/model and return supporting configs."""
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    logging.warning(f"Model name: {model_name}")

    print("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir="/app/weights", force_download=False
    )
    print("Loading model")
    model = PointLLMLlamaForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=False,
        use_cache=True,
        torch_dtype=args.torch_dtype,
        cache_dir="/app/weights",
        force_download=False,
    ).to(args.device)
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    model.eval()

    point_backbone_config = model.get_model().point_backbone_config
    conv = conv_templates["vicuna_v1_1"].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]

    mm_use_point_start_end = getattr(model.config, "mm_use_point_start_end", False)

    return model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv


def _build_plot(points: np.ndarray, colors: np.ndarray) -> go.Figure:
    """Create a 3D scatter plot for the uploaded cloud."""
    rgb_strings = [
        f"rgb({int(r)}, {int(g)}, {int(b)})"
        for r, g, b in np.clip(colors * 255, 0, 255)
    ]
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(size=2, color=rgb_strings),
            )
        ]
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        paper_bgcolor="white",
    )
    return fig


def _prepare_tensor(points: np.ndarray, colors: np.ndarray, args) -> torch.Tensor:
    """Downsample, normalize, and convert the cloud into the model tensor."""
    points_with_color = np.concatenate((points, colors), axis=1)
    if points_with_color.shape[0] > MAX_POINTS:
        points_with_color = farthest_point_sample(points_with_color, MAX_POINTS)
    normed = pc_norm(points_with_color)
    tensor = torch.from_numpy(normed).unsqueeze(0).to(dtype=args.torch_dtype, device=args.device)
    return tensor


def start_demo(args, model, tokenizer, point_backbone_config, keywords, mm_use_point_start_end, conv):
    """Spin up the simplified Gradio UI."""
    point_token_len = point_backbone_config["point_token_len"]
    default_point_patch_token = point_backbone_config["default_point_patch_token"]
    default_point_start_token = point_backbone_config["default_point_start_token"]
    default_point_end_token = point_backbone_config["default_point_end_token"]

    def confirm_point_cloud(file_obj, chatbot, answer_time, conv_state):
        chatbot = [] if chatbot is None else chatbot
        if file_obj is None:
            warning = "<span style='color: red;'>[System] Please upload a .npy point cloud first.</span>"
            chatbot = chatbot + [[None, warning]]
            return gr.Plot.update(value=None), chatbot, answer_time, None

        file_path = file_obj.name
        if not file_path.endswith(".npy"):
            warning = "<span style='color: red;'>[System] Only .npy files are supported in this demo.</span>"
            chatbot = chatbot + [[None, warning]]
            return gr.Plot.update(value=None), chatbot, answer_time, None

        try:
            data = np.load(file_path)
            if data.ndim != 2 or data.shape[1] < 3:
                raise ValueError("Expected an array with at least three columns (xyz).")

            points = data[:, :3].astype(np.float32)
            colors = (
                data[:, 3:6].astype(np.float32)
                if data.shape[1] >= 6
                else np.zeros_like(points)
            )
            if colors.max() > 1:
                colors = colors / 255.0

            fig = _build_plot(points, colors)
            point_tensor = _prepare_tensor(points, colors, args)

            conv_state.reset()
            answer_time = 0
            chatbot = chatbot + [[None, "<span style='color: red;'>[System] New Point Cloud</span>"]]
            return fig, chatbot, answer_time, point_tensor
        except Exception as exc:
            logging.warning(f"Failed to load point cloud: {exc}")
            warning = "<span style='color: red;'>[System] Failed to read the .npy file.</span>"
            chatbot = chatbot + [[None, warning]]
            return gr.Plot.update(value=None), chatbot, answer_time, None

    def user(message, history):
        history = [] if history is None else history
        return "", history + [[message, None]]

    def clear_conv(history, conv_state):
        conv_state.reset()
        return [], 0

    def ask_point_description(x, y, z, history):
        history = [] if history is None else history
        question = f"What is loacted at coordinates ({x}, {y}, {z}) of point cloud?"
        return history + [[question, None]]

    def answer_generate(history, answer_time, point_clouds, conv_state):
        history = [] if history is None else history
        if not history:
            history.append(["", ""])

        if point_clouds is None:
            outputs = "<span style='color: red;'>[System] Please upload a point cloud first.</span>"
            history[-1][1] = outputs
            yield history
            return

        print(f"Answer Time: {answer_time}")
        logging.warning(f"Answer Time: {answer_time}")
        input_text = history[-1][0]
        qs = input_text

        if answer_time == 0:
            if mm_use_point_start_end:
                qs = (
                    default_point_start_token
                    + default_point_patch_token * point_token_len
                    + default_point_end_token
                    + "\n"
                    + qs
                )
            else:
                qs = default_point_patch_token * point_token_len + "\n" + qs

        conv_state.append_message(conv_state.roles[0], qs)
        conv_state.append_message(conv_state.roles[1], None)
        prompt = conv_state.get_prompt()

        print("#" * 80)
        print(prompt.replace("<point_patch>" * point_token_len, f"<point_patch> * {point_token_len}"))
        print("#" * 80)

        logging.warning("#" * 80)
        logging.warning(
            prompt.replace("<point_patch>" * point_token_len, f"<point_patch> * {point_token_len}")
        )
        logging.warning("#" * 80)

        inputs = tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids, device=args.device)
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        stop_str = keywords[0]

        try:
            if input_ids.shape[1] >= 2047:
                raise ValueError("Context length exceeds the 2048 token limit.")

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    point_clouds=point_clouds,
                    do_sample=True,
                    temperature=1.0,
                    top_k=50,
                    max_length=2048,
                    top_p=0.95,
                    stopping_criteria=[stopping_criteria],
                )

            input_token_len = input_ids.shape[1]
            outputs = tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True
            )[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()

            conv_state.pop_last_none_message()
            conv_state.append_message(conv_state.roles[1], outputs)
            history[-1][1] = ""
            for character in outputs:
                history[-1][1] += character
                yield history
        except Exception as exc:
            logging.warning(f"[ERROR] {exc}")
            if input_ids.shape[1] >= 2047:
                chatbot_system_message = (
                    "Context length exceeds the model limit. Please press 'Clear' to restart."
                )
            else:
                chatbot_system_message = (
                    "Something went wrong while generating. Please confirm the point cloud again."
                )
            outputs = f"<span style='color: red;'>[System] {chatbot_system_message}</span>"
            history[-1][1] = outputs
            yield history

    with gr.Blocks(title="PointLLM Minimal Chat") as demo:
        answer_time = gr.State(value=0)
        point_clouds = gr.State(value=None)
        conv_state = gr.State(value=conv.copy())

        gr.Markdown(
            """
            # PointLLM Minimal Demo
            Upload a `.npy` point cloud, visualize it, and chat with PointLLM.
            """
        )

        available_devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        device_dropdown = gr.Dropdown(
            choices=available_devices,
            value=args.device.type if args.device.type in available_devices else available_devices[0],
            label="Compute Device",
            interactive=True,
        )

        with gr.Row():
            with gr.Column(scale=1):
                point_cloud_input = gr.File(
                    label="Upload Point Cloud (.npy only)",
                    file_types=[".npy"],
                    file_count="single",
                )
                confirm_btn = gr.Button("Load Point Cloud")
            with gr.Column(scale=2):
                plot = gr.Plot(label="Point Cloud Preview")

        chatbot = gr.Chatbot([], elem_id="chatbot", height=560, type='tuples')

        with gr.Row():
            text_input = gr.Textbox(
                show_label=False,
                placeholder="Ask PointLLM about the uploaded point cloud",
                container=False,
            )
            send_btn = gr.Button("Send")
        with gr.Row():
            coord_x = gr.Number(label="X coordinate", value=0.0)
            coord_y = gr.Number(label="Y coordinate", value=0.0)
            coord_z = gr.Number(label="Z coordinate", value=0.0)
            describe_btn = gr.Button("Describe Point")
        clear_btn = gr.Button("Clear Conversation")

        def change_device(selected_device, chatbot, point_clouds):
            chatbot = [] if chatbot is None else chatbot
            target_device = _resolve_device(selected_device)
            target_dtype = args.torch_dtype
            dtype_note = None
            note_parts = []
            dropdown_value = (
                target_device.type if target_device.type in available_devices else available_devices[0]
            )

            if selected_device == "cuda" and target_device.type != "cuda":
                note_parts.append("CUDA not available; using CPU instead.")

            if target_device.type == "cpu" and target_dtype == torch.float16:
                target_dtype = torch.float32
                args.torch_dtype = target_dtype
                dtype_note = "float16 is not fully supported on CPU. Using float32 instead."

            if args.device.type != target_device.type:
                model.to(device=target_device, dtype=target_dtype)
                if point_clouds is not None:
                    point_clouds = point_clouds.to(device=target_device, dtype=target_dtype)
                args.device = target_device
                if target_device.type == "cpu" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                note_parts.insert(0, f"Switched to {target_device.type.upper()}.")
            else:
                note_parts.append(f"Already using {target_device.type.upper()}.")

            if dtype_note:
                note_parts.append(dtype_note)

            note = f"<span style='color: red;'>[System] {' '.join(note_parts)}</span>"
            return gr.update(value=dropdown_value), chatbot + [[None, note]], point_clouds

        confirm_btn.click(
            confirm_point_cloud,
            inputs=[point_cloud_input, chatbot, answer_time, conv_state],
            outputs=[plot, chatbot, answer_time, point_clouds],
        )
        text_input.submit(user, [text_input, chatbot], [text_input, chatbot], queue=False).then(
            answer_generate, [chatbot, answer_time, point_clouds, conv_state], chatbot
        ).then(lambda x: x + 1, answer_time, answer_time)
        send_btn.click(user, [text_input, chatbot], [text_input, chatbot], queue=False).then(
            answer_generate, [chatbot, answer_time, point_clouds, conv_state], chatbot
        ).then(lambda x: x + 1, answer_time, answer_time)
        describe_btn.click(
            ask_point_description, inputs=[coord_x, coord_y, coord_z, chatbot], outputs=chatbot
        ).then(
            answer_generate, [chatbot, answer_time, point_clouds, conv_state], chatbot
        ).then(lambda x: x + 1, answer_time, answer_time)
        clear_btn.click(clear_conv, inputs=[chatbot, conv_state], outputs=[chatbot, answer_time], queue=False)
        device_dropdown.change(
            change_device,
            inputs=[device_dropdown, chatbot, point_clouds],
            outputs=[device_dropdown, chatbot, point_clouds],
        )

    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="RunsenXu/PointLLM_7B_v1.2")
    parser.add_argument("--data_path", type=str, default="data/chosen")
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--log_file", type=str, default="serving_workdirs/serving_log.txt")
    parser.add_argument("--tmp_dir", type=str, default="serving_workdirs/tmp")
    parser.add_argument("--port", type=int, default=7810)
    parser.add_argument(
        "--torch_dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"]
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    dtype_mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    args.torch_dtype = dtype_mapping[args.torch_dtype]
    args.device = _resolve_device(args.device)
    if args.device.type == "cpu" and args.torch_dtype == torch.float16:
        logging.warning("float16 not fully supported on CPU. Using float32 instead.")
        args.torch_dtype = torch.float32

    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)
    args.log_file = args.log_file.replace(
        ".txt", f"_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}.txt"
    )
    logging.basicConfig(
        filename=args.log_file,
        level=logging.WARNING,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    os.environ["GRADIO_TEMP_DIR"] = args.tmp_dir
    logging.warning("-----New Run-----")
    logging.warning(f"args: {args}")
    print("-----New Run-----")

    (
        model,
        tokenizer,
        point_backbone_config,
        keywords,
        mm_use_point_start_end,
        conv,
    ) = init_model(args)
    start_demo(
        args,
        model,
        tokenizer,
        point_backbone_config,
        keywords,
        mm_use_point_start_end,
        conv,
    )


if __name__ == "__main__":
    main()
