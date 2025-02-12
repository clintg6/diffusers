import os
import sys

import torch
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight, float8_weight_only

from tqdm import tqdm
from diffusers.utils import load_image
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting,
    AutoPipelineForText2Image,
    ControlNetModel,
    LCMScheduler,
    StableDiffusionAdapterPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLAdapterPipeline,
    StableDiffusionXLControlNetPipeline,
    T2IAdapter,
    WuerstchenCombinedPipeline,
    FluxPipeline,
)

sys.path.append(".")

from utils import (  # noqa: E402
    BASE_PATH,
    PROMPT,
    BenchmarkInfo,
    benchmark_fn,
    bytes_to_giga_bytes,
    flush,
    generate_csv_dict,
    write_to_csv,
)

torch.set_float32_matmul_precision("high")
#torch._inductor.config.conv_1x1_as_mm = True
#torch._inductor.config.coordinate_descent_tuning = True
#torch._inductor.config.epilogue_fusion = True
#torch._inductor.config.coordinate_descent_check_all_directions = True
#torch._inductor.config.freezing = True

RESOLUTION_MAPPING = {
    "runwayml/stable-diffusion-v1-5": (512, 512),
    "lllyasviel/sd-controlnet-canny": (512, 512),
    "diffusers/controlnet-canny-sdxl-1.0": (1024, 1024),
    "TencentARC/t2iadapter_canny_sd14v1": (512, 512),
    "TencentARC/t2i-adapter-canny-sdxl-1.0": (1024, 1024),
    "stabilityai/stable-diffusion-2-1": (768, 768),
    "stabilityai/stable-diffusion-xl-base-1.0": (1024, 1024),
    "stabilityai/stable-diffusion-xl-refiner-1.0": (1024, 1024),
    "stabilityai/sdxl-turbo": (512, 512),
    "etri-vilab/koala-1b": (1024, 1024),
    "black-forest-labs/FLUX.1-dev": (1024,1024),
}


class BaseBenchmak:
    pipeline_class = None

    def __init__(self, args):
        super().__init__()

    def run_inference(self, args):
        raise NotImplementedError

    def benchmark(self, args):
        raise NotImplementedError

    def get_result_filepath(self, args):
        pipeline_class_name = str(self.pipe.__class__.__name__)
        name = (
            args.ckpt.replace("/", "_")
            + "_"
            + pipeline_class_name
            + f"-bs@{args.batch_size}-steps@{args.num_inference_steps}-mco@{args.model_cpu_offload}-compile@{args.run_compile}.csv"
        )
        filepath = os.path.join(BASE_PATH, name)
        return filepath


class TextToImageBenchmark(BaseBenchmak):
    pipeline_class = AutoPipelineForText2Image

    def __init__(self, args):
        if args.dtype == "FP16":
            pipe = self.pipeline_class.from_pretrained(args.ckpt, torch_dtype=torch.float16)
        elif args.dtype == "BF16" or args.dtype == "FP8":
            pipe = self.pipeline_class.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16)
        elif args.dtype == "FP32":
            pipe = self.pipeline_class.from_pretrained(args.ckpt, torch_dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported data type: {args.dtype}. "
                        f"Supported types are: BF16, FP32, FP16.")
        pipe = pipe.to("cuda")
        #pipe.enable_attention_slicing()
        #pipe.enable_vae_slicing()
        #pipe.enable_vae_tiling()

        if args.dtype == "FP8":
            quantize_(pipe.transformer, float8_dynamic_activation_float8_weight(torch.float8_e4m3fnuz, torch.float8_e4m3fnuz))
            
        if args.cache_opt:
            apply_cache_on_pipe(pipe, residual_diff_threshold=0.08)

        if args.run_compile:
            if isinstance(pipe, FluxPipeline):
                #pipe.transformer.to(memory_format=torch.channels_last)
                #pipe.vae.to(memory_format=torch.channels_last)
                print("Run torch compile")
                if args.cache_opt or args.dtype == "FP8":
                    pipe.transformer = torch.compile(pipe.transformer, mode="default", fullgraph=False)
                else:
                    pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
                    #pipe.vae.decode = torch.compile(pipe.vae.decode, mode="reduce-overhead", fullgraph=True)

            elif not isinstance(pipe, WuerstchenCombinedPipeline):
                pipe.unet.to(memory_format=torch.channels_last)
                print("Run torch compile")
                pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

                if hasattr(pipe, "movq") and getattr(pipe, "movq", None) is not None:
                    pipe.movq.to(memory_format=torch.channels_last)
                    pipe.movq = torch.compile(pipe.movq, mode="reduce-overhead", fullgraph=True)
            else:
                print("Run torch compile")
                pipe.decoder = torch.compile(pipe.decoder, mode="reduce-overhead", fullgraph=True)
                pipe.vqgan = torch.compile(pipe.vqgan, mode="reduce-overhead", fullgraph=True)

        pipe.scheduler = pipe.scheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    def run_inference(self, pipe, args):
        #print(f"****the resolution for the args are {RESOLUTION_MAPPING[args.ckpt]}****")
        print(f"********height: {args.height}, width: {args.width}**********")
        image_output = pipe(
            prompt=PROMPT,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.batch_size,
            height=args.height,
            width=args.width,
        )
        # Print the type of the image object
        images = image_output.images
        print(f"**********the prompt is {PROMPT}**********")
        print(f"Type of generated image: {type(image_output)}")

        # Inspect the attributes of the image object
        print(f"Attributes of generated image: {dir(image_output)}")
        if isinstance(images, list) and len(images) > 0:
            print(f"******************Generated image shape: {images[0].size}*****************")
        else:
            print("Generated images is not a list or is empty.")

    def benchmark(self, args):
        flush()

        print(f"[INFO] {self.pipe.__class__.__name__}: Running benchmark with: {vars(args)}\n")

        time = benchmark_fn(self.run_inference, self.pipe, args)  # in seconds.
        memory = bytes_to_giga_bytes(torch.cuda.max_memory_allocated())  # in GBs.
        benchmark_info = BenchmarkInfo(time=time, memory=memory)

        pipeline_class_name = str(self.pipe.__class__.__name__)
        flush()
        csv_dict = generate_csv_dict(
            pipeline_cls=pipeline_class_name, ckpt=args.ckpt, args=args, benchmark_info=benchmark_info
        )
        filepath = self.get_result_filepath(args)
        write_to_csv(filepath, csv_dict)
        print(f"Logs written to: {filepath}")
        flush()


class TextToImageBenchmark_multi_image(BaseBenchmak):
    pipeline_class = AutoPipelineForText2Image

    def __init__(self, args):
        if args.dtype == "FP16":
            pipe = self.pipeline_class.from_pretrained(args.ckpt, torch_dtype=torch.float16)
        elif args.dtype == "BF16":
            pipe = self.pipeline_class.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16)
        elif args.dtype == "FP32":
            pipe = self.pipeline_class.from_pretrained(args.ckpt, torch_dtype=torch.float32)
        else:
            raise TypeError(f"Unsupported data type: {args.dtype}. "
                        f"Supported types are: BF16, FP32, FP16.")
        pipe = pipe.to("cuda")

        if args.run_compile:
            if isinstance(pipe, FluxPipeline):
                #pipe.transformer.to(memory_format=torch.channels_last)
                pipe.vae.to(memory_format=torch.channels_last)
                print("Run torch compile")
                pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead", fullgraph=True)
                #pipe.vae.decode = torch.compile(pipe.vae.decode, mode="reduce-overhead", fullgraph=True)

            elif not isinstance(pipe, WuerstchenCombinedPipeline):
                pipe.unet.to(memory_format=torch.channels_last)
                print("Run torch compile")
                pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

                if hasattr(pipe, "movq") and getattr(pipe, "movq", None) is not None:
                    pipe.movq.to(memory_format=torch.channels_last)
                    pipe.movq = torch.compile(pipe.movq, mode="reduce-overhead", fullgraph=True)
            else:
                print("Run torch compile")
                pipe.decoder = torch.compile(pipe.decoder, mode="reduce-overhead", fullgraph=True)
                pipe.vqgan = torch.compile(pipe.vqgan, mode="reduce-overhead", fullgraph=True)

        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    def run_inference(self, pipe, args):
        total_images = args.no_of_images
        batch_size = args.batch_size
        num_batches = (total_images + batch_size - 1) // batch_size  # Calculate the number of batches needed

        all_images = []

        for _ in tqdm(range(num_batches), desc="Processing Batches"):
            current_batch_size = min(batch_size, total_images - len(all_images))
            image_output = pipe(
                prompt=PROMPT,
                num_inference_steps=args.num_inference_steps,
                num_images_per_prompt=current_batch_size,
                height=args.resolution,
                width=args.resolution,
            )
            images = image_output.images
            all_images.extend(images)

            # Print the type and attributes of the image object
            '''
            print(f"**********the prompt is {PROMPT}**********")
            print(f"Type of generated image: {type(image_output)}")
            print(f"Attributes of generated image: {dir(image_output)}")
            if isinstance(images, list) and len(images) > 0:
                print(f"******************Generated image size: {images[0].size}*****************")
            else:
                print("Generated images is not a list or is empty.")
            '''
        # Now all_images contains 1000 images
        print(f"Total images generated: {len(all_images)}")

    def benchmark(self, args):
        flush()

        print(f"[INFO] {self.pipe.__class__.__name__}: Running benchmark with: {vars(args)}\n")

        time = benchmark_fn(self.run_inference, self.pipe, args)  # in seconds.
        memory = bytes_to_giga_bytes(torch.cuda.max_memory_allocated())  # in GBs.
        benchmark_info = BenchmarkInfo(time=time, memory=memory)

        pipeline_class_name = str(self.pipe.__class__.__name__)
        flush()
        csv_dict = generate_csv_dict(
            pipeline_cls=pipeline_class_name, ckpt=args.ckpt, args=args, benchmark_info=benchmark_info
        )
        filepath = self.get_result_filepath(args)
        write_to_csv(filepath, csv_dict)
        print(f"Logs written to: {filepath}")
        flush()

class TurboTextToImageBenchmark(TextToImageBenchmark):
    def __init__(self, args):
        super().__init__(args)

    def run_inference(self, pipe, args):
        _ = pipe(
            prompt=PROMPT,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.batch_size,
            guidance_scale=0.0,
        )


class LCMLoRATextToImageBenchmark(TextToImageBenchmark):
    lora_id = "latent-consistency/lcm-lora-sdxl"

    def __init__(self, args):
        super().__init__(args)
        self.pipe.load_lora_weights(self.lora_id)
        self.pipe.fuse_lora()
        self.pipe.unload_lora_weights()
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)

    def get_result_filepath(self, args):
        pipeline_class_name = str(self.pipe.__class__.__name__)
        name = (
            self.lora_id.replace("/", "_")
            + "_"
            + pipeline_class_name
            + f"-bs@{args.batch_size}-steps@{args.num_inference_steps}-mco@{args.model_cpu_offload}-compile@{args.run_compile}.csv"
        )
        filepath = os.path.join(BASE_PATH, name)
        return filepath

    def run_inference(self, pipe, args):
        _ = pipe(
            prompt=PROMPT,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.batch_size,
            guidance_scale=1.0,
        )

    def benchmark(self, args):
        flush()

        print(f"[INFO] {self.pipe.__class__.__name__}: Running benchmark with: {vars(args)}\n")

        time = benchmark_fn(self.run_inference, self.pipe, args)  # in seconds.
        memory = bytes_to_giga_bytes(torch.cuda.max_memory_allocated())  # in GBs.
        benchmark_info = BenchmarkInfo(time=time, memory=memory)

        pipeline_class_name = str(self.pipe.__class__.__name__)
        flush()
        csv_dict = generate_csv_dict(
            pipeline_cls=pipeline_class_name, ckpt=self.lora_id, args=args, benchmark_info=benchmark_info
        )
        filepath = self.get_result_filepath(args)
        write_to_csv(filepath, csv_dict)
        print(f"Logs written to: {filepath}")
        flush()


class ImageToImageBenchmark(TextToImageBenchmark):
    pipeline_class = AutoPipelineForImage2Image
    url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/benchmarking/1665_Girl_with_a_Pearl_Earring.jpg"
    image = load_image(url).convert("RGB")

    def __init__(self, args):
        super().__init__(args)
        self.image = self.image.resize(RESOLUTION_MAPPING[args.ckpt])

    def run_inference(self, pipe, args):
        _ = pipe(
            prompt=PROMPT,
            image=self.image,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.batch_size,
        )


class TurboImageToImageBenchmark(ImageToImageBenchmark):
    def __init__(self, args):
        super().__init__(args)

    def run_inference(self, pipe, args):
        _ = pipe(
            prompt=PROMPT,
            image=self.image,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.batch_size,
            guidance_scale=0.0,
            strength=0.5,
        )


class InpaintingBenchmark(ImageToImageBenchmark):
    pipeline_class = AutoPipelineForInpainting
    mask_url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/benchmarking/overture-creations-5sI6fQgYIuo_mask.png"
    mask = load_image(mask_url).convert("RGB")

    def __init__(self, args):
        super().__init__(args)
        self.image = self.image.resize(RESOLUTION_MAPPING[args.ckpt])
        self.mask = self.mask.resize(RESOLUTION_MAPPING[args.ckpt])

    def run_inference(self, pipe, args):
        _ = pipe(
            prompt=PROMPT,
            image=self.image,
            mask_image=self.mask,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.batch_size,
        )


class IPAdapterTextToImageBenchmark(TextToImageBenchmark):
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_neg_embed.png"
    image = load_image(url)

    def __init__(self, args):
        pipe = self.pipeline_class.from_pretrained(args.ckpt, torch_dtype=torch.float16).to("cuda")
        pipe.load_ip_adapter(
            args.ip_adapter_id[0],
            subfolder="models" if "sdxl" not in args.ip_adapter_id[1] else "sdxl_models",
            weight_name=args.ip_adapter_id[1],
        )

        if args.run_compile:
            pipe.unet.to(memory_format=torch.channels_last)
            print("Run torch compile")
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    def run_inference(self, pipe, args):
        _ = pipe(
            prompt=PROMPT,
            ip_adapter_image=self.image,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.batch_size,
        )


class ControlNetBenchmark(TextToImageBenchmark):
    pipeline_class = StableDiffusionControlNetPipeline
    aux_network_class = ControlNetModel
    root_ckpt = "runwayml/stable-diffusion-v1-5"

    url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/benchmarking/canny_image_condition.png"
    image = load_image(url).convert("RGB")

    def __init__(self, args):
        aux_network = self.aux_network_class.from_pretrained(args.ckpt, torch_dtype=torch.float16)
        pipe = self.pipeline_class.from_pretrained(self.root_ckpt, controlnet=aux_network, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

        if args.run_compile:
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.controlnet.to(memory_format=torch.channels_last)

            print("Run torch compile")
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            pipe.controlnet = torch.compile(pipe.controlnet, mode="reduce-overhead", fullgraph=True)

        self.image = self.image.resize(RESOLUTION_MAPPING[args.ckpt])

    def run_inference(self, pipe, args):
        _ = pipe(
            prompt=PROMPT,
            image=self.image,
            num_inference_steps=args.num_inference_steps,
            num_images_per_prompt=args.batch_size,
        )


class ControlNetSDXLBenchmark(ControlNetBenchmark):
    pipeline_class = StableDiffusionXLControlNetPipeline
    root_ckpt = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self, args):
        super().__init__(args)


class T2IAdapterBenchmark(ControlNetBenchmark):
    pipeline_class = StableDiffusionAdapterPipeline
    aux_network_class = T2IAdapter
    root_ckpt = "CompVis/stable-diffusion-v1-4"

    url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/benchmarking/canny_for_adapter.png"
    image = load_image(url).convert("L")

    def __init__(self, args):
        aux_network = self.aux_network_class.from_pretrained(args.ckpt, torch_dtype=torch.float16)
        pipe = self.pipeline_class.from_pretrained(self.root_ckpt, adapter=aux_network, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")

        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

        if args.run_compile:
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.adapter.to(memory_format=torch.channels_last)

            print("Run torch compile")
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            pipe.adapter = torch.compile(pipe.adapter, mode="reduce-overhead", fullgraph=True)

        self.image = self.image.resize(RESOLUTION_MAPPING[args.ckpt])


class T2IAdapterSDXLBenchmark(T2IAdapterBenchmark):
    pipeline_class = StableDiffusionXLAdapterPipeline
    root_ckpt = "stabilityai/stable-diffusion-xl-base-1.0"

    url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/benchmarking/canny_for_adapter_sdxl.png"
    image = load_image(url)

    def __init__(self, args):
        super().__init__(args)

