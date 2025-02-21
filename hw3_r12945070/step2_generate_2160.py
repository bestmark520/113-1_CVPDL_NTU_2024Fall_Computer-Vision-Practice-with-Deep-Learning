import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from diffusers import StableDiffusionGLIGENTextImagePipeline
from transformers import Blip2Processor, Blip2ForConditionalGeneration


# Function to generate descriptions using BLIP-2
def generate_descriptions(input_file, image_dir, model_name="Salesforce/blip2-opt-2.7b",
                          output_file="result/blip2/blip2-descriptions.json"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        revision="51572668da0eb669e01a189dc22abe6088589a24",
        load_in_8bit=True,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )

    with open(input_file, "r") as f:
        input_data = json.load(f)

    output_list = []

    for image_entry in tqdm(input_data, desc="Generating Descriptions"):
        image_file_path = os.path.join(image_dir, image_entry["image"])
        image_file = Image.open(image_file_path)

        inputs = processor(images=image_file, return_tensors="pt").to(
            device=device, dtype=torch.bfloat16
        )

        generated_ids = model.generate(
            **inputs,
            max_length=50,
            num_beams=4,
            early_stopping=True
        )
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        labels_str = ", ".join(set(image_entry.get("labels", [])))
        labels_bboxes = ", ".join([
            f"{label} at location {bbox}"
            for label, bbox in zip(image_entry['labels'], image_entry['bboxes'])
        ])

        output_entry = {
            **image_entry,
            "generated_text": generated_text,
            "prompt_w_label": f"{generated_text}, Detailed scene with {labels_bboxes}, high resolution",
            "prompt_w_suffix": f"{generated_text}, Photorealistic scene with {labels_bboxes}, professional quality"
        }
        output_list.append(output_entry)

    Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(output_list, f, indent=4)

    return output_file

# New function to generate images with different prompts
def generate_images_text(input_file, output_dir):
    """
    這個新函數會根據不同的prompt類型生成圖片並保存。
    """
    output_dirs = {
        "generated_text": "result/text/generated",
        "prompt_w_label": "result/text/prompt_w_label",
        "prompt_w_suffix": "result/text/prompt_w_suffix"
    }

    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionGLIGENTextImagePipeline.from_pretrained(
        "anhnct/Gligen_Text_Image"
    ).to(device)

    with open(input_file, "r") as f:
        data = json.load(f)

    for entry in tqdm(data, desc="Generating Images with GLIGEN"):
        image_name = os.path.splitext(entry['image'])[0] + ".jpeg"

        width, height = entry['width'], entry['height']
        gligen_boxes = torch.tensor([
            [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
            for bbox in entry['bboxes']
        ]).to(device)
        gligen_phrases = entry['labels']

        for prompt_type in ['generated_text', 'prompt_w_label', 'prompt_w_suffix']:
            prompt = entry[prompt_type]
            try:
                result = pipe(
                    prompt=prompt,
                    gligen_phrases=gligen_phrases,
                    gligen_boxes=gligen_boxes,
                    height=512,
                    width=512,
                    num_inference_steps=75,
                    guidance_scale=11.0,  # 提高 guidance_scale
                    output_type="pil"
                )
                image = result.images[0]
                output_path = os.path.join(output_dirs[prompt_type], image_name)
                image.save(output_path)
            except Exception as e:
                print(f"Error in {prompt_type}: {e}")

# Function to generate images from descriptions and save them
def generate_images(input_file, image_dir):
    output_dirs = {
        "generated_text": "result/generated_text",
        "prompt_w_label": "result/prompt_w_label",
        "prompt_w_suffix": "result/prompt_w_suffix"
    }
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionGLIGENTextImagePipeline.from_pretrained(
        "anhnct/Gligen_Text_Image"
    ).to(device)

    with open(input_file, "r") as f:
        data = json.load(f)

    for entry in tqdm(data, desc="Generating Images"):
        image_name = os.path.splitext(entry['image'])[0] + ".jpeg"

        width, height = entry['width'], entry['height']
        gligen_boxes = torch.tensor([
            [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
            for bbox in entry['bboxes']
        ]).to(device)
        gligen_phrases = entry['labels']

        for prompt_type in ['generated_text', 'prompt_w_label', 'prompt_w_suffix']:
            prompt = entry[prompt_type]
            try:
                result = pipe(
                    prompt=prompt,
                    gligen_phrases=gligen_phrases,
                    gligen_boxes=gligen_boxes,
                    height=512,
                    width=512,
                    num_inference_steps=75,
                    guidance_scale=12.0,  # 提高 guidance_scale
                    output_type="pil"
                )
                image = result.images[0]
                output_path = os.path.join(output_dirs[prompt_type], image_name)
                image.save(output_path)
            except Exception as e:
                print(f"Error in {prompt_type}: {e}")


# Main function to call the entire process
def main():
    os.environ["TRANSFORMERS_CACHE"] = "/mnt/lab/1.projects/bestmark520/huggingface_cache"
    os.environ["HF_HOME"] = "/mnt/lab/1.projects/bestmark520/huggingface"

    input_file = "label.json"
    image_dir = "images_512x512"

    descriptions_file = generate_descriptions(input_file, image_dir)
    # generate_images_text(descriptions_file, "result/gligen/text/generated")
    generate_images(descriptions_file, image_dir)
    
if __name__ == "__main__":
    main()
