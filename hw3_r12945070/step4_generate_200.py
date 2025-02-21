import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from diffusers import StableDiffusionGLIGENTextImagePipeline
from transformers import Blip2Processor, Blip2ForConditionalGeneration


def generate_descriptions(input_file, image_dir, model_name="Salesforce/blip2-opt-2.7b",
                          output_file="result_200/blip2/blip2-descriptions.json"):
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
            f"{label} at location ({round(bbox[0], 1)}, {round(bbox[1], 1)}, {round(bbox[2], 1)}, {round(bbox[3], 1)})"
            for label, bbox in zip(image_entry['labels'], image_entry['bboxes'])
        ])

        output_entry = {
            **image_entry,
            "generated_text": generated_text,
            "prompt_w_label": f"{generated_text},focus on on {labels_str},focus on bboxes, high resolution, highly detailed",
            "prompt_w_suffix": f"{generated_text}, professional quality, highly detailed"

        }
        output_list.append(output_entry)

        '''
            "prompt_w_label": f"{generated_text}, Detailed scene with {labels_bboxes}, high resolution",
            "prompt_w_suffix": f"{generated_text}, Photorealistic scene with {labels_bboxes}, professional quality"
            output_entry["prompt_w_label"] = f"{generated_text}, {labels_str}, height: {image_entry['height']}, width: {image_entry['width']}"
            output_entry["prompt_w_suffix"] = f"{generated_text}, {labels_str}, height: {image_entry['height']}, width: {image_entry['width']}, HD quality, highly detailed"
        '''

    Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(output_list, f, indent=4)

    return output_file


def generate_images(input_file, image_dir):
    output_dirs = {
        "generated_text": "result_200/generated",
        "prompt_w_label": "result_200/prompt_w_label",
        "prompt_w_suffix": "result_200/prompt_w_suffix"
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
                    guidance_scale=11.0,  # 提高 guidance_scale
                    output_type="pil"
                )
                image = result.images[0]
                output_path = os.path.join(output_dirs[prompt_type], image_name)
                image.save(output_path)
            except Exception as e:
                print(f"Error in {prompt_type}: {e}")


def main():
    os.environ["TRANSFORMERS_CACHE"] = "/mnt/lab/1.projects/bestmark520/huggingface_cache"
    os.environ["HF_HOME"] = "/mnt/lab/1.projects/bestmark520/huggingface"

    input_file = "visualiztion_200.json"
    image_dir = "images_512x512"

    descriptions_file = generate_descriptions(input_file, image_dir)
    generate_images(descriptions_file, image_dir)


if __name__ == "__main__":
    main()