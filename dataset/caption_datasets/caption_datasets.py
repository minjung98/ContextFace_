import os
import cv2
import json
import random
import torch
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from tools.utils import DEFAULT_IMAGE_TOKEN


class LLaVAInstructDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, tokenizer, clip_image_encoder, epoch_samples=10000,precision="fp32", validation=False, random_sampling=True):

        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.precision = precision
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(clip_image_encoder)
        self.epoch_samples = epoch_samples
        self.validation = validation
        self.random_sampling = random_sampling

        # Defining paths
        mode = "train"
        self.base_dir = os.path.join(dataset_dir, "llava_dataset")
        self.image_folder = os.path.join(self.base_dir, f"{mode}2017")
        annotations_file = os.path.join(self.base_dir, "llava_instruct_150k.json")
        self.data_infos = self._load_annotations(annotations_file)
        print('\033[92m' + "----CAP-{}: LLaVA-Instruct VQA dataset initialized----".format(mode) + '\033[0m')

    def _load_annotations(self, ann_file):
        with open(ann_file, 'r') as f:
            all_data_infos = json.load(f)
        
        random.seed(42)
        if self.validation:
            data_infos = random.sample(all_data_infos, 200)
        else:
            val_indices = set(random.sample(range(len(all_data_infos)), 1500))
            remaining_data = [data for i, data in enumerate(all_data_infos) if i not in val_indices]
            data_infos = random.sample(remaining_data, 42196)
        
        print(f"Loaded {len(data_infos)} samples for {'validation' if self.validation else 'training'}")
        return data_infos

    def __len__(self):
        return len(self.data_infos)

    def create_conversations(self, conv_ann):
        # Preprocess:
        for sentence in conv_ann:
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = (sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip())
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>"
                    )
        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        if roles[conv_ann[0]["from"]] != conv.roles[0]:
            conv_ann = conv_ann[1:]

        for j, sentence in enumerate(conv_ann):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{j}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
        questions = conversations
        return questions, conversations

    def __getitem__(self, idx):
        ann_info = self.data_infos[idx] 
        image_path = os.path.join(self.image_folder, ann_info["image"])
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            raise ValueError(f"Could not load image at {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        clip_enc_images = self.clip_image_processor.preprocess(image, 
                                                               return_tensors="pt",
                                                               size=336)["pixel_values"][0]

        conv_ann = ann_info["conversations"]
        # print(f"conv_ann: {conv_ann}")
        questions, conversations = self.create_conversations(conv_ann)

        exps = None

        assert len(conversations) == 1

        return (image_path, clip_enc_images, conversations, exps, questions)
