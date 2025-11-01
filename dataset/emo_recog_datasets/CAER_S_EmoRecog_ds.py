import os
import cv2
import json
import random
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
from model.llava import conversation as conversation_lib
from tools.utils import DEFAULT_IMAGE_TOKEN
from PIL import Image

class CAER_S_EmoRecog_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, tokenizer, clip_image_encoder, epoch_samples=10000, precision="fp32", validation=False, random_sampling=False):

        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.precision = precision
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(clip_image_encoder)
        self.epoch_samples = epoch_samples
        self.validation = validation
        self.random_sampling = random_sampling

        # Defining paths
        mode = "Test" if validation else "Train"
        self.base_dir = os.path.join(dataset_dir, "CAER-S_dataset_bbox")
        self.image_folder = os.path.join(self.base_dir, "images", mode)
        json_dir = "/home/minjung2/groundingLMM/final_dataset/CAER-S"
        annotations_file = os.path.join(json_dir, mode,"CAER-S_emotion_recognition_shuffled.json")
        self.data_infos = self._load_annotations(annotations_file)
        print('\033[92m' + "----emotion recognition -{}: CAER-S emotion recognition dataset initialized----".format(mode) + '\033[0m')
    
    def _load_annotations(self, ann_file):
        with open(ann_file, 'r') as f:
            all_data_infos = json.load(f)
        
        random.seed(42)
        if self.validation:
            data_infos = random.sample(all_data_infos, 100)
        else:
            data_infos = all_data_infos
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
        # image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.open(image_path)
        clip_enc_images = self.clip_image_processor.preprocess(image, 
                                                               return_tensors="pt",
                                                               size=336)["pixel_values"][0]
    
        conv_ann = ann_info["conversations"]
        # print(conv_ann)
        questions, conversations = self.create_conversations(conv_ann)

        exps = None

        assert len(conversations) == 1

        return (image_path, clip_enc_images, conversations, exps, questions)
