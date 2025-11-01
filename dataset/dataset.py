import numpy as np
import torch

from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from dataset.caption_datasets.LLavaInstruct_vqa_ds import LLaVAInstructDataset
from dataset.category_datasets.CAER_S_category_ds import CAER_S_Cate_Dataset
from dataset.generation_datasets.CAER_S_generation_quote_ds import CAER_S_Quote_Gen_Dataset
from dataset.generation_datasets.CAER_S_generation_situation_ds import CAER_S_Situation_Gen_Dataset
from dataset.reasoning_datasets.CAER_S_reasoning_ds import CAER_S_Reason_Dataset
from dataset.visual_dataset.CAER_S_visual_ds import CAER_S_Visual_Dataset
from tools.utils import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN


class HybridDatasetBase(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, tokenizer, clip_image_encoder, dataset, datasets_config, epoch_samples=10000, batch_size=4, precision="fp32", validation=False, sample_rate=None):
        self.dataset_dir = dataset_dir
        self.tokenizer = tokenizer
        self.clip_image_encoder = clip_image_encoder
        self.dataset = dataset
        self.datasets_config = datasets_config
        self.epoch_samples = epoch_samples
        self.batch_size = batch_size
        self.precision = precision
        # task당 데이터셋이 여러개 있는 경우
        self.dataset_list = dataset.split("||")
        # sample_rate이라는 것은 하나의 task 내에서 각 데이터셋의 비율을 정해주는 것(지정 안 하면 그냥 모두 같은 비율로 뽑음)
        self.sample_rate = np.array(sample_rate or [1.0] * len(self.dataset_list))
        self.sample_rate /= self.sample_rate.sum()
        self.validation = validation
        self.all_datasets = self.create_datasets()
    
    def create_datasets(self):
        datasets = []
        for ds in self.dataset_list:
            dataset_cls = self.datasets_config.get(ds)
            print("ds: ", ds)
            if dataset_cls:
                datasets.append(
                    dataset_cls(
                        self.dataset_dir, self.tokenizer, self.clip_image_encoder, self.epoch_samples, self.precision, self.validation, self.sample_rate)
                    )
        print(f"datasets: {datasets}")
        return datasets

    def __len__(self):
        return self.epoch_samples

    def __getitem__(self, idx):
        dataset_idx = np.random.choice(len(self.dataset_list), p=self.sample_rate)
        selected_dataset = self.all_datasets[dataset_idx]
        data = selected_dataset[idx]
        print("idx: ", idx)
        return (*data,)

class HybridGenDataset(HybridDatasetBase):
    def __init__(self, dataset_dir, tokenizer, clip_image_encoder, epoch_samples=10000, batch_size=4,
                 precision="fp32",dataset="CAER_S_generation_quote_dataset||CAER_S_generation_situation_dataset", validation=False, sample_rate=None):
        datasets_config = {"CAER_S_generation_quote_dataset": CAER_S_Quote_Gen_Dataset,
                           "CAER_S_generation_situation_dataset": CAER_S_Situation_Gen_Dataset}
        super().__init__(
            dataset_dir=dataset_dir, tokenizer = tokenizer, clip_image_encoder = clip_image_encoder, 
            dataset=dataset, datasets_config=datasets_config, epoch_samples=epoch_samples, batch_size=batch_size,
            precision=precision, validation=validation, sample_rate=sample_rate)

class HybridCateDataset(HybridDatasetBase):
    def __init__(self, dataset_dir, tokenizer, clip_image_encoder, epoch_samples=10000, batch_size=4,
                 precision="fp32", dataset="CAER_S_category_dataset", validation=False, sample_rate=None):
        datasets_config = {"CAER_S_category_dataset": CAER_S_Cate_Dataset
                           }
        super().__init__(
            dataset_dir=dataset_dir, tokenizer = tokenizer, clip_image_encoder = clip_image_encoder, 
            dataset=dataset, datasets_config=datasets_config, epoch_samples=epoch_samples, batch_size=batch_size,
            precision=precision, validation=validation, sample_rate=sample_rate)

class HybridCapDataset(HybridDatasetBase):
    def __init__(self, dataset_dir, tokenizer, clip_image_encoder,epoch_samples=10000, batch_size=4,
                 precision="fp32", dataset="llava_instruct_150k", validation=False, sample_rate=None):
        datasets_config = {"llava_instruct_150k": LLaVAInstructDataset}
        super().__init__(
            dataset_dir=dataset_dir, tokenizer = tokenizer, clip_image_encoder = clip_image_encoder, 
            dataset=dataset, datasets_config=datasets_config, epoch_samples=epoch_samples, batch_size=batch_size,
            precision=precision, validation=validation, sample_rate=sample_rate)
        
class HybridReasonDataset(HybridDatasetBase):
    def __init__(self, dataset_dir, tokenizer, clip_image_encoder,epoch_samples=10000, batch_size=4,
                 precision="fp32", dataset="CAER_S_reasoning_dataset", validation=False, sample_rate=None):
        datasets_config = {"CAER_S_reasoning_dataset": CAER_S_Reason_Dataset}
        super().__init__(
            dataset_dir=dataset_dir, tokenizer = tokenizer, clip_image_encoder = clip_image_encoder, 
            dataset=dataset, datasets_config=datasets_config, epoch_samples=epoch_samples, batch_size=batch_size,
            precision=precision, validation=validation, sample_rate=sample_rate)

class HybridVisualDataset(HybridDatasetBase):
    def __init__(self, dataset_dir, tokenizer, clip_image_encoder,epoch_samples=10000, batch_size=4,
                 precision="fp32", dataset="CAER_S_visual_dataset", validation=False, sample_rate=None):
        datasets_config = {"CAER_S_visual_dataset": CAER_S_Visual_Dataset}
        super().__init__(
            dataset_dir=dataset_dir, tokenizer = tokenizer, clip_image_encoder = clip_image_encoder, 
            dataset=dataset, datasets_config=datasets_config, epoch_samples=epoch_samples, batch_size=batch_size,
            precision=precision, validation=validation, sample_rate=sample_rate)
        
def custom_collate_fn(batch, tokenizer=None, use_mm_start_end=True, inference=False, local_rank=-1):
    # Initializing lists and counters
    image_path_list, clip_enc_image_list = [], []
    conversation_list, gt_exps = [], []
    questions_list = []
    offset_list, inferences = [0], []
    cnt = 0

    # Iterating through the batch
    for (image_path, clip_enc_image, conversations, exps, questions) in batch:
        image_path_list.append(image_path)
        clip_enc_image_list.append(clip_enc_image)
        conversation_list.extend(conversations)
        gt_exps.append(torch.empty(0,103) if exps is None else exps.float())
        questions_list.append(questions)
        offset_list.append(cnt := cnt + len(conversations))
        inferences.append(inference)

    # Handling the conversation list
    if use_mm_start_end:
        replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        conversation_list = [conv.replace(DEFAULT_IMAGE_TOKEN, replace_token) for conv in conversation_list]
    # Tokenizing and padding input ids
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversation_list],
        batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    # Preparing targets and handling conversation types
    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()
    # conv_type == "llava_llama2"
    # sep = conv.sep + conv.roles[1] + ": "
    sep = "[/INST] "
    sep2 = conv.sep2
    for conversation, target in zip(conversation_list, targets):
        _process_conversation(conversation, target, tokenizer, sep, sep2)

    # Adjusting for inferences
    if not inferences[0]:
        truncate_len = tokenizer.model_max_length - 575
        if input_ids.shape[1] > truncate_len:
            input_ids, targets, attention_masks = map(
                lambda x: x[:, :truncate_len], [input_ids, targets, attention_masks]
                )
    return {
        "image_paths": image_path_list,
        "clip_enc_images": torch.stack(clip_enc_image_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "gt_exps": gt_exps,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }


def _process_conversation(conversation, target, tokenizer, sep, sep2):
    total_len = target.ne(tokenizer.pad_token_id).sum().item()
    rounds = conversation.split(sep2)
    
    cur_len = 1
    target[:cur_len] = IGNORE_INDEX
    for rou in rounds:
        if not rou:
            break

        parts = rou.split(sep)
        assert len(parts) == 2, (len(parts), rou)
        parts[0] += sep

        if DEFAULT_IMAGE_TOKEN in conversation:
            round_len = len(tokenizer_image_token(rou, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
        else:
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

        target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
        cur_len += round_len

    target[cur_len:] = IGNORE_INDEX
    if cur_len < tokenizer.model_max_length:
        assert cur_len == total_len
