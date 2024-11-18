
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

class PubMedAlignDataset(Dataset):
    def __init__(self, annotation_file='', vis_root='', transform=None):
        """
        Initialize the dataset.

        Parameters:
            annotation_file (str): Path to the annotation file containing image IDs and captions.
            vis_root (str): Root directory where images are stored.
            transform (callable, optional): Optional transform to be applied on a PIL image.
        """
        with open(annotation_file, 'r') as file:
            self.annotation = json.load(file)['annotations']

        self.vis_root = vis_root
        self.img_ids = {}
        for idx, ann in enumerate(self.annotation):
            ann['id'] = idx
            self.img_ids[ann['id']] = idx
        self.transform = transform

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.annotation)

    def __getitem__(self, index):
        """
        Retrieve a sample from the dataset at the specified index.
        """
        ann = self.annotation[index]
        img_files = ann["image_id"][:2]
        images = []
        for img_file in img_files:
            image_path = os.path.join(self.vis_root,  img_file)
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)



        question = ann['conversations'][0]['value'].replace('<image>', '').replace('\n', '').strip()
        answer = ann['conversations'][1]['value'].replace('\n', '').strip()

        return {
            "images": images,
            "question": question,
            "answer": answer,
            "image_id": self.img_ids[ann["id"]]
        }


class PubMedVQADataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        assert len(examples) == 1, 'Phi-3-V only supports batch_size == 1'
        example = examples[0]

        images = example['images']
        image_references = ''.join([f"<|image_{i + 1}|>\n" for i in range(len(images))])

        question = example['question']
        answer = example['answer']
        prompt_message = {
            'role': 'user',
            'content': f"{image_references}{question}",
        }

        prompt = self.processor.tokenizer.apply_chat_template(
            [prompt_message], tokenize=False, add_generation_prompt=True
        )
        answer = f'{answer}<|end|>\n<|endoftext|>'

        # mask questions for labels
        batch = self.processor(prompt, images, return_tensors='pt')
        prompt_input_ids = batch['input_ids']
        # Do not add bos token to answer
        answer_input_ids = self.processor.tokenizer(
            answer, add_special_tokens=False, return_tensors='pt'
        )['input_ids']
        input_ids = torch.cat([prompt_input_ids, answer_input_ids], dim=1)
        ignore_index = -100
        labels = torch.cat(
            [
                torch.tensor([ignore_index] * len(prompt_input_ids[0])).unsqueeze(0),
                answer_input_ids,
            ],
            dim=1,
        )

        batch['input_ids'] = input_ids
        del batch['attention_mask']
        batch['labels'] = labels

        return batch
