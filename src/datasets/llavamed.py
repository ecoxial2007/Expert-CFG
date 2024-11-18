
import os
import json
import random
import torch
from torch.utils.data import Dataset
from PIL import Image

class LlavaMedAlignDataset(Dataset):
    def __init__(self, annotation_file='', vis_root='', transform=None):
        """
        Initialize the dataset.

        Parameters:
            annotation_file (str): Path to the annotation file containing image IDs and captions.
            vis_root (str): Root directory where images are stored.
            transform (callable, optional): Optional transform to be applied on a PIL image.
        """
        with open(annotation_file, 'r') as file:
            self.annotation = json.load(file)

        self.vis_root = vis_root
        self.img_ids = {ann['id']: idx for idx, ann in enumerate(self.annotation)}
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

        img_file = ann['image']
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        

        first_instruction = ann['conversations'][0]['value'].replace('<image>', '').replace('\n', '').strip()
        questions = [first_instruction]
        answers = []

        for i, item in enumerate(ann["conversations"][1:]):
            if i % 2 == 0:  # assistant
                assistant_answer = item["value"]
                answers.append(assistant_answer)
            else:
                human_instruction = item["value"] + " "
                questions.append(human_instruction)

        return {
            "image": image,
            "question": questions,
            "answer": answers,
            "image_id": self.img_ids[ann["id"]]
        }



class LlavaMedVQADataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        assert len(examples) == 1, 'Phi-3-V only supports batch_size == 1'
        example = examples[0]

        image = example['image']
        idx = random.randint(0, len(example['question']) - 1)
        question = example['question'][idx]
        answer = example['answer'][idx]
        prompt_message = {
            'role': 'user',
            'content': f'<|image_1|>\n{question}',
        }

        prompt = self.processor.tokenizer.apply_chat_template(
            [prompt_message], tokenize=False, add_generation_prompt=True
        )
        answer = f'{answer}<|end|>\n<|endoftext|>'

        # mask questions for labels
        batch = self.processor(prompt, [image], return_tensors='pt')
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

# Usage example
if __name__ == "__main__":
    from torchvision import transforms

    from tqdm import tqdm

    dataset = LlavaMedAlignDataset(annotation_file='/workspace/LLaVA-Med/align_train.json', vis_root='/workspace/medical_conversation')


    for sample in tqdm(dataset):
        print(sample['question'], sample['answer'])
        len += 1

    print(len)