# coding=utf-8
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Processor class for Phi3-V.
"""
import re
from typing import List, Optional, Union

import torch

import transformers
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PaddingStrategy, TextInput, TruncationStrategy
from src.highlighter_modules.utils import txt_highlight_mask
from transformers.utils import TensorType
from .image_processing_phi3_v import Phi3VImageProcessor

transformers.Phi3VImageProcessor = Phi3VImageProcessor


class Phi3VProcessor(ProcessorMixin):
    r"""
    Constructs a Phi3-V processor which wraps a Phi3-V image processor and a LLaMa tokenizer into a single processor.

    [`Phi3VProcessor`] offers all the functionalities of [`Phi3VImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~Phi3VProcessor.__call__`] and [`~Phi3VProcessor.decode`] for more information.

    Args:
        image_processor ([`Phi3VImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Phi3VImageProcessor"
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")
    special_image_token = "<|image|>"

    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'
        self.num_img_tokens = image_processor.num_img_tokens
        self.img_tokens = [f"<|image_{i + 1}|>" for i in range(1000000)]

    def __call__(
            self,
            text: Union[TextInput, List[TextInput]],
            images: ImageInput = None,
            padding: Union[bool, str, PaddingStrategy] = False,
            truncation: Union[bool, str, TruncationStrategy] = None,
            max_length=None,
            return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
            qs_highlighted_parts: List[str] = None
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        Phi3ImageProcessor's [`~Phi3ImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """
        if images is not None:
            image_inputs = self.image_processor(images, return_tensors=return_tensors)
        else:
            image_inputs = {}
        inputs = self._convert_images_texts_to_inputs(image_inputs, text, padding=padding, truncation=truncation,
                                                      max_length=max_length, return_tensors=return_tensors,
                                                      qs_highlighted_parts=qs_highlighted_parts)
        return inputs

    def calc_num_image_tokens(self, images: ImageInput):
        """ Calculate the number of image tokens for each image.
        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
        """
        return self.image_processor.calc_num_image_tokens(images)

    def calc_num_image_tokens_from_image_size(self, width, height):
        """ Calculate the number of image token for an image with given width and height.
        Args:
            width (`int`):
                Width of the image.
            height (`int`):
                Height of the image.
        """
        return self.image_processor.calc_num_image_tokens_from_image_size(width, height)

    @property
    def special_image_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.special_image_token)

    def get_special_image_token_id(self):
        return self.tokenizer.convert_tokens_to_ids(self.special_image_token)

    def tokenize_and_create_masks(self, pattern, texts, highlighted_parts):
        # 分割文本并保留图像标签
        prompt_chunks = re.split(pattern, texts)
        image_tags = re.findall(pattern, texts)

        input_ids = []
        highlight_attention_mask = []

        # 处理每个chunk和对应的图像标签
        for i, chunk in enumerate(prompt_chunks):
            # Tokenize the chunk
            chunk_ids = self.tokenizer.encode(chunk, add_special_tokens=False)

            # 生成对应chunk的高亮掩码
            chunk_highlight_mask = [0] * len(chunk_ids)
            for part in highlighted_parts:
                start = 0
                while start < len(chunk):
                    start_index = chunk.find(part, start)
                    if start_index == -1:
                        break
                    end_index = start_index + len(part)
                    # 将字符索引转换为token索引
                    start_token_idx = len(self.tokenizer.encode(chunk[:start_index], add_special_tokens=False))
                    end_token_idx = len(self.tokenizer.encode(chunk[:end_index], add_special_tokens=False))
                    for idx in range(start_token_idx, end_token_idx):
                        if idx < len(chunk_highlight_mask):
                            chunk_highlight_mask[idx] = 1
                    start = end_index

            # 添加chunk的token ids和高亮掩码到总列表
            input_ids.extend(chunk_ids)
            highlight_attention_mask.extend(chunk_highlight_mask)

            # 如果还有图像标签，处理图像标签
            if i < len(image_tags):
                image_tag_ids = self.tokenizer.encode(image_tags[i], add_special_tokens=False)
                input_ids.extend(image_tag_ids)
                # 图像标签不高亮
                highlight_attention_mask.extend([0] * len(image_tag_ids))

        return input_ids, highlight_attention_mask
        # 转换为torch tensor
        # input_ids_tensor = torch.tensor(input_ids).unsqueeze(0)
        # highlight_attention_mask_tensor = torch.tensor(highlight_attention_mask).unsqueeze(0)
        #
        #
        # return input_ids_tensor, highlight_attention_mask_tensor

    def _convert_images_texts_to_inputs(self, images, texts, padding=False, truncation=None, max_length=None,
                                        return_tensors=None, qs_highlighted_parts=[]):

        if not len(images):
            model_inputs = self.tokenizer(texts, return_tensors=return_tensors, padding=padding, truncation=truncation,
                                          max_length=max_length)
            return BatchFeature(data={**model_inputs})

        # print(texts, qs_highlighted_parts, '!!!!!this is in processing.py')

        pattern = r"<\|image_\d+\|>"

        if 'num_img_tokens' in images:
            num_img_tokens = images['num_img_tokens']
        else:
            assert 'num_crops' in images, 'num_crops must be provided in images if num_img_tokens is not provided'
            num_crops = images['num_crops']
            num_img_tokens = [_num_crops * self.num_img_tokens for _num_crops in num_crops]

        images, image_sizes = images['pixel_values'], images['image_sizes']

        # image_tags needs to start from 1 to n
        image_tags = re.findall(pattern, texts)
        # image_ids = [int(s.split("|")[1].split("_")[-1]) * -1 for s in image_tags]
        # image_ids_pad = [[iid]*num_img_tokens[i] for i, iid in enumerate(image_ids)]
        image_ids = [int(s.split("|")[1].split("_")[-1]) for s in image_tags]
        unique_image_ids = sorted(list(set(image_ids)))
        # image_ids must start from 1, and must be continuous int, e.g. [1, 2, 3], cannot be [1, 4, 5]
        # check the condition
        assert unique_image_ids == list(range(1, len(unique_image_ids) + 1)), f"image_ids must start from 1, and must be continuous int, e.g. [1, 2, 3], cannot be {unique_image_ids}"
        # total images must be the same as the number of image tags
        assert len(unique_image_ids) == len(
            images), f"total images must be the same as the number of image tags, got {len(unique_image_ids)} image tags and {len(images)} images"

        image_ids_pad = [[-iid] * num_img_tokens[iid - 1] for iid in image_ids]

        def insert_separator(X, sep_list):
            if len(X) > len(sep_list):
                sep_list.append([])
            return [ele for sublist in zip(X, sep_list) for ele in sublist]

        highlight_attention_mask = []
        prompt_chunks = []


        # Generate highlight masks for each chunk
        for chunk in re.split(pattern, texts):
            chunk_mask, _ = txt_highlight_mask(self.tokenizer, chunk, qs_highlighted_parts)
            highlight_attention_mask.append([0] + chunk_mask)
            a = self.tokenizer(chunk)
            prompt_chunks.append(a.input_ids)

        offset = 0
        input_ids = []
        combined_highlight_mask = []
        zero_mask_padding = [[0] * len(pad) for pad in
                             image_ids_pad]  # Create zero padding mask with the same length as image ids

        for tokens, mask in zip(insert_separator(prompt_chunks, image_ids_pad),
                                insert_separator(highlight_attention_mask,
                                                 zero_mask_padding)):  # Use zero_mask_padding here
            input_ids.extend(tokens[offset:])
            combined_highlight_mask.extend(mask[offset:])

        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = (input_ids > -1000000).to(torch.long)
        combined_highlight_mask = torch.tensor(combined_highlight_mask, dtype=torch.long).unsqueeze(0)

        return BatchFeature(data={"input_ids": input_ids,
                                  "attention_mask": attention_mask,
                                  "highlight_attention_mask": combined_highlight_mask,
                                  "pixel_values": images,
                                  "image_sizes": image_sizes})

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))