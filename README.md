# Negative Prompt Generator
> [!CAUTION]
> This project is a work in progress.

The trained model is a finetuned [GPT2](https://github.com/openai/gpt-2) text generation model that takes a positive prompt as an input, and outputs a negative prompt that is supposed to match the input prompt.

However, the results are very random and they are mostly unrelated to the prompt, and sometime they even output a positive prompt.

As the results are not good, I have not yet cleaned up the project and made it presentable.

Use this mostly for your own curiosity or experimentation.

## Model Weights
Model weights and checkpoint can be downloaded from my [huggingface](https://huggingface.co/mnemic/NegativePromptGenerator/tree/main).

The [weights.pt](https://huggingface.co/mnemic/NegativePromptGenerator/blob/main/weights.pt) contains the weights. Use them freely for inference.

The [model.safetensors](https://huggingface.co/mnemic/NegativePromptGenerator/blob/main/model.safetensors) contains the checkpoint. Use this to continue training.

> [!CAUTION]
> These files are not included in this repository due to file size. They should be manually downloaded from the huggingface space and placed in the folder:
>
> `models/negativeprompt_054/`

## Training
The model was trained using positive prompt and negative prompt pairs scraped from the CivitAI API. In total a set of 18k pairs was trained on.

The final trained epoch was #54.

The [tokenized data](https://huggingface.co/mnemic/NegativePromptGenerator/blob/main/tokenized_data.pt) can be downloaded from [huggingface](https://huggingface.co/mnemic/NegativePromptGenerator/tree/main). Place this in the: `dataset/`-folder

1. Create a virtual environment. You can use the [venv_create.bat](https://github.com/MNeMoNiCuZ/create_venv) to easily set an virtual environment up.
2. Install the required libraries using `pip install -r requirements.txt` from inside the environment.
3. Install [PyTorch](https://pytorch.org/get-started/locally/) according to your GPU's capabilities.
4. Run `py finetune_gpt2.py` to continue training, or train a model from scratch.

Be sure to edit the script and make sure the TOKENIZED_DATA_PATH and CHECKPOINT_PATH are correct (if you are resuming training). Leave it empty to start a new training from scratch.

## Local Inference
1. Ensure you have the project set up according to steps 1-3 in the Training section.
2. Run `py inference.py "Insert Your Prompt Here"` to get a negative prompt returned. You could also update the script to make it save the output, or generate a bunch of files with outputs.

## Comfy Node
To test the model in a realistic environment I created a ComfyUI node for the inference. It's a simple implementation but it works fine.

I have included it in my [ComfyUI Node Pack](https://github.com/MNeMoNiCuZ/ComfyUI-mnemic-nodes).

![image](https://github.com/MNeMoNiCuZ/NegativePromptGenerator/assets/60541708/6b7614e6-2510-4b02-8696-8a6d7e1c59d3)

## Results & Conclusions
**Is this better than having no negative prompt?**

> Yes, sometimes it is. It's a bit random, but I more often preferred the results with the model than not. I used my [A/B Tester](https://github.com/MNeMoNiCuZ/ABTester) to get a less biased result test.

**What's next?**

> Re-training of the model at some point. For now paused and put on hold.

> Check out [Prompt Quill](https://github.com/osi1880vr/prompt_quill). A similar but seemingly successful project to improve an input prompt.

## Future steps to experiment with
- [ ] Re-train the model with proper separator between the inputs and the outputs, and call on this token during inference
- [ ] Consider RAG implementation of the dataset or additional data

## Special Thanks
Thanks to [DonMischo](https://civitai.com/user/DonMischo/models) for testing out the training script and lending me the compute to train the epochs for this version.
