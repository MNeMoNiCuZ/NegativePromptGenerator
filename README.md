# Negative Prompt Generator
> [!CAUTION]
> This project is a work in progress.

The trained model is a finetuned [GPT2](https://github.com/openai/gpt-2) text generation model that takes a positive prompt as an input, and outputs a negative prompt that is supposed to match the input prompt.

However, the results are very random and they are mostly unrelated to the prompt, and sometime they even output a positive prompt.

As the results are not good, I have not yet cleaned up the project and made it presentable.

Use this mostly for your own curiosity or experimentation.

## Training
The model was trained using positive prompt and negative prompt pairs scraped from the CivitAI API. In total a set of 18k pairs was trained on.

The final trained epoch was #54.

## Model Weights
Model weights and checkpoint can be downloaded from my [huggingface](https://huggingface.co/mnemic/NegativePromptGenerator/tree/main).

The [weights.pt](https://huggingface.co/mnemic/NegativePromptGenerator/blob/main/weights.pt) contains the weights. Use them freely.

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
- [ ] Re-train the model with proper stopping
- [ ] Consider RAG implementation of the dataset or additional data
