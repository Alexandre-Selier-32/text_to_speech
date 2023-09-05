!pip install git+https://github.com/huggingface/transformers.git


from transformers import AutoProcessor, AutoModel


processor = AutoProcessor.from_pretrained("suno/bark-small")
model = AutoModel.from_pretrained("suno/bark-small")

inputs = processor(
    text=["Hello, my name is dylan"],
    return_tensors="pt",
)

speech_values = model.generate(**inputs, do_sample=True)


from IPython.display import Audio

sampling_rate = model.generation_config.sample_rate
Audio(speech_values.cpu().numpy().squeeze(), rate=sampling_rate)
