import ctranslate2
import transformers

# Load the original model
model_name = "models/whisper-large"
model = transformers.WhisperForConditionalGeneration.from_pretrained(model_name)
tokenizer = transformers.WhisperTokenizer.from_pretrained(model_name)

# Convert and save the model in CTranslate2 format
output_dir = "models/whisper-large-ct2"
ctranslate2.converters.WhisperConverter(model_name).convert(output_dir, force=True)

print("Conversion complete! CTranslate2 model saved in:", output_dir)

