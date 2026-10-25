
import os
import traceback
from threading import Thread

import torch
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextStreamer,
    TextIteratorStreamer,
)

from pdf_reader import PdfReader
from local_embedding import LocalEmbedding


class AiModel:
    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct"):
        """
        Initialize the tokenizer and language model.
        """
        self.model_name = model_name

        print("Running hardware checks...")
        self.hardware_check()

        print("Checking Hugging Face authentication...")
        self.hugging_face_auth()

        print("Loading tokenizer. This may take a while...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        print("Loading language model. Please wait...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
        )

        self.model.eval()
        print("Model loaded successfully.")

    def hardware_check(self):
        """
        Detect available hardware, including Apple Silicon's MPS backend.
        """
        if torch.cuda.is_available():
            print(f"NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            print("Apple Silicon GPU detected: MPS")
        else:
            print("No supported GPU detected. The model may use the CPU.")

    def hugging_face_auth(self):
        """
        Authenticate with Hugging Face if HF_TOKEN is configured.
        """
        hf_token = os.environ.get("HF_TOKEN")

        if hf_token:
            print("Authenticating with Hugging Face...")
            login(token=hf_token)
            print("Hugging Face authentication successful.")
        else:
            print(
                "HF_TOKEN is not set. Continuing without explicit login. "
                "A token may be required for gated models."
            )

    def _prepare_inputs(self, prompt):
        """
        Format a user prompt using the tokenizer's chat template.
        """
        messages = [{"role": "user", "content": prompt}]

        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
        )

        # Place inputs on the device used by the model's first parameter.
        model_device = next(self.model.parameters()).device
        return inputs.to(model_device)

    def ask_a_question(self, prompt="Hello there!"):
        """
        Generate a response and print it to the terminal.
        """
        inputs = self._prepare_inputs(prompt)

        streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        with torch.inference_mode():
            self.model.generate(
                **inputs,
                max_new_tokens=512,
                streamer=streamer,
            )

    def ask_a_question_from_pdf(
        self,
        pdf_path,
        prompt="Tell me what this PDF is about",
    ):
        """
        Retrieve relevant PDF passages and generate an answer.
        Prints the answer to the terminal.
        """
        pdf_reader = PdfReader(pdf_path)
        pdf_paragraphs = pdf_reader.get_paragraphs()

        local_embedding = LocalEmbedding()
        local_embedding.build_index(pdf_paragraphs)

        relevant_sections = local_embedding.get_context(prompt, 10)

        full_prompt = self.full_prompt_for_rag(
            relevant_sections,
            prompt,
        )

        inputs = self._prepare_inputs(full_prompt)

        streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        with torch.inference_mode():
            self.model.generate(
                **inputs,
                max_new_tokens=512,
                streamer=streamer,
            )

    def ask_a_question_from_pdf_stream(
        self,
        pdf_path,
        prompt="Tell me what this PDF is about",
        local_embedding=None,
    ):
        """
        Stream generated text chunks to callers such as Streamlit.

        If a pre-built LocalEmbedding instance is supplied, reuse its index.
        Otherwise, read the PDF and build an index.
        """
        if local_embedding is None:
            pdf_reader = PdfReader(pdf_path)
            pdf_paragraphs = pdf_reader.get_paragraphs()

            local_embedding = LocalEmbedding()
            local_embedding.build_index(pdf_paragraphs)

        relevant_sections = local_embedding.get_context(prompt, 10)

        full_prompt = self.full_prompt_for_rag(
            relevant_sections,
            prompt,
        )

        inputs = self._prepare_inputs(full_prompt)

        # No timeout: the generation thread signals the end of the stream,
        # even if it fails before producing its first token.
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=None,
        )

        generation_errors = []

        def generate():
            try:
                with torch.inference_mode():
                    self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        streamer=streamer,
                    )
            except Exception as exc:
                generation_errors.append(exc)
                traceback.print_exc()
            finally:
                # Always signal the end so the consumer cannot wait forever
                # just because generation raised an exception.
                streamer.end()

        thread = Thread(target=generate, daemon=True)
        thread.start()

        try:
            for chunk in streamer:
                if chunk:
                    yield chunk
        finally:
            thread.join()

        if generation_errors:
            raise RuntimeError(
                f"Language model generation failed: {generation_errors[0]}"
            ) from generation_errors[0]

    def full_prompt_for_rag(self, relevant_sections, question_prompt):
        """
        Build the RAG prompt. The tokenizer adds the model-specific chat
        formatting separately through apply_chat_template().
        """
        return f"""You are an AI assistant. Answer the question using
only the provided document excerpts.

If the excerpts do not contain the answer, say:
"The document does not contain information on this topic."

Document excerpts:
---
{relevant_sections}
---

Question: {question_prompt}

Answer based only on the document excerpts.""" 
