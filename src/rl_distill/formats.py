"""Per-model CoT format adapters.

The two models format reasoning differently and this is the single most likely
place for the pipeline to break silently, so every format detail is declarative
(loaded from config) rather than hard-coded. Nothing here assumes <think> tags;
a format without think tags treats the whole pre-answer span as reasoning.
"""

from dataclasses import dataclass

from .parsing import extract_between, extract_boxed


@dataclass
class CotFormat:
    name: str
    chat: bool = False
    system_prompt: str = None
    user_template: str = "{problem}"
    prompt_template: str = None
    think_open: str = None
    think_close: str = None
    answer_open: str = None
    answer_close: str = None
    # True when prompt_template already seeds think_open (ORZ). Controls whether
    # build_resample_prompt re-adds the tag.
    think_open_in_prompt: bool = False

    def build_prompt(self, problem, tokenizer=None):
        if self.chat:
            if tokenizer is None:
                raise ValueError(f"format '{self.name}' is chat but no tokenizer given")
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            messages.append({"role": "user", "content": self.user_template.format(problem=problem)})
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        if self.prompt_template is None:
            raise ValueError(f"format '{self.name}' needs prompt_template when chat=False")
        return self.prompt_template.format(problem=problem)

    def build_resample_prompt(self, problem, thinking_prefix, tokenizer=None):
        """Prompt that makes the model continue reasoning from thinking_prefix.

        thinking_prefix is a slice of the reasoning region (no think_open). We
        re-seed think_open only when the base prompt doesn't already contain it.
        """
        base = self.build_prompt(problem, tokenizer)
        if self.think_open and not self.think_open_in_prompt:
            base = base + self.think_open
        return base + thinking_prefix

    def extract_thinking(self, text):
        """Reasoning span only.

        Handles a think-open tag that was seeded into the prompt (so the
        completion contains only the close tag): start defaults to 0 when the
        open tag is absent. Falls back to the answer region, then whole text.
        """
        if self.think_close:
            start = 0
            if self.think_open:
                pos = text.find(self.think_open)
                if pos != -1:
                    start = pos + len(self.think_open)
            end = text.find(self.think_close, start)
            if end != -1:
                return text[start:end]
        if self.answer_open:
            cut = text.find(self.answer_open)
            if cut != -1:
                return text[:cut]
        return text

    def extract_answer(self, text):
        """Final answer string. Prefer explicit answer tags, then \\boxed{}."""
        if self.answer_open and self.answer_close:
            span = extract_between(text, self.answer_open, self.answer_close)
            if span is not None:
                boxed = extract_boxed(span)
                return (boxed if boxed is not None else span).strip()
        boxed = extract_boxed(text)
        return boxed.strip() if boxed is not None else None


def load_formats(config):
    """Build {name: CotFormat} from the `formats` block of a run config."""
    return {name: CotFormat(name=name, **spec) for name, spec in config.items()}
