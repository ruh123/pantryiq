"""Constrained generation — what the model is shown, and what it is told.

The model's *output* is checked by the guardrail, not here. What these pin is the input: that the
context is the only source of facts, that untrusted text is fenced, and that a refusal is not
read as an answer.
"""
from types import SimpleNamespace

import pytest

from pantryiq.agent.claude import Refused, injection_paragraph
from pantryiq.agent.context import AnswerContext, RecipeFact
from pantryiq.agent.generate import SYSTEM, answer
from pantryiq.agent.retrieval import PantryQuery
from test_context import make_candidate


def fake_client(text: str = "An answer.", stop_reason: str = "end_turn"):
    """Duck-types `client.beta.messages.stream(...)` as a context manager."""
    sent = {}
    blocks = [] if stop_reason == "refusal" else [SimpleNamespace(type="text", text=text)]
    final = SimpleNamespace(stop_reason=stop_reason, content=blocks)

    class Stream:
        text_stream = [] if stop_reason == "refusal" else [text]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            return final

    def stream(**kwargs):
        sent.update(kwargs)
        return Stream()

    client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=stream)))
    return client, sent


def a_context(question: str = "what can I make?") -> AnswerContext:
    return AnswerContext(question=question, query=PantryQuery(pantry=("chicken",)),
                         recipes=(RecipeFact.from_candidate(make_candidate()),))


def test_the_recipe_data_is_fenced_as_untrusted():
    """Titles are third-party scraped text reaching the prompt verbatim."""
    client, sent = fake_client()

    answer(a_context(), client)

    assert injection_paragraph("recipes", "recipe data") in SYSTEM
    assert "<recipes>" in sent["messages"][0]["content"]
    assert "<user_question>" in sent["messages"][0]["content"]


def test_the_model_is_given_no_tools_and_no_corpus_access():
    """The property that makes 'only speaks from verified data' checkable: the request carries
    the context and nothing else. A tool would be a second source of facts."""
    client, sent = fake_client()

    answer(a_context(), client)

    assert "tools" not in sent
    assert sent["messages"][0]["content"].count("<recipe>") == 1


def test_the_rules_that_prevent_a_plausible_wrong_answer_are_present():
    """Each of these exists because the warehouse can be honestly misread — a bare total that
    hides missing coverage, a divided per-serving figure, a cost floor read as a price."""
    assert "coverage" in SYSTEM
    assert "divid" in SYSTEM.lower(), "the rule against computing per-serving figures"
    assert "floor" in SYSTEM.lower(), "the rule about partial cost coverage"
    assert "nothing in the warehouse matches" in SYSTEM


def test_a_correction_is_passed_back_on_a_regeneration():
    """The guardrail's second attempt has to be told what was wrong with the first, or it is just
    a re-roll of the same dice."""
    client, sent = fake_client()

    answer(a_context(), client, note="You wrote 433.75, which is not in the data.")

    assert "<correction>" in sent["messages"][0]["content"]
    assert "433.75" in sent["messages"][0]["content"]


def test_a_refusal_is_raised_rather_than_returned_as_an_answer():
    """A mid-stream classifier decline leaves partial text already streamed. The check runs on
    the FINAL message, so the partial is discarded rather than shipped as a complete answer."""
    client, _ = fake_client(stop_reason="refusal")

    with pytest.raises(Refused):
        answer(a_context(), client)


def test_streaming_is_used_so_the_first_words_arrive_before_the_last():
    """Two sequential model calls cost 7-12 s end to end with no effort budget left to cut. The
    answer streams so the measured target is time to first token, and `on_text` is what makes
    that measurable rather than asserted."""
    client, _ = fake_client("Hello world")
    seen = []

    result = answer(a_context(), client, on_text=seen.append)

    assert seen == ["Hello world"]
    assert result == "Hello world"
