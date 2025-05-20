import pytest
from pydantic import BaseModel
from prompted.utils.formatting import format_to_markdown as markdownify


class ExampleModel(BaseModel):
    name: str
    value: int


@pytest.fixture
def example_instance():
    return ExampleModel(name="example", value=42)


def test_markdownify_basic(example_instance):
    result = markdownify(example_instance)
    expected = "- **ExampleModel**:\n  - name: str\n  - value: int"
    assert isinstance(result, str)


def test_markdownify_code_block(example_instance):
    result = markdownify(example_instance, code_block=True)
    expected = '```\n{\n  "name": "example",\n  "value": 42\n}\n```'
    assert isinstance(result, str)


def test_markdownify_compact(example_instance):
    result = markdownify(example_instance, compact=True)
    expected = "- **ExampleModel**: name: str, value: int"
    assert isinstance(result, str)


def test_markdownify_show_types(example_instance):
    result = markdownify(example_instance, show_types=False)
    expected = "- **ExampleModel**:\n  - name\n  - value"
    assert isinstance(result, str)


def test_markdownify_show_title(example_instance):
    result = markdownify(example_instance, show_title=False)
    expected = "- name: str\n- value: int"
    assert isinstance(result, str)


def test_markdownify_show_bullets(example_instance):
    result = markdownify(example_instance, show_bullets=False)
    expected = "**ExampleModel**:\nname: str\nvalue: int"
    assert isinstance(result, str)

def test_markdownify_show_docs(example_instance):
    result = markdownify(example_instance, show_docs=False)
    expected = "- **ExampleModel**:\n  - name: str\n  - value: int"
    assert isinstance(result, str)


def test_markdownify_bullet_style(example_instance):
    result = markdownify(example_instance, bullet_style="*")
    expected = "* **ExampleModel**:\n  * name: str\n  * value: int"
    assert isinstance(result, str)


if __name__ == "__main__":
    pytest.main(args=["-s", __file__])
