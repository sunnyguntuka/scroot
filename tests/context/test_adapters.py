"""extract_text() adapter tests for all chunk types."""

from scroot.context.adapters import extract_text


class FakeLangChainDocument:
    def __init__(self, page_content):
        self.page_content = page_content


class FakeLlamaIndexNode:
    def __init__(self, text):
        self.text = text


class FakeNodeWithScore:
    def __init__(self, text):
        self.node = FakeLlamaIndexNode(text)
        # NodeWithScore itself has no .text attribute
        self.score = 0.9


class FakeChromaResult:
    def __init__(self, documents):
        self.documents = documents


class FakePineconeVector:
    def __init__(self, text):
        self.metadata = {"text": text}


class TestExtractText:
    def test_plain_string(self):
        assert extract_text("hello") == "hello"

    def test_langchain_document(self):
        assert extract_text(FakeLangChainDocument("doc text")) == "doc text"

    def test_dict_text_key(self):
        assert extract_text({"text": "from dict"}) == "from dict"

    def test_dict_content_key(self):
        assert extract_text({"content": "c"}) == "c"

    def test_dict_page_content_key(self):
        assert extract_text({"page_content": "pc"}) == "pc"

    def test_dict_body_and_chunk_keys(self):
        assert extract_text({"body": "b"}) == "b"
        assert extract_text({"chunk": "ck"}) == "ck"

    def test_llamaindex_text_node(self):
        assert extract_text(FakeLlamaIndexNode("node text")) == "node text"

    def test_llamaindex_node_with_score(self):
        assert extract_text(FakeNodeWithScore("scored text")) == "scored text"

    def test_chromadb_query_result(self):
        result = FakeChromaResult([["chunk a", "chunk b"], ["chunk c"]])
        out = extract_text(result)
        assert "chunk a" in out and "chunk b" in out and "chunk c" in out

    def test_chromadb_dict_result(self):
        out = extract_text({"documents": [["x", "y"]]})
        assert "x" in out and "y" in out

    def test_pinecone_scored_vector(self):
        assert extract_text(FakePineconeVector("pine text")) == "pine text"

    def test_unrecognised_type_returns_none(self):
        assert extract_text(42) is None
        assert extract_text(None) is None
        assert extract_text(object()) is None

    def test_dict_without_text_keys_returns_none(self):
        assert extract_text({"id": "abc", "score": 0.5}) is None

    def test_never_raises(self):
        class Hostile:
            @property
            def page_content(self):
                raise RuntimeError("boom")

        assert extract_text(Hostile()) is None
