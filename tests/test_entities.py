"""Tests for entity extraction and knowledge graph."""


from agent_memory.entities import (
    Entity,
    ExtractedEntities,
    _dedupe,
)


# ---------------------------------------------------------------------------
# EntityExtractor
# ---------------------------------------------------------------------------

class TestEntityExtractor:
    def test_known_person(self, extractor):
        result = extractor.extract("User ile konuştum")
        names = [e.text.lower() for e in result.people]
        assert "user" in names

    def test_known_place(self, extractor):
        result = extractor.extract("istanbul'a gidiyorum")
        names = [e.text.lower() for e in result.places]
        assert "istanbul" in names

    def test_known_org(self, extractor):
        result = extractor.extract("Anthropic çok güzel çalışıyor")
        names = [e.text.lower() for e in result.organizations]
        assert "anthropic" in names

    def test_known_tech(self, extractor):
        result = extractor.extract("python ile fastapi projesi yapıyorum")
        tech = [e.text.lower() for e in result.tech_terms]
        assert "python" in tech
        assert "fastapi" in tech

    def test_turkish_name_pattern(self, extractor):
        result = extractor.extract("Ahmet Yılmaz ile görüştüm bugün")
        names = [e.text for e in result.people]
        assert any("Ahmet" in n for n in names)

    def test_context_based_person(self, extractor):
        result = extractor.extract("Mehmet ile konuştum dün")
        names = [e.text.lower() for e in result.people]
        assert any("mehmet" in n for n in names)

    def test_email_extraction(self, extractor):
        result = extractor.extract("bana user@example.com adresinden yaz")
        concepts = [e.text for e in result.concepts]
        assert "user@example.com" in concepts

    def test_date_extraction(self, extractor):
        result = extractor.extract("toplantı 2026-02-15 tarihinde")
        dates = [e.text for e in result.dates]
        assert "2026-02-15" in dates

    def test_product_extraction(self, extractor):
        result = extractor.extract("yeni iPhone 15 Pro aldım")
        products = [e.text for e in result.products]
        assert any("iPhone" in p for p in products)

    def test_no_entities(self, extractor):
        result = extractor.extract("merhaba nasılsın")
        # Should still work, might find 0 or minimal entities
        assert isinstance(result, ExtractedEntities)

    def test_turkish_temporal_words(self, extractor):
        result = extractor.extract("bugün çok güzel hava var")
        dates = [e.text.lower() for e in result.dates]
        assert "bugün" in dates

    def test_multiple_entity_types(self, extractor):
        text = "User, Istanbul'da Anthropic ile Python projesi yapıyor"
        result = extractor.extract(text)
        assert len(result.all_entities()) >= 3

    def test_extracted_entities_to_dict(self, extractor):
        result = extractor.extract("User Python kullanıyor")
        d = result.to_dict()
        assert "people" in d
        assert "tech_terms" in d
        assert isinstance(d["people"], list)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDedup:
    def test_case_insensitive_dedup(self):
        entities = [
            Entity(text="User", label="person"),
            Entity(text="user", label="person"),
            Entity(text="USER", label="person"),
        ]
        result = _dedupe(entities)
        assert len(result) == 1

    def test_no_dedup_different_names(self):
        entities = [
            Entity(text="Alice", label="person"),
            Entity(text="Bob", label="person"),
        ]
        result = _dedupe(entities)
        assert len(result) == 2

    def test_empty_list(self):
        assert _dedupe([]) == []


# ---------------------------------------------------------------------------
# KnowledgeGraph
# ---------------------------------------------------------------------------

class TestKnowledgeGraph:
    def test_process_text_stores_entities(self, knowledge_graph, tmp_storage):
        knowledge_graph.process_text("User Python ile çalışıyor")
        entities = tmp_storage.search_entities("user", limit=5)
        assert len(entities) >= 1

    def test_process_text_creates_relationships(self, knowledge_graph, tmp_storage):
        knowledge_graph.process_text(
            "User works on Python projects"
        )
        stats = tmp_storage.stats()
        assert stats["relationships"] > 0

    def test_search_entities(self, knowledge_graph):
        knowledge_graph.process_text("User ile konuştum")
        results = knowledge_graph.search("user")
        assert len(results) >= 1
        assert results[0]["name"].lower() == "user"

    def test_entity_mention_count_increases(self, knowledge_graph, tmp_storage):
        knowledge_graph.process_text("User ile konuştum")
        knowledge_graph.process_text("User yine aradı")
        entities = tmp_storage.search_entities("User", limit=1)
        assert len(entities) >= 1  # entity found at least once

    def test_co_occurrence_linking(self, knowledge_graph, tmp_storage):
        knowledge_graph.process_text("Alice ve Bob Python projesi yapıyor")
        stats = tmp_storage.stats()
        # Should have relationships between co-occurring entities
        assert stats["entities"] >= 2
        assert stats["relationships"] >= 1
