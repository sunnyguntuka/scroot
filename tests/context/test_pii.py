"""PII scrubber unit tests - all entity types."""

from scroot.context.pii import scrub


class TestEntityTypes:
    def test_email(self):
        r = scrub("Contact john@acme.com for details.")
        assert "[EMAIL]" in r.scrubbed_text
        assert "john@acme.com" not in r.scrubbed_text
        assert r.summary["EMAIL"] == 1

    def test_phone(self):
        r = scrub("Call +1-555-0172 today.")
        assert "[PHONE]" in r.scrubbed_text
        assert "555-0172" not in r.scrubbed_text
        assert r.summary["PHONE"] == 1

    def test_phone_standard_us(self):
        r = scrub("Call 555-867-5309 today.")
        assert "[PHONE]" in r.scrubbed_text

    def test_ssn(self):
        r = scrub("SSN is 123-45-6789.")
        assert "[SSN]" in r.scrubbed_text
        assert "123-45-6789" not in r.scrubbed_text
        assert r.summary["SSN"] == 1

    def test_credit_card(self):
        r = scrub("Card: 4111-1111-1111-1111 expires soon.")
        assert "[CARD]" in r.scrubbed_text
        assert "4111" not in r.scrubbed_text
        assert r.summary["CARD"] == 1

    def test_person_name(self):
        r = scrub("The account belongs to John Smith.")
        assert "[PERSON]" in r.scrubbed_text
        assert "John Smith" not in r.scrubbed_text
        assert r.summary["PERSON"] == 1

    def test_person_honorific(self):
        r = scrub("Please see Dr. Watson immediately.")
        assert "[PERSON]" in r.scrubbed_text
        assert "Watson" not in r.scrubbed_text

    def test_ip_address(self):
        r = scrub("Server at 192.168.1.1 responded.")
        assert "[IP]" in r.scrubbed_text
        assert "192.168.1.1" not in r.scrubbed_text
        assert r.summary["IP"] == 1

    def test_date_of_birth(self):
        r = scrub("Born Jan 15, 1985 in Ohio.")
        assert "[DOB]" in r.scrubbed_text
        assert "1985" not in r.scrubbed_text
        assert r.summary["DOB"] == 1

    def test_street_address(self):
        r = scrub("Ship to 123 Main St please.")
        assert "[ADDRESS]" in r.scrubbed_text
        assert "Main St" not in r.scrubbed_text
        assert r.summary["ADDRESS"] == 1


class TestSecrets:
    def test_openai_key(self):
        r = scrub("Use sk-abcdefghij1234567890abcd for auth.")
        assert "[SECRET]" in r.scrubbed_text
        assert "sk-abcdefghij" not in r.scrubbed_text

    def test_anthropic_key(self):
        r = scrub("Key: sk-ant-api03-abcdefghij1234567890")
        assert "[SECRET]" in r.scrubbed_text
        assert "sk-ant" not in r.scrubbed_text

    def test_aws_key(self):
        r = scrub("AWS: AKIAIOSFODNN7EXAMPLE")
        assert "[SECRET]" in r.scrubbed_text
        assert "AKIA" not in r.scrubbed_text

    def test_github_token(self):
        r = scrub("Token ghp_" + "a" * 36 + " leaked.")
        assert "[SECRET]" in r.scrubbed_text
        assert "ghp_" not in r.scrubbed_text

    def test_long_hex_string(self):
        r = scrub("Hash: " + "a1b2c3d4" * 4 + " found.")
        assert "[SECRET]" in r.scrubbed_text


class TestSummary:
    def test_counts_only_no_original_values(self):
        r = scrub("Email john@acme.com and jane@acme.com, SSN 123-45-6789.")
        assert r.summary["EMAIL"] == 2
        assert r.summary["SSN"] == 1
        assert r.summary["total_entities_scrubbed"] == 3
        # Summary must contain only ints - never original text
        for value in r.summary.values():
            assert isinstance(value, int)

    def test_clean_text_not_scrubbed(self):
        r = scrub("The refund policy allows returns within 30 days.")
        assert r.was_scrubbed is False
        assert r.summary["total_entities_scrubbed"] == 0
        assert r.scrubbed_text == "The refund policy allows returns within 30 days."

    def test_was_scrubbed_flag(self):
        assert scrub("mail me: a@b.co").was_scrubbed is True
        assert scrub("no pii here").was_scrubbed is False
