import pytest

from sec_financial_analyst.sec import SecError, extract_item_7, html_to_text


def test_item_7_extractor_skips_short_table_of_contents_entry() -> None:
    body = " ".join(
        "Revenue changed because demand, pricing, and product mix evolved."
        for _ in range(100)
    )
    html = f"""
    <p>Item 7. Management's Discussion and Analysis ..... 40</p>
    <p>Item 7A. Quantitative and Qualitative Disclosures ..... 65</p>
    <h1>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</h1>
    <p>{body}</p>
    <h1>ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES</h1>
    """

    result = extract_item_7(html, min_characters=4_000)

    assert result.startswith("ITEM 7.")
    assert "Revenue changed" in result
    assert "QUANTITATIVE AND QUALITATIVE DISCLOSURES" not in result


def test_extractor_rejects_toc_only_document() -> None:
    with pytest.raises(SecError, match="complete Item 7"):
        extract_item_7("<p>Item 7. MD&A 40</p><p>Item 8. Statements 70</p>", min_characters=500)


def test_hidden_inline_xbrl_content_is_removed() -> None:
    html = '<div style="display:none"><b>secret</b></div><p>visible filing text</p>'
    assert html_to_text(html) == "visible filing text"
