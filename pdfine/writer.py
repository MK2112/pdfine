from typing import List, Dict

"""
Convert refined pages into Markdown strings
"""

def pages_to_markdown(refined_pages: List[Dict]) -> List[str]:
    """
    Turn refined page data into clean Markdown strings.
    Each page dict: {'page_num': int, 'text_blocks': List[str]}.
    Returns a list of Markdown strings, one per page.
    """
    md_pages = []
    for page in refined_pages:
        header = f"## Page {page['page_num']}\n"
        body = '\n\n'.join(block.strip() for block in page.get('text_blocks', []) if block.strip())
        md = header + body if body else header
        md_pages.append(md.strip())
    return md_pages
