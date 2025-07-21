from typing import List

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_core.documents import Document
from langchain_text_splitters import HTMLSemanticPreservingSplitter



def get_polars_user_guide() -> List[str]:
    BASE_URL = 'https://docs.pola.rs/'
    urls = []

    user_guide = BeautifulSoup(requests.get(BASE_URL).text, 'html.parser')

    ul_navbar = user_guide.find('ul', class_='md-nav__list')
    lis = ul_navbar.find_all('li')

    anchors = lis[1].find_all('a')
    anchors.pop(0)

    for a in anchors:
        urls.append(BASE_URL + a.get('href'))

    return urls


def get_polars_api() -> List[str]:
    BASE_URL = 'https://docs.pola.rs/api/python/stable/reference/'
    urls = []

    api = BeautifulSoup(requests.get(BASE_URL).text, 'html.parser')

    div_ul = api.find(class_='bd-toc-item navbar-nav')
    uls = div_ul.find_all('ul', class_='nav bd-sidenav')

    for ul in uls:
        lis = ul.find_all('li')
        for li in lis:
            urls.append(BASE_URL + li.find('a').get('href'))

    return urls


def split_polars(urls: List[str]) -> List[Document]:
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()

    filtred_docs = []
    for doc in docs:
        soup = BeautifulSoup(doc.page_content, 'html.parser')
        article = soup.find('article', class_='bd-article')

        if not article:  # User guide
            article = soup.find('article', class_='md-content__inner md-typeset')

        filtred_docs.append(Document(str(article), metadata=doc.metadata))

    splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=[('h1', 'Page'), ('h2', 'Seciton'),
                             ('h3', 'Subsection'), ('h4', 'Subsection')],
        separators=["\n\n", "\n", ". ", "! ", "? "],
        max_chunk_size=1000,
        preserve_links=True,
        preserve_images=False,
        elements_to_preserve=['table', 'ul', 'ol',
                              'code', 'span', 'pre',
                              'mjx-container', 'mjx-math'],
        custom_handlers={
            "code": lambda element: f"<code:Python>{element.get_text()}</code>"}
    )

    split_docs = []
    for doc in filtred_docs:
        new_doc = splitter.split_text(doc.page_content)
        for d in new_doc:
            d.metadata.update(doc.metadata)
            split_docs.append(d)
    return split_docs


def split_polars_changelogs() -> List[Document]:
    loader = AsyncHtmlLoader(['https://github.com/pola-rs/polars/releases'])
    doc = loader.load()
    doc = doc[0]

    soup = BeautifulSoup(doc.page_content, 'html.parser')
    div = soup.find('div', class_='clearfix container-xl px-3 px-md-4 px-lg-5 mt-4')
    sections = div.find_all('section')

    filtred_docs = []
    for section in sections:
        version = section.find('a', class_='Link--primary Link').getText()
        if 'Rust' in version:
            continue
        doc.metadata['title'] = version

        content = section.find('div', class_='markdown-body my-3')
        filtred_docs.append(Document(str(content), metadata=doc.metadata))

    splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=[('h1', 'Page'), ('h2', 'Seciton'),
                             ('h3', 'Subsection'), ('h4', 'Subsection')],
        separators=["\n\n", "\n", ". ", "! ", "? "],
        max_chunk_size=1000,
        preserve_links=True,
        preserve_images=False,
        elements_to_preserve=['table', 'ul', 'ol',
                              'code', 'span', 'pre',
                              'mjx-container', 'mjx-math'],
    )

    split_docs = []
    for doc in filtred_docs:
        new_doc = splitter.split_text(doc.page_content)
        for d in new_doc:
            d.metadata.update(doc.metadata)
            split_docs.append(d)
    return split_docs



def main():
    sp = split_polars_changelogs()
    print(sp[0])
    print('')
    print(sp[-1])
    print(len(sp))


if __name__ == '__main__':
    main()


