from typing import List

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_core.documents import Document
from langchain_text_splitters import HTMLSemanticPreservingSplitter


BASE_URL = 'https://pandas.pydata.org/docs/'


def get_pandas_user_guide():
    user_guide = BeautifulSoup(
        requests.get(BASE_URL + 'user_guide/index.html').text,
        'html.parser'
    )
    urls = []

    links_navbar = user_guide.find(class_='nav bd-sidenav')
    anchors = links_navbar.find_all('a')

    for a in anchors:
        urls.append(BASE_URL + 'user_guide/' + a.get('href'))

    return urls


def get_pandas_api():
    api = BeautifulSoup(
    requests.get(BASE_URL + 'reference/index.html').text,
    'html.parser'
    )
    urls = []

    links_navbar = api.find(class_='nav bd-sidenav')
    anchors = links_navbar.find_all('a')

    for a in anchors:
        req = BeautifulSoup(
        requests.get(BASE_URL + 'reference/' + a.get('href')).text,
        'html.parser'
        )
        li_navbar = req.find(class_='toctree-l1 current active has-children')
        all_anchors = li_navbar.find_all('a')
        for an in all_anchors:
            urls.append(BASE_URL + 'reference/' + an.get('href'))

    urls.pop(0)
    return urls


def get_pandas_changelogs():
    user_guide = BeautifulSoup(
        requests.get(BASE_URL + 'whatsnew/index.html').text,
        'html.parser'
    )
    urls = []

    links_navbar = user_guide.find(class_='bd-toc-item navbar-nav')
    anchors = links_navbar.find_all('a')

    for a in anchors:
        urls.append(BASE_URL + 'whatsnew/'+ a.get('href'))

    return urls


def split_pandas(urls: List[str]) -> List[Document]:
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()

    filtred_docs = []
    for doc in docs:
        soup = BeautifulSoup(doc.page_content, 'html.parser')
        article = soup.find('article', class_='bd-article')
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
            "pre": lambda element: f"<code:Python>{element.get_text()}</code>"}
    )

    split_docs = []
    for doc in filtred_docs:
        new_doc = splitter.split_text(doc.page_content)
        for d in new_doc:
            d.metadata.update(doc.metadata)
            split_docs.append(d)
    return split_docs


