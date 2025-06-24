from typing import List

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_core.documents import Document
from langchain_text_splitters import (HTMLHeaderTextSplitter,
                                      RecursiveCharacterTextSplitter)

load_dotenv()

BASE_URL = 'https://scikit-learn.org/stable/'


def get_urls_scikit_user_guide() -> List[str]:
    """Get all urls of "user guide" from a navbar on the site

    Returns:
        List[str]: List with URLs
    """
    user_guide = BeautifulSoup(
        requests.get(BASE_URL + '/user_guide.html').text,
        'html.parser'
    )
    urls = []

    links_navbar = user_guide.find(class_='nav bd-sidenav')
    all_li = links_navbar.find_all('li')

    for li in all_li:
        details = li.find('details')
        if details:
            ul = details.find('ul')
            anchors = ul.find_all('a')
            for a in anchors:
                urls.append(BASE_URL + a.get('href'))
        else:
            anchor = li.find('a')
            urls.append(BASE_URL + anchor.get('href'))
    return urls


def get_urls_scikit_api() -> List[str]:
    """Get all urls of "API" from a index page

    Returns:
        List[str]: List with URLs
    """
    api = BeautifulSoup(
        requests.get(BASE_URL + 'api/index.html').text
    )
    urls = []

    tbody = api.find('tbody')
    all_tr = tbody.find_all('tr')

    for tr in all_tr:
        td = tr.find('td')
        link = td.find('a').get('href')
        urls.append(BASE_URL + link[3:])
    return urls


def split_scikit_user_guide(urls: List[str]) -> List[Document]:
    """Scrap and split all urls passed of user guide pages
    Only the content inside the article tag will be read.

    Args:
        urls (List[str]): List with URLs

    Returns:
        List[Document]: List of langchain documents
    """
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()

    filtred_docs = []
    for doc in docs:
        soup = BeautifulSoup(doc.page_content, 'html.parser')
        article = soup.find('article', class_='bd-article')
        filtred_docs.append(Document(str(article), metadata=doc.metadata))

    splitter = HTMLHeaderTextSplitter(
        headers_to_split_on=[
            ('h1', 'Page'), ('h2', 'Section'),
            ('h3', 'Sub Section'), ('h4', 'Sub Section'),
        ])

    split_docs = []
    for doc in filtred_docs:
        new_doc = splitter.split_text(doc.page_content)
        for d in new_doc:
            if len(d.page_content) > 40:
                d.metadata.update(doc.metadata)
                split_docs.append(d)
            else:
                continue
    return split_docs


def split_scikit_api(urls: List[str]) -> List[Document]:
    """Scrap and split all urls passed of api pages
    Only the content inside the article tag will be read.

    Args:
        urls (List[str]): List with URLs

    Returns:
        List[Document]: List of langchain documents
    """
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()

    filtred_docs = BeautifulSoupTransformer().transform_documents(
        docs,
        tags_to_extract=['article']
    )

    splitter = RecursiveCharacterTextSplitter(  # Checar a qualidade da configuração
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    split_docs = splitter.split_documents(filtred_docs)
    return split_docs


def scikit_pipeline() -> List[Document]:
    """Automates the scikit pipeline

    Returns:
        List[Document]: (User guide + api) langchain docs 
    """
    user_guide = split_scikit_user_guide(
        get_urls_scikit_user_guide()
    )
    api = split_scikit_api(
        get_urls_scikit_user_guide()
    )
    user_guide.extend(api)
    return user_guide


split = split_scikit_user_guide(['https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification'])

print(split[2])  # Blocos de código e textos em destaque ficam totalmente bugados na raspagem.