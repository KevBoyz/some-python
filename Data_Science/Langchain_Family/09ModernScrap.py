from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import HTMLHeaderTextSplitter

from langchain_core.documents import Document
from bs4 import BeautifulSoup
from typing import List
import requests


from dotenv import load_dotenv
load_dotenv()

base_url = 'https://scikit-learn.org/stable/'


def get_urls_scikit_user_guide() -> List[str]:
    user_guide = BeautifulSoup(
        requests.get(base_url + '/user_guide.html').text,
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
                urls.append(base_url + a.get('href'))
        else:
            anchor = li.find('a')
            urls.append(base_url + anchor.get('href'))
    return urls


def split_scikit_user_guide(urls: List[str]) -> List[Document]:
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()

    splitter = HTMLHeaderTextSplitter(
        headers_to_split_on=[
            ('h1', 'Page'), ('h2', 'Section'),
            ('h3', 'Sub Section'), ('h4', 'Sub Section'),
        ])

    split_docs = []
    for doc in docs:
        if len(doc.page_content) > 50:
            new_doc = splitter.split_text(doc.page_content)
            for d in new_doc:
                d.metadata.update(doc.metadata)
            split_docs.extend(new_doc)
    return split_docs


def get_urls_scikit_api() -> List[str]:
    api = BeautifulSoup(
        requests.get(base_url + 'api/index.html').text
    )
    urls = []

    tbody = api.find('tbody')
    all_tr = tbody.find_all('tr')

    for tr in all_tr:
        td = tr.find('td')
        link = td.find('a').get('href')
        urls.append(base_url + link[3:])
    return urls


def split_scikit_api(urls: List[str]) -> List[Document]:
    loader = AsyncHtmlLoader(urls)
    docs = loader.load()

    filtred_docs = BeautifulSoupTransformer().transform_documents(
        docs,
        tags_to_extract=['dl']
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    split_docs = splitter.split_documents(filtred_docs)
    return split_docs


def scikit_pipeline() -> List[Document]:
    user_guide = split_scikit_user_guide(
        get_urls_scikit_user_guide()
    )
    api = split_scikit_api(
        get_urls_scikit_user_guide()
    )
    user_guide.extend(api)
    return user_guide
