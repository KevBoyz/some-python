from typing import List

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_core.documents import Document
from langchain_text_splitters import HTMLSemanticPreservingSplitter
                                     

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
        requests.get(BASE_URL + 'api/index.html').text,
        'html.parser'
    )
    urls = []

    tbody = api.find('tbody')
    all_tr = tbody.find_all('tr')

    for tr in all_tr:
        td = tr.find('td')
        link = td.find('a').get('href')
        urls.append(BASE_URL + link[3:])
    return urls


def get_urls_scikit_examples() -> List[str]:
    """Get all urls of "examples" from a index page

    Returns:
        List[str]: List with URLs
    """
    examples = BeautifulSoup(
        requests.get(BASE_URL + 'auto_examples/index.html').text,
        'html.parser'
    )

    urls = []

    main_section = examples.find('section', id='examples')
    all_sections = main_section.find_all('section')
    for section in all_sections:
        div = section.find('div', class_='sphx-glr-thumbnails')
        anchors = div.find_all('a')
        for a in anchors:
            urls.append(BASE_URL + 'auto_examples/' + a.get('href'))

    return urls


def split_scikit(urls: List[str]) -> List[Document]:
    """Scrap and split all urls passed of user guide, api and examples
    Only the content inside the article (central tag) will be read.

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
            # Add source, description, language
            d.metadata.update(doc.metadata)
            split_docs.append(d)
    return split_docs


def scikit_pipeline() -> List[Document]:
    """Automates the scikit pipeline

    Returns:
        List[Document]: (User guide + api) langchain docs 
    """
    split = split_scikit(
        get_urls_scikit_api() + get_urls_scikit_user_guide() + get_urls_scikit_examples()
    )
    return split


def main():
    results = scikit_pipeline()
    print(results[:4])


if __name__ == '__main__':
    main()
