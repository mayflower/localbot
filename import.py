"""Import einer Website in den Vectorstore"""
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


def import_docs():
    """Lese alle Dokumente im Verzeichnis data und wandle zu Vektoren"""
    sitemap_loader = SitemapLoader(web_path="https://langchain.readthedocs.io/sitemap.xml")

    scraped_documents = sitemap_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(scraped_documents)
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    vectorstore.save_local('./store/')

if __name__ == "__main__":
    import_docs()
