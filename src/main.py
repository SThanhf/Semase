# from contextlib import asynccontextmanager
# from datetime import datetime

# from azure.core.credentials import AzureKeyCredential
# from azure.search.documents.aio import SearchClient
# from azure.search.documents.models import QueryType, SearchMode, VectorizedQuery
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from openai import AsyncAzureOpenAI
# from src.settings import SETTINGS
# from src.setup import setup
# from src.utils import DocumentMetaData, SearchHit, aggregate_search_hits

# documents_search_client = SearchClient(
#     SETTINGS.search_endpoint,
#     SETTINGS.search_document_index_name,
#     AzureKeyCredential(SETTINGS.search_key),
# )

# chunks_search_client = SearchClient(
#     SETTINGS.search_endpoint,
#     SETTINGS.search_chunk_index_name,
#     AzureKeyCredential(SETTINGS.search_key),
# )

# embedder = AsyncAzureOpenAI(
#     azure_endpoint=SETTINGS.ai_endpoint,
#     api_key=SETTINGS.ai_key,
#     api_version=SETTINGS.ai_api_version,
# )

# TOKEN_LIMIT = 2048


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     await setup()
#     yield
#     await documents_search_client.close()
#     await chunks_search_client.close()
#     await embedder.close()


# app = FastAPI(lifespan=lifespan)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# @app.get("/")
# async def search(query: str, page: int = 1, page_size: int = 10):
#     text_search_hits = await text_search(query)
#     vector_search_hits = await vector_search(query)

#     documents = aggregate_search_hits([*text_search_hits, *vector_search_hits])

#     total_results = len(documents)
#     total_pages = (total_results + page_size - 1) // page_size

#     start = (page - 1) * page_size
#     end = start + page_size
#     paginated = documents[start:end]

#     for doc in paginated:
#         meta = await documents_search_client.get_document(doc.id)
#         doc.meta = DocumentMetaData(
#             title=meta["metadata_storage_name"],
#             url=meta["metadata_storage_path"],
#             content_type=meta["metadata_storage_content_type"],
#             last_modified=datetime.fromisoformat(
#                 meta["metadata_storage_last_modified"]
#             ),
#             storage_size=meta["metadata_storage_size"],
#         )

#     return {
#         "query": query,
#         "page": page,
#         "page_size": page_size,
#         "total_results": total_results,
#         "total_pages": total_pages,
#         "results": paginated,
#     }


# async def text_search(query: str) -> list[SearchHit]:
#     results = await chunks_search_client.search(
#         query,
#         query_type=QueryType.SIMPLE,
#         search_mode=SearchMode.ANY,
#     )

#     hits = [
#         SearchHit(id=item["document_id"], bm25=item["@search.score"])
#         async for item in results
#     ]

#     return hits


# async def vector_search(query: str) -> list[SearchHit]:
#     query = query if len(query) <= TOKEN_LIMIT else query[:TOKEN_LIMIT]
#     response = await embedder.embeddings.create(
#         input=query, model=SETTINGS.search_embedding_model
#     )
#     embedding = response.data[0].embedding

#     results = await chunks_search_client.search(
#         vector_queries=[VectorizedQuery(fields="embedding", vector=embedding)]
#     )

#     hits = [
#         SearchHit(id=item["document_id"], cosine=item["@search.score"])
#         async for item in results
#     ]

#     return hits

from contextlib import asynccontextmanager
from datetime import datetime

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import QueryType, SearchMode, VectorizedQuery
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncAzureOpenAI

from src.settings import SETTINGS
from src.setup import setup
from src.utils import DocumentMetaData, SearchHit, Passage, aggregate_search_hits

documents_search_client = SearchClient(
    SETTINGS.search_endpoint,
    SETTINGS.search_document_index_name,
    AzureKeyCredential(SETTINGS.search_key),
)

chunks_search_client = SearchClient(
    SETTINGS.search_endpoint,
    SETTINGS.search_chunk_index_name,
    AzureKeyCredential(SETTINGS.search_key),
)

embedder = AsyncAzureOpenAI(
    azure_endpoint=SETTINGS.ai_endpoint,
    api_key=SETTINGS.ai_key,
    api_version=SETTINGS.ai_api_version,
)

TOKEN_LIMIT = 2048

def make_snippet(text: str, query: str, window: int = 400) -> str:
    t = text or ""
    q = (query or "").strip()
    if not t:
        return ""
    if not q:
        return t[:window] + ("..." if len(t) > window else "")

    idx = t.lower().find(q.lower())
    if idx == -1:
        return t[:window] + ("..." if len(t) > window else "")

    start = max(0, idx - window // 2)
    end = min(len(t), idx + window // 2)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(t) else ""
    return prefix + t[start:end] + suffix



@asynccontextmanager
async def lifespan(app: FastAPI):
    await setup()
    yield
    await documents_search_client.close()
    await chunks_search_client.close()
    await embedder.close()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _recall_k(page: int, page_size: int) -> int:
    # Lấy nhiều hơn để hybrid + paginate không bị hụt
    return max(50, page * page_size * 5)


@app.get("/")
async def search(query: str, page: int = 1, page_size: int = 10):
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 10

    k = _recall_k(page, page_size)

    text_search_hits = await text_search(query, top_k=k)
    vector_search_hits = await vector_search(query, top_k=k)

    documents = aggregate_search_hits([*text_search_hits, *vector_search_hits])

    total_results = len(documents)
    total_pages = (total_results + page_size - 1) // page_size

    start = (page - 1) * page_size
    end = start + page_size
    paginated = documents[start:end]

    # Enrich metadata cho từng document
    for doc in paginated:
        meta = await documents_search_client.get_document(doc.id)
        doc.meta = DocumentMetaData(
            title=meta["metadata_storage_name"],
            url=meta["metadata_storage_path"],
            content_type=meta["metadata_storage_content_type"],
            last_modified=datetime.fromisoformat(meta["metadata_storage_last_modified"]),
            storage_size=meta["metadata_storage_size"],
        )

    return {
        "query": query,
        "page": page,
        "page_size": page_size,
        "total_results": total_results,
        "total_pages": total_pages,
        "results": paginated,
    }


# Alias endpoint cho dễ dùng như tool
@app.get("/search")
async def search_alias(query: str, page: int = 1, page_size: int = 10):
    return await search(query=query, page=page, page_size=page_size)


async def text_search(query: str, top_k: int = 50) -> list[SearchHit]:
    results = await chunks_search_client.search(
        search_text=query,
        query_type=QueryType.SIMPLE,
        search_mode=SearchMode.ANY,
        top=top_k,
        # chunk index schema: id, document_id, text, embedding
        select=["id", "document_id", "text"],
    )

    hits: list[SearchHit] = []
    async for item in results:
        bm25_score = float(item["@search.score"])
        hits.append(
            SearchHit(
                id=item["document_id"],
                bm25=bm25_score,
                passages=[
                    Passage(
                        chunk_id=item["id"],  # chunk key field trong index
                        text=make_snippet(item.get("text", "") or "", query),
                        score=bm25_score,
                        retriever="bm25",
                    )
                ],
            )
        )
    return hits


async def vector_search(query: str, top_k: int = 50) -> list[SearchHit]:
    query = query if len(query) <= TOKEN_LIMIT else query[:TOKEN_LIMIT]

    response = await embedder.embeddings.create(
        input=query,
        model=SETTINGS.search_embedding_model,
    )
    embedding = response.data[0].embedding

    results = await chunks_search_client.search(
        search_text=None,
        vector_queries=[
            VectorizedQuery(
                fields="embedding",
                vector=embedding,
                k_nearest_neighbors=top_k,
            )
        ],
        top=top_k,
        select=["id", "document_id", "text"],
    )

    hits: list[SearchHit] = []
    async for item in results:
        vec_score = float(item["@search.score"])
        hits.append(
            SearchHit(
                id=item["document_id"],
                cosine=vec_score,
                passages=[
                    Passage(
                        chunk_id=item["id"],
                        text=make_snippet(item.get("text", "") or "", query),
                        score=vec_score,
                        retriever="vector",
                    )
                ],
            )
        )
    return hits
