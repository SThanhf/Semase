import logging
from datetime import timedelta

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.search.documents.indexes.aio import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    AzureOpenAIEmbeddingSkill,
    HnswAlgorithmConfiguration,
    HnswParameters,
    IndexingSchedule,
    IndexProjectionMode,
    InputFieldMappingEntry,
    OutputFieldMappingEntry,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchIndexer,
    SearchIndexerDataContainer,
    SearchIndexerDataSourceConnection,
    SearchIndexerIndexProjection,
    SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters,
    SearchIndexerSkillset,
    SimpleField,
    SplitSkill,
    VectorSearch,
    VectorSearchProfile,
)
from src.settings import SETTINGS

logger = logging.getLogger("uvicorn")

DOCUMENT_INDEX_SCHEMA = SearchIndex(
    name=SETTINGS.search_document_index_name,
    fields=[
        SimpleField(
            name="id", type=SearchFieldDataType.String, key=True, filterable=True
        ),
        SimpleField(
            name="metadata_storage_path",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SimpleField(
            name="metadata_storage_name",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="metadata_storage_content_type",
            type=SearchFieldDataType.String,
            filterable=True,
        ),
        SimpleField(
            name="metadata_storage_last_modified",
            type=SearchFieldDataType.DateTimeOffset,
            filterable=True,
            sortable=True,
        ),
        SimpleField(
            name="metadata_storage_size",
            type=SearchFieldDataType.Int64,
            filterable=True,
            sortable=True,
        ),
    ],
)

CHUNK_INDEX_SCHEMA = SearchIndex(
    name=SETTINGS.search_chunk_index_name,
    fields=[
        SearchField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
            analyzer_name="keyword",
        ),
        SimpleField(
            name="document_id", type=SearchFieldDataType.String, filterable=True
        ),
        SearchableField(
            name="text", type=SearchFieldDataType.String, analyzer_name="en.lucene"
        ),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=SETTINGS.search_embedding_dim,
            vector_search_profile_name="vs-default",
        ),
    ],
    vector_search=VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-cosine",
                parameters=HnswParameters(
                    metric="cosine", m=16, ef_construction=400, ef_search=100
                ),
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="vs-default", algorithm_configuration_name="hnsw-cosine"
            )
        ],
    ),
)

DATA_SOURCE = SearchIndexerDataSourceConnection(
    name=SETTINGS.search_data_source_name,
    type="azureblob",
    connection_string=SETTINGS.search_data_source_connection,
    container=SearchIndexerDataContainer(name=SETTINGS.search_data_container_name),
)

SKILLSET = SearchIndexerSkillset(
    name=SETTINGS.search_skillset_name,
    skills=[
        SplitSkill(
            inputs=[InputFieldMappingEntry(name="text", source="/document/content")],
            outputs=[OutputFieldMappingEntry(name="textItems", target_name="pages")],
            context="/document",
            text_split_mode="pages",
            maximum_page_length=2000,
            page_overlap_length=200,
        ),
        AzureOpenAIEmbeddingSkill(
            inputs=[InputFieldMappingEntry(name="text", source="/document/pages/*")],
            outputs=[
                OutputFieldMappingEntry(name="embedding", target_name="embedding")
            ],
            context="/document/pages/*",
            resource_url=SETTINGS.ai_endpoint,
            api_key=SETTINGS.ai_key,
            deployment_name=SETTINGS.search_embedding_model,
            model_name=SETTINGS.search_embedding_model,
            dimensions=SETTINGS.search_embedding_dim,
        ),
    ],
    index_projection=SearchIndexerIndexProjection(
        selectors=[
            SearchIndexerIndexProjectionSelector(
                target_index_name=SETTINGS.search_chunk_index_name,
                parent_key_field_name="document_id",
                source_context="/document/pages/*",
                mappings=[
                    InputFieldMappingEntry(name="text", source="/document/pages/*"),
                    InputFieldMappingEntry(
                        name="embedding", source="/document/pages/*/embedding"
                    ),
                ],
            )
        ],
        parameters=SearchIndexerIndexProjectionsParameters(
            projection_mode=IndexProjectionMode.INCLUDE_INDEXING_PARENT_DOCUMENTS
        ),
    ),
)

INDEXER = SearchIndexer(
    name=SETTINGS.search_indexer_name,
    data_source_name=SETTINGS.search_data_source_name,
    target_index_name=SETTINGS.search_document_index_name,
    skillset_name=SETTINGS.search_skillset_name,
    schedule=IndexingSchedule(interval=timedelta(days=1)),
)


async def setup():
    index_client = SearchIndexClient(
        SETTINGS.search_endpoint, credential=AzureKeyCredential(SETTINGS.search_key)
    )

    indexer_client = SearchIndexerClient(
        SETTINGS.search_endpoint, credential=AzureKeyCredential(SETTINGS.search_key)
    )

    try:
        _ = await index_client.get_index(SETTINGS.search_document_index_name)
        _ = await index_client.get_index(SETTINGS.search_chunk_index_name)
        logger.info("Azure AI Search's indexes are healthy")
    except HttpResponseError as e:
        logger.info(str(e))
        logger.info("Setting up Azure AI Search ...")
        try:
            await index_client.delete_index(DOCUMENT_INDEX_SCHEMA)
            await index_client.delete_index(CHUNK_INDEX_SCHEMA)

            await indexer_client.delete_indexer(INDEXER)
            await indexer_client.delete_skillset(SKILLSET)
            await indexer_client.delete_data_source_connection(DATA_SOURCE)

            await index_client.create_index(DOCUMENT_INDEX_SCHEMA)
            await index_client.create_index(CHUNK_INDEX_SCHEMA)

            await indexer_client.create_data_source_connection(DATA_SOURCE)
            await indexer_client.create_skillset(SKILLSET)
            await indexer_client.create_indexer(INDEXER)
            logger.info("Completed")
        except Exception as e:
            logger.error(str(e))

    await index_client.close()
    await indexer_client.close()
