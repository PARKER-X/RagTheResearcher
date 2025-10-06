import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import load_and_chunk_pdf,embed_texts
from vector_db import QdrantStorage
from custom_types import *

# env 
load_dotenv()

# Ingest client
inngest_client = inngest.Inngest(
    app_id="rag_the_researcher",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)


# Create an Inngest function
@inngest_client.create_function(
    fn_id="Rag: Ingest Pdf",
    # Event that triggers this function
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context) -> str:
    ctx.logger.info(ctx.event)
    return "done"


app=FastAPI()

# Serve the Inngest endpoint
inngest.fast_api.serve(app, inngest_client, [])

