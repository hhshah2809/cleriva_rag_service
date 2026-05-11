from langchain_community.vectorstores import SupabaseVectorStore
from supabase import create_client
from core.config import SUPABASE_URL, SUPABASE_KEY, SUPABASE_TABLE, SUPABASE_QUERY_NAME
from core.embeddings import embeddings

supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Export a ready-to-use `vectorstore` instance for the rest of the app to import.
vectorstore = SupabaseVectorStore(
    client=supabase_client,
    embedding=embeddings,
    table_name=SUPABASE_TABLE,
    query_name=SUPABASE_QUERY_NAME,
)