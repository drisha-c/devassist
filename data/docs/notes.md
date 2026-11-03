\# Deployment



DevAssist is deployed on Render's free tier at devassist-1.onrender.com. The free tier sleeps after 15 minutes of no traffic, so the first request afterwards takes 30 to 60 seconds. Every push to the main branch on GitHub triggers an automatic redeploy.



\# Secrets



The Gemini API key lives in a local .env file, which is listed in .gitignore so it never reaches GitHub. On Render the key is set as the GEMINI\_API\_KEY environment variable in the service settings.



\# Dependency conflict



The first Render build failed because requirements.txt was missing. The second failed with a grpcio-status version conflict on Python 3.14. The fix was a minimal requirements.txt listing only fastapi, uvicorn, google-genai and python-dotenv.



\# Error handling



Google sometimes returns a 503 error saying the model is under high demand. The ask endpoint retries up to 3 times, waiting 2 seconds and then 4 seconds, and returns a clean 503 message if all attempts fail.



\# Model names



The model gemini-2.5-flash was deprecated for new users. The project uses gemini-3.6-flash for answers and gemini-embedding-001 for embeddings.





\# Retrieval limitations



Testing showed that natural language questions about the logs retrieve correctly, for example asking about a rate limit error correctly found the relevant log chunk. However, exact value lookups such as asking what happened at a specific timestamp often fail, because embeddings capture semantic meaning rather than exact literal values like numbers or timestamps. A production system would likely combine semantic search with keyword or structured filtering for this kind of query.

