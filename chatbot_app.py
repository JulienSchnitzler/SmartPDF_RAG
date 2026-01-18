import os

import streamlit as st

try:
    from langfuse.callback import CallbackHandler
    HAS_LANGFUSE = True
except ImportError:
    HAS_LANGFUSE = False

from rag_pipeline import (
    MANUAL_PROMPT_TEMPLATE,
    PDF_FOLDER_PATH,
    initialize_rag_pipeline,
)

st.set_page_config(page_title="SmartPDF - RAG Pro (BM25)", page_icon="🚀", layout="wide")

# --- Configuration de la reformulation ---
REWRITE_PROMPT = """
Sur la base de l'historique de la conversation et de la dernière
question de l'utilisateur, reformule une question autonome qui 
peut être comprise sans l'historique. 
Cette question servira à faire une recherche dans des documents PDF.

Historique :
{history}

Dernière question : {question}

Question reformulée (sois précis et direct) :"""

# --- Barre latérale : Gestion des documents et Aide ---
with st.sidebar:
    st.title("📁 Documents")
    os.makedirs(PDF_FOLDER_PATH, exist_ok=True)

    uploaded_files = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        for f in uploaded_files:
            file_path = os.path.join(PDF_FOLDER_PATH, f.name)
            with open(file_path, "wb") as file:
                file.write(f.getbuffer())
        st.success("PDF prêts.")

    st.divider()

    if st.sidebar.button("🔄 Ré-indexer les PDFs", use_container_width=True):
        st.cache_resource.clear()
        initialize_rag_pipeline(force_reindex=True)
        st.success("Index mis à jour !")
        st.rerun()

    st.divider()

    # --- BLOC D'EXPLICATION BM25 ---
    st.info(
        """
        **💡 Score BM25 (Confiance) :**
        * **> 10** : Très pertinent ✅
        * **Bas (< 2)** : Peu précis / Aléatoire ⚠️
        
        *Plus le score est élevé, plus la source est fiable.*
        """
    )


st.title("🤖 Assistant RAG Intelligent")

# Initialisation du pipeline
llm, retriever = initialize_rag_pipeline()

# Récupération des clés
pk = os.getenv("LANGFUSE_PUBLIC_KEY")
sk = os.getenv("LANGFUSE_SECRET_KEY")
host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# N'active le handler que si la librairie ET les clés sont présentes
if HAS_LANGFUSE and pk and sk:
    langfuse_handler = CallbackHandler(public_key=pk, secret_key=sk, host=host)
    st.sidebar.caption("✅ Monitoring Langfuse actif")
else:
    langfuse_handler = None
    st.sidebar.caption("ℹ️ Monitoring Langfuse désactivé (ou module absent)")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage du chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Logique Principale ---
if prompt := st.chat_input("Votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not llm or not retriever:
            st.error("Pipeline non prêt.")
        else:
            try:
                # --- ÉTAPE 1 : RÉ-ÉCRITURE DE LA REQUÊTE ---
                search_query = prompt
                history_str = ""
                if len(st.session_state.messages) > 1:
                    past_messages = st.session_state.messages[-4:-1]
                    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in past_messages])

                    rewrite_input = REWRITE_PROMPT.format(history=history_str, question=prompt)


                    # Initialisation d'une config vide (puisque Langfuse est retiré)
                    #config = {"callbacks": [langfuse_handler]} if langfuse_handler else {}
                    #config = {}
                    #rewrite_res = llm.invoke(rewrite_input, config=config)
                    rewrite_res = llm.invoke(rewrite_input)

                    # Extraction sécurisée du texte
                    if isinstance(rewrite_res.content, list):
                        search_query = rewrite_res.content[0].get("text", str(rewrite_res.content[0]))
                    else:
                        search_query = rewrite_res.content

                # ASTUCE PRO : Affichage de la requête de recherche générée
                st.caption(f"🔍 **Requête de recherche optimisée :** *{search_query}*")

                # --- ÉTAPE 2 : RECHERCHE AVEC LA REQUÊTE OPTIMISÉE ---
                with st.spinner("Recherche dans les documents..."):
                    hits = retriever(search_query, k=4)
                    doc_context = "\n---\n".join([h["text"] for h in hits])

                # --- ÉTAPE 3 : GÉNÉRATION DE LA RÉPONSE FINALE ---
                with st.spinner("Rédaction de la réponse..."):
                    combined_context = f"[HISTORIQUE]\n{history_str}\n\n[DOCUMENTS]\n{doc_context}"
                    final_prompt = MANUAL_PROMPT_TEMPLATE.format(context=combined_context, question=prompt)
                    #response = llm.invoke(final_prompt, config=config)
                    response = llm.invoke(final_prompt)

                    # Extraction sécurisée pour la réponse finale
                    if isinstance(response.content, list):
                        answer = response.content[0].get("text", str(response.content[0]))
                    else:
                        answer = response.content

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    # AFFICHAGE DES SCORES DANS LES SOURCES
                    if hits:
                        with st.expander("🔍 Sources consultées & Pertinence"):
                            for h in hits:
                                score = h.get("score", 0)
                                source_name = os.path.basename(h["meta"]["source"])
                                page_num = h["meta"]["page"] + 1

                                st.write(f"📄 **{source_name}** (Page {page_num})")
                                # Affichage du score avec un code couleur simple
                                st.code(f"Score BM25 : {score:.2f}", language="markdown")
                                st.text(f"“{h['text'][:200]}…”")
                                st.divider()

            except Exception as e:
                st.error(f"Erreur : {e}")
