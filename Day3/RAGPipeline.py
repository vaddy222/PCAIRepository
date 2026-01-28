class RAGPipeline:
    def __init__(self, retriever, gpt_client, reranker_client, topic_control_client, topic_control_model):
        self.retriever = retriever
        self.gpt_client = gpt_client
        self.reranker_client = reranker_client
        self.topic_control_client = topic_control_client
        self.topic_control_model = topic_control_model
        self.topic_control_prompt = ""

    def query(self, user_input, enable_topic_control=False, enable_rerank=False):
        # 1. Topic Control
        if enable_topic_control:
            print("Running topic control check...")
            # (Your guardrail logic here)

        # 2. Retrieval
        docs = self.retriever.invoke(user_input)

        # 3. Optional Reranking
        if enable_rerank and self.reranker_client:
            print("Reranking documents...")
            # Note: The syntax depends on your specific Reranker client implementation
            docs = self.reranker_client.compress_documents(query=user_input, documents=docs)

        # 4. Final Generation
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"Use the context below to answer: {context}\n\nQuestion: {user_input}"
        response = self.gpt_client.invoke(prompt)
        
        return response.content