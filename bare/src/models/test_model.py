"""The purpose of this file is to easily ensure new models checked into 
the LLMEngine are working as expected. To run this from the root dir:

make test_model

"""

from src.models.llm_engine import EmbeddingEngine, LLMEngine


def test_model():
    model = LLMEngine(model_name="vllm/llama-3.1-70b-base")
    engine_response = model.get_response(
        [
            {
                "role": "user",
                "content": "Fire truck : red. Tree : green. Sky : blue. Corn : ",
            }
        ]
    )
    print(f"Response:\n{engine_response.response}\n****\nCost: {engine_response.cost}")


def test_embedding_model():
    model = EmbeddingEngine(model_name="text-embedding-ada-002")
    engine_response = model.get_embedding("Is Ottawa the capital of Canada?")
    print(f"Embedding: {engine_response.embedding}")
    print(f"Cost: {engine_response.cost}")


if __name__ == "__main__":
    test_model()
    # test_embedding_model()
