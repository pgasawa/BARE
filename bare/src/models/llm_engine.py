import litellm
from load_dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()
litellm.set_verbose = False
litellm.suppress_debug_info = True

# For adding shorthand forms for non-OAI models, using vLLM hosted models
MODEL_MAP = {
    "claude-3.5-sonnet": "anthropic/claude-3-5-sonnet-20241022",  # Example
    "vllm/llama-3.1-8b-base": "text-completion-openai/meta-llama/Llama-3.1-8B",  # Example
    "vllm/llama-3.1-8b-instruct": "openai/meta-llama/Llama-3.1-8B-Instruct",  # Example
}
API_BASE_MAP = {
    "vllm/llama-3.1-8b-base": "http://34.83.38.89:5152/v1/",  # Example: ip subject to change
    "vllm/llama-3.1-8b-instruct": "http://35.192.184.95:5152/v1/",  # Example: ip subject to changect to change
}
VLLM_SK = "example-key"  # Example
API_KEY_OVERRIDE = {
    "vllm/llama-3.1-8b-base": VLLM_SK,
    "vllm/llama-3.1-8b-instruct": VLLM_SK,
}


class LLMResponseObject:
    def __init__(self, response: str, cost: float):
        self.response = response
        self.cost = cost


class LLMEngine:
    """Class for interacting with LLM calls."""

    def __init__(self, model_name: str, system_prompt: str = ""):
        self.model_name = (
            MODEL_MAP[model_name] if model_name in MODEL_MAP else model_name
        )
        self.api_base = API_BASE_MAP[model_name] if model_name in API_BASE_MAP else None
        self.api_key = (
            API_KEY_OVERRIDE[model_name] if model_name in API_KEY_OVERRIDE else None
        )
        if system_prompt:
            self.system_prompt = {"role": "system", "content": system_prompt}
        else:
            self.system_prompt = None
        self.params = {}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=8),
        retry=retry_if_exception_type((TimeoutError, Exception)),
        reraise=True,
    )
    def get_response(
        self, messages: list[dict], timeout: int = 300
    ) -> LLMResponseObject:
        """Get response from LLM engine.

        Args:
            messages (list[dict]): The messages to send to the LLM engine.
            timeout (int, optional): Timeout in seconds. Defaults to 30.

        Returns:
            LLMResponseObject: The response from the LLM engine.

        Raises:
            TimeoutError: If the request times out after retries
            Exception: For other types of errors
        """
        if self.system_prompt is not None:
            messages = [self.system_prompt] + messages

        if "text-completion-openai" in self.model_name:
            timeout = 300

        try:
            litellm_response = litellm.completion(
                model=self.model_name,
                messages=messages,
                api_base=self.api_base,
                api_key=self.api_key,
                max_tokens=1000,
                timeout=timeout,
                **self.params,
            )
        except Exception as e:
            if "timeout" in str(e).lower():
                raise TimeoutError(f"Request timed out after {timeout} seconds") from e
            raise

        response = litellm_response["choices"][0]["message"]["content"]
        cost = litellm_response._hidden_params["response_cost"]
        if cost is None:
            cost = 0

        return LLMResponseObject(response=response, cost=cost)

    def set_params(self, **kwargs):
        """Set parameters for LLM engine."""
        for arg in kwargs:
            self.params[arg] = kwargs[arg]


class EmbeddingResponseObject:
    def __init__(self, embedding: list[float], cost: float):
        self.embedding = embedding
        self.cost = cost


class EmbeddingEngine:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def get_embedding(self, text: str):
        response = litellm.embedding(model=self.model_name, input=[text])
        return EmbeddingResponseObject(
            response["data"][0]["embedding"], response._hidden_params["response_cost"]
        )
